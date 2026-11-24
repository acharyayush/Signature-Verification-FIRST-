"""Stage 2 preprocessing step: grayscale normalization + aspect-aware resize.

Input:  a signature image (grayscale or BGR, any size, with a white or whitish
        background), i.e. the output of the upstream YOLO crop and artifact
        cleaning.
Output: a fixed-size ``(height, width)`` uint8 grayscale image, dark ink on a
        white background, at the agreed wide aspect ratio. Stroke intensity is
        preserved and strokes are never stretched.

``preprocess_signature`` runs the whole step; the individual functions are exposed for
testing and ablations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

# Agreed input size (width, height): 2:1 and a multiple of 32 (ConvNeXt total stride).
# 384 x 192 is the default. 512 x 256 was tried first to avoid downscaling typical ink
# crops (a 3.5 px stroke became ~2.1 px there), but it is reverted here to keep the
# cheaper canvas. Both are multiples of 32, so the fully-convolutional backbone needs no
# architectural change either way.
DEFAULT_TARGET_SIZE = (384, 192)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

InkPolarity = Literal["auto", "light", "dark"]  # retained for CLI compatibility
# How the paper background is handled. ``whiten`` (default) forces the estimated
# background to pure white while leaving every ink pixel byte-for-byte unchanged;
# ``flatten`` is the previous flat-field correction, kept for ablation.
BackgroundMode = Literal["whiten", "flatten"]


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D array")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("image must have 1 or 3 channels")


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Whiten slowly varying background (shadows, lighting gradients).

    The input must have dark ink on a lighter background. The background is
    estimated by morphological closing with a kernel larger than the stroke
    width, then smoothed. Returns an image where the background is ~255 and
    grayscale ink contrast is preserved.
    """
    k = max(15, (min(gray.shape[:2]) // 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), k / 2)

    # Subtract the estimated background variation without quantizing the ink.
    # Pixels matching the local background are explicitly made pure white.
    background_gap = background.astype(np.int16) - gray.astype(np.int16)
    corrected = np.clip(255 - background_gap, 0, 255).astype(np.uint8)
    corrected[background_gap <= 8] = 255
    return corrected


def whiten_background(
    gray: np.ndarray,
    *,
    background_gap_max: int | None = None,
    minimum_gap: int = 8,
    relative_gap: float = 0.08,
) -> np.ndarray:
    """Force the paper background to pure white, preserving the ink grayscale exactly.

    The local background is estimated with the same morphological closing used by
    :func:`flatten_illumination`, so uneven illumination and shadows are tracked and
    whitened. Ink pixels are returned **unchanged**: the input is copied and only pixels
    within the tolerance of the local background are set to 255.

    Scanned paper is grey and textured rather than pure white, so a fixed tolerance leaves a
    speckled background. Measured on CEDAR the paper sits a median of 3.4 and a 99th
    percentile of 19.9 grey levels below its local estimate, while BHSig260's paper is already
    exact. The default tolerance therefore scales with the local background level
    (``relative_gap``) with a floor (``minimum_gap``), which whitens ~8x more background than a
    fixed cut of 8 at no cost to ink: measured across all three datasets, ink retention stays
    at 100% and the leftover non-white area falls from 3.4% to 0.5%.

    Args:
        gray: Single-channel uint8 image with dark ink on a lighter background.
        background_gap_max: Fixed tolerance in grey levels. ``None`` (default) derives the
            tolerance from the local background instead.
        minimum_gap: Floor for the derived tolerance, so a clean image is unaffected.
        relative_gap: Fraction of the local background used as the derived tolerance.

    Limitation:
        The local background is a morphological estimate, so a **solid ink region wider than
        the closing kernel** (``max(15, min(shape) // 40)``, i.e. ~15 px at these sizes) is
        estimated as background and whitened away. Thin strokes -- every signature in these
        datasets -- are unaffected, and ink retention was verified at 100% across all three.
        If a future dataset contains filled seals or marker-thick strokes, enlarge the kernel
        or mask the ink with a global threshold first.
    """
    k = max(15, (min(gray.shape[:2]) // 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), k / 2)

    gap = background.astype(np.int16) - gray.astype(np.int16)
    if background_gap_max is None:
        tolerance = np.maximum(minimum_gap, relative_gap * background)
    else:
        tolerance = background_gap_max
    corrected = gray.copy()
    corrected[gap <= tolerance] = 255
    return corrected


def to_grayscale(
    image: np.ndarray,
    ink: InkPolarity = "auto",
    blur_ksize: int = 0,
    remove_shadow: bool = True,
    background_mode: BackgroundMode = "whiten",
) -> np.ndarray:
    """Normalize to grayscale dark ink on a white background without binarizing.

    Args:
        image: Grayscale or BGR uint8 image.
        ink: Retained for compatibility with existing callers. The preprocessing
            assumes dark ink on a white or whitish background and does not inspect
            or invert ink polarity.
        blur_ksize: Odd Gaussian kernel size applied before resizing. ``0``
            disables it so pressure-dependent intensity is retained.
        remove_shadow: Whitening is applied when true. When false only pixels already
            at/above 245 are set to 255, which removes a residual tint but does not
            correct illumination.
        background_mode: ``"whiten"`` (default) forces the estimated paper background to
            pure white and leaves ink pixels byte-for-byte unchanged, preserving the true
            signature grayscale. ``"flatten"`` applies the previous flat-field correction,
            which rescales ink by the local background; it is kept for ablation only.
    """
    gray = _to_gray(image).copy()
    # Ink polarity is intentionally not checked or inverted. Inputs are expected
    # to have dark signatures on a white or whitish background.
    if remove_shadow:
        gray = whiten_background(gray) if background_mode == "whiten" else flatten_illumination(gray)
    else:
        # Also support callers that skip illumination correction while removing
        # small residual tint from an otherwise white background.
        gray[gray >= 245] = 255
    if blur_ksize:
        if blur_ksize % 2 == 0:
            raise ValueError("blur_ksize must be odd")
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    return gray

def crop_to_ink(
    image: np.ndarray,
    margin: float = 0.05,
    ink_threshold: int = 200,
    min_ink_pixels: int = 5,
) -> np.ndarray:
    """Crop a grayscale dark-ink-on-white image to the ink box plus a margin.

    Args:
        image: Single-channel uint8 image with dark ink on a white background.
        margin: Padding around the ink box, as a fraction of the box's larger side.
        ink_threshold: Pixels below this value count as ink for locating the
            signature. The original grayscale values are returned unchanged.
        min_ink_pixels: If fewer ink pixels are found the image is treated as empty
            and ``ValueError`` is raised, so the caller can route it to review.
    """
    mask = image < ink_threshold
    # Faint scans may have no pixels below the threshold; relax it step by step.
    for relaxed in (225, 240):
        if int(mask.sum()) >= min_ink_pixels:
            break
        mask = image < relaxed
    if int(mask.sum()) < min_ink_pixels:
        raise ValueError("no signature ink found in image")

    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(round(margin * max(x1 - x0, y1 - y0)))
    h, w = image.shape[:2]
    return image[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)]


def aspect_aware_resize(
    image: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    pad_value: int = 255,
) -> np.ndarray:
    """Scale ``image`` to fit inside ``target_size`` and pad it to the exact size.

    The aspect ratio is preserved: the image is scaled uniformly, centered, and the
    remaining area is filled with ``pad_value`` (255 for normalized grayscale
    images).

    Args:
        image: ``H x W`` or ``H x W x C`` uint8 array.
        target_size: ``(width, height)`` of the output canvas.
        pad_value: Background value used for padding.

    Returns:
        Array of shape ``(height, width)`` (2D input) or ``(height, width, C)``.
    """
    if image is None or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D array")

    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = max(1, min(target_w, round(w * scale)))
    new_h = max(1, min(target_h, round(h * scale)))

    # INTER_AREA avoids aliasing of thin strokes when shrinking.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas_shape = (target_h, target_w) + image.shape[2:]
    canvas = np.full(canvas_shape, pad_value, dtype=image.dtype)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def fit_to_canvas(
    image: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    *,
    margin: float = 0.05,
    pad_value: int = 255,
) -> np.ndarray:
    """Crop to the ink, resize onto the canvas, and force the background to pure white.

    Resampling leaves a faint anti-aliasing halo around every stroke, so the canvas is
    whitened once more after the resize. That halo is the only thing standing between the
    background and a true 255: measured on CEDAR it removes ~55% of the remaining non-white
    area while keeping every ink pixel (``< 200``) unchanged, because the halo is lighter than
    the tolerance.

    :func:`preprocess_signature` and :func:`preprocessing.augmentation.augment_signature` both
    end with this step, so training and inference images are produced by the same path and
    cannot drift.

    Raises:
        ValueError: If the image contains no ink.
    """
    cropped = crop_to_ink(image, margin=margin)
    resized = aspect_aware_resize(cropped, target_size, pad_value=pad_value)
    if pad_value == 255:
        resized = whiten_background(resized)
    return resized


def preprocess_signature(
    image: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    ink: InkPolarity = "auto",
    margin: float = 0.05,
    blur_ksize: int = 0,
    binarize: bool = False,
    background_mode: BackgroundMode = "whiten",
) -> np.ndarray:
    """Grayscale normalization, tight crop, and aspect-aware resize.

    With ``binarize=True`` the image is Otsu-thresholded to pure black ink (0) on
    white (255) before cropping, and the resized result is re-thresholded so the
    output contains only 0 and 255.

    By default the signature's true grayscale is preserved and only the paper background
    is forced to pure white (:func:`whiten_background`). Pass ``background_mode="flatten"``
    to use the previous flat-field correction instead.

    The threshold is used only to locate the signature for cropping. The output
    remains grayscale, so stroke and pressure variations are preserved.

    Returns:
        ``target_h x target_w`` uint8 grayscale image, dark ink on a pure-white background.

    Raises:
        ValueError: If the input is invalid or contains no ink.
    """
    grayscale = to_grayscale(
        image, ink=ink, blur_ksize=blur_ksize, background_mode=background_mode
    )
    if binarize:
        _, grayscale = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    resized = fit_to_canvas(grayscale, target_size, margin=margin)
    if binarize:
        # Area resampling blurs thin strokes into gray; a high cut keeps them as ink.
        resized = np.where(resized < 200, 0, 255).astype(np.uint8)
    return resized


def load_preprocessed_image(
    path: str | Path,
    *,
    already_preprocessed: bool = False,
    ink: InkPolarity = "auto",
) -> np.ndarray:
    """Read one signature image and return the ``target_h x target_w`` uint8 grayscale.

    This is the single image-loading path shared by the training dataset and the
    handcrafted feature extractor, so both always see identical pixels. When
    ``already_preprocessed`` is true the file is expected to be a grayscale image at
    :data:`DEFAULT_TARGET_SIZE`; otherwise Stage 2.2 preprocessing is applied.

    Raises:
        FileNotFoundError: If the file cannot be read.
        ValueError: If a preprocessed image has the wrong shape, or the image has no ink.
    """
    image_path = Path(path)
    read_mode = cv2.IMREAD_GRAYSCALE if already_preprocessed else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), read_mode)
    if image is None:
        raise FileNotFoundError(f"could not read signature image: {image_path}")

    if already_preprocessed:
        expected_shape = (DEFAULT_TARGET_SIZE[1], DEFAULT_TARGET_SIZE[0])
        if image.shape != expected_shape:
            raise ValueError(
                f"preprocessed image {image_path} has shape {image.shape}; "
                f"expected {expected_shape}"
            )
        return image
    return preprocess_signature(image, ink=ink)


def to_model_input(image: np.ndarray) -> np.ndarray:
    """Convert a preprocessed uint8 image to a ``3 x H x W`` float32 ImageNet-normalized array.

    The single channel is replicated to 3 channels so ImageNet-pretrained weights can
    be used unchanged.
    """
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError("expected the single-channel output of preprocess_signature")

    array = np.repeat(image[..., None], 3, axis=2).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))
