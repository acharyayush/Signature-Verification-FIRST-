"""Training-time augmentation for Stage 2.2 preprocessed signatures.

Augmentation is applied only to the **training** split, on the ``uint8`` grayscale output
of :func:`preprocessing.signature_preprocessing.preprocess_signature` (or of an already
preprocessed ``256 x 512`` file), immediately before ``to_model_input``. It is never
applied to validation or test data: the checkpoint is selected by the validation EER
(``--select-metric eer``), so augmenting the validation split would corrupt the selector.

Why this particular set of transforms
-------------------------------------
Stage 2.2 is a *tight ink crop* followed by an *aspect-aware uniform resize* onto a fixed
canvas. Position and absolute scale are therefore deleted by the pipeline, so translating
or uniformly rescaling an image before it is cropped changes nothing the model can see.
The transforms that do survive the pipeline, and are implemented here, are:

* **Geometry** -- rotation (a cheque signature may be tilted), shear and a mild
  perspective (scan/photo skew), and elastic warping (natural handwriting wobble).
* **Stroke width** -- a grayscale min/max filter thickens or thins the ink, simulating a
  different pen (ballpoint vs marker). Intensity stays grayscale, so pressure detail is
  not lost.
* **Photometric** -- ink contrast and gamma (pen pressure, ink darkness), blur and sensor
  noise (scan quality). These are background-preserving: white is a fixed point, so the
  pure-white background cannot become grey.

The chain always ends by restoring a pure-white background
(:func:`preprocessing.signature_preprocessing.whiten_background`) and re-running the same
crop + resize the inference path uses, so an augmented image keeps the canvas geometry and
the white background the model is trained on. That is the train/serve consistency rule:
augmentation changes *how* a signature is drawn, never the representation contract.

Determinism
-----------
Every transform draws from a caller-supplied :class:`numpy.random.Generator`. The training
dataset seeds one generator per sample from ``(augment_seed, epoch, index)``, so a run is
reproducible byte for byte while the augmentation still varies across epochs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from preprocessing.signature_preprocessing import (
    DEFAULT_TARGET_SIZE,
    aspect_aware_resize,
    fit_to_canvas,
    whiten_background,
)

AugmentStrength = Literal["light", "standard", "strong"]

STRENGTHS: tuple[AugmentStrength, ...] = ("light", "standard", "strong")


@dataclass(frozen=True)
class AugmentationConfig:
    """Ranges for one augmentation draw. Every limit is symmetric about zero."""

    # Geometry
    # Cheque signatures are tilted by hand; +-20 deg covers the realistic range. Over-rotating
    # destroys the fine identity cues a verification model needs, so this is deliberately
    # moderate (the original design note suggested as little as +-5 deg).
    rotate_degrees: float = 20.0
    shear_degrees: float = 6.0
    perspective: float = 0.01
    # Perspective stacks a third geometric warp on rotation + shear, so it is occasional
    # rather than applied to every draw.
    perspective_probability: float = 0.3
    elastic_alpha: float = 4.0  # peak displacement in pixels
    elastic_sigma: float = 10.0
    elastic_probability: float = 0.3
    # Stroke width (pen)
    stroke_width_probability: float = 0.5
    stroke_width_kernel: int = 3
    # Photometric
    ink_contrast: float = 0.25
    # Gamma is off by default. It is a monotonic intensity remap with white as a fixed point,
    # exactly like ``ink_contrast``, so stacking both double-counts "different pen" while
    # adding no new axis. The field is kept for the ablation demo.
    gamma: float = 0.0
    blur_probability: float = 0.15
    blur_sigma: float = 1.2
    noise_probability: float = 0.15
    noise_std: float = 8.0
    # Representation contract
    fill: int = 255
    refit: bool = True
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE


# Indic scripts (BHSig260) carry meaning in connected strokes, so heavy elastic warping
# destroys the very structure that discriminates writers. The preset keeps it mild.
_DATASET_OVERRIDES: dict[str, AugmentationConfig] = {
    "bhsig": AugmentationConfig(
        elastic_alpha=2.5,
        elastic_sigma=8.0,
        elastic_probability=0.2,
        shear_degrees=5.0,
    ),
}


def augmentation_config(
    strength: AugmentStrength = "standard", dataset: str | None = None
) -> AugmentationConfig:
    """Return the recommended config for a strength preset and an optional dataset.

    ``dataset`` is matched case-insensitively against the presets in
    :data:`_DATASET_OVERRIDES` (for example ``"BHSig260"``). An unknown dataset keeps the
    strength preset unchanged, so an unseen script degrades to the default rather than
    failing.
    """
    if strength == "light":
        config = AugmentationConfig(
            rotate_degrees=10.0,
            shear_degrees=4.0,
            perspective=0.0,
            elastic_probability=0.0,
            stroke_width_probability=0.4,
            ink_contrast=0.15,
            blur_probability=0.1,
            noise_probability=0.1,
        )
    elif strength == "strong":
        config = AugmentationConfig(
            rotate_degrees=26.0,
            shear_degrees=8.0,
            perspective=0.02,
            perspective_probability=0.4,
            elastic_alpha=6.0,
            elastic_sigma=12.0,
            elastic_probability=0.4,
            stroke_width_probability=0.6,
            ink_contrast=0.35,
            blur_probability=0.3,
            noise_probability=0.3,
        )
    else:
        config = AugmentationConfig()

    if dataset:
        lowered = dataset.lower()
        for key, override in _DATASET_OVERRIDES.items():
            if key in lowered:
                # Only the geometric fields are overridden; the strength preset still
                # decides everything else.
                return AugmentationConfig(
                    **{
                        **config.__dict__,
                        **{
                            field: getattr(override, field)
                            for field in ("elastic_alpha", "elastic_sigma",
                                          "elastic_probability", "shear_degrees")
                        },
                    }
                )
    return config


def _apply_affine(
    gray: np.ndarray, angle_degrees: float, shear_degrees: float, *, fill: int
) -> np.ndarray:
    """Rotate and shear about the image centre, expanding the canvas to fit.

    The output is large enough to hold every transformed corner, so no ink is clipped
    before the caller re-crops. Border pixels are ``fill`` (white).
    """
    height, width = gray.shape[:2]
    center = np.array([width / 2.0, height / 2.0])

    angle = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    shear = np.array([[1.0, math.tan(math.radians(shear_degrees))], [0.0, 1.0]])
    linear = rotation @ shear

    corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64
    )
    transformed = (linear @ corners.T).T + (center - linear @ center)
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    out_width = int(math.ceil(float(maximum[0] - minimum[0])))
    out_height = int(math.ceil(float(maximum[1] - minimum[1])))

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = linear
    matrix[:, 2] = center - linear @ center - minimum
    return cv2.warpAffine(
        gray,
        matrix,
        (max(out_width, 1), max(out_height, 1)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _apply_perspective(
    gray: np.ndarray, fraction: float, rng: np.random.Generator, *, fill: int
) -> np.ndarray:
    """Jitter the four corners by up to ``fraction`` of the shorter side.

    The output keeps the input size. The Stage 2.2 crop already leaves a margin around the
    ink, so a mild jitter moves the border without clipping the signature.
    """
    height, width = gray.shape[:2]
    margin = fraction * min(height, width)
    source = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    destination = source + rng.uniform(-margin, margin, size=(4, 2)).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _elastic_offsets(
    height: int,
    width: int,
    alpha: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dx, dy)`` displacement fields whose peak magnitude is ``alpha`` pixels.

    The noise is smoothed with ``sigma`` and then normalised to unit peak before scaling, so
    ``alpha`` means "peak displacement in pixels". Without the normalisation the field scales
    as ``1/sqrt(sigma)`` and a nominal ``alpha`` of 1 moves the ink by only a fraction of a
    pixel -- i.e. an elastic transform that does nothing.
    """

    def _field() -> np.ndarray:
        noise = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
        smoothed = cv2.GaussianBlur(noise, (0, 0), sigma)
        peak = float(np.abs(smoothed).max())
        return smoothed / peak if peak > 0.0 else smoothed

    return _field() * alpha, _field() * alpha


def _apply_elastic(
    gray: np.ndarray,
    alpha: float,
    sigma: float,
    rng: np.random.Generator,
    *,
    fill: int,
) -> np.ndarray:
    """Warp with a smoothed random displacement field (elastic deformation).

    ``alpha`` is the peak displacement in pixels and ``sigma`` the smoothing of the field: a
    small ``sigma`` gives high-frequency wobble, a large one a smooth global bend.
    """
    height, width = gray.shape[:2]
    offset_x, offset_y = _elastic_offsets(height, width, alpha, sigma, rng)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return cv2.remap(
        gray,
        grid_x + offset_x,
        grid_y + offset_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _apply_stroke_width(
    gray: np.ndarray, kernel: int, rng: np.random.Generator
) -> np.ndarray:
    """Thicken or thin the ink with a grayscale min/max filter.

    A minimum filter (``cv2.erode``) spreads dark pixels outward, thickening the stroke; a
    maximum filter (``cv2.dilate``) shrinks them, thinning it. Grayscale intensity is
    preserved, so ``to_model_input`` needs no change and pen-pressure detail survives.
    """
    size = int(kernel) | 1
    if size < 3:
        return gray
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if rng.random() < 0.5:
        return cv2.erode(gray, element)
    return cv2.dilate(gray, element)


def _apply_ink_contrast(
    gray: np.ndarray, delta: float, rng: np.random.Generator
) -> np.ndarray:
    """Scale the ink-to-background distance, keeping white a fixed point.

    ``factor > 1`` darkens the ink (heavier pressure, darker pen); ``factor < 1`` lightens
    it (a fainter pen). Because the transform is ``255 - (255 - gray) * factor``, a pure
    white background maps to white exactly, so it cannot create a grey background. The
    factor is bounded, so the ink cannot be washed out to near-background either -- unlike a
    global brightness/contrast shift, which lifts dark ink into the background range.
    """
    factor = 1.0 + float(rng.uniform(-delta, delta))
    adjusted = 255.0 - (255.0 - gray.astype(np.float32)) * factor
    return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)


# A draw that leaves fewer than this many clearly-dark pixels has washed the signature out;
# the geometric stage is used instead, so a near-blank sample never reaches training.
MIN_INK_LEVEL = 180
MIN_INK_PIXELS = 50


def _ensure_dark_ink(augmented: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return ``fallback`` when the photometric/stroke stages washed the ink out.

    The representation contract is dark ink on a pure-white background. Thinning, blurring
    and lightening can combine to lift every ink pixel above the ink threshold; that would
    inject a nearly blank image into training. The geometric stage preserves ink contrast,
    so it is the safe fallback.
    """
    if int((augmented < MIN_INK_LEVEL).sum()) >= MIN_INK_PIXELS:
        return augmented
    return fallback


def _apply_gamma(gray: np.ndarray, delta: float, rng: np.random.Generator) -> np.ndarray:
    """Apply a gamma curve. White (255) is a fixed point, so the background is untouched."""
    exponent = 1.0 / max(1.0 + float(rng.uniform(-delta, delta)), 1e-6)
    table = np.clip(((np.arange(256) / 255.0) ** exponent) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, table)


def _apply_noise(gray: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian sensor noise; the background is re-whitened by the caller."""
    noisy = gray.astype(np.float32) + rng.normal(0.0, std, size=gray.shape)
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def augment_signature(
    image: np.ndarray,
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one augmented, re-fitted copy of a preprocessed signature.

    Args:
        image: ``uint8`` grayscale signature (dark ink on white), as produced by Stage 2.2.
        config: Ranges to draw from. See :class:`AugmentationConfig`.
        rng: Seeded generator; the only source of randomness, so a run is reproducible.

    Returns:
        ``target_h x target_w`` ``uint8`` grayscale image with a pure-white background. The
        canvas geometry matches the inference path exactly because the chain ends with the
        same :func:`~preprocessing.signature_preprocessing.crop_to_ink` +
        :func:`~preprocessing.signature_preprocessing.aspect_aware_resize` re-fit.

    Raises:
        ValueError: If ``image`` is not a valid uint8 grayscale signature.
    """
    gray = np.asarray(image)
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("augment_signature expects a non-empty uint8 grayscale image")
    original = gray

    if config.rotate_degrees or config.shear_degrees:
        gray = _apply_affine(
            gray,
            float(rng.uniform(-config.rotate_degrees, config.rotate_degrees)),
            float(rng.uniform(-config.shear_degrees, config.shear_degrees)),
            fill=config.fill,
        )
    if config.perspective > 0.0 and rng.random() < config.perspective_probability:
        gray = _apply_perspective(gray, config.perspective, rng, fill=config.fill)
    if config.elastic_probability > 0.0 and rng.random() < config.elastic_probability:
        gray = _apply_elastic(
            gray, config.elastic_alpha, config.elastic_sigma, rng, fill=config.fill
        )
    # The geometric stage preserves ink contrast, so it is the safe fallback if the
    # photometric stages below wash the signature out.
    geometric = gray

    if (
        config.stroke_width_probability > 0.0
        and rng.random() < config.stroke_width_probability
    ):
        gray = _apply_stroke_width(gray, config.stroke_width_kernel, rng)

    if config.ink_contrast:
        gray = _apply_ink_contrast(gray, config.ink_contrast, rng)
    if config.gamma:
        gray = _apply_gamma(gray, config.gamma, rng)
    if config.blur_probability > 0.0 and rng.random() < config.blur_probability:
        gray = cv2.GaussianBlur(gray, (0, 0), config.blur_sigma)
    if config.noise_probability > 0.0 and rng.random() < config.noise_probability:
        gray = _apply_noise(gray, config.noise_std, rng)
    gray = _ensure_dark_ink(gray, geometric)

    # Restore the representation contract: pure-white background, then the inference
    # crop + resize. ``fit_to_canvas`` is the same helper Stage 2.2 ends with, so the
    # augmented and the inference image cannot drift.
    gray = whiten_background(gray)
    if config.refit:
        try:
            gray = fit_to_canvas(gray, config.target_size, pad_value=config.fill)
        except ValueError:
            # A transform removed every ink pixel (should not happen for the ranges
            # above). Fall back to the untouched input rather than emitting a blank crop.
            return aspect_aware_resize(original, config.target_size, pad_value=config.fill)
    return gray
