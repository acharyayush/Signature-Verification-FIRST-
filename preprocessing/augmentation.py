"""Training-time augmentation for Stage 2.2 preprocessed signatures.

Augmentation is applied only to the **training** split, on the ``uint8`` grayscale output of
:func:`preprocessing.signature_preprocessing.preprocess_signature` (or of an already
preprocessed file), immediately before ``to_model_input``. It is never applied to validation
or test data: the checkpoint is selected by the validation EER, so augmenting the validation
split would corrupt the selector.

The four transforms
-------------------
One transform per real-world cause of variation, and nothing else:

===================  ==========================================================
transform            cause it models
===================  ==========================================================
rotation             the cheque signature is tilted by hand
stroke width         the signer used a different pen (ballpoint vs marker)
ink contrast         the signer pressed harder, or the ink is darker
blur                 the scan or photo is soft
===================  ==========================================================

Shear, perspective, elastic warping, gamma and sensor noise were removed. Gamma duplicated
ink contrast (both are white-preserving monotonic intensity remaps), and the others added
variation these datasets do not exhibit, so they widened the training distribution without
covering a new axis.

What is *not* augmented is just as important: Stage 2.2 already deletes position and
absolute scale (tight ink crop + aspect-aware uniform resize), so translating or rescaling an
image before the crop changes nothing the model can see.

The chain always ends by restoring a pure-white background
(:func:`preprocessing.signature_preprocessing.whiten_background`) and re-running the same
:func:`~preprocessing.signature_preprocessing.fit_to_canvas` the inference path uses, so an
augmented image keeps the canvas geometry and the white background the model is trained on.
That is the train/serve consistency rule: augmentation changes *how* a signature is drawn,
never the representation contract.

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

    # Rotation: the cheque signature is tilted by hand. +-20 deg covers the realistic range.
    # Over-rotating destroys the fine identity cues a verification model needs, so this is
    # deliberately moderate (the original design note suggested as little as +-5 deg).
    rotate_degrees: float = 20.0
    # Stroke width: the signer used a different pen.
    stroke_width_probability: float = 0.5
    stroke_width_kernel: int = 3
    # Ink contrast: pen pressure / ink darkness.
    ink_contrast: float = 0.25
    # Blur: scan quality.
    blur_probability: float = 0.15
    blur_sigma: float = 1.2
    # Representation contract.
    fill: int = 255
    refit: bool = True
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE


def augmentation_config(strength: AugmentStrength = "standard") -> AugmentationConfig:
    """Return the recommended config for a strength preset."""
    if strength == "light":
        return AugmentationConfig(
            rotate_degrees=10.0,
            stroke_width_probability=0.4,
            ink_contrast=0.15,
            blur_probability=0.1,
        )
    if strength == "strong":
        return AugmentationConfig(
            rotate_degrees=26.0,
            stroke_width_probability=0.6,
            ink_contrast=0.35,
            blur_probability=0.3,
        )
    return AugmentationConfig()


def _apply_rotation(gray: np.ndarray, angle_degrees: float, *, fill: int) -> np.ndarray:
    """Rotate about the image centre, expanding the canvas so no ink is clipped.

    The output is large enough to hold every transformed corner, so nothing is lost before
    the caller re-crops. Border pixels are ``fill`` (white).
    """
    height, width = gray.shape[:2]
    center = np.array([width / 2.0, height / 2.0])

    angle = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    linear = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

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

    ``factor > 1`` darkens the ink (heavier pressure, darker pen); ``factor < 1`` lightens it
    (a fainter pen). Because the transform is ``255 - (255 - gray) * factor``, a pure white
    background maps to white exactly, so it cannot create a grey background. The factor is
    bounded, so the ink cannot be washed out to near-background either -- unlike a global
    brightness/contrast shift, which lifts dark ink into the background range.
    """
    factor = 1.0 + float(rng.uniform(-delta, delta))
    adjusted = 255.0 - (255.0 - gray.astype(np.float32)) * factor
    return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)


# A draw that leaves fewer than this many clearly-dark pixels has washed the signature out;
# the rotation stage is used instead, so a near-blank sample never reaches training.
MIN_INK_LEVEL = 180
MIN_INK_PIXELS = 50


def _ensure_dark_ink(augmented: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return ``fallback`` when the stroke/blur/contrast stages washed the ink out.

    The representation contract is dark ink on a pure-white background. Thinning, blurring
    and lightening can combine to lift every ink pixel above the ink threshold; that would
    inject a nearly blank image into training. The rotation stage preserves ink contrast, so
    it is the safe fallback.
    """
    if int((augmented < MIN_INK_LEVEL).sum()) >= MIN_INK_PIXELS:
        return augmented
    return fallback


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
        same :func:`~preprocessing.signature_preprocessing.fit_to_canvas` the inference path
        uses.

    Raises:
        ValueError: If ``image`` is not a valid uint8 grayscale signature.
    """
    gray = np.asarray(image)
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("augment_signature expects a non-empty uint8 grayscale image")
    original = gray

    # 1. Rotation (tilt). Applied first, so the ink is still un-warped by any later stage.
    if config.rotate_degrees:
        gray = _apply_rotation(
            gray,
            float(rng.uniform(-config.rotate_degrees, config.rotate_degrees)),
            fill=config.fill,
        )
    # The rotation stage preserves ink contrast, so it is the safe fallback if the stages
    # below wash the signature out.
    rotated = gray

    # 2. Stroke width (pen). 3. Ink contrast (pen pressure). 4. Blur (scan quality).
    if (
        config.stroke_width_probability > 0.0
        and rng.random() < config.stroke_width_probability
    ):
        gray = _apply_stroke_width(gray, config.stroke_width_kernel, rng)
    if config.ink_contrast:
        gray = _apply_ink_contrast(gray, config.ink_contrast, rng)
    if config.blur_probability > 0.0 and rng.random() < config.blur_probability:
        gray = cv2.GaussianBlur(gray, (0, 0), config.blur_sigma)
    gray = _ensure_dark_ink(gray, rotated)

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
