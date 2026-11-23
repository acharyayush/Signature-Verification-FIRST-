"""Tests for Stage 2.2 preprocessing and the training-time augmentation.

These pin the two contracts the training pipeline now depends on:

* :func:`preprocessing.signature_preprocessing.whiten_background` -- the paper background
  is forced to pure white while every ink pixel is returned byte-for-byte unchanged, so
  the signature's true grayscale (pen pressure, ink flow) survives;
* :func:`preprocessing.augmentation.augment_signature` -- augmentation is deterministic for
  a fixed seed, keeps the ``256 x 512`` canvas and the pure-white background, and is only
  reachable through ``SignatureDataset(augment=True)`` (never validation).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from preprocessing.augmentation import (
    MIN_INK_LEVEL,
    MIN_INK_PIXELS,
    STRENGTHS,
    AugmentationConfig,
    _apply_elastic,
    _apply_stroke_width,
    _elastic_offsets,
    augment_signature,
    augmentation_config,
)
from preprocessing.signature_preprocessing import (
    DEFAULT_TARGET_SIZE,
    load_preprocessed_image,
    preprocess_signature,
    to_model_input,
    whiten_background,
)
from training.stage3_metric_learning import SignatureDataset, SignatureRecord

WIDTH, HEIGHT = DEFAULT_TARGET_SIZE
INK_LEVEL = 60


def _signature_image(background: int = 255, seed: int = 0) -> np.ndarray:
    """Draw a deterministic dark-on-``background`` signature-like image."""
    rng = np.random.default_rng(seed)
    image = np.full((HEIGHT, WIDTH), background, dtype=np.uint8)
    wave = np.column_stack(
        [
            np.linspace(40, WIDTH - 40, 24),
            HEIGHT / 2.0 + 40.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, 24)),
        ]
    ).astype(np.int32)
    cv2.polylines(image, [wave.reshape(-1, 1, 2)], False, INK_LEVEL, 3)
    second = np.column_stack(
        [
            np.linspace(WIDTH * 0.3, WIDTH * 0.8, 12),
            np.linspace(HEIGHT * 0.3, HEIGHT * 0.7, 12),
        ]
    ).astype(np.int32) + rng.integers(-6, 7, size=(12, 2))
    cv2.polylines(image, [second.reshape(-1, 1, 2)], False, INK_LEVEL + 30, 2)
    return image


def _ink_bbox_aspect(image: np.ndarray) -> float:
    """Aspect ratio of the tight ink bounding box (``0`` when there is no ink)."""
    mask = image < 200
    if not mask.any():
        return 0.0
    _, _, width, height = cv2.boundingRect(mask.astype(np.uint8))
    return width / max(height, 1)


class _FixedRandom:
    """Stand-in for ``numpy.random.Generator`` that returns fixed draws.

    Only the methods the augmentation helpers use are implemented, so a test can force a
    specific branch (for example ``random() < 0.5`` selects ink thickening) without relying
    on a lucky seed.
    """

    def __init__(self, random_value: float = 1.0, uniform_value: float = 0.0) -> None:
        self.random_value = random_value
        self.uniform_value = uniform_value

    def random(self) -> float:
        return self.random_value

    def uniform(self, low: float = 0.0, high: float = 1.0, size=None):
        # Clipped to the requested bounds so a zero-width range (e.g. shear 0) stays 0.
        value = float(np.clip(self.uniform_value, low, high))
        return np.full(size, value) if size is not None else value

    def normal(self, loc: float = 0.0, scale: float = 1.0, size=None):
        return np.zeros(size) if size is not None else float(loc)


def _geometry_only_config(**overrides) -> AugmentationConfig:
    """A config with every transform disabled, so a test can enable exactly one."""
    disabled = dict(
        rotate_degrees=0.0,
        shear_degrees=0.0,
        perspective=0.0,
        elastic_probability=0.0,
        stroke_width_probability=0.0,
        ink_contrast=0.0,
        gamma=0.0,
        blur_probability=0.0,
        noise_probability=0.0,
    )
    disabled.update(overrides)
    return AugmentationConfig(**disabled)


def _wide_bar() -> np.ndarray:
    """A wide, low ink bar: rotation visibly changes its bounding-box aspect."""
    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    cv2.rectangle(image, (100, 80), (400, 180), INK_LEVEL, -1)
    return image


class WhitenBackgroundTests(unittest.TestCase):
    def test_background_becomes_pure_white_even_under_shadow(self) -> None:
        # A uniformly grey "shadow" must be whitened, not kept as grey.
        image = _signature_image(background=200)
        whitened = whiten_background(image)
        background = image >= 200
        self.assertTrue((whitened[background] == 255).all())

    def test_ink_pixels_are_preserved_exactly(self) -> None:
        image = _signature_image(background=200)
        whitened = whiten_background(image)
        ink = image < 200
        # The old flatten_illumination rescaled ink by the local background; this must not.
        np.testing.assert_array_equal(whitened[ink], image[ink])

    def test_already_white_image_is_unchanged(self) -> None:
        image = _signature_image(background=255)
        np.testing.assert_array_equal(whiten_background(image), image)

    def test_does_not_binarize(self) -> None:
        # Grayscale intensity is preserved, so the output has more than two levels.
        whitened = whiten_background(_signature_image())
        self.assertGreater(np.unique(whitened).size, 2)

    def test_derived_tolerance_whitens_grey_scanned_paper(self) -> None:
        # CEDAR paper is grey (mean ~232) and textured, so a fixed gap of 8 left ~14% of the
        # background un-whitened. The derived tolerance must clear that without touching ink.
        strokes = _signature_image()
        rng = np.random.default_rng(0)
        paper = np.clip(232 + rng.integers(-12, 1, size=strokes.shape), 0, 255).astype(np.uint8)
        paper[strokes < 200] = strokes[strokes < 200]  # thin strokes stay dark
        ink = paper < 200

        whitened = whiten_background(paper)
        leftover = float(((whitened >= 200) & (whitened < 255)).mean())
        self.assertLess(leftover, 0.02, "grey scanned paper was not whitened")
        np.testing.assert_array_equal(whitened[ink], paper[ink])

        fixed = whiten_background(paper, background_gap_max=8)
        fixed_leftover = float(((fixed >= 200) & (fixed < 255)).mean())
        self.assertGreater(
            fixed_leftover, leftover * 3, "the derived tolerance should beat a fixed 8"
        )

    def test_explicit_fixed_tolerance_is_still_honoured(self) -> None:
        image = _signature_image(background=200)
        whitened = whiten_background(image, background_gap_max=8)
        self.assertTrue((whitened[image >= 200] == 255).all())


class PreprocessSignatureTests(unittest.TestCase):
    def test_output_shape_and_dtype(self) -> None:
        processed = preprocess_signature(_signature_image(background=200))
        self.assertEqual(processed.shape, (HEIGHT, WIDTH))
        self.assertEqual(processed.dtype, np.uint8)

    def test_background_is_pure_white(self) -> None:
        processed = preprocess_signature(_signature_image(background=210))
        # Everything that is not ink is 255, so a clear majority of the canvas is white.
        self.assertGreater((processed == 255).mean(), 0.5)

    def test_grayscale_is_preserved_by_default(self) -> None:
        processed = preprocess_signature(_signature_image())
        self.assertGreater(np.unique(processed).size, 2)

    def test_flatten_mode_is_still_available(self) -> None:
        image = _signature_image(background=200)
        self.assertFalse(
            np.array_equal(
                preprocess_signature(image, background_mode="whiten"),
                preprocess_signature(image, background_mode="flatten"),
            )
        )

    def test_binarize_still_produces_two_levels(self) -> None:
        processed = preprocess_signature(_signature_image(), binarize=True)
        self.assertTrue(set(np.unique(processed).tolist()).issubset({0, 255}))

    def test_blank_image_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no signature ink"):
            preprocess_signature(np.full((64, 64), 255, dtype=np.uint8))


class AugmentationTests(unittest.TestCase):
    def test_shape_dtype_and_white_background_for_every_strength(self) -> None:
        for strength in STRENGTHS:
            augmented = augment_signature(
                _signature_image(),
                augmentation_config(strength),
                np.random.default_rng([1, 0, 3]),
            )
            self.assertEqual(augmented.shape, (HEIGHT, WIDTH))
            self.assertEqual(augmented.dtype, np.uint8)
            self.assertGreater((augmented == 255).mean(), 0.5)

    def test_keeps_ink(self) -> None:
        augmented = augment_signature(
            _signature_image(), augmentation_config("standard"), np.random.default_rng([2, 0, 1])
        )
        self.assertGreater(int((augmented < 200).sum()), 0)

    def test_deterministic_for_same_seed(self) -> None:
        config = augmentation_config("standard")
        first = augment_signature(_signature_image(), config, np.random.default_rng([7, 0, 5]))
        second = augment_signature(_signature_image(), config, np.random.default_rng([7, 0, 5]))
        np.testing.assert_array_equal(first, second)

    def test_varies_across_epoch_and_index(self) -> None:
        config = augmentation_config("standard")
        base = _signature_image()
        epoch0 = augment_signature(base, config, np.random.default_rng([7, 0, 5]))
        epoch1 = augment_signature(base, config, np.random.default_rng([7, 1, 5]))
        other = augment_signature(base, config, np.random.default_rng([7, 0, 6]))
        self.assertFalse(np.array_equal(epoch0, epoch1))
        self.assertFalse(np.array_equal(epoch0, other))

    def test_rotation_changes_the_bounding_box_aspect(self) -> None:
        # A 30 degree tilt must change the ink geometry, not merely resample it.
        rotated = augment_signature(
            _wide_bar(),
            _geometry_only_config(rotate_degrees=20.0),
            _FixedRandom(uniform_value=20.0),
        )
        upright = augment_signature(_wide_bar(), _geometry_only_config(), _FixedRandom())
        self.assertLess(_ink_bbox_aspect(rotated), _ink_bbox_aspect(upright) * 0.9)

    def test_stroke_width_erode_thickens_and_dilate_thins(self) -> None:
        image = _signature_image()
        baseline = int((image < 200).sum())
        thicker = _apply_stroke_width(image, 3, _FixedRandom(random_value=0.0))
        thinner = _apply_stroke_width(image, 3, _FixedRandom(random_value=0.9))
        self.assertGreater(int((thicker < 200).sum()), baseline)
        self.assertLess(int((thinner < 200).sum()), baseline)

    def test_invalid_input_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "uint8 grayscale"):
            augment_signature(
                np.zeros((4, 4), dtype=np.float32), augmentation_config(), np.random.default_rng(0)
            )

    def test_strength_presets_do_not_decrease_rotation(self) -> None:
        light = augmentation_config("light").rotate_degrees
        standard = augmentation_config("standard").rotate_degrees
        strong = augmentation_config("strong").rotate_degrees
        self.assertLessEqual(light, standard)
        self.assertLessEqual(standard, strong)

    def test_standard_preset_tilt_is_20_degrees(self) -> None:
        # Pinned deliberately: the tilt range is the largest single source of augmentation
        # variance, and it was reduced from 30 to 20 because over-rotating destroys the fine
        # identity cues a verification model depends on.
        self.assertEqual(augmentation_config("standard").rotate_degrees, 20.0)
        self.assertEqual(AugmentationConfig().rotate_degrees, 20.0)

    def test_bhsig_preset_uses_milder_elastic(self) -> None:
        default = augmentation_config("standard")
        indic = augmentation_config("standard", dataset="BHSig260")
        self.assertLess(indic.elastic_alpha, default.elastic_alpha)
        self.assertLess(indic.shear_degrees, default.shear_degrees)

    def test_unknown_dataset_keeps_the_strength_preset(self) -> None:
        self.assertEqual(
            augmentation_config("standard", dataset="CEDAR").elastic_alpha,
            augmentation_config("standard").elastic_alpha,
        )

    def test_elastic_actually_moves_the_ink(self) -> None:
        # Regression: the displacement field was normalised noise times a nominal alpha of
        # 1, so the ink moved by ~0.2 px and elastic was effectively a no-op. alpha is now
        # the peak displacement in pixels, so the ink mask must visibly shift.
        image = _signature_image()
        warped = _apply_elastic(image, 4.0, 10.0, np.random.default_rng(0), fill=255)
        before, after = image < 200, warped < 200
        overlap = float((before & after).sum()) / max(int((before | after).sum()), 1)
        self.assertLess(overlap, 0.95, "elastic warp barely moved the ink")

    def test_elastic_displacement_is_bounded_by_alpha(self) -> None:
        offsets_x, offsets_y = _elastic_offsets(128, 128, 5.0, 10.0, np.random.default_rng(0))
        self.assertAlmostEqual(float(np.abs(offsets_x).max()), 5.0, places=4)
        self.assertAlmostEqual(float(np.abs(offsets_y).max()), 5.0, places=4)

    def test_photometric_stage_preserves_a_dark_ink_contract(self) -> None:
        # Regression: thinning + blur + lightening once combined to lift every ink pixel
        # into the 200..255 background range on ~4% of draws, injecting near-blank samples.
        # The representation contract is dark ink on white, so dark ink must always survive.
        for strength in STRENGTHS:
            config = augmentation_config(strength)
            for seed in range(25):
                augmented = augment_signature(
                    _signature_image(),
                    config,
                    np.random.default_rng([seed, seed + 1, seed + 2]),
                )
                self.assertGreaterEqual(
                    int((augmented < MIN_INK_LEVEL).sum()),
                    MIN_INK_PIXELS,
                    msg=f"{strength} seed {seed} washed the signature out",
                )


class SignatureDatasetAugmentationTests(unittest.TestCase):
    def _record(self, directory: str) -> SignatureRecord:
        path = Path(directory) / "sig.png"
        self.assertTrue(cv2.imwrite(str(path), _signature_image()))
        return SignatureRecord(image_path=path, signer_id="a", is_forgery=False)

    def test_augment_disabled_matches_the_shared_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = self._record(directory)
            dataset = SignatureDataset([record], {"a": 0}, already_preprocessed=True)
            tensor, signer, is_forgery = dataset[0]
            expected = to_model_input(
                load_preprocessed_image(record.image_path, already_preprocessed=True)
            )
            np.testing.assert_allclose(tensor.numpy(), expected)
            self.assertEqual(signer, 0)
            self.assertFalse(is_forgery)

    def test_augmentation_is_deterministic_per_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = self._record(directory)
            dataset = SignatureDataset(
                [record], {"a": 0}, already_preprocessed=True, augment=True, augment_seed=11
            )
            first = dataset[0][0].numpy()
            np.testing.assert_array_equal(first, dataset[0][0].numpy())
            dataset.set_epoch(1)
            self.assertFalse(np.array_equal(first, dataset[0][0].numpy()))

    def test_augmentation_changes_the_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = self._record(directory)
            plain = SignatureDataset([record], {"a": 0}, already_preprocessed=True)
            augmented = SignatureDataset(
                [record], {"a": 0}, already_preprocessed=True, augment=True, augment_seed=3
            )
            self.assertFalse(np.array_equal(plain[0][0].numpy(), augmented[0][0].numpy()))

    def test_per_record_configs_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = self._record(directory)
            light = augmentation_config("light")
            dataset = SignatureDataset(
                [record],
                {"a": 0},
                already_preprocessed=True,
                augment=True,
                augment_seed=5,
                record_datasets=["CEDAR"],
                dataset_configs={"CEDAR": light},
            )
            expected = augment_signature(
                load_preprocessed_image(record.image_path, already_preprocessed=True),
                light,
                np.random.default_rng([5, 0, 0]),
            )
            np.testing.assert_allclose(dataset[0][0].numpy(), to_model_input(expected))

    def test_record_datasets_must_align_with_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = self._record(directory)
            with self.assertRaisesRegex(ValueError, "one label per record"):
                SignatureDataset(
                    [record], {"a": 0}, already_preprocessed=True, record_datasets=["a", "b"]
                )
