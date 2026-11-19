"""Tests for the target-FAR operating point and per-dataset handcrafted normalisation.

These cover the additions that make a multi-script manifest comparable and auditable:

* ``scoring.metrics.threshold_at_far`` -- pick the *largest* threshold meeting a FAR
  budget, which is also the FRR-minimising threshold subject to that budget;
* ``scoring.fusion.HandNormalizer`` -- the opt-in per-dataset standardisation of the
  absolute handcrafted features;
* ``training.fit_fusion._to_query_level`` -- fit the fusion rule in the unit that
  ``training.evaluate`` actually thresholds;
* ``training.stage3_metric_learning.dataset_labels`` -- the single mapping from a
  record to its source dataset.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scoring.fusion import (
    FusionConfig,
    FusionScorer,
    HandNormalizer,
    PairMatrix,
)
from scoring.handcrafted_features import FEATURE_NAMES
from scoring.metrics import frr_at_threshold, rates_at_threshold, threshold_at_far
from training.fit_fusion import _to_query_level
from training.stage3_metric_learning import SignatureRecord, dataset_labels


class ThresholdAtFarTests(unittest.TestCase):
    def test_meets_the_budget_and_minimises_frr(self) -> None:
        negatives = np.linspace(0.0, 1.0, 101)
        threshold, achieved = threshold_at_far(negatives, 0.03)
        # 3% of 101 negatives is 3 allowed acceptances, and the decision rule is
        # ``score >= threshold``, so exactly 3 negatives may be accepted.
        self.assertLessEqual(achieved, 0.03)
        self.assertEqual(int((negatives >= threshold).sum()), 3)
        # No larger threshold can satisfy the budget, so FRR cannot be lower.
        _, higher_far = threshold_at_far(negatives, 0.02)
        self.assertLessEqual(higher_far, achieved)

    def test_zero_budget_accepts_no_negative(self) -> None:
        negatives = np.array([0.1, 0.5, 0.9])
        threshold, achieved = threshold_at_far(negatives, 0.0)
        self.assertEqual(achieved, 0.0)
        self.assertGreater(threshold, negatives.max())
        self.assertEqual(int((negatives >= threshold).sum()), 0)

    def test_full_budget_accepts_every_negative(self) -> None:
        negatives = np.array([0.1, 0.5, 0.9])
        threshold, achieved = threshold_at_far(negatives, 1.0)
        self.assertEqual(achieved, 1.0)
        self.assertAlmostEqual(threshold, negatives.min())

    def test_raises_on_empty_or_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            threshold_at_far(np.array([]), 0.1)
        with self.assertRaisesRegex(ValueError, "target_far"):
            threshold_at_far(np.array([0.5]), 1.5)

    def test_frr_at_threshold_counts_strictly_below(self) -> None:
        self.assertAlmostEqual(frr_at_threshold([0.1, 0.5, 0.9], 0.5), 1 / 3)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            frr_at_threshold([], 0.5)

    def test_target_far_is_more_conservative_than_eer(self) -> None:
        rng = np.random.default_rng(3)
        genuine = rng.normal(0.7, 0.10, size=500)
        skilled = rng.normal(0.5, 0.12, size=500)
        far_threshold, far = threshold_at_far(skilled, 0.03)
        self.assertLessEqual(far, 0.03)
        # A 3% FAR budget must sit at or above the equal-error threshold.
        _, eer = rates_at_threshold(genuine, skilled, far_threshold)
        self.assertGreaterEqual(eer, far)



def _feature_row(value: float) -> dict[str, float]:
    """A full, valid handcrafted feature dict with every feature set to ``value``."""
    return {name: value for name in FEATURE_NAMES}


class HandNormalizerTests(unittest.TestCase):
    def _matrix(self) -> tuple[np.ndarray, list[str]]:
        rng = np.random.default_rng(5)
        first = rng.normal(0.8, 0.05, size=(40, len(FEATURE_NAMES)))
        second = rng.normal(0.2, 0.10, size=(60, len(FEATURE_NAMES)))
        return np.vstack([first, second]), ["easy"] * 40 + ["hard"] * 60

    def test_fit_standardises_each_dataset(self) -> None:
        matrix, labels = self._matrix()
        normalizer = HandNormalizer.fit(matrix, labels)
        transformed = normalizer.transform(matrix, labels)
        for label in ("easy", "hard"):
            rows = transformed[np.asarray(labels) == label]
            np.testing.assert_allclose(rows.mean(axis=0), 0.0, atol=1e-12)
            np.testing.assert_allclose(rows.std(axis=0), 1.0, atol=1e-12)

    def test_unknown_dataset_is_left_untouched(self) -> None:
        matrix, labels = self._matrix()
        normalizer = HandNormalizer.fit(matrix, labels)
        untouched = normalizer.transform(matrix, ["unseen"] * len(labels))
        np.testing.assert_allclose(untouched, matrix)
        features = _feature_row(0.1)
        self.assertEqual(normalizer.transform_features(features, "unseen"), features)
        self.assertEqual(normalizer.transform_features(features, None), features)

    def test_transform_features_matches_transform(self) -> None:
        matrix, labels = self._matrix()
        normalizer = HandNormalizer.fit(matrix, labels)
        transformed = normalizer.transform(matrix, labels)
        features = {
            name: float(matrix[7, index]) for index, name in enumerate(FEATURE_NAMES)
        }
        row = normalizer.transform_features(features, labels[7])
        np.testing.assert_allclose(
            [row[name] for name in FEATURE_NAMES], transformed[7], atol=1e-12
        )

    def test_metadata_round_trip(self) -> None:
        import json

        matrix, labels = self._matrix()
        normalizer = HandNormalizer.fit(matrix, labels)
        expected = normalizer.transform(matrix, labels)
        for payload in (
            normalizer.to_metadata(),
            # The payload lives in FusionConfig.metadata, so it must survive JSON.
            json.loads(json.dumps(normalizer.to_metadata())),
        ):
            restored = HandNormalizer.from_metadata(payload)
            self.assertIsNotNone(restored)
            np.testing.assert_allclose(restored.transform(matrix, labels), expected)

    def test_from_metadata_rejects_junk(self) -> None:
        for payload in (None, {}, {"mean": {}}, "nope", {"mean": {"a": 1}, "scale": {}}):
            self.assertIsNone(HandNormalizer.from_metadata(payload))

    def test_single_row_dataset_is_skipped(self) -> None:
        matrix = np.vstack(
            [
                np.full((3, len(FEATURE_NAMES)), 0.5),
                np.full((1, len(FEATURE_NAMES)), 0.9),
            ]
        )
        normalizer = HandNormalizer.fit(matrix, ["many", "many", "many", "lonely"])
        self.assertIn("many", normalizer.mean)
        self.assertNotIn("lonely", normalizer.mean)

    def test_fit_validates_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            HandNormalizer.fit(np.zeros((2, 3)), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "equal length"):
            HandNormalizer.fit(np.zeros((2, len(FEATURE_NAMES))), ["a"])


def _pair_matrix() -> PairMatrix:
    """Two queries with three references each: one genuine, one skilled forgery."""
    return PairMatrix(
        deep=np.array([0.9, 0.8, 0.7, 0.2, 0.3, 0.4]),
        hand=np.arange(
            6 * len(FEATURE_NAMES), dtype=np.float64
        ).reshape(6, len(FEATURE_NAMES)),
        labels=np.array([1, 1, 1, 0, 0, 0]),
        signers=np.array(["a"] * 3 + ["b"] * 3, dtype=object),
        kinds=np.array(["genuine"] * 3 + ["skilled"] * 3, dtype=object),
        query_ids=np.array([0, 0, 0, 1, 1, 1]),
        datasets=np.array(["easy"] * 3 + ["hard"] * 3, dtype=object),
    )


class ToQueryLevelTests(unittest.TestCase):
    def test_averages_rows_per_query(self) -> None:
        collapsed = _to_query_level(_pair_matrix())
        self.assertEqual(len(collapsed), 2)
        np.testing.assert_allclose(collapsed.deep, [0.8, 0.3])
        expected_hand = np.arange(
            3 * len(FEATURE_NAMES), dtype=np.float64
        ).reshape(3, len(FEATURE_NAMES)).mean(axis=0)
        np.testing.assert_allclose(collapsed.hand[0], expected_hand)
        np.testing.assert_array_equal(collapsed.labels, [1, 0])
        np.testing.assert_array_equal(collapsed.kinds, ["genuine", "skilled"])
        np.testing.assert_array_equal(collapsed.signers, ["a", "b"])
        np.testing.assert_array_equal(collapsed.datasets, ["easy", "hard"])

    def test_weighted_fused_score_is_unchanged(self) -> None:
        # Averaging is linear, so fusing the averaged row equals the average of the fused
        # rows: aggregating only moves the fitting objective to the reported unit.
        scorer = FusionScorer(
            FusionConfig(type="weighted", deep_weight=0.6, hand_weight=0.4)
        )
        matrix = _pair_matrix()
        per_query = scorer.score_matrix(matrix.deep, matrix.hand).reshape(2, 3).mean(axis=1)
        collapsed = _to_query_level(matrix)
        np.testing.assert_allclose(
            scorer.score_matrix(collapsed.deep, collapsed.hand), per_query, atol=1e-12
        )

    def test_handles_missing_datasets(self) -> None:
        matrix = _pair_matrix()
        bare = PairMatrix(
            deep=matrix.deep,
            hand=matrix.hand,
            labels=matrix.labels,
            signers=matrix.signers,
            kinds=matrix.kinds,
            query_ids=matrix.query_ids,
        )
        self.assertIsNone(_to_query_level(bare).datasets)


class DatasetLabelsTests(unittest.TestCase):
    def _records(self, root: Path) -> list[SignatureRecord]:
        return [
            SignatureRecord(
                root / "BHSig260" / "BHSig100_Bengali" / "Genuine" / "a.png", "s1", False
            ),
            SignatureRecord(root / "cedar" / "full_org" / "b.png", "s2", False),
        ]

    def test_depth_one_and_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            records = self._records(root)
            self.assertEqual(dataset_labels(records, root), ["BHSig260", "cedar"])
            self.assertEqual(
                dataset_labels(records, root, depth=2),
                ["BHSig260/BHSig100_Bengali", "cedar/full_org"],
            )

    def test_path_outside_root_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            record = SignatureRecord(Path("/somewhere/else/x.png"), "s1", False)
            with self.assertRaisesRegex(ValueError, "not under"):
                dataset_labels([record], root)

