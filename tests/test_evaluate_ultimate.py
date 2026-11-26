"""Tests for the "Ultimate Test Signs" pairwise evaluation protocol.

The Ultimate benchmark has no enrolment and no averaging over references: every signer
contributes nine single-pair comparisons, and each pair is scored by the cosine similarity
of its two embeddings. These tests pin that protocol down:

* ``training.evaluate_ultimate.pair_indices`` -- three genuine-genuine combinations, three
  positionally paired genuine-forged pairs, and three genuine-random pairs, with loud
  failures instead of silently dropped pairs;
* ``training.evaluate_ultimate.score_pairs`` -- a pair's score is the raw embedding dot
  product, which is the cosine similarity of L2-normalised embeddings.
"""

from __future__ import annotations

import unittest

import numpy as np

from training.evaluate_ultimate import (
    DEFAULT_THRESHOLD,
    fusion_decision_threshold,
    pair_indices,
    score_pairs,
)


def _manifest(spec: dict[str, tuple[int, int, int]]):
    """Build ``(rows, by_folder)`` with ``genuine, forged, random`` counts per folder.

    File names are ``<kind>/00.png``, ``<kind>/01.png``, ... so the deterministic ordering
    by path is easy to reason about.
    """
    rows: list[dict[str, str]] = []
    by_folder: dict[str, list[int]] = {}
    for folder, (genuine, forged, random_count) in spec.items():
        indices = []
        for kind, count in (
            ("genuine", genuine),
            ("forged", forged),
            ("random", random_count),
        ):
            for position in range(count):
                indices.append(len(rows))
                rows.append(
                    {
                        "folder": folder,
                        "class": kind,
                        "source_dataset": "dataset",
                        "source_signer": folder,
                        "image_path": f"{folder}/{kind}/{position:02d}.png",
                    }
                )
        by_folder[folder] = indices
    return rows, by_folder


class PairIndicesTests(unittest.TestCase):
    def test_three_of_each_class_yields_nine_pairs(self) -> None:
        rows, by_folder = _manifest({"Signer_001": (3, 3, 3)})
        pairs = pair_indices(rows, by_folder)
        self.assertEqual(
            {kind: len(group) for kind, group in pairs.items()},
            {"genuine": 3, "skilled": 3, "random": 3},
        )
        self.assertEqual(sum(len(group) for group in pairs.values()), 9)

    def test_genuine_pairs_are_every_unordered_combination(self) -> None:
        rows, by_folder = _manifest({"Signer_001": (3, 3, 3)})
        pairs = pair_indices(rows, by_folder)
        genuine = [
            tuple(rows[index]["image_path"] for index in couple)
            for couple in pairs["genuine"]
        ]
        self.assertEqual(
            genuine,
            [
                ("Signer_001/genuine/00.png", "Signer_001/genuine/01.png"),
                ("Signer_001/genuine/00.png", "Signer_001/genuine/02.png"),
                ("Signer_001/genuine/01.png", "Signer_001/genuine/02.png"),
            ],
        )

    def test_impostors_are_paired_positionally_with_the_genuine_signatures(self) -> None:
        rows, by_folder = _manifest({"Signer_001": (3, 3, 3)})
        pairs = pair_indices(rows, by_folder)
        skilled = [
            (rows[first]["image_path"], rows[second]["image_path"])
            for first, second in pairs["skilled"]
        ]
        self.assertEqual(
            skilled,
            [
                ("Signer_001/genuine/00.png", "Signer_001/forged/00.png"),
                ("Signer_001/genuine/01.png", "Signer_001/forged/01.png"),
                ("Signer_001/genuine/02.png", "Signer_001/forged/02.png"),
            ],
        )
        random_pairs = [
            (rows[first]["image_path"], rows[second]["image_path"])
            for first, second in pairs["random"]
        ]
        self.assertEqual(
            random_pairs,
            [
                ("Signer_001/genuine/00.png", "Signer_001/random/00.png"),
                ("Signer_001/genuine/01.png", "Signer_001/random/01.png"),
                ("Signer_001/genuine/02.png", "Signer_001/random/02.png"),
            ],
        )

    def test_pairs_are_built_per_signer(self) -> None:
        rows, by_folder = _manifest({"a": (3, 3, 3), "b": (3, 3, 3)})
        pairs = pair_indices(rows, by_folder)
        self.assertEqual(sum(len(group) for group in pairs.values()), 18)
        for group in pairs.values():
            signers = {rows[first]["folder"] for first, _ in group}
            self.assertEqual(signers, {"a", "b"})

    def test_more_genuine_signatures_add_combinations_not_forgeries(self) -> None:
        rows, by_folder = _manifest({"Signer_001": (4, 4, 4)})
        pairs = pair_indices(rows, by_folder)
        self.assertEqual(
            {kind: len(group) for kind, group in pairs.items()},
            {"genuine": 6, "skilled": 4, "random": 4},
        )

    def test_one_genuine_signature_cannot_form_a_pair(self) -> None:
        rows, by_folder = _manifest({"Signer_001": (1, 1, 1)})
        with self.assertRaises(ValueError):
            pair_indices(rows, by_folder)

    def test_unequal_impostor_counts_raise_instead_of_dropping_pairs(self) -> None:
        rows, by_folder = _manifest({"Signer_001": (3, 2, 3)})
        with self.assertRaises(ValueError):
            pair_indices(rows, by_folder)
        rows, by_folder = _manifest({"Signer_001": (3, 3, 2)})
        with self.assertRaises(ValueError):
            pair_indices(rows, by_folder)


class ScorePairsTests(unittest.TestCase):
    def test_scores_are_raw_embedding_dot_products(self) -> None:
        # Pairing with two genuine (0, 1), two forged (2, 3), two random (4, 5):
        #   genuine (0, 1), skilled (0, 2) and (1, 3), random (0, 4) and (1, 5).
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=np.float64,
        )
        rows, by_folder = _manifest({"Signer_001": (2, 2, 2)})
        pairs = pair_indices(rows, by_folder)
        scores = score_pairs(embeddings, rows, pairs)
        self.assertEqual([round(score, 6) for score in scores["genuine"]], [1.0])
        self.assertEqual([round(score, 6) for score in scores["skilled"]], [0.0, 0.0])
        self.assertEqual([round(score, 6) for score in scores["random"]], [1.0, 0.0])

    def test_each_kind_keeps_its_own_score_list(self) -> None:
        embeddings = np.eye(6, dtype=np.float64)
        rows, by_folder = _manifest({"Signer_001": (2, 2, 2)})
        pairs = pair_indices(rows, by_folder)
        scores = score_pairs(embeddings, rows, pairs)
        self.assertEqual(set(scores), {"genuine", "skilled", "random"})
        self.assertEqual(sum(len(scores[kind]) for kind in scores), 1 + 2 + 2)


class FusionDecisionThresholdTests(unittest.TestCase):
    """The fused operating point lives in a different space from the deep cosine."""

    def test_explicit_fused_threshold_wins(self) -> None:
        threshold, source = fusion_decision_threshold(
            fused_threshold=0.5, config_threshold=0.8, deep_threshold=0.9
        )
        self.assertEqual(threshold, 0.5)
        self.assertEqual(source, "fused_threshold")

    def test_config_threshold_is_used_without_an_override(self) -> None:
        threshold, source = fusion_decision_threshold(
            fused_threshold=None, config_threshold=0.8, deep_threshold=0.9
        )
        self.assertEqual(threshold, 0.8)
        self.assertEqual(source, "fusion_config")

    def test_deep_threshold_is_only_borrowed_as_a_last_resort(self) -> None:
        threshold, source = fusion_decision_threshold(
            fused_threshold=None, config_threshold=None, deep_threshold=0.9
        )
        self.assertEqual(threshold, 0.9)
        self.assertEqual(source, "threshold")

    def test_default_applies_when_nothing_is_supplied(self) -> None:
        threshold, source = fusion_decision_threshold(
            fused_threshold=None, config_threshold=None, deep_threshold=None
        )
        self.assertEqual(threshold, DEFAULT_THRESHOLD)
        self.assertEqual(source, "default")


if __name__ == "__main__":
    unittest.main()