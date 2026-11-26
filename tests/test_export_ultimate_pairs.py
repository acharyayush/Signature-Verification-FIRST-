"""Tests for the per-pair CSV export of the "Ultimate Test Signs" benchmark.

:mod:`scripts.export_ultimate_pairs` turns the aggregate pairwise protocol into one CSV row
per comparison. These tests pin down the parts that a reader of that CSV relies on:

* ``comparison_type`` -- the ``genuine``/``skilled``/``random`` kinds become
  ``genuine-genuine`` / ``genuine-forge`` / ``genuine-random``;
* ``build_pair_rows`` -- one row per pair, grouped by signer then by comparison type, with
  the raw embedding dot product and ``matched == score >= threshold``;
* ``overall_summary`` -- the confusion-matrix counts, FAR, FRR and EER appended after the
  rows;
* ``write_report`` -- the file layout (pair header, pair rows, blank line, ``metric,value``
  block).
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scoring.metrics import eer_threshold
from scripts.export_ultimate_pairs import (
    FUSION_COLUMNS,
    PAIR_COLUMNS,
    build_pair_rows,
    comparison_type,
    display_path,
    fusion_summary,
    overall_summary,
    write_report,
)
from training.evaluate_ultimate import pair_indices


def _manifest(spec: dict[str, tuple[int, int, int]]):
    """Build ``(rows, by_folder)`` with ``genuine, forged, random`` counts per folder.

    Mirrors the helper in ``test_evaluate_ultimate`` so the two suites exercise the same
    layout: file names are ``<kind>/00.png``, ``<kind>/01.png``, ...
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


class ComparisonTypeTests(unittest.TestCase):
    def test_each_pair_kind_maps_to_its_label(self) -> None:
        self.assertEqual(comparison_type("genuine"), "genuine-genuine")
        self.assertEqual(comparison_type("skilled"), "genuine-forge")
        self.assertEqual(comparison_type("random"), "genuine-random")

    def test_unknown_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            comparison_type("mystery")


class DisplayPathTests(unittest.TestCase):
    def test_paths_are_relative_to_the_image_root(self) -> None:
        root = Path("/data/Ultimate Test Signs")
        path = root / "Signer_001" / "Genuine" / "a.png"
        self.assertEqual(display_path(path, root), "Signer_001/Genuine/a.png")

    def test_paths_outside_the_root_are_left_alone(self) -> None:
        self.assertEqual(display_path("/elsewhere/a.png", Path("/data")), "/elsewhere/a.png")

    def test_no_root_keeps_the_path_as_is(self) -> None:
        self.assertEqual(
            display_path("Signer_001/Genuine/a.png", None), "Signer_001/Genuine/a.png"
        )


class BuildPairRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        rows, by_folder = _manifest({"Signer_001": (3, 3, 3)})
        pairs = pair_indices(rows, by_folder)
        # Unit-ish vectors so every dot product is 0.0 or 1.0 and easy to assert.
        self.embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # 0 genuine
                [1.0, 0.0, 0.0],  # 1 genuine
                [0.0, 1.0, 0.0],  # 2 genuine
                [0.0, 1.0, 0.0],  # 3 forged
                [1.0, 0.0, 0.0],  # 4 forged
                [0.0, 1.0, 0.0],  # 5 forged
                [0.0, 0.0, 1.0],  # 6 random
                [0.0, 0.0, 1.0],  # 7 random
                [0.0, 0.0, 1.0],  # 8 random
            ],
            dtype=np.float64,
        )
        self.table = build_pair_rows(rows, by_folder, pairs, self.embeddings, 0.5)

    def test_one_row_per_pair(self) -> None:
        self.assertEqual(len(self.table), 9)

    def test_rows_are_grouped_by_comparison_type(self) -> None:
        self.assertEqual(
            [row["comparison_type"] for row in self.table],
            [
                "genuine-genuine",
                "genuine-genuine",
                "genuine-genuine",
                "genuine-forge",
                "genuine-forge",
                "genuine-forge",
                "genuine-random",
                "genuine-random",
                "genuine-random",
            ],
        )

    def test_first_genuine_pair_carries_paths_score_and_match(self) -> None:
        first = self.table[0]
        self.assertEqual(first["sign1_path"], "Signer_001/genuine/00.png")
        self.assertEqual(first["sign2_path"], "Signer_001/genuine/01.png")
        self.assertAlmostEqual(first["cosine_similarity"], 1.0)
        self.assertEqual(first["threshold"], 0.5)
        self.assertTrue(first["matched"])

    def test_matched_follows_the_threshold_rule(self) -> None:
        # (1, 4) is genuine/forged and both point the same way -> 1.0 -> accepted.
        self.assertTrue(self.table[4]["matched"])
        self.assertAlmostEqual(self.table[4]["cosine_similarity"], 1.0)
        # (0, 3) is genuine/forged but orthogonal -> 0.0 -> rejected.
        self.assertFalse(self.table[3]["matched"])
        self.assertAlmostEqual(self.table[3]["cosine_similarity"], 0.0)

    def test_every_genuine_signature_takes_part(self) -> None:
        genuine_paths = {
            row["sign1_path"] for row in self.table if "genuine/" in row["sign1_path"]
        }
        self.assertEqual(
            genuine_paths,
            {
                "Signer_001/genuine/00.png",
                "Signer_001/genuine/01.png",
                "Signer_001/genuine/02.png",
            },
        )


class OverallSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.genuine = np.array([0.9, 0.1])
        self.negatives = np.array([0.8, 0.2])
        self.summary = overall_summary(
            self.genuine,
            self.negatives,
            0.5,
            signers=1,
            pair_counts={"genuine": 2, "skilled": 1, "random": 1},
            skilled=np.array([0.8]),
            random_scores=np.array([0.2]),
        )

    def test_confusion_counts(self) -> None:
        self.assertEqual(self.summary["true_positive"], 1)
        self.assertEqual(self.summary["false_negative"], 1)
        self.assertEqual(self.summary["false_positive"], 1)
        self.assertEqual(self.summary["true_negative"], 1)

    def test_rates_and_totals(self) -> None:
        self.assertAlmostEqual(self.summary["far"], 0.5)
        self.assertAlmostEqual(self.summary["frr"], 0.5)
        self.assertAlmostEqual(self.summary["accuracy"], 0.5)
        self.assertEqual(self.summary["total_sign_pairs"], 4)

    def test_signers_and_pair_counts_come_from_the_caller(self) -> None:
        self.assertEqual(self.summary["total_signers"], 1)
        self.assertEqual(self.summary["genuine_genuine_pairs"], 2)
        self.assertEqual(self.summary["genuine_forge_pairs"], 1)
        self.assertEqual(self.summary["genuine_random_pairs"], 1)

    def test_eer_matches_the_shared_helper(self) -> None:
        expected_threshold, expected_eer = eer_threshold(self.genuine, self.negatives)
        self.assertAlmostEqual(self.summary["eer"], expected_eer)
        self.assertAlmostEqual(self.summary["eer_threshold"], expected_threshold)

    def test_per_threat_breakdown_is_included(self) -> None:
        self.assertIn("far_skilled", self.summary)
        self.assertIn("eer_skilled", self.summary)
        self.assertIn("far_random", self.summary)
        self.assertIn("eer_random", self.summary)


class WriteReportTests(unittest.TestCase):
    def test_file_layout_has_pairs_then_metric_block(self) -> None:
        pair_rows = [
            {
                "sign1_path": "Signer_001/Genuine/a.png",
                "sign2_path": "Signer_001/Genuine/b.png",
                "comparison_type": "genuine-genuine",
                "cosine_similarity": 0.91,
                "threshold": 0.82,
                "matched": True,
            },
            {
                "sign1_path": "Signer_001/Genuine/a.png",
                "sign2_path": "Signer_001/Forged/a.png",
                "comparison_type": "genuine-forge",
                "cosine_similarity": 0.4,
                "threshold": 0.82,
                "matched": False,
            },
        ]
        summary = {"total_signers": 1, "true_positive": 1, "far": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pairs.csv"
            write_report(destination, pair_rows, summary)
            with destination.open(newline="", encoding="utf-8") as handle:
                table = list(csv.reader(handle))

        self.assertEqual(table[0], list(PAIR_COLUMNS))
        self.assertEqual(
            table[1],
            [
                "Signer_001/Genuine/a.png",
                "Signer_001/Genuine/b.png",
                "genuine-genuine",
                "0.91",
                "0.82",
                "true",
            ],
        )
        self.assertEqual(table[2][2], "genuine-forge")
        self.assertEqual(table[2][5], "false")
        # Blank separator, then the metric,value block.
        self.assertEqual(table[3], [])
        self.assertEqual(table[4], ["metric", "value"])
        self.assertEqual(table[5], ["total_signers", "1"])
        self.assertEqual(table[6], ["true_positive", "1"])
        self.assertEqual(table[7], ["far", "0.0"])

    def test_parent_directories_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "nested" / "deeper" / "pairs.csv"
            write_report(destination, [], {"threshold": 0.82})
            self.assertTrue(destination.is_file())


class FusionSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.genuine = np.array([0.9, 0.1])
        self.negatives = np.array([0.8, 0.2])
        self.deep = overall_summary(
            self.genuine,
            self.negatives,
            0.5,
            signers=1,
            pair_counts={"genuine": 2, "skilled": 1, "random": 1},
            skilled=np.array([0.8]),
            random_scores=np.array([0.2]),
        )
        self.fused = fusion_summary(
            self.genuine,
            self.negatives,
            0.5,
            skilled=np.array([0.8]),
            random_scores=np.array([0.2]),
        )

    def test_every_deep_metric_has_a_fused_twin(self) -> None:
        for key, value in self.deep.items():
            if key in {"total_signers", "total_sign_pairs", "genuine_genuine_pairs",
                       "genuine_forge_pairs", "genuine_random_pairs", "threshold"}:
                continue
            self.assertIn(f"{key}_fused", self.fused, key)
            self.assertEqual(self.fused[f"{key}_fused"], value, key)

    def test_threshold_is_reported_separately(self) -> None:
        self.assertEqual(self.fused["threshold_fused"], 0.5)
        self.assertNotIn("threshold", self.fused)


class FusedPairRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows, self.by_folder = _manifest({"Signer_001": (3, 3, 3)})
        self.pairs = pair_indices(self.rows, self.by_folder)
        self.embeddings = np.eye(9, dtype=np.float64)
        self.fused = {
            (first, second): 0.9
            for group in self.pairs.values()
            for first, second in group
        }

    def test_fused_columns_are_added_when_scores_are_supplied(self) -> None:
        table = build_pair_rows(
            self.rows, self.by_folder, self.pairs, self.embeddings, 0.8,
            fused_scores=self.fused, fused_threshold=0.85,
        )
        self.assertEqual(len(table), 9)
        for row in table:
            self.assertAlmostEqual(row["fused_similarity"], 0.9)
            self.assertEqual(row["fused_threshold"], 0.85)
            self.assertTrue(row["fused_matched"])

    def test_a_higher_fused_threshold_rejects(self) -> None:
        table = build_pair_rows(
            self.rows, self.by_folder, self.pairs, self.embeddings, 0.8,
            fused_scores=self.fused, fused_threshold=0.95,
        )
        self.assertFalse(table[0]["fused_matched"])

    def test_no_fused_columns_without_scores(self) -> None:
        table = build_pair_rows(
            self.rows, self.by_folder, self.pairs, self.embeddings, 0.8
        )
        self.assertNotIn("fused_similarity", table[0])

    def test_write_report_uses_the_fused_header(self) -> None:
        table = build_pair_rows(
            self.rows, self.by_folder, self.pairs, self.embeddings, 0.8,
            fused_scores=self.fused, fused_threshold=0.85,
        )
        columns = PAIR_COLUMNS + FUSION_COLUMNS
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pairs.csv"
            write_report(destination, table, {"threshold": 0.8}, columns)
            with destination.open(newline="", encoding="utf-8") as handle:
                written = list(csv.reader(handle))

        self.assertEqual(written[0], list(columns))
        self.assertEqual(written[1][6:], ["0.9", "0.85", "true"])


if __name__ == "__main__":
    unittest.main()
