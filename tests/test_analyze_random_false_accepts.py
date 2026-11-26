"""Tests for the false-accept logger in :mod:`scripts.analyze_random_false_accepts`.

The script replays the enrolment protocol, keeps every random query whose mean score
reaches the threshold, and writes a per-pair CSV plus contact sheets. These tests pin down
the per-reference pattern classifier, the hub concentration, the summary, the CSV layout,
the image metadata, and the contact sheet rendering. ``compute_false_accepts`` is exercised
with a patched embedding step so no model or GPU is needed.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scoring.fusion import Enrollment, Query
from scripts.analyze_random_false_accepts import (
    _concentration,
    _pattern,
    build_contact_sheet,
    compute_false_accepts,
    count_random_queries,
    image_metadata,
    normalize_dataset,
    raw_path_for,
    read_false_accepts_csv,
    summarize,
    write_csv,
    write_hub_analysis_csv,
)
from training.stage3_metric_learning import SignatureRecord


def _write_png(path: Path, size: tuple[int, int] = (40, 20), ink: bool = True) -> None:
    """Write a small white PNG with an optional dark rectangle so ink metrics are non-zero."""
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", size, 255)
    if ink:
        draw = ImageDraw.Draw(image)
        draw.rectangle([2, 2, size[0] - 3, size[1] - 3], fill=0)
    image.save(path)


def _row(
    *,
    query_signer: str,
    ref_signer: str,
    query_dataset: str,
    ref_dataset: str,
    score: float,
    per_ref: list[float],
    decision: str = "deep",
    fused: float | None = None,
) -> dict[str, object]:
    """Build one false-accept row in the shape ``compute_false_accepts`` returns."""
    row: dict[str, object] = {
        "query_id": f"{query_dataset}/Genuine/{query_signer}.png",
        "query_signer_id": query_signer,
        "query_dataset": query_dataset,
        "ref_signer_id": ref_signer,
        "ref_dataset": ref_dataset,
        "score": score,
        "decision": decision,
        "decision_score": score if fused is None else fused,
        "per_ref_scores": list(per_ref),
        "ref_ids": [
            f"{ref_dataset}/Genuine/{ref_signer}_{i}.png" for i in range(len(per_ref))
        ],
        "ref_signer_ids": [ref_signer] * len(per_ref),
        "n_refs": len(per_ref),
        "query_metadata": {
            "source": "processed",
            "width": 40,
            "height": 20,
            "aspect_ratio": 2.0,
            "ink_density": 0.8,
            "bbox": [2, 2, 37, 17],
        },
        "ref_metadata": [
            {
                "source": "processed",
                "width": 40,
                "height": 20,
                "aspect_ratio": 2.0,
                "ink_density": 0.8,
                "bbox": [2, 2, 37, 17],
            }
            for _ in per_ref
        ],
    }
    if fused is not None:
        row["fused_score"] = fused
        row["per_ref_fused_scores"] = [fused] * len(per_ref)
    return row


class TestNormalizeDataset(unittest.TestCase):
    def test_aliases_map_to_short_labels(self) -> None:
        self.assertEqual(normalize_dataset("BHSig260/BHSig100_Bengali"), "bhsig_bn")
        self.assertEqual(normalize_dataset("BHSig260/BHSig160_Hindi"), "bhsig_hi")
        self.assertEqual(normalize_dataset("CEDAR/full_org"), "cedar")

    def test_unknown_label_is_returned_unchanged(self) -> None:
        self.assertEqual(normalize_dataset("something/else"), "something/else")


class TestPattern(unittest.TestCase):
    def test_all_refs_similar_when_spread_is_small(self) -> None:
        self.assertEqual(_pattern([0.90, 0.91, 0.905, 0.895, 0.902]), "all_refs_similar")

    def test_single_reference_when_one_score_dominates(self) -> None:
        self.assertEqual(_pattern([0.95, 0.80, 0.79, 0.81, 0.80]), "single_reference")

    def test_mixed_when_spread_is_between(self) -> None:
        self.assertEqual(_pattern([0.90, 0.85, 0.86, 0.84, 0.87]), "mixed")

    def test_single_reference_count_is_reported(self) -> None:
        self.assertEqual(_pattern([0.9]), "single_reference")

    def test_unknown_when_empty(self) -> None:
        self.assertEqual(_pattern([]), "unknown")


class TestConcentration(unittest.TestCase):
    def test_top_shares_and_distinct_signers(self) -> None:
        from collections import Counter

        counts = Counter({"a": 6, "b": 3, "c": 1})
        result = _concentration(counts)
        self.assertEqual(result["total_false_accepts"], 10)
        self.assertEqual(result["distinct_signers"], 3)
        self.assertEqual(result["top_signers"][0], {"signer": "a", "false_accepts": 6})
        self.assertEqual(result["top_1_share"], 0.6)
        self.assertEqual(result["top_3_share"], 1.0)


class TestSummarize(unittest.TestCase):
    def test_counts_datasets_and_patterns(self) -> None:
        rows = [
            _row(
                query_signer="q1",
                ref_signer="r1",
                query_dataset="bhsig_hi",
                ref_dataset="bhsig_hi",
                score=0.9,
                per_ref=[0.9, 0.91, 0.9, 0.9, 0.9],
            ),
            _row(
                query_signer="q2",
                ref_signer="r2",
                query_dataset="bhsig_hi",
                ref_dataset="bhsig_hi",
                score=0.88,
                per_ref=[0.95, 0.8, 0.8, 0.8, 0.8],
            ),
            _row(
                query_signer="q3",
                ref_signer="r3",
                query_dataset="cedar",
                ref_dataset="cedar",
                score=0.85,
                per_ref=[0.9, 0.85, 0.86, 0.84, 0.87],
            ),
        ]
        summary = summarize(rows, threshold=0.823, decision="deep")
        self.assertEqual(summary["false_accepts"], 3)
        self.assertEqual(summary["threshold"], 0.823)
        self.assertEqual(summary["decision"], "deep")
        by_dataset = {
            row["dataset"]: row["false_accepts"] for row in summary["by_dataset"]
        }
        self.assertEqual(by_dataset, {"bhsig_hi": 2, "cedar": 1})
        by_pattern = {
            row["pattern"]: row["false_accepts"] for row in summary["patterns"]
        }
        self.assertEqual(by_pattern["all_refs_similar"], 1)
        self.assertEqual(by_pattern["single_reference"], 1)
        self.assertEqual(by_pattern["mixed"], 1)
        self.assertEqual(
            summary["hub_analysis"]["query_signers"]["distinct_signers"], 3
        )


class TestImageMetadata(unittest.TestCase):
    def test_processed_image_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Genuine" / "a.png"
            _write_png(path, size=(40, 20))
            meta = image_metadata(path)
            self.assertEqual(meta["source"], "processed")
            self.assertEqual(meta["width"], 40)
            self.assertEqual(meta["height"], 20)
            self.assertEqual(meta["aspect_ratio"], 2.0)
            self.assertGreater(meta["ink_density"], 0.0)
            self.assertEqual(meta["bbox"], [2, 2, 37, 17])

    def test_raw_image_is_preferred_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed" / "Genuine" / "a.png"
            raw = root / "raw" / "Genuine" / "a.png"
            _write_png(processed, size=(40, 20))
            _write_png(raw, size=(80, 40))
            meta = image_metadata(
                processed, raw_root=root / "raw", image_root=root / "processed"
            )
            self.assertEqual(meta["source"], "raw")
            self.assertEqual(meta["width"], 80)

    def test_missing_image_returns_none_fields(self) -> None:
        meta = image_metadata(Path("/nonexistent/x.png"))
        self.assertEqual(meta["source"], "missing")
        self.assertIsNone(meta["width"])

    def test_raw_path_for_tries_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "Genuine" / "a.tif"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"not really a tif")
            found = raw_path_for(
                root / "processed" / "Genuine" / "a.png",
                root / "processed",
                root / "raw",
            )
            self.assertEqual(found, raw)


class TestCountRandomQueries(unittest.TestCase):
    def test_counts_only_random_queries(self) -> None:
        records = [
            SignatureRecord(Path("/x/a.png"), "s1", False),
            SignatureRecord(Path("/x/b.png"), "s1", False),
            SignatureRecord(Path("/x/c.png"), "s1", False),
            SignatureRecord(Path("/x/d.png"), "s2", False),
            SignatureRecord(Path("/x/e.png"), "s2", False),
            SignatureRecord(Path("/x/f.png"), "s2", False),
        ]
        # 2 references per signer leaves 1 genuine query each; s1's query is random against
        # s2 and s2's query is random against s1 -> 2 random queries.
        self.assertEqual(count_random_queries(records, 2), 2)


class TestWriteCsv(unittest.TestCase):
    def test_ranks_by_decision_score_and_writes_summary(self) -> None:
        rows = [
            _row(
                query_signer="q1",
                ref_signer="r1",
                query_dataset="bhsig_hi",
                ref_dataset="bhsig_hi",
                score=0.86,
                per_ref=[0.86, 0.87, 0.86, 0.86, 0.86],
            ),
            _row(
                query_signer="q2",
                ref_signer="r2",
                query_dataset="cedar",
                ref_dataset="cedar",
                score=0.94,
                per_ref=[0.94, 0.95, 0.94, 0.94, 0.94],
            ),
        ]
        summary = summarize(rows, threshold=0.823, decision="deep")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.csv"
            write_csv(path, rows, summary, total_random_pairs=10)
            with path.open(newline="", encoding="utf-8") as handle:
                lines = list(csv.reader(handle))
        header = lines[0]
        self.assertEqual(header[0], "rank")
        self.assertIn("per_ref_scores", header)
        self.assertIn("query_bbox", header)
        # Highest score first.
        self.assertEqual(lines[1][0], "1")
        self.assertEqual(lines[1][1], "cedar/Genuine/q2.png")
        self.assertEqual(lines[2][0], "2")
        # Summary block at the end.
        self.assertIn(["metric", "value"], lines)
        flat = {row[0]: row[1] for row in lines if len(row) == 2}
        self.assertEqual(flat["false_accepts"], "2")
        self.assertEqual(flat["far_random"], "0.200000")

    def test_reader_splits_pairs_from_summary(self) -> None:
        rows = [
            _row(
                query_signer="q1",
                ref_signer="r1",
                query_dataset="cedar",
                ref_dataset="cedar",
                score=0.94,
                per_ref=[0.94, 0.95, 0.94, 0.94, 0.94],
            )
        ]
        summary = summarize(rows, threshold=0.823, decision="deep")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.csv"
            write_csv(path, rows, summary, total_random_pairs=10)
            read_rows, read_summary = read_false_accepts_csv(path)
        self.assertEqual(len(read_rows), 1)
        self.assertEqual(read_rows[0]["query_signer_id"], "q1")
        self.assertIsInstance(read_rows[0]["per_ref_scores"], list)
        self.assertEqual(read_summary["false_accepts"], "1")
        self.assertEqual(read_summary["decision"], "deep")


class TestHubAnalysisCsv(unittest.TestCase):
    def test_writes_query_and_reference_rows_ranked(self) -> None:
        rows = [
            _row(
                query_signer="hub",
                ref_signer="r1",
                query_dataset="bhsig_hi",
                ref_dataset="bhsig_hi",
                score=0.9,
                per_ref=[0.9, 0.91, 0.905, 0.895, 0.9],
            ),
            _row(
                query_signer="hub",
                ref_signer="r2",
                query_dataset="bhsig_hi",
                ref_dataset="bhsig_hi",
                score=0.88,
                per_ref=[0.88, 0.89, 0.885, 0.875, 0.88],
            ),
            _row(
                query_signer="other",
                ref_signer="hub",
                query_dataset="cedar",
                ref_dataset="cedar",
                score=0.86,
                per_ref=[0.86, 0.87, 0.865, 0.855, 0.86],
            ),
        ]
        summary = summarize(rows, threshold=0.823, decision="deep")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hub.csv"
            write_hub_analysis_csv(path, summary)
            with path.open(newline="", encoding="utf-8") as handle:
                lines = list(csv.reader(handle))
        self.assertEqual(lines[0], ["role", "signer", "false_accepts", "share"])
        by_role_signer = {(row[0], row[1]): row[2] for row in lines[1:]}
        self.assertEqual(by_role_signer[("query", "hub")], "2")
        self.assertEqual(by_role_signer[("query", "other")], "1")
        self.assertEqual(by_role_signer[("reference", "hub")], "1")
        # The query block is ranked, so the hub signer comes first.
        self.assertEqual(lines[1][:3], ["query", "hub", "2"])


class TestContactSheet(unittest.TestCase):
    def test_renders_a_png_with_query_and_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_png(root / "bhsig_hi" / "Genuine" / "q1.png")
            for index in range(5):
                _write_png(root / "bhsig_hi" / "Genuine" / f"r1_{index}.png")
            row = _row(
                query_signer="q1",
                ref_signer="r1",
                query_dataset="bhsig_hi",
                ref_dataset="bhsig_hi",
                score=0.9,
                per_ref=[0.9, 0.91, 0.9, 0.9, 0.9],
            )
            row["query_id"] = str(root / "bhsig_hi" / "Genuine" / "q1.png")
            row["ref_ids"] = [
                str(root / "bhsig_hi" / "Genuine" / f"r1_{i}.png") for i in range(5)
            ]
            destination = root / "sheet.png"
            build_contact_sheet(row, destination, rank=1, image_root=root)
            self.assertTrue(destination.is_file())
            from PIL import Image

            with Image.open(destination) as image:
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)


class TestComputeFalseAccepts(unittest.TestCase):
    def test_keeps_pairs_at_or_above_threshold(self) -> None:
        records = [
            SignatureRecord(Path("/x/r1.png"), "s1", False),
            SignatureRecord(Path("/x/r2.png"), "s1", False),
            SignatureRecord(Path("/x/q1.png"), "s2", False),
            SignatureRecord(Path("/x/q2.png"), "s2", False),
        ]
        enrollment = Enrollment(
            references={"s1": [0, 1], "s2": [2, 3]},
            queries=[
                Query(index=2, target_signer="s1", kind="random"),
                Query(index=3, target_signer="s1", kind="random"),
            ],
        )
        # q1 matches r1/r2 strongly; q2 is weak.
        embeddings = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.95, 0.312], [0.2, 0.98]], dtype=np.float64
        )
        with mock.patch(
            "scripts.analyze_random_false_accepts.embed_records",
            return_value=embeddings,
        ), mock.patch(
            "scripts.analyze_random_false_accepts.enroll_and_build_queries",
            return_value=enrollment,
        ):
            rows = compute_false_accepts(
                object(),
                records,
                threshold=0.9,
                references=2,
                device="cpu",
                batch_size=1,
                workers=0,
                datasets=["bhsig_hi"] * 4,
                report_datasets=["bhsig_hi"] * 4,
            )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["query_signer_id"], "s2")
        self.assertEqual(row["ref_signer_id"], "s1")
        self.assertEqual(row["decision"], "deep")
        self.assertAlmostEqual(row["score"], 0.95, places=6)
        self.assertEqual(len(row["per_ref_scores"]), 2)

    def test_fused_decision_requires_a_scorer(self) -> None:
        with self.assertRaises(ValueError):
            compute_false_accepts(
                object(),
                [],
                threshold=0.5,
                references=2,
                device="cpu",
                batch_size=1,
                workers=0,
                decision="fused",
            )


if __name__ == "__main__":
    unittest.main()
