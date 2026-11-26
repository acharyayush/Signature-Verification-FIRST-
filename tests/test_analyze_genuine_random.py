"""Tests for the genuine-random review script in :mod:`scripts.analyze_genuine_random`.

The script isolates the ``genuine-random`` rows of a per-pair CSV -- a genuine signature
against a genuine signature of a *different* signer -- and evaluates the subset. These
tests pin down the filtering, the decision rules, the summary numbers, the filtered CSV,
and the HTML report.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_genuine_random import (
    _decision_accepts,
    build_html_report,
    evaluate_genuine_random,
    filter_genuine_random,
    read_manifest_datasets,
    read_manifest_signers,
    write_filtered_csv,
)
from scripts.build_report_site import FUSION_COLUMNS, PAIR_COLUMNS


def _pair(
    signer: str,
    kind: str,
    score: float,
    matched: bool,
    *,
    fused: float | None = None,
    fused_matched: bool | None = None,
) -> dict[str, object]:
    """Build one parsed-pair dict in the shape ``read_pairs_csv`` returns."""
    pair: dict[str, object] = {
        "a": f"{signer}/Genuine/01.png",
        "b": f"{signer}/Random/01.png" if kind == "genuine-random" else f"{signer}/Genuine/02.png",
        "type": kind,
        "cos": score,
        "threshold": 0.823,
        "matched": matched,
    }
    if fused is not None:
        pair["fused"] = fused
        pair["fused_threshold"] = 0.8
        pair["fused_matched"] = bool(fused_matched)
    return pair


def _write_csv(path: Path, pairs, summary, header=PAIR_COLUMNS) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(pairs)
        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerows(summary)


def _write_manifest(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["folder", "class", "source_dataset", "source_signer", "image_path"])
        writer.writerows(rows)


def _write_png(path: Path) -> None:
    """Write a 1x1 PNG so the HTML builder has a real image to embed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000d49444154789c6360000002000100ffff03000006000557bfabd400"
            "00000049454e44ae426082"
        )
    )


class DecisionTests(unittest.TestCase):
    def test_deep_decision_uses_matched(self) -> None:
        pair = _pair("Signer_001", "genuine-random", 0.9, True, fused=0.7, fused_matched=False)
        self.assertTrue(_decision_accepts(pair, "deep"))
        self.assertFalse(_decision_accepts(pair, "fused"))
        self.assertTrue(_decision_accepts(pair, "either"))
        self.assertFalse(_decision_accepts(pair, "both"))

    def test_fused_decision_uses_fused_matched(self) -> None:
        pair = _pair("Signer_001", "genuine-random", 0.5, False, fused=0.9, fused_matched=True)
        self.assertFalse(_decision_accepts(pair, "deep"))
        self.assertTrue(_decision_accepts(pair, "fused"))
        self.assertTrue(_decision_accepts(pair, "either"))
        self.assertFalse(_decision_accepts(pair, "both"))

    def test_unknown_decision_raises(self) -> None:
        pair = _pair("Signer_001", "genuine-random", 0.9, True)
        with self.assertRaises(ValueError):
            _decision_accepts(pair, "nope")


class FilterTests(unittest.TestCase):
    def test_keeps_only_genuine_random(self) -> None:
        pairs = [
            _pair("Signer_001", "genuine-genuine", 0.9, True),
            _pair("Signer_001", "genuine-random", 0.9, True),
            _pair("Signer_001", "genuine-forge", 0.4, False),
            _pair("Signer_002", "genuine-random", 0.1, False),
        ]
        filtered = filter_genuine_random(pairs)
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(pair["type"] == "genuine-random" for pair in filtered))
        self.assertEqual(
            [pair["a"] for pair in filtered],
            ["Signer_001/Genuine/01.png", "Signer_002/Genuine/01.png"],
        )


class ManifestTests(unittest.TestCase):
    def test_reads_datasets_and_signers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            _write_manifest(
                manifest,
                [
                    ["Signer_001", "genuine", "bhsig260", "bhsig_bn_002", "Signer_001/Genuine/01.png"],
                    ["Signer_001", "forged", "bhsig260", "bhsig_bn_002", "Signer_001/Forged/01.png"],
                    ["Signer_002", "genuine", "cedar", "cedar_001", "Signer_002/Genuine/01.png"],
                ],
            )
            datasets = read_manifest_datasets(manifest)
            signers = read_manifest_signers(manifest)
        self.assertEqual(datasets, {"Signer_001": "bhsig260", "Signer_002": "cedar"})
        self.assertEqual(signers, ["Signer_001", "Signer_002"])


class EvaluateTests(unittest.TestCase):
    def _pairs(self):
        return [
            _pair("Signer_001", "genuine-random", 0.9, True, fused=0.85, fused_matched=True),
            _pair("Signer_001", "genuine-random", 0.5, False, fused=0.6, fused_matched=False),
            _pair("Signer_002", "genuine-random", 0.85, True, fused=0.7, fused_matched=False),
            _pair("Signer_002", "genuine-random", 0.2, False, fused=0.3, fused_matched=False),
        ]

    def test_deep_decision_summary(self) -> None:
        summary = evaluate_genuine_random(self._pairs(), decision="deep")
        self.assertEqual(summary["total_pairs"], 4)
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["rejected"], 2)
        self.assertAlmostEqual(summary["far_random"], 0.5)
        self.assertEqual(summary["accepted_deep"], 2)
        self.assertEqual(summary["accepted_fused"], 1)
        self.assertAlmostEqual(summary["far_random_fused"], 0.25)
        self.assertEqual(summary["signers_with_pairs"], 2)
        self.assertEqual(summary["signers_with_accepted"], 2)
        self.assertAlmostEqual(summary["score_max"], 0.9)
        self.assertAlmostEqual(summary["score_min"], 0.2)

    def test_fused_decision_summary(self) -> None:
        summary = evaluate_genuine_random(self._pairs(), decision="fused")
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["rejected"], 3)
        self.assertAlmostEqual(summary["far_random"], 0.25)

    def test_per_signer_breakdown(self) -> None:
        summary = evaluate_genuine_random(self._pairs(), decision="deep")
        by_signer = {row["signer"]: row for row in summary["per_signer"]}
        self.assertEqual(by_signer["Signer_001"]["pairs"], 2)
        self.assertEqual(by_signer["Signer_001"]["accepted"], 1)
        self.assertEqual(by_signer["Signer_002"]["pairs"], 2)
        self.assertEqual(by_signer["Signer_002"]["accepted"], 1)

    def test_manifest_adds_dataset_and_missing_signers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            _write_manifest(
                manifest,
                [
                    ["Signer_001", "genuine", "bhsig260", "bhsig_bn_002", "Signer_001/Genuine/01.png"],
                    ["Signer_002", "genuine", "cedar", "cedar_001", "Signer_002/Genuine/01.png"],
                    ["Signer_003", "genuine", "sigresize", "sig_001", "Signer_003/Genuine/01.png"],
                ],
            )
            summary = evaluate_genuine_random(
                self._pairs(), decision="deep", manifest=manifest
            )
        self.assertEqual(summary["manifest_signers"], 3)
        self.assertEqual(summary["signers_without_pairs"], ["Signer_003"])
        by_dataset = {row["dataset"]: row for row in summary["per_dataset"]}
        self.assertEqual(by_dataset["bhsig260"]["pairs"], 2)
        self.assertEqual(by_dataset["cedar"]["pairs"], 2)


class WriteCsvTests(unittest.TestCase):
    def test_writes_pair_rows_and_summary(self) -> None:
        pairs = filter_genuine_random(
            [
                _pair("Signer_001", "genuine-random", 0.9, True, fused=0.85, fused_matched=True),
                _pair("Signer_001", "genuine-random", 0.5, False, fused=0.6, fused_matched=False),
            ]
        )
        summary = evaluate_genuine_random(pairs, decision="deep")
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "out.csv"
            write_filtered_csv(destination, pairs, summary, decision="deep")
            with destination.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        header = rows[0]
        self.assertEqual(header[-3:], ["signer", "dataset", "accepted"])
        self.assertEqual(rows[1][-3:], ["Signer_001", "unknown", "true"])
        self.assertEqual(rows[2][-3:], ["Signer_001", "unknown", "false"])
        metric_start = next(
            index for index, row in enumerate(rows) if row == ["metric", "value"]
        )
        metrics = {row[0]: row[1] for row in rows[metric_start + 1 :]}
        self.assertEqual(metrics["total_pairs"], "2")
        self.assertEqual(metrics["accepted"], "1")
        self.assertEqual(metrics["far_random"], "0.5")


class HtmlTests(unittest.TestCase):
    def test_writes_cards_and_embedded_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_png(root / "Signer_001" / "Genuine" / "01.png")
            _write_png(root / "Signer_001" / "Random" / "01.png")
            pairs = filter_genuine_random(
                [
                    _pair("Signer_001", "genuine-random", 0.9, True, fused=0.85, fused_matched=True),
                ]
            )
            summary = evaluate_genuine_random(pairs, decision="deep")
            destination = root / "report.html"
            build_html_report(
                destination,
                pairs,
                summary,
                image_root=root,
                decision="deep",
                show="accepted",
            )
            page = destination.read_text(encoding="utf-8")
        self.assertIn('<article class="card">', page)
        self.assertIn("data:image/png;base64", page)
        self.assertIn("Signer_001", page)
        self.assertIn("Per-signer breakdown", page)
        self.assertIn("Reviewed pairs (1)", page)

    def test_show_all_includes_rejected_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_png(root / "Signer_001" / "Genuine" / "01.png")
            _write_png(root / "Signer_001" / "Random" / "01.png")
            pairs = filter_genuine_random(
                [
                    _pair("Signer_001", "genuine-random", 0.9, True),
                    _pair("Signer_001", "genuine-random", 0.5, False),
                ]
            )
            summary = evaluate_genuine_random(pairs, decision="deep")
            destination = root / "report.html"
            build_html_report(
                destination,
                pairs,
                summary,
                image_root=root,
                decision="deep",
                show="all",
            )
            page = destination.read_text(encoding="utf-8")
        self.assertEqual(page.count('<article class="card">'), 2)
        self.assertIn("Reviewed pairs (2)", page)


if __name__ == "__main__":
    unittest.main()
