"""Tests for the report-site builder in :mod:`scripts.build_report_site`.

The builder reads the per-pair CSV that :mod:`scripts.export_ultimate_pairs` writes and
turns it into the files the HTML report reads. These tests pin down the CSV parsing (pairs
plus the trailing ``metric,value`` block), the image copying, and the ``data.js`` payload.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_report_site import (
    FUSION_COLUMNS,
    PAIR_COLUMNS,
    copy_images,
    read_pairs_csv,
    write_data_js,
)


def _write_csv(path: Path, pairs, summary, header=PAIR_COLUMNS) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(pairs)
        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerows(summary)


class ReadPairsCsvTests(unittest.TestCase):
    def test_reads_pairs_and_the_summary_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            _write_csv(
                path,
                [
                    ["a/1.png", "a/2.png", "genuine-genuine", "0.9", "0.82", "true"],
                    ["a/1.png", "a/3.png", "genuine-forge", "0.4", "0.82", "false"],
                ],
                [["threshold", "0.82"], ["far", "0.09"]],
            )
            pairs, summary = read_pairs_csv(path)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["a"], "a/1.png")
        self.assertEqual(pairs[0]["type"], "genuine-genuine")
        self.assertAlmostEqual(pairs[0]["cos"], 0.9)
        self.assertTrue(pairs[0]["matched"])
        self.assertFalse(pairs[1]["matched"])
        self.assertEqual(summary, [["threshold", "0.82"], ["far", "0.09"]])

    def test_reads_the_fused_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            _write_csv(
                path,
                [["a/1.png", "a/2.png", "genuine-forge", "0.9", "0.82", "true",
                  "0.7", "0.88", "false"]],
                [["threshold", "0.82"], ["threshold_fused", "0.88"]],
                header=PAIR_COLUMNS + FUSION_COLUMNS,
            )
            pairs, summary = read_pairs_csv(path)

        self.assertAlmostEqual(pairs[0]["fused"], 0.7)
        self.assertEqual(pairs[0]["fused_threshold"], 0.88)
        self.assertFalse(pairs[0]["fused_matched"])
        self.assertTrue(pairs[0]["matched"])
        self.assertEqual(summary, [["threshold", "0.82"], ["threshold_fused", "0.88"]])

    def test_invalid_fused_matched_value_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            _write_csv(
                path,
                [["a/1.png", "a/2.png", "genuine-forge", "0.9", "0.82", "true",
                  "0.7", "0.88", "sometimes"]],
                [["threshold", "0.82"]],
                header=PAIR_COLUMNS + FUSION_COLUMNS,
            )
            with self.assertRaises(ValueError):
                read_pairs_csv(path)

    def test_wrong_header_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            path.write_text("nope,value\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_pairs_csv(path)

    def test_invalid_matched_value_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            _write_csv(
                path,
                [["a/1.png", "a/2.png", "genuine-genuine", "0.9", "0.82", "maybe"]],
                [["threshold", "0.82"]],
            )
            with self.assertRaises(ValueError):
                read_pairs_csv(path)

    def test_missing_summary_block_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(PAIR_COLUMNS)
                writer.writerow(["a/1.png", "a/2.png", "genuine-genuine", "0.9", "0.82", "true"])
            with self.assertRaises(ValueError):
                read_pairs_csv(path)


class CopyImagesTests(unittest.TestCase):
    def test_copies_referenced_images_keeping_the_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            (root / "Signer_001" / "Genuine").mkdir(parents=True)
            (root / "Signer_001" / "Genuine" / "a.png").write_bytes(b"a")
            (root / "Signer_001" / "Forged").mkdir(parents=True)
            (root / "Signer_001" / "Forged" / "b.png").write_bytes(b"b")
            images = Path(tmp) / "images"
            pairs = [
                {"a": "Signer_001/Genuine/a.png", "b": "Signer_001/Forged/b.png"},
                {"a": "Signer_001/Genuine/a.png", "b": "Signer_001/Forged/b.png"},
            ]
            copied = copy_images(pairs, root, images)

            # a.png is referenced twice but copied once.
            self.assertEqual(copied, 2)
            self.assertTrue((images / "Signer_001" / "Genuine" / "a.png").is_file())
            self.assertTrue((images / "Signer_001" / "Forged" / "b.png").is_file())

    def test_missing_image_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            with self.assertRaises(ValueError):
                copy_images(
                    [{"a": "Signer_001/Genuine/missing.png", "b": "x/y.png"}],
                    root,
                    Path(tmp) / "images",
                )


class WriteDataJsTests(unittest.TestCase):
    def test_writes_the_window_global_with_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            pairs = [
                {"a": "a/1.png", "b": "a/2.png", "type": "genuine-forge", "cos": 0.5,
                 "threshold": 0.82, "matched": False}
            ]
            summary = [
                ["checkpoint", "/models/best.pt"],
                ["device", "cpu"],
                ["threshold", "0.82"],
            ]
            destination = write_data_js(site, pairs, summary, source_csv="pairs.csv")

            self.assertEqual(destination, site / "data" / "data.js")
            text = destination.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("window.ULTIMATE_REPORT = "))
            payload = json.loads(text[len("window.ULTIMATE_REPORT = "):].rstrip().rstrip(";"))
            self.assertEqual(payload["pairs"], pairs)
            self.assertEqual(payload["summary"], summary)
            self.assertEqual(payload["meta"]["checkpoint"], "/models/best.pt")
            self.assertEqual(payload["meta"]["source_csv"], "pairs.csv")


if __name__ == "__main__":
    unittest.main()
