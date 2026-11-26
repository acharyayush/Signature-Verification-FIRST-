"""Build the self-contained HTML report site for an Ultimate Test Signs run.

Reads the per-pair CSV produced by :mod:`scripts.export_ultimate_pairs` and assembles a
folder that can be zipped and opened offline::

    <site>/index.html, styles.css, app.js     the report site (static)
    <site>/data/pairs.csv                     the per-pair CSV, copied verbatim
    <site>/data/data.js                       the same data embedded for file:// use
    <site>/images/<Signer_...>/*.png          the Ultimate Test Signs images
    <site>/model/<checkpoint>.pt              the checkpoint the run used

The page reads ``data/data.js`` rather than fetching the CSV, because a page opened straight
from disk (``file://``) is not allowed to fetch local files. The CSV is still shipped so the
raw numbers travel with the report.

Usage:
    python -m scripts.build_report_site \\
        --csv ultimate_pairs_grayscale.csv \\
        --image-root "Ultimate Test Signs" \\
        --model ultimate_best_model.pt \\
        --site-dir ultimate_report_site
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PAIR_COLUMNS = (
    "sign1_path",
    "sign2_path",
    "comparison_type",
    "cosine_similarity",
    "threshold",
    "matched",
)
# Optional columns appended when the CSV was exported with a fusion config.
FUSION_COLUMNS = ("fused_similarity", "fused_threshold", "fused_matched")
META_KEYS = ("checkpoint", "manifest", "device", "fusion_type", "fusion_config")


def _parse_bool(value: str, *, row_number: int, column: str) -> bool:
    """Parse a ``true``/``false`` CSV cell, reporting the offending row on failure."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"row {row_number}: {column} must be true/false, got {value!r}"
    )


def read_pairs_csv(path: Path) -> tuple[list[dict[str, object]], list[list[str]]]:
    """Return ``(pairs, summary)`` from a CSV written by ``export_ultimate_pairs``.

    Both the deep-only header and the deep-plus-fused header are accepted; when the fused
    columns are present each pair also carries ``fused``, ``fused_threshold``, and
    ``fused_matched``.

    Raises:
        ValueError: If the header, a pair row, or the ``metric,value`` block is malformed.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = tuple(cell.strip() for cell in rows[0])
    if header not in (PAIR_COLUMNS, PAIR_COLUMNS + FUSION_COLUMNS):
        raise ValueError(f"{path} does not start with the expected pair header")
    fused = header == PAIR_COLUMNS + FUSION_COLUMNS
    width = len(header)

    pairs: list[dict[str, object]] = []
    index = 1
    while index < len(rows) and len(rows[index]) == width:
        cells = rows[index]
        pair: dict[str, object] = {
            "a": cells[0],
            "b": cells[1],
            "type": cells[2],
            "cos": float(cells[3]),
            "threshold": float(cells[4]),
            "matched": _parse_bool(
                cells[5], row_number=index + 1, column="matched"
            ),
        }
        if fused:
            pair["fused"] = float(cells[6])
            pair["fused_threshold"] = float(cells[7])
            pair["fused_matched"] = _parse_bool(
                cells[8], row_number=index + 1, column="fused_matched"
            )
        pairs.append(pair)
        index += 1
    if not pairs:
        raise ValueError(f"{path} has no pair rows")

    while index < len(rows) and not rows[index]:
        index += 1
    summary: list[list[str]] = []
    if index < len(rows) and [cell.strip().lower() for cell in rows[index]] == ["metric", "value"]:
        index += 1
        while index < len(rows) and len(rows[index]) >= 2:
            summary.append([rows[index][0].strip(), rows[index][1].strip()])
            index += 1
    if not summary:
        raise ValueError(f"{path} has no metric,value summary block")
    return pairs, summary


def write_data_js(
    site_dir: Path,
    pairs: list[dict[str, object]],
    summary: list[list[str]],
    *,
    source_csv: str,
) -> Path:
    """Write ``data/data.js`` holding ``window.ULTIMATE_REPORT`` for the page to read."""
    meta: dict[str, str] = {
        "source_csv": source_csv,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    }
    for metric, value in summary:
        if metric in META_KEYS:
            meta[metric] = value
    payload = {"meta": meta, "summary": summary, "pairs": pairs}
    destination = site_dir / "data" / "data.js"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "window.ULTIMATE_REPORT = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    return destination


def copy_images(
    pairs: list[dict[str, object]], image_root: Path, images_dir: Path
) -> int:
    """Copy every signature the CSV references into ``images_dir``, keeping its layout.

    Raises:
        ValueError: If a referenced image is missing from ``image_root``.
    """
    copied = 0
    for pair in pairs:
        for relative in (pair["a"], pair["b"]):
            source = image_root / str(relative)
            target = images_dir / str(relative)
            if target.is_file():
                continue
            if not source.is_file():
                raise ValueError(f"image referenced by the CSV is missing: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--site-dir", type=Path, required=True)
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    model = args.model.expanduser().resolve()
    site = args.site_dir.expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"checkpoint not found: {model}")
    site.mkdir(parents=True, exist_ok=True)

    pairs, summary = read_pairs_csv(csv_path)
    data_js = write_data_js(site, pairs, summary, source_csv=csv_path.name)

    data_dir = site / "data"
    shutil.copy2(csv_path, data_dir / "pairs.csv")

    images_copied = copy_images(pairs, image_root, site / "images")

    model_dir = site / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, model_dir / model.name)

    print(f"Pairs         : {len(pairs)}")
    print(f"Summary rows  : {len(summary)}")
    print(f"Images copied : {images_copied}")
    print(f"Data file     : {data_js}")
    print(f"Model         : {model_dir / model.name}")
    print(f"Site folder   : {site}  (zip this folder to share the report)")


if __name__ == "__main__":
    main()
