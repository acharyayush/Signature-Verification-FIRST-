"""Log and inspect false-accept random pairs of the enrolment protocol.

The enrolment protocol (``training.evaluate``) enrols the first ``--references`` genuine
signatures of every signer and scores queries against those references. A *random* query
is a genuine signature of a **different** signer: when its mean cosine similarity to the
reference signer's references reaches the threshold, the pair is a false accept (FAR).

This script replays that protocol, keeps every random pair whose score is at or above the
threshold, and writes everything needed to inspect the false accepts:

* ``random_false_accepts.csv`` -- one row per false accept with ``query_id``,
  ``query_signer_id``, ``query_dataset``, ``ref_signer_id``, ``score``, ``per_ref_scores``
  (JSON), the reference image ids, and image metadata (width, height, aspect ratio, ink
  density, crop bounding box) for the query and every reference.
* ``random_false_accepts_summary.json`` -- the FAR, the per-dataset / per-signer counts,
  and the hub analysis (which query signers and reference signers dominate).
* ``hub_analysis.csv`` -- per query signer and per reference signer false-accept counts,
  ranked, so the "hub" signers are the first rows.
* ``contact_sheets/<rank>_<query>_vs_<ref>.png`` -- the top-scoring false accepts as
  side-by-side contact sheets: the query, the five references, and each reference's score.

The per-reference scores are what make a false accept diagnosable: a single reference
pulling the mean up looks very different from five references that all look similar. The
summary counts both cases (``single_reference`` when the best reference is far above the
other references, ``all_refs_similar`` when the spread across references is small).

Usage:
    python -m scripts.analyze_random_false_accepts \\
        --checkpoint ultimate_best_model.pt \\
        --manifest data/grayscale_processed/manifest.csv \\
        --image-root data/grayscale_processed \\
        --raw-root data/raw \\
        --output-dir random_false_accepts

    python -m scripts.analyze_random_false_accepts ... --fusion-config fusion.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scoring.fusion import (
    FusionScorer,
    HandNormalizer,
    HAND_NORMALIZATION_KEY,
    enroll_and_build_queries,
    load_fusion_config,
)
from scoring.handcrafted_features import describe_records, pair_features
from training.evaluate_ultimate import DEFAULT_THRESHOLD
from training.stage3_metric_learning import (
    SiameseSignatureModel,
    dataset_labels,
    embed_records,
    read_split,
)

# The sub-dataset labels the preprocessed manifest uses, mapped to short report labels.
# ``dataset_labels`` runs at depth 2 for reporting so BHSig's two scripts stay separate,
# while scoring keeps the depth-1 labels the fusion config was fitted with.
DATASET_ALIASES = {
    "BHSig260/BHSig100_Bengali": "bhsig_bn",
    "BHSig260/BHSig160_Hindi": "bhsig_hi",
    "CEDAR/full_org": "cedar",
    "CEDAR/full_forg": "cedar",
    "SigResize/Genuine": "sigresize",
    "SigResize/Forgery": "sigresize",
}
# A false accept counts as "single reference" when the best reference score is this far
# above the average of the *other* references. A lift this large means one reference carried
# the decision rather than the set as a whole.
SINGLE_REFERENCE_GAP = 0.15
# A false accept counts as "all references similar" when the reference scores span less
# than this. A small spread means the model found every reference equally convincing.
ALL_REFS_SPREAD = 0.05
# Extensions tried when the raw image is not a .png.
RAW_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
# Which mean score decides a false accept. ``deep`` uses the cosine mean, ``fused`` the
# fused mean; the two live in different score spaces and need their own thresholds.
DECISIONS = ("deep", "fused")


def normalize_dataset(label: str) -> str:
    """Map a path-derived dataset label to the short report label."""
    return DATASET_ALIASES.get(label, label)


def raw_path_for(processed: Path, image_root: Path, raw_root: Path) -> Path | None:
    """Return the raw image beside ``raw_root`` for a preprocessed image, if it exists."""
    try:
        relative = processed.relative_to(image_root)
    except ValueError:
        return None
    base = raw_root / relative.with_suffix("")
    for suffix in RAW_EXTENSIONS:
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def image_metadata(
    path: Path,
    *,
    raw_root: Path | None = None,
    image_root: Path | None = None,
) -> dict[str, object]:
    """Return width, height, aspect ratio, ink density, and ink bbox for one image.

    The raw image is preferred when ``raw_root`` and ``image_root`` are given, because the
    preprocessed image is always 384 x 192 and its ink box is the canvas box. The raw image
    keeps the original resolution and the original crop, which is what makes the metadata
    useful. When no raw image is found the preprocessed image is measured instead.
    """
    raw = (
        raw_path_for(path, image_root, raw_root)
        if raw_root is not None and image_root is not None
        else None
    )
    source = raw if raw is not None else path
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {
            "source": "missing",
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "ink_density": None,
            "bbox": None,
        }
    height, width = image.shape[:2]
    mask = image < 200
    rows, columns = np.where(mask)
    bbox = (
        [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]
        if rows.size
        else None
    )
    return {
        "source": "raw" if raw is not None else "processed",
        "width": int(width),
        "height": int(height),
        "aspect_ratio": round(width / height, 4) if height else None,
        "ink_density": round(float(mask.mean()), 6),
        "bbox": bbox,
    }


def _fused_scores(
    scorer: FusionScorer,
    descriptors: Sequence[object],
    ref_indices: Sequence[int],
    query_index: int,
    deep_scores: np.ndarray,
    *,
    datasets: Sequence[str] | None,
    hand_normalizer: HandNormalizer | None,
) -> list[float]:
    """Return one fused score per reference for one query."""
    scores: list[float] = []
    for position, reference in enumerate(ref_indices):
        features = pair_features(descriptors[reference], descriptors[query_index])
        if hand_normalizer is not None and datasets is not None:
            features = hand_normalizer.transform_features(features, datasets[query_index])
        scores.append(float(scorer.score(float(deep_scores[position]), features)))
    return scores


def compute_false_accepts(
    model: SiameseSignatureModel,
    records: Sequence[object],
    *,
    threshold: float,
    references: int,
    device: str,
    batch_size: int,
    workers: int,
    datasets: Sequence[str] | None = None,
    dataset_of: dict[str, str] | None = None,
    scorer: FusionScorer | None = None,
    hand_normalizer: HandNormalizer | None = None,
    descriptors: Sequence[object] | None = None,
    raw_root: Path | None = None,
    image_root: Path | None = None,
    report_datasets: Sequence[str] | None = None,
    decision: str = "deep",
) -> list[dict[str, object]]:
    """Return one row per random query whose mean score reaches ``threshold``.

    ``datasets`` is the per-record label the *scorer* uses (depth 1, matching the fusion
    config); ``dataset_of`` maps signer id to that label and restricts random queries to the
    same dataset when given. ``report_datasets`` is the finer label the report groups by
    (depth 2, e.g. ``bhsig_bn``); it defaults to ``datasets``. ``scorer``/``descriptors``
    add fused per-reference scores next to the deep ones. ``decision`` picks which mean score
    is compared against ``threshold``: ``deep`` uses the cosine mean, ``fused`` uses the fused
    mean and requires a scorer.
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    if decision == "fused" and scorer is None:
        raise ValueError("decision='fused' requires a fusion scorer")
    embeddings = embed_records(
        model,
        records,
        already_preprocessed=True,
        device=device,
        batch_size=batch_size,
        workers=workers,
    )
    enrollment = enroll_and_build_queries(
        records, references, dataset_of=dataset_of
    )
    rows: list[dict[str, object]] = []
    for query in enrollment.queries:
        if query.kind != "random":
            continue
        ref_indices = enrollment.references[query.target_signer]
        deep_scores = embeddings[ref_indices] @ embeddings[query.index]
        fused = (
            _fused_scores(
                scorer,
                descriptors or [],
                ref_indices,
                query.index,
                deep_scores,
                datasets=datasets,
                hand_normalizer=hand_normalizer,
            )
            if scorer is not None and descriptors is not None
            else None
        )
        score = float(deep_scores.mean())
        decision_score = float(np.mean(fused)) if decision == "fused" else score
        if decision_score < threshold:
            continue
        query_record = records[query.index]
        ref_records = [records[index] for index in ref_indices]
        labels = report_datasets if report_datasets is not None else datasets
        query_dataset = normalize_dataset(labels[query.index]) if labels else "unknown"
        ref_dataset = (
            normalize_dataset(labels[ref_indices[0]]) if labels else "unknown"
        )
        row: dict[str, object] = {
            "query_id": str(query_record.image_path),
            "query_signer_id": str(query_record.signer_id),
            "query_dataset": query_dataset,
            "ref_signer_id": str(query.target_signer),
            "ref_dataset": ref_dataset,
            "score": score,
            "decision": decision,
            "decision_score": decision_score,
            "per_ref_scores": [float(value) for value in deep_scores],
            "ref_ids": [str(record.image_path) for record in ref_records],
            "ref_signer_ids": [str(record.signer_id) for record in ref_records],
            "n_refs": len(ref_indices),
            "query_metadata": image_metadata(
                Path(query_record.image_path), raw_root=raw_root, image_root=image_root
            ),
            "ref_metadata": [
                image_metadata(
                    Path(record.image_path), raw_root=raw_root, image_root=image_root
                )
                for record in ref_records
            ],
        }
        if fused is not None:
            row["fused_score"] = float(np.mean(fused))
            row["per_ref_fused_scores"] = fused
        rows.append(row)
    return rows


def count_random_queries(
    records: Sequence[object],
    references: int,
    *,
    dataset_of: dict[str, str] | None = None,
) -> int:
    """Return how many random queries the protocol scores, i.e. the FAR denominator."""
    enrollment = enroll_and_build_queries(records, references, dataset_of=dataset_of)
    return sum(1 for query in enrollment.queries if query.kind == "random")


def _pattern(per_ref_scores: Sequence[float]) -> str:
    """Classify how the per-reference scores produced the false accept.

    ``single_reference`` means the best reference sits well above the *other* references, so
    it is what lifted the mean over the threshold. ``all_refs_similar`` means every reference
    scored about the same, so the whole set looked convincing. ``mixed`` is in between.
    """
    if not per_ref_scores:
        return "unknown"
    values = np.asarray(per_ref_scores, dtype=np.float64)
    if values.size == 1:
        return "single_reference"
    spread = float(values.max() - values.min())
    if spread <= ALL_REFS_SPREAD:
        return "all_refs_similar"
    best = int(np.argmax(values))
    others = np.delete(values, best)
    # How far the best reference sits above the average of the rest: this is the lift the
    # single reference contributes to the mean, which is what "pulled the average up" means.
    lift = round(float(values[best] - others.mean()), 6)
    if lift >= SINGLE_REFERENCE_GAP:
        return "single_reference"
    return "mixed"


def _concentration(counts: Counter) -> dict[str, object]:
    """Return how concentrated a counter is: top-N shares and the hub signers."""
    total = sum(counts.values())
    ordered = counts.most_common()
    top = [
        {"signer": signer, "false_accepts": count}
        for signer, count in ordered[:5]
    ]
    shares = {}
    for n in (1, 3, 5):
        shares[f"top_{n}_share"] = (
            round(sum(count for _, count in ordered[:n]) / total, 4) if total else 0.0
        )
    return {
        "total_false_accepts": total,
        "distinct_signers": len(counts),
        "top_signers": top,
        **shares,
    }


def summarize(
    rows: Sequence[dict[str, object]],
    *,
    threshold: float,
    decision: str = "deep",
) -> dict[str, object]:
    """Return the FAR, per-dataset counts, and the hub analysis for the false accepts."""
    by_dataset: Counter = Counter()
    by_query_signer: Counter = Counter()
    by_ref_signer: Counter = Counter()
    by_query_dataset: Counter = Counter()
    by_ref_dataset: Counter = Counter()
    patterns: Counter = Counter()
    for row in rows:
        by_dataset[str(row["query_dataset"])] += 1
        by_query_signer[str(row["query_signer_id"])] += 1
        by_ref_signer[str(row["ref_signer_id"])] += 1
        by_query_dataset[str(row["query_dataset"])] += 1
        by_ref_dataset[str(row["ref_dataset"])] += 1
        patterns[_pattern(row["per_ref_scores"])] += 1

    return {
        "threshold": threshold,
        "decision": decision,
        "false_accepts": len(rows),
        "by_dataset": [
            {"dataset": dataset, "false_accepts": count}
            for dataset, count in by_dataset.most_common()
        ],
        "patterns": [
            {"pattern": pattern, "false_accepts": count}
            for pattern, count in patterns.most_common()
        ],
        "hub_analysis": {
            "query_signers": _concentration(by_query_signer),
            "reference_signers": _concentration(by_ref_signer),
            "query_datasets": _concentration(by_query_dataset),
            "reference_datasets": _concentration(by_ref_dataset),
        },
    }


CSV_COLUMNS = (
    "rank",
    "query_id",
    "query_signer_id",
    "query_dataset",
    "ref_signer_id",
    "ref_dataset",
    "decision",
    "score",
    "fused_score",
    "decision_score",
    "per_ref_scores",
    "per_ref_fused_scores",
    "n_refs",
    "pattern",
    "ref_ids",
    "query_width",
    "query_height",
    "query_aspect_ratio",
    "query_ink_density",
    "query_bbox",
    "ref_widths",
    "ref_heights",
    "ref_aspect_ratios",
    "ref_ink_densities",
    "ref_bboxes",
)


def _metadata_column(metadata: dict[str, object], key: str) -> object:
    """Return one metadata field, or an empty string when the image was missing."""
    value = metadata.get(key)
    return "" if value is None else value


def write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    summary: dict[str, object],
    *,
    total_random_pairs: int,
) -> Path:
    """Write the false accepts plus a ``metric,value`` summary block to ``path``."""
    ordered = sorted(rows, key=lambda row: float(row["decision_score"]), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        for rank, row in enumerate(ordered, start=1):
            query_meta = row["query_metadata"]
            ref_meta = row["ref_metadata"]
            writer.writerow(
                [
                    rank,
                    row["query_id"],
                    row["query_signer_id"],
                    row["query_dataset"],
                    row["ref_signer_id"],
                    row["ref_dataset"],
                    row["decision"],
                    f"{float(row['score']):.6f}",
                    f"{float(row['fused_score']):.6f}" if "fused_score" in row else "",
                    f"{float(row['decision_score']):.6f}",
                    json.dumps([round(float(value), 6) for value in row["per_ref_scores"]]),
                    json.dumps(
                        [round(float(value), 6) for value in row["per_ref_fused_scores"]]
                    )
                    if "per_ref_fused_scores" in row
                    else "",
                    row["n_refs"],
                    _pattern(row["per_ref_scores"]),
                    json.dumps(row["ref_ids"]),
                    _metadata_column(query_meta, "width"),
                    _metadata_column(query_meta, "height"),
                    _metadata_column(query_meta, "aspect_ratio"),
                    _metadata_column(query_meta, "ink_density"),
                    json.dumps(query_meta.get("bbox")),
                    json.dumps([meta.get("width") for meta in ref_meta]),
                    json.dumps([meta.get("height") for meta in ref_meta]),
                    json.dumps([meta.get("aspect_ratio") for meta in ref_meta]),
                    json.dumps([meta.get("ink_density") for meta in ref_meta]),
                    json.dumps([meta.get("bbox") for meta in ref_meta]),
                ]
            )
        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerow(["decision", summary.get("decision", "deep")])
        writer.writerow(["threshold", f"{summary['threshold']:.6f}"])
        writer.writerow(["total_random_pairs", total_random_pairs])
        writer.writerow(["false_accepts", summary["false_accepts"]])
        writer.writerow(
            [
                "far_random",
                f"{summary['false_accepts'] / total_random_pairs:.6f}"
                if total_random_pairs
                else "0.000000",
            ]
        )
        for entry in summary["by_dataset"]:
            writer.writerow(
                [f"false_accepts::{entry['dataset']}", entry["false_accepts"]]
            )
        for entry in summary["patterns"]:
            writer.writerow([f"pattern::{entry['pattern']}", entry["false_accepts"]])
    return path


def read_false_accepts_csv(path: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Read a CSV written by :func:`write_csv` into ``(rows, summary)``.

    The summary block starts at the ``metric,value`` marker, so this keeps the pair rows and
    the summary separate. Notebooks should use this instead of handing the raw file to
    ``pandas.read_csv``, which would try to parse the summary block as pair rows.
    """
    rows: list[dict[str, object]] = []
    summary: dict[str, str] = {}
    in_summary = False
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        for line in reader:
            if not line:
                continue
            if line[0] == "metric":
                in_summary = True
                continue
            if in_summary:
                if len(line) >= 2:
                    summary[line[0]] = line[1]
                continue
            if header is None:
                continue
            row = dict(zip(header, line))
            for key in ("per_ref_scores", "per_ref_fused_scores", "ref_ids", "query_bbox", "ref_bboxes"):
                if key in row and row[key]:
                    row[key] = json.loads(row[key])
            rows.append(row)
    return rows, summary


def write_hub_analysis_csv(path: Path, summary: dict[str, object]) -> Path:
    """Write the per-signer false-accept counts, ranked, to ``path``.

    One row per ``(role, signer)`` pair, where ``role`` is ``query`` or ``reference``. A
    signer appearing as a query hub means its own signatures are being accepted against
    other signers; a signer appearing as a reference hub means other signers' signatures are
    being accepted against its references. The two are different failure modes, so both are
    written.
    """
    hub = summary["hub_analysis"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["role", "signer", "false_accepts", "share"])
        for role, key in (("query", "query_signers"), ("reference", "reference_signers")):
            block = hub[key]
            total = int(block["total_false_accepts"])
            for entry in block["top_signers"]:
                count = int(entry["false_accepts"])
                writer.writerow(
                    [role, entry["signer"], count, round(count / total, 4) if total else 0.0]
                )
    return path


def _font(size: int) -> ImageFont.ImageFont:
    """Return a font of ``size`` pixels, falling back to the bitmap default."""
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _thumbnail(path: Path, box: tuple[int, int], *, raw_root: Path | None, image_root: Path | None) -> Image.Image:
    """Return ``path`` scaled to fit ``box`` on a white canvas, preserving aspect ratio."""
    raw = (
        raw_path_for(path, image_root, raw_root)
        if raw_root is not None and image_root is not None
        else None
    )
    source = raw if raw is not None else path
    try:
        image = Image.open(source).convert("L")
    except OSError:
        placeholder = Image.new("RGB", box, "white")
        draw = ImageDraw.Draw(placeholder)
        draw.text((8, 8), "missing image", fill="red", font=_font(14))
        return placeholder
    image.thumbnail(box, Image.LANCZOS)
    canvas = Image.new("RGB", box, "white")
    canvas.paste(image.convert("RGB"), ((box[0] - image.width) // 2, (box[1] - image.height) // 2))
    return canvas


def build_contact_sheet(
    row: dict[str, object],
    destination: Path,
    *,
    rank: int,
    image_root: Path,
    raw_root: Path | None = None,
    thumb: tuple[int, int] = (220, 110),
) -> Path:
    """Render one false accept as a contact sheet: query, five references, per-ref scores."""
    per_ref = [float(value) for value in row["per_ref_scores"]]
    ref_paths = [Path(path) for path in row["ref_ids"]]
    query_path = Path(str(row["query_id"]))
    best = int(np.argmax(per_ref)) if per_ref else -1

    columns = len(ref_paths)
    label_height = 46
    query_width = thumb[0] + 20
    refs_width = columns * (thumb[0] + 12) + 12
    width = query_width + refs_width
    height = 60 + thumb[1] + label_height + 12
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    title = (
        f"#{rank}  {row.get('decision', 'deep')}={float(row['decision_score']):.4f}  "
        f"deep={float(row['score']):.4f}  "
        f"query={row['query_signer_id']} ({row['query_dataset']})  "
        f"-> ref={row['ref_signer_id']} ({row['ref_dataset']})  "
        f"pattern={_pattern(per_ref)}"
    )
    if "fused_score" in row:
        title += f"  fused={float(row['fused_score']):.4f}"
    draw.text((8, 8), title, fill="black", font=_font(15))

    draw.text((8, 34), "QUERY", fill="#555555", font=_font(12))
    query_thumb = _thumbnail(query_path, thumb, raw_root=raw_root, image_root=image_root)
    sheet.paste(query_thumb, (8, 60))
    draw.rectangle([8, 60, 8 + thumb[0], 60 + thumb[1]], outline="#999999")

    draw.text((query_width, 34), "REFERENCES", fill="#555555", font=_font(12))
    for position, (path, score) in enumerate(zip(ref_paths, per_ref)):
        x = query_width + position * (thumb[0] + 12)
        y = 60
        ref_thumb = _thumbnail(path, thumb, raw_root=raw_root, image_root=image_root)
        sheet.paste(ref_thumb, (x, y))
        colour = "#0a7d32" if position == best else "#999999"
        draw.rectangle([x, y, x + thumb[0], y + thumb[1]], outline=colour, width=2 if position == best else 1)
        marker = " *" if position == best else ""
        draw.text(
            (x + 2, y + thumb[1] + 4),
            f"r{position + 1} {score:.4f}{marker}",
            fill=colour,
            font=_font(12),
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination


def build_contact_sheets(
    rows: Sequence[dict[str, object]],
    output_dir: Path,
    *,
    image_root: Path,
    raw_root: Path | None = None,
    top: int = 20,
) -> list[Path]:
    """Render the ``top`` highest-scoring false accepts as contact sheets."""
    ordered = sorted(rows, key=lambda row: float(row["decision_score"]), reverse=True)[:top]
    paths: list[Path] = []
    for rank, row in enumerate(ordered, start=1):
        query_stem = Path(str(row["query_id"])).stem
        destination = output_dir / f"{rank:02d}_{query_stem}_vs_{row['ref_signer_id']}.png"
        paths.append(
            build_contact_sheet(
                row,
                destination,
                rank=rank,
                image_root=image_root,
                raw_root=raw_root,
            )
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained Stage 3 checkpoint.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest.")
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Directory the manifest paths resolve against (default: the manifest's parent).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Raw dataset root, for original-resolution metadata and contact sheets.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for the CSV, JSON, and sheets."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Score threshold for a false accept (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--references", type=int, default=5, help="References enrolled per signer (default: 5)."
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        help="Fusion config; adds fused per-reference scores next to the deep ones.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--random-negatives",
        choices=("within-dataset", "all"),
        default="within-dataset",
        help="Which signers random queries come from (default: within-dataset).",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="How many contact sheets to render (default: 20)."
    )
    parser.add_argument(
        "--dataset-depth",
        type=int,
        default=2,
        help=(
            "Path depth used to label datasets in the report (default: 2, which separates "
            "bhsig_bn / bhsig_hi). Scoring always uses depth 1 to match the fusion config."
        ),
    )
    parser.add_argument(
        "--decision",
        choices=DECISIONS,
        default="deep",
        help="Which mean score decides a false accept (default: deep).",
    )
    parser.add_argument(
        "--scoring-depth",
        type=int,
        default=1,
        help=(
            "Path depth used to restrict random negatives to the same dataset. Default 1 "
            "matches training.evaluate; pass 2 to keep impostors within one script."
        ),
    )
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    image_root = (
        args.image_root.expanduser().resolve() if args.image_root else manifest.parent
    )
    raw_root = args.raw_root.expanduser().resolve() if args.raw_root else None
    output_dir = args.output_dir.expanduser().resolve()
    if not manifest.is_file():
        raise SystemExit(f"manifest not found: {manifest}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    scorer: FusionScorer | None = None
    hand_normalizer: HandNormalizer | None = None
    config = None
    if args.fusion_config is not None:
        config = load_fusion_config(args.fusion_config)
        scorer = FusionScorer(config)
        hand_normalizer = HandNormalizer.from_metadata(
            config.metadata.get(HAND_NORMALIZATION_KEY)
        )

    # The deep and fused scores live in different spaces, so the threshold has to come from
    # the space the decision uses. ``--threshold`` always wins; otherwise the fused decision
    # reads the calibrated ``config.threshold`` and the deep decision falls back to the
    # documented default.
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif args.decision == "fused" and config is not None and config.threshold is not None:
        threshold = float(config.threshold)
    else:
        threshold = DEFAULT_THRESHOLD
    if args.decision == "fused" and scorer is None:
        raise SystemExit("--decision fused requires --fusion-config")

    records = read_split(manifest, image_root, {"test"})
    # Depth 1 keeps the fusion-config label space; depth 2 splits BHSig into its scripts.
    datasets = dataset_labels(records, image_root, depth=args.scoring_depth)
    report_datasets = dataset_labels(records, image_root, depth=args.dataset_depth)
    dataset_of = (
        {record.signer_id: label for record, label in zip(records, datasets)}
        if args.random_negatives == "within-dataset"
        else None
    )
    descriptors = (
        describe_records(records, already_preprocessed=True) if scorer is not None else None
    )

    rows = compute_false_accepts(
        model,
        records,
        threshold=threshold,
        references=args.references,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        datasets=datasets,
        dataset_of=dataset_of,
        scorer=scorer,
        hand_normalizer=hand_normalizer,
        descriptors=descriptors,
        raw_root=raw_root,
        image_root=image_root,
        report_datasets=report_datasets,
        decision=args.decision,
    )

    total_random_pairs = count_random_queries(
        records, args.references, dataset_of=dataset_of
    )
    summary = summarize(rows, threshold=threshold, decision=args.decision)
    summary["total_random_pairs"] = total_random_pairs
    summary["far_random"] = (
        len(rows) / total_random_pairs if total_random_pairs else 0.0
    )
    summary["references"] = args.references
    summary["random_negatives"] = args.random_negatives
    summary["scoring_depth"] = args.scoring_depth
    summary["dataset_depth"] = args.dataset_depth
    summary["fusion"] = args.fusion_config is not None

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = write_csv(
        output_dir / "random_false_accepts.csv",
        rows,
        summary,
        total_random_pairs=total_random_pairs,
    )
    json_out = output_dir / "random_false_accepts_summary.json"
    json_out.write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    hub_out = write_hub_analysis_csv(output_dir / "hub_analysis.csv", summary)
    sheets = build_contact_sheets(
        rows,
        output_dir / "contact_sheets",
        image_root=image_root,
        raw_root=raw_root,
        top=args.top,
    )

    print(f"threshold              : {threshold:.4f}")
    print(f"random pairs scored    : {total_random_pairs}")
    print(f"false accepts          : {len(rows)}")
    print(f"FAR_random             : {summary['far_random']:.4f}")
    print(f"CSV                    : {csv_out}")
    print(f"JSON                   : {json_out}")
    print(f"hub analysis           : {hub_out}")
    print(f"contact sheets         : {len(sheets)} in {output_dir / 'contact_sheets'}")
    print(json.dumps(summary["hub_analysis"], indent=2))


if __name__ == "__main__":
    main()
