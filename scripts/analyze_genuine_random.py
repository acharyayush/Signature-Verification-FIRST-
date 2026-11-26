"""Review the accepted "genuine-random" pairs of an Ultimate Test Signs run.

Reads the per-pair CSV written by :mod:`scripts.export_ultimate_pairs` and keeps only the
``genuine-random`` comparisons -- a genuine signature against a genuine signature of a
*different* signer. A genuine-random pair that is accepted is a false positive: the random
impostor was let through. Those pairs are the ones worth looking at when ``far_random`` is
higher than expected, so this script isolates them, evaluates the subset, and writes a
self-contained HTML page with the accepted pairs' images side by side.

The pair CSV already carries the deep cosine decision (``matched``) and, when the run used
a fusion config, the fused decision (``fused_matched``). ``--decision`` selects which rule
decides what "accepted" means for the review; the summary always reports both rules so the
two operating points can be compared without re-running the export.

Outputs (inside ``--output-dir``)::

    genuine_random_pairs.csv     every genuine-random row, plus a metric,value summary block
    genuine_random_report.html   a self-contained page with the reviewed pairs' images
                                 embedded as base64 (no server, no internet needed)

Usage:
    python -m scripts.analyze_genuine_random \\
        --csv ultimate_pairs_grayscale_fusion_thr0.80.csv \\
        --image-root "Ultimate Test Signs" \\
        --output-dir genuine_random_review

    # Review fused-accepted pairs instead of deep-accepted ones
    python -m scripts.analyze_genuine_random ... --decision fused

    # Show every genuine-random pair, not just the accepted ones
    python -m scripts.analyze_genuine_random ... --show all

    # Add per-dataset counts by reading the manifest the CSV was built from
    python -m scripts.analyze_genuine_random ... \\
        --manifest "Ultimate Test Signs/manifest_ultimate.csv"
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path
from typing import Sequence

from scripts.build_report_site import read_pairs_csv

GENUINE_RANDOM = "genuine-random"
DECISIONS = ("deep", "fused", "either", "both")
SHOW_MODES = ("accepted", "all")
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _parse_bool(value: object) -> bool:
    """Return a CSV boolean cell as ``bool``."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _decision_accepts(pair: dict[str, object], decision: str) -> bool:
    """Return whether ``pair`` is accepted under ``decision``."""
    deep = _parse_bool(pair.get("matched", False))
    fused = _parse_bool(pair.get("fused_matched", False))
    if decision == "deep":
        return deep
    if decision == "fused":
        return fused
    if decision == "either":
        return deep or fused
    if decision == "both":
        return deep and fused
    raise ValueError(f"unknown decision {decision!r}")


def _pair_scores(pair: dict[str, object]) -> tuple[float, float]:
    """Return ``(deep_score, fused_score)``; fused falls back to the deep score."""
    deep = float(pair.get("cos", 0.0))
    fused = float(pair.get("fused", deep))
    return deep, fused


def _pair_thresholds(pair: dict[str, object]) -> tuple[float, float]:
    """Return ``(deep_threshold, fused_threshold)``; fused falls back to the deep cut."""
    deep = float(pair.get("threshold", 0.0))
    fused = float(pair.get("fused_threshold", deep))
    return deep, fused


def _quantile(values: Sequence[float], fraction: float) -> float:
    """Return the ``fraction`` quantile of ``values`` (linear interpolation)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _score_stats(values: Sequence[float]) -> dict[str, float]:
    """Return min/max/mean/median/p95/p99 for a score list."""
    if not values:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
    }


def _signer_of(pair: dict[str, object]) -> str:
    """Return the signer folder a genuine-random pair belongs to."""
    return str(pair["a"]).split("/")[0]


def _dataset_of(pair: dict[str, object], manifest: dict[str, str] | None) -> str:
    """Return the source dataset for a pair, when the manifest supplies it."""
    if manifest is None:
        return "unknown"
    return manifest.get(_signer_of(pair), "unknown")


def read_manifest_datasets(manifest: str | Path) -> dict[str, str]:
    """Return ``folder -> source_dataset`` from an Ultimate manifest.

    The manifest stores one row per image, so the same folder repeats; the first
    ``source_dataset`` seen for a folder wins. The file is read with ``csv.DictReader`` so
    the ``\\r\\n`` line endings the builder writes are handled.
    """
    path = Path(manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            folder = (row.get("folder") or "").strip()
            dataset = (row.get("source_dataset") or "").strip()
            if folder and dataset and folder not in mapping:
                mapping[folder] = dataset
    return mapping


def read_manifest_signers(manifest: str | Path) -> list[str]:
    """Return the manifest's signer folders in first-seen order."""
    path = Path(manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    signers: list[str] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            folder = (row.get("folder") or "").strip()
            if folder and folder not in seen:
                seen.add(folder)
                signers.append(folder)
    return signers


def filter_genuine_random(pairs: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Return only the ``genuine-random`` pairs, preserving the CSV order."""
    return [pair for pair in pairs if str(pair.get("type", "")) == GENUINE_RANDOM]


def _new_signer_row(signer: str, dataset: str) -> dict[str, object]:
    """Return a zeroed per-signer accumulator."""
    return {
        "signer": signer,
        "dataset": dataset,
        "pairs": 0,
        "accepted": 0,
        "accepted_deep": 0,
        "accepted_fused": 0,
        "max_score": 0.0,
        "max_fused": 0.0,
    }


def _new_dataset_row(dataset: str) -> dict[str, object]:
    """Return a zeroed per-dataset accumulator."""
    return {
        "dataset": dataset,
        "pairs": 0,
        "accepted": 0,
        "accepted_deep": 0,
        "accepted_fused": 0,
    }


def _tally(
    row: dict[str, object],
    *,
    is_decision: bool,
    is_deep: bool,
    is_fused: bool,
    deep: float,
    fused: float,
) -> None:
    """Add one pair to a per-signer or per-dataset accumulator."""
    row["pairs"] = int(row["pairs"]) + 1
    row["accepted"] = int(row["accepted"]) + int(is_decision)
    row["accepted_deep"] = int(row["accepted_deep"]) + int(is_deep)
    row["accepted_fused"] = int(row["accepted_fused"]) + int(is_fused)
    if "max_score" in row:
        row["max_score"] = max(float(row["max_score"]), deep)
        row["max_fused"] = max(float(row["max_fused"]), fused)


def evaluate_genuine_random(
    pairs: Sequence[dict[str, object]],
    *,
    decision: str = "deep",
    manifest: str | Path | None = None,
) -> dict[str, object]:
    """Summarise the genuine-random subset and return a JSON-friendly dict.

    The summary always carries both decision rules so a fused run can be compared with
    its deep baseline without re-exporting. ``accepted``/``rejected``/``far_random`` follow
    ``decision``; the ``_deep`` and ``_fused`` keys are the per-rule views.

    Args:
        pairs: The ``genuine-random`` rows from :func:`filter_genuine_random`.
        decision: ``deep``, ``fused``, ``either``, or ``both`` -- what "accepted" means.
        manifest: Optional Ultimate manifest, used for the per-dataset breakdown and for
            the "signers with no genuine-random pairs" list.

    Raises:
        ValueError: If ``decision`` is not one of :data:`DECISIONS`.
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")

    dataset_map = read_manifest_datasets(manifest) if manifest is not None else None
    all_signers = read_manifest_signers(manifest) if manifest is not None else None

    deep_scores: list[float] = []
    fused_scores: list[float] = []
    accepted_deep = accepted_fused = accepted_decision = 0
    deep_threshold = fused_threshold = 0.0
    per_signer: dict[str, dict[str, object]] = {}
    per_dataset: dict[str, dict[str, object]] = {}

    for pair in pairs:
        deep, fused = _pair_scores(pair)
        deep_cut, fused_cut = _pair_thresholds(pair)
        deep_threshold = deep_cut
        fused_threshold = fused_cut
        deep_scores.append(deep)
        fused_scores.append(fused)

        is_deep = _parse_bool(pair.get("matched", False))
        is_fused = _parse_bool(pair.get("fused_matched", False))
        is_decision = _decision_accepts(pair, decision)
        accepted_deep += int(is_deep)
        accepted_fused += int(is_fused)
        accepted_decision += int(is_decision)

        signer = _signer_of(pair)
        dataset = _dataset_of(pair, dataset_map)
        signer_row = per_signer.setdefault(signer, _new_signer_row(signer, dataset))
        _tally(
            signer_row,
            is_decision=is_decision,
            is_deep=is_deep,
            is_fused=is_fused,
            deep=deep,
            fused=fused,
        )
        dataset_row = per_dataset.setdefault(dataset, _new_dataset_row(dataset))
        _tally(
            dataset_row,
            is_decision=is_decision,
            is_deep=is_deep,
            is_fused=is_fused,
            deep=deep,
            fused=fused,
        )

    total = len(pairs)
    deep_stats = _score_stats(deep_scores)
    fused_stats = _score_stats(fused_scores)
    summary: dict[str, object] = {
        "decision": decision,
        "comparison_type": GENUINE_RANDOM,
        "total_pairs": total,
        "accepted": accepted_decision,
        "rejected": total - accepted_decision,
        "far_random": accepted_decision / total if total else 0.0,
        "threshold": deep_threshold,
        "threshold_fused": fused_threshold,
        "accepted_deep": accepted_deep,
        "rejected_deep": total - accepted_deep,
        "far_random_deep": accepted_deep / total if total else 0.0,
        "accepted_fused": accepted_fused,
        "rejected_fused": total - accepted_fused,
        "far_random_fused": accepted_fused / total if total else 0.0,
        "signers_with_pairs": len(per_signer),
        "signers_with_accepted": sum(
            1 for row in per_signer.values() if int(row["accepted"]) > 0
        ),
        "score_min": deep_stats["min"],
        "score_max": deep_stats["max"],
        "score_mean": deep_stats["mean"],
        "score_median": deep_stats["median"],
        "score_p95": deep_stats["p95"],
        "score_p99": deep_stats["p99"],
        "fused_min": fused_stats["min"],
        "fused_max": fused_stats["max"],
        "fused_mean": fused_stats["mean"],
        "fused_median": fused_stats["median"],
        "fused_p95": fused_stats["p95"],
        "fused_p99": fused_stats["p99"],
    }
    if all_signers is not None:
        summary["manifest_signers"] = len(all_signers)
        summary["signers_without_pairs"] = [
            signer for signer in all_signers if signer not in per_signer
        ]
    summary["per_signer"] = [per_signer[key] for key in sorted(per_signer)]
    summary["per_dataset"] = [per_dataset[key] for key in sorted(per_dataset)]
    return summary


def write_filtered_csv(
    destination: Path,
    pairs: Sequence[dict[str, object]],
    summary: dict[str, object],
    *,
    decision: str,
    dataset_map: dict[str, str] | None = None,
) -> None:
    """Write the genuine-random pairs plus a ``metric,value`` summary block.

    The pair columns mirror :mod:`scripts.export_ultimate_pairs` so the file reads the
    same way, with three review columns appended: ``signer``, ``dataset``, and
    ``accepted`` (the pair's decision under ``--decision``).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    fused = any("fused" in pair for pair in pairs)
    columns = [
        "sign1_path",
        "sign2_path",
        "comparison_type",
        "cosine_similarity",
        "threshold",
        "matched",
    ]
    if fused:
        columns += ["fused_similarity", "fused_threshold", "fused_matched"]
    columns += ["signer", "dataset", "accepted"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for pair in pairs:
            signer = _signer_of(pair)
            dataset = _dataset_of(pair, dataset_map)
            row: list[object] = [
                pair["a"],
                pair["b"],
                pair["type"],
                pair["cos"],
                pair["threshold"],
                "true" if _parse_bool(pair.get("matched", False)) else "false",
            ]
            if fused:
                row += [
                    pair.get("fused", ""),
                    pair.get("fused_threshold", ""),
                    "true" if _parse_bool(pair.get("fused_matched", False)) else "false",
                ]
            row += [
                signer,
                dataset,
                "true" if _decision_accepts(pair, decision) else "false",
            ]
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["metric", "value"])
        for metric, value in summary.items():
            if metric in ("per_signer", "per_dataset", "signers_without_pairs"):
                continue
            writer.writerow([metric, value])


def _image_data_uri(path: Path) -> str:
    """Return a base64 ``data:`` URI for an image, or a placeholder when missing."""
    if not path.is_file():
        return ""
    media_type = IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _image_html(path: Path, label: str) -> str:
    """Return an ``<figure>`` with the image, or a missing-file placeholder."""
    data_uri = _image_data_uri(path)
    if data_uri:
        body = f'<img src="{data_uri}" alt="{html.escape(label)}">'
    else:
        body = f'<div class="missing">image not found<br><code>{html.escape(str(path))}</code></div>'
    return (
        f'<figure class="sign">'
        f'<figcaption>{html.escape(label)}</figcaption>'
        f'{body}'
        f'<div class="path">{html.escape(str(path))}</div>'
        f'</figure>'
    )


def _badge(text: str, accepted: bool) -> str:
    """Return a coloured accept/reject badge."""
    kind = "accept" if accepted else "reject"
    return f'<span class="badge {kind}">{html.escape(text)}</span>'


def _card_html(
    pair: dict[str, object],
    image_root: Path,
    decision: str,
    dataset_map: dict[str, str] | None,
) -> str:
    """Return the HTML for one genuine-random pair card."""
    signer = _signer_of(pair)
    dataset = _dataset_of(pair, dataset_map)
    deep, fused = _pair_scores(pair)
    deep_cut, fused_cut = _pair_thresholds(pair)
    is_deep = _parse_bool(pair.get("matched", False))
    is_fused = _parse_bool(pair.get("fused_matched", False))
    is_decision = _decision_accepts(pair, decision)
    image_a = _image_html(image_root / str(pair["a"]), "Genuine")
    image_b = _image_html(image_root / str(pair["b"]), "Random (other signer)")
    fused_block = ""
    if "fused" in pair:
        fused_block = (
            f'<div class="score-row"><span>Fused</span>'
            f'<span>{fused:.4f} &middot; threshold {fused_cut:.4f}</span>'
            f'{_badge("matched" if is_fused else "rejected", is_fused)}</div>'
        )
    return (
        f'<article class="card">'
        f'<header><h3>{html.escape(signer)}</h3>'
        f'<span class="dataset">{html.escape(dataset)}</span>'
        f'{_badge("ACCEPTED" if is_decision else "REJECTED", is_decision)}</header>'
        f'<div class="images">{image_a}{image_b}</div>'
        f'<div class="scores">'
        f'<div class="score-row"><span>Deep cosine</span>'
        f'<span>{deep:.4f} &middot; threshold {deep_cut:.4f}</span>'
        f'{_badge("matched" if is_deep else "rejected", is_deep)}</div>'
        f'{fused_block}'
        f'</div>'
        f'</article>'
    )


METRIC_KEYS = (
    "decision",
    "comparison_type",
    "total_pairs",
    "accepted",
    "rejected",
    "far_random",
    "threshold",
    "threshold_fused",
    "accepted_deep",
    "far_random_deep",
    "accepted_fused",
    "far_random_fused",
    "signers_with_pairs",
    "signers_with_accepted",
    "score_min",
    "score_max",
    "score_mean",
    "score_median",
    "score_p95",
    "score_p99",
    "fused_min",
    "fused_max",
    "fused_mean",
    "fused_median",
    "fused_p95",
    "fused_p99",
)


def _metric_table_html(summary: dict[str, object]) -> str:
    """Return the summary metrics table."""
    rows = []
    for key in METRIC_KEYS:
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, float):
            value = f"{value:.4f}"
        rows.append(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        )
    return f'<table class="metrics"><tbody>{"".join(rows)}</tbody></table>'


def _breakdown_table_html(rows: Sequence[dict[str, object]], title: str) -> str:
    """Return a per-signer or per-dataset breakdown table."""
    if not rows:
        return ""
    preferred = (
        "signer",
        "dataset",
        "pairs",
        "accepted",
        "accepted_deep",
        "accepted_fused",
        "max_score",
        "max_fused",
    )
    columns = [column for column in preferred if column in rows[0]]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<h2>{html.escape(title)}</h2>'
        f'<table class="breakdown"><thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


PAGE_CSS = """
:root {
  --bg: #f4f6fb; --card: #ffffff; --ink: #1f2937; --muted: #6b7280;
  --accept: #047857; --reject: #b91c1c; --border: #e5e7eb;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       background: var(--bg); color: var(--ink); }
header.page { padding: 24px 32px; background: #111827; color: #fff; }
header.page h1 { margin: 0 0 4px; font-size: 24px; }
header.page p { margin: 0; color: #d1d5db; font-size: 14px; }
main { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }
h2 { font-size: 18px; margin: 32px 0 12px; }
table { border-collapse: collapse; width: 100%; background: var(--card);
        border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border);
         font-size: 13px; }
th { background: #f9fafb; font-weight: 600; }
table.metrics th { width: 220px; }
table.metrics tbody tr:last-child th, table.metrics tbody tr:last-child td { border-bottom: none; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
        gap: 20px; margin-top: 16px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
        padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.card header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.card header h3 { margin: 0; font-size: 16px; }
.card header .dataset { color: var(--muted); font-size: 12px; }
.images { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.sign { margin: 0; background: #f9fafb; border: 1px solid var(--border);
        border-radius: 8px; padding: 8px; }
.sign figcaption { font-size: 12px; font-weight: 600; color: var(--muted);
                   margin-bottom: 6px; }
.sign img { width: 100%; height: auto; display: block; background: #fff;
            border-radius: 4px; }
.sign .path { font-size: 10px; color: var(--muted); word-break: break-all;
              margin-top: 6px; }
.sign .missing { padding: 20px; text-align: center; color: var(--reject);
                 font-size: 12px; }
.scores { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
.score-row { display: flex; align-items: center; gap: 8px; font-size: 13px;
             background: #f9fafb; border: 1px solid var(--border);
             border-radius: 6px; padding: 6px 10px; }
.score-row span:first-child { font-weight: 600; width: 90px; }
.score-row span:nth-child(2) { color: var(--muted); flex: 1; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
         font-size: 11px; font-weight: 700; text-transform: uppercase; }
.badge.accept { background: #d1fae5; color: var(--accept); }
.badge.reject { background: #fee2e2; color: var(--reject); }
"""


def build_html_report(
    destination: Path,
    pairs: Sequence[dict[str, object]],
    summary: dict[str, object],
    *,
    image_root: Path,
    decision: str,
    show: str,
    dataset_map: dict[str, str] | None = None,
) -> None:
    """Write the self-contained HTML review page.

    ``show="accepted"`` (the default) lists only the pairs the decision accepts;
    ``show="all"`` lists every genuine-random pair. Images are embedded as base64 so the
    page works from ``file://`` with no server and no internet.
    """
    if show not in SHOW_MODES:
        raise ValueError(f"show must be one of {SHOW_MODES}, got {show!r}")
    reviewed = (
        [pair for pair in pairs if _decision_accepts(pair, decision)]
        if show == "accepted"
        else list(pairs)
    )
    cards = "".join(
        _card_html(pair, image_root, decision, dataset_map) for pair in reviewed
    )
    metrics = _metric_table_html(summary)
    per_signer = _breakdown_table_html(
        summary.get("per_signer", []), "Per-signer breakdown"  # type: ignore[arg-type]
    )
    per_dataset = _breakdown_table_html(
        summary.get("per_dataset", []), "Per-dataset breakdown"  # type: ignore[arg-type]
    )
    without = summary.get("signers_without_pairs", [])
    without_html = ""
    if without:
        without_html = (
            '<h2>Signers with no genuine-random pairs</h2>'
            f'<p class="muted">{html.escape(", ".join(str(s) for s in without))}</p>'
        )
    title = "Genuine-random review"
    subtitle = (
        f"{len(pairs)} genuine-random pairs &middot; "
        f"{summary['accepted']} accepted by {decision} &middot; "
        f"FAR_random {float(summary['far_random']):.4f}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<header class="page">
  <h1>{html.escape(title)}</h1>
  <p>{subtitle}</p>
</header>
<main>
  <h2>Summary</h2>
  {metrics}
  {per_signer}
  {per_dataset}
  {without_html}
  <h2>Reviewed pairs ({len(reviewed)})</h2>
  <div class="grid">{cards}</div>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Per-pair CSV written by scripts.export_ultimate_pairs.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory the CSV's relative image paths resolve against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the review CSV and the HTML report.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Ultimate manifest (folder,class,source_dataset,...), for per-dataset counts.",
    )
    parser.add_argument(
        "--decision",
        choices=DECISIONS,
        default="deep",
        help="Which rule decides 'accepted' (default: deep).",
    )
    parser.add_argument(
        "--show",
        choices=SHOW_MODES,
        default="accepted",
        help="Which pairs the HTML lists (default: accepted).",
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve() if args.manifest else None

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not image_root.is_dir():
        raise SystemExit(f"image root not found: {image_root}")

    pairs, _summary_rows = read_pairs_csv(csv_path)
    genuine_random = filter_genuine_random(pairs)
    if not genuine_random:
        raise SystemExit(f"no genuine-random pairs in {csv_path}")

    dataset_map = read_manifest_datasets(manifest) if manifest else None
    summary = evaluate_genuine_random(
        genuine_random, decision=args.decision, manifest=manifest
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = output_dir / "genuine_random_pairs.csv"
    html_out = output_dir / "genuine_random_report.html"
    write_filtered_csv(
        csv_out,
        genuine_random,
        summary,
        decision=args.decision,
        dataset_map=dataset_map,
    )
    build_html_report(
        html_out,
        genuine_random,
        summary,
        image_root=image_root,
        decision=args.decision,
        show=args.show,
        dataset_map=dataset_map,
    )

    printable = {
        key: value
        for key, value in summary.items()
        if key not in ("per_signer", "per_dataset", "signers_without_pairs")
    }
    print(f"Genuine-random pairs : {len(genuine_random)}")
    print(f"Accepted ({args.decision}) : {summary['accepted']}")
    print(f"FAR_random           : {float(summary['far_random']):.4f}")
    print(f"CSV                  : {csv_out}")
    print(f"HTML                 : {html_out}")
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
