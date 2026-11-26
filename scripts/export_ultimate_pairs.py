"""Export the per-pair decisions of the "Ultimate Test Signs" benchmark to CSV.

:mod:`training.evaluate_ultimate` reports only aggregate FAR / FRR numbers. When a run
looks wrong it is much easier to debug from the individual comparisons, so this script
replays the *same* protocol and writes one CSV row per comparison pair::

    sign1_path,sign2_path,comparison_type,cosine_similarity,threshold,matched

``comparison_type`` is ``genuine-genuine`` (two genuine signatures of one signer),
``genuine-forge`` (a genuine signature against a skilled forgery of that signer), or
``genuine-random`` (a genuine signature against a genuine signature of a *different*
signer). ``cosine_similarity`` is the dot product of the two L2-normalised embeddings, and
``matched`` is ``score >= threshold`` -- the repository-wide accept rule -- written as
``true``/``false``.

After the pair rows the file holds a blank line and a two-column ``metric,value`` block with
the overall result: the confusion-matrix counts (true/false positive/negative), FAR, FRR,
the number of signers and sign pairs, the equal error rate, and the threshold and checkpoint
used, so the file is self-describing.

The pairing, the embeddings, and the metrics are all imported from
:mod:`training.evaluate_ultimate` and :mod:`scoring.metrics`, so the CSV can never disagree
with the aggregate report.

Score-level fusion (optional). With ``--fusion-config`` the file gains three more columns --
``fused_similarity``, ``fused_threshold``, ``fused_matched`` -- and the summary block gains
the ``*_fused`` mirror of every metric, so the deep and fused decisions sit side by side. The
fused operating point comes from ``--fused-threshold``, then the threshold stored in the
config, then ``--threshold``, then :data:`DEFAULT_THRESHOLD`.

Usage:
    python -m scripts.export_ultimate_pairs \\
        --checkpoint ultimate_best_model.pt \\
        --manifest "Ultimate Test Signs/manifest_ultimate.csv" \\
        --output ultimate_pairs_grayscale.csv
    python -m scripts.export_ultimate_pairs ... --fusion-config fusion.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from scoring.fusion import (
    HAND_NORMALIZATION_KEY,
    FusionScorer,
    HandNormalizer,
    load_fusion_config,
)
from scoring.handcrafted_features import describe_records
from scoring.metrics import eer_threshold, rates_at_threshold
from training.evaluate_ultimate import (
    DEFAULT_THRESHOLD,
    GENUINE,
    RANDOM,
    build_records,
    fusion_decision_threshold,
    iter_fused_pair_scores,
    pair_indices,
    read_ultimate_manifest,
    score_pairs,
)
from training.stage3_metric_learning import SiameseSignatureModel, embed_records

# The three pair kinds ``pair_indices`` produces, in the order rows are written, mapped to
# the human-readable ``comparison_type`` label the CSV uses.
COMPARISON_TYPES = {
    GENUINE: "genuine-genuine",
    "skilled": "genuine-forge",
    RANDOM: "genuine-random",
}
PAIR_COLUMNS = (
    "sign1_path",
    "sign2_path",
    "comparison_type",
    "cosine_similarity",
    "threshold",
    "matched",
)
# Appended to the pair header when a fusion config supplies a fused score per pair.
FUSION_COLUMNS = ("fused_similarity", "fused_threshold", "fused_matched")


def comparison_type(pair_kind: str) -> str:
    """Map a ``pair_indices`` kind (``genuine``/``skilled``/``random``) to a CSV label.

    Raises:
        ValueError: If ``pair_kind`` is not one of the three supported kinds.
    """
    try:
        return COMPARISON_TYPES[pair_kind]
    except KeyError as error:
        raise ValueError(f"unknown pair kind {pair_kind!r}") from error


def display_path(image_path: str | Path, image_root: Path | None) -> str:
    """Return ``image_path`` relative to ``image_root`` when it lives under it."""
    path = Path(image_path)
    if image_root is not None:
        try:
            return path.relative_to(image_root).as_posix()
        except ValueError:
            return str(path)
    return str(path)


def build_pair_rows(
    rows: list[dict[str, str]],
    by_folder: dict[str, list[int]],
    pairs: dict[str, list[tuple[int, int]]],
    embeddings: np.ndarray,
    threshold: float,
    image_root: Path | None = None,
    fused_scores: dict[tuple[int, int], float] | None = None,
    fused_threshold: float | None = None,
) -> list[dict[str, object]]:
    """Build one CSV row per comparison pair, grouped by signer then comparison type.

    The order follows the manifest's signer order; within a signer the rows are the
    genuine-genuine pairs, then the genuine-forge pairs, then the genuine-random pairs, so
    each signer's comparisons read as one block. ``matched`` is ``score >= threshold``.

    When ``fused_scores`` and ``fused_threshold`` are given, three more keys are added to
    every row (``fused_similarity``, ``fused_threshold``, ``fused_matched``) so the deep and
    fused decisions travel side by side in the same file.
    """
    grouped: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for kind in COMPARISON_TYPES:
        for first, second in pairs[kind]:
            grouped[rows[first]["folder"]].append((kind, first, second))

    table: list[dict[str, object]] = []
    for folder in by_folder:
        for kind, first, second in grouped.get(folder, []):
            score = float(embeddings[first] @ embeddings[second])
            row: dict[str, object] = {
                "sign1_path": display_path(rows[first]["image_path"], image_root),
                "sign2_path": display_path(rows[second]["image_path"], image_root),
                "comparison_type": comparison_type(kind),
                "cosine_similarity": score,
                "threshold": float(threshold),
                "matched": score >= threshold,
            }
            if fused_scores is not None and fused_threshold is not None:
                fused = float(fused_scores[(first, second)])
                row["fused_similarity"] = fused
                row["fused_threshold"] = float(fused_threshold)
                row["fused_matched"] = fused >= fused_threshold
            table.append(row)
    return table


def confusion_counts(
    genuine: np.ndarray, negatives: np.ndarray, threshold: float
) -> dict[str, object]:
    """Return the confusion-matrix counts and rates at ``threshold``.

    Shared by the deep and fused blocks so the two can never disagree about what a count or
    a rate means.
    """
    far, frr = rates_at_threshold(genuine, negatives, threshold)
    true_positive = int((genuine >= threshold).sum())
    false_negative = int((genuine < threshold).sum())
    false_positive = int((negatives >= threshold).sum())
    true_negative = int((negatives < threshold).sum())
    total = true_positive + true_negative + false_positive + false_negative
    eer_threshold_value, equal_error_rate = eer_threshold(genuine, negatives)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "far": far,
        "frr": frr,
        "accuracy": (true_positive + true_negative) / max(1, total),
        "eer": equal_error_rate,
        "eer_threshold": float(eer_threshold_value),
    }


def overall_summary(
    genuine: np.ndarray,
    negatives: np.ndarray,
    threshold: float,
    *,
    signers: int,
    pair_counts: dict[str, int],
    skilled: np.ndarray | None = None,
    random_scores: np.ndarray | None = None,
) -> dict[str, object]:
    """Return the overall result block appended after the pair rows.

    Counts come from the same ``score >= threshold`` rule as ``rates_at_threshold``, so the
    confusion matrix and the FAR/FRR rates can never disagree. ``skilled``/``random_scores``
    add the per-threat FAR and EER when the caller has them.
    """
    counts = confusion_counts(genuine, negatives, threshold)
    total = sum(
        int(counts[key])
        for key in ("true_positive", "false_positive", "true_negative", "false_negative")
    )
    summary: dict[str, object] = {
        "threshold": float(threshold),
        "total_signers": int(signers),
        "total_sign_pairs": total,
        "genuine_genuine_pairs": int(pair_counts.get(GENUINE, 0)),
        "genuine_forge_pairs": int(pair_counts.get("skilled", 0)),
        "genuine_random_pairs": int(pair_counts.get(RANDOM, 0)),
        **counts,
    }
    if skilled is not None and skilled.size:
        summary["far_skilled"] = rates_at_threshold(genuine, skilled, threshold)[0]
        summary["eer_skilled"] = eer_threshold(genuine, skilled)[1]
    if random_scores is not None and random_scores.size:
        summary["far_random"] = rates_at_threshold(genuine, random_scores, threshold)[0]
        summary["eer_random"] = eer_threshold(genuine, random_scores)[1]
    return summary


def fusion_summary(
    genuine: np.ndarray,
    negatives: np.ndarray,
    threshold: float,
    *,
    skilled: np.ndarray | None = None,
    random_scores: np.ndarray | None = None,
) -> dict[str, object]:
    """Return the ``*_fused`` metric block for the fused score.

    Keys mirror :func:`overall_summary` with a ``_fused`` suffix, so one ``metric,value``
    block can describe both score families without ambiguity.
    """
    summary: dict[str, object] = {"threshold_fused": float(threshold)}
    summary.update(
        {
            f"{key}_fused": value
            for key, value in confusion_counts(genuine, negatives, threshold).items()
        }
    )
    if skilled is not None and skilled.size:
        summary["far_skilled_fused"] = rates_at_threshold(genuine, skilled, threshold)[0]
        summary["eer_skilled_fused"] = eer_threshold(genuine, skilled)[1]
    if random_scores is not None and random_scores.size:
        summary["far_random_fused"] = rates_at_threshold(genuine, random_scores, threshold)[0]
        summary["eer_random_fused"] = eer_threshold(genuine, random_scores)[1]
    return summary


def write_report(
    destination: Path,
    pair_rows: Sequence[dict[str, object]],
    summary: dict[str, object],
    columns: Sequence[str] = PAIR_COLUMNS,
) -> None:
    """Write the pair rows and the overall summary block to ``destination`` as one CSV.

    ``columns`` selects the pair columns, so the same writer serves the deep-only header and
    the deep-plus-fused header. Boolean cells are written as ``true``/``false``.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in pair_rows:
            cells = []
            for column in columns:
                value = row[column]
                if isinstance(value, bool):
                    cells.append("true" if value else "false")
                else:
                    cells.append(value)
            writer.writerow(cells)
        writer.writerow([])
        writer.writerow(["metric", "value"])
        for metric, value in summary.items():
            writer.writerow([metric, value])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="CSV file to write.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Accept a pair when its cosine similarity reaches this value "
            f"(default {DEFAULT_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        help="Fusion config written by training.fit_fusion; adds fused columns per pair.",
    )
    parser.add_argument(
        "--fused-threshold",
        type=float,
        default=None,
        help=(
            "Fixed decision threshold for the fused score. Default: the threshold stored "
            "in the fusion config, then --threshold, then 0.823."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.fused_threshold is not None and not 0.0 <= args.fused_threshold <= 1.0:
        raise ValueError("--fused-threshold must be in [0, 1]")

    scorer: FusionScorer | None = None
    fusion_config = None
    hand_normalizer: HandNormalizer | None = None
    if args.fusion_config is not None:
        fusion_config = load_fusion_config(args.fusion_config)
        scorer = FusionScorer(fusion_config)
        hand_normalizer = HandNormalizer.from_metadata(
            fusion_config.metadata.get(HAND_NORMALIZATION_KEY)
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    manifest = args.manifest.expanduser().resolve()
    image_root = (
        args.image_root.expanduser().resolve() if args.image_root else manifest.parent
    )
    rows = read_ultimate_manifest(manifest, image_root)
    records, by_folder = build_records(rows)
    pairs = pair_indices(rows, by_folder)

    embeddings = embed_records(
        model,
        records,
        already_preprocessed=True,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    scores = score_pairs(embeddings, rows, pairs)
    genuine = np.asarray(scores[GENUINE], dtype=np.float64)
    skilled = np.asarray(scores["skilled"], dtype=np.float64)
    random_scores = np.asarray(scores[RANDOM], dtype=np.float64)
    negatives = np.concatenate([skilled, random_scores])

    fused_by_pair: dict[tuple[int, int], float] | None = None
    fused_threshold_value: float | None = None
    fused_metrics: dict[str, object] = {}
    columns: tuple[str, ...] = PAIR_COLUMNS
    if scorer is not None and fusion_config is not None:
        descriptors = describe_records(records, already_preprocessed=True)
        fused_pair_list = list(
            iter_fused_pair_scores(
                embeddings, rows, pairs, descriptors, scorer, hand_normalizer
            )
        )
        fused_by_pair = {
            (first, second): score for first, second, _, score in fused_pair_list
        }
        fused_by_kind: dict[str, list[float]] = {GENUINE: [], "skilled": [], RANDOM: []}
        for _, _, pair_kind, score in fused_pair_list:
            fused_by_kind[pair_kind].append(score)
        fused_genuine = np.asarray(fused_by_kind[GENUINE], dtype=np.float64)
        fused_skilled = np.asarray(fused_by_kind["skilled"], dtype=np.float64)
        fused_random = np.asarray(fused_by_kind[RANDOM], dtype=np.float64)
        fused_threshold_value, _ = fusion_decision_threshold(
            fused_threshold=args.fused_threshold,
            config_threshold=fusion_config.threshold,
            deep_threshold=args.threshold,
        )
        fused_metrics = {
            "fusion_type": fusion_config.type,
            "fusion_config": str(args.fusion_config.expanduser().resolve()),
            **fusion_summary(
                fused_genuine,
                np.concatenate([fused_skilled, fused_random]),
                fused_threshold_value,
                skilled=fused_skilled,
                random_scores=fused_random,
            ),
        }
        columns = PAIR_COLUMNS + FUSION_COLUMNS

    pair_rows = build_pair_rows(
        rows,
        by_folder,
        pairs,
        embeddings,
        args.threshold,
        image_root,
        fused_scores=fused_by_pair,
        fused_threshold=fused_threshold_value,
    )
    pair_counts = {kind: len(group) for kind, group in pairs.items()}
    metrics = overall_summary(
        genuine,
        negatives,
        args.threshold,
        signers=len(by_folder),
        pair_counts=pair_counts,
        skilled=skilled,
        random_scores=random_scores,
    )
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "manifest": str(manifest),
        "device": str(args.device),
        **metrics,
        **fused_metrics,
    }

    destination = args.output.expanduser().resolve()
    write_report(destination, pair_rows, summary, columns)
    print(f"Wrote {len(pair_rows)} pair rows to {destination}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

