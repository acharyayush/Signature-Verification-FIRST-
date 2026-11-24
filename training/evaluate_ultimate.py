"""Evaluate the "Ultimate Test Signs" benchmark: FAR / FRR and a confusion matrix.

The benchmark is built by :mod:`scripts.build_ultimate_test_signs` and described by
``Ultimate Test Signs/manifest_ultimate.csv``. Every signer folder holds three genuine
signatures, three skilled forgeries of that signer, and three genuine signatures of a
*different* signer of the same dataset::

    Signer_001/Genuine/  3 genuine signatures   -> references + genuine queries (positives)
    Signer_001/Forged/   3 skilled forgeries    -> skilled impostors          (negatives)
    Signer_001/Random/   3 other-signer genuine -> random impostors           (negatives)

Protocol (identical to :mod:`training.evaluate`): the first ``--references`` genuine
signatures of every signer are enrolled as references, and a query is scored by its mean
cosine similarity to those references. A query is accepted when ``score >= threshold``,
because a higher score means "more likely the same signer". Genuine queries are the
positives and both impostor kinds are negatives::

    true positives   (TP): genuine query accepted
    false negatives  (FN): genuine query rejected
    true negatives   (TN): impostor query rejected
    false positives  (FP): impostor query accepted

with ``FAR = FP / (FP + TN)`` and ``FRR = FN / (FN + TP)``. Skilled-forgery and random FAR
are also reported separately, because a learned forgery and an unrelated signer are very
different threats.

Operating point: by default the equal-error threshold of this benchmark is used and the
report labels it ``test_eer`` -- the Ultimate set is a *test* set, so nothing here is tuned
on it. Pass ``--threshold`` to apply a fixed, externally calibrated threshold (for example
the deep threshold :mod:`training.evaluate` derived on the validation signers), or
``--target-far`` to take the largest threshold whose combined-impostor FAR meets a budget.

Usage:
    python -m training.evaluate_ultimate \\
        --checkpoint best_model_first.pt \\
        --manifest "Ultimate Test Signs/manifest_ultimate.csv" --references 2
    python -m training.evaluate_ultimate ... --target-far 0.03 \\
        --report ultimate_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scoring.metrics import eer_threshold, rates_at_threshold, threshold_at_far
from training.stage3_metric_learning import (
    SiameseSignatureModel,
    SignatureRecord,
    embed_records,
)


def read_ultimate_manifest(
    manifest: str | Path,
    image_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Read the ultimate manifest and resolve every image path.

    Args:
        manifest: CSV with ``folder,class,source_dataset,source_signer,image_path``.
        image_root: Directory relative paths resolve against. Defaults to the manifest's
            own directory, which is how the builder laid the set out.

    Raises:
        ValueError: If a required column or an image path is missing, or a row uses an
            unsupported ``class`` value.
    """
    manifest_path = Path(manifest).expanduser().resolve()
    root = Path(image_root).expanduser().resolve() if image_root else manifest_path.parent
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"ultimate manifest is missing required columns: {', '.join(sorted(missing))}"
            )
        for row_number, row in enumerate(reader, start=2):
            kind = row["class"].strip().lower()
            if kind not in {GENUINE, FORGED, RANDOM}:
                raise ValueError(
                    f"manifest row {row_number}: class must be genuine/forged/random, "
                    f"got {kind!r}"
                )
            folder = row["folder"].strip()
            if not folder:
                raise ValueError(f"manifest row {row_number}: folder cannot be empty")
            path = Path(row["image_path"].strip()).expanduser()
            if not path.is_absolute():
                path = (root / path).resolve()
            rows.append(
                {
                    "folder": folder,
                    "class": kind,
                    "source_dataset": row["source_dataset"].strip(),
                    "source_signer": row["source_signer"].strip(),
                    "image_path": str(path),
                }
            )
    if not rows:
        raise ValueError(f"no rows found in {manifest_path}")
    return rows


def build_records(
    rows: list[dict[str, str]],
) -> tuple[list[SignatureRecord], dict[str, list[int]]]:
    """Turn manifest rows into records plus a ``folder -> row indices`` grouping.

    ``signer_id`` is the signer *folder* the row lives in, so a query is scored against
    the references of the signer it is meant to match. ``is_forgery`` is only set for the
    explicit forgeries; the genuine/forged/random distinction needed for the confusion
    matrix is carried by the row itself, not by this record flag.
    """
    records: list[SignatureRecord] = []
    by_folder: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        records.append(
            SignatureRecord(
                image_path=Path(row["image_path"]),
                signer_id=row["folder"],
                is_forgery=row["class"] == FORGED,
            )
        )
        by_folder[row["folder"]].append(index)
    return records, by_folder


# The three ``class`` values the ultimate manifest uses, in report order.
GENUINE = "genuine"
FORGED = "forged"
RANDOM = "random"
_REQUIRED_COLUMNS = {"folder", "class", "source_dataset", "source_signer", "image_path"}


def reference_indices(
    rows: list[dict[str, str]],
    by_folder: dict[str, list[int]],
    references: int,
) -> dict[str, list[int]]:
    """Return the first ``references`` genuine images enrolled per signer folder.

    Ordered by file name so the enrolment is deterministic.

    Raises:
        ValueError: If a signer has too few genuine signatures to enrol ``references`` and
            still keep at least one genuine query.
    """
    enrolled: dict[str, list[int]] = {}
    for folder, indices in by_folder.items():
        genuine = sorted(
            (index for index in indices if rows[index]["class"] == GENUINE),
            key=lambda index: rows[index]["image_path"],
        )
        if len(genuine) <= references:
            raise ValueError(
                f"{folder} has {len(genuine)} genuine signatures; more than {references} "
                "are needed to enrol references and keep a genuine query"
            )
        enrolled[folder] = genuine[:references]
    return enrolled


def iter_query_scores(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    enrolled: dict[str, list[int]],
):
    """Yield ``(row_index, query_kind, score)`` for every non-reference query.

    A query's score is its mean cosine similarity to its signer's references, the same
    aggregation :mod:`training.evaluate` uses. Embeddings are L2-normalised, so their dot
    product is the cosine similarity. ``query_kind`` is ``genuine``, ``skilled``, or
    ``random``.
    """
    for index, row in enumerate(rows):
        kind = row["class"]
        refs = enrolled[row["folder"]]
        if kind == GENUINE and index in refs:
            continue  # an enrolled reference is not a query
        similarity = embeddings[refs] @ embeddings[index]
        yield index, ("skilled" if kind == FORGED else kind), float(similarity.mean())


def score_queries(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    enrolled: dict[str, list[int]],
) -> dict[str, list[float]]:
    """Collect :func:`iter_query_scores` into one score list per query kind."""
    scores: dict[str, list[float]] = {GENUINE: [], "skilled": [], RANDOM: []}
    for _, query_kind, score in iter_query_scores(embeddings, rows, enrolled):
        scores[query_kind].append(score)
    return scores


def confusion(
    genuine: np.ndarray,
    negatives: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Return the FAR / FRR confusion matrix at ``threshold``.

    ``rates_at_threshold`` supplies FAR and FRR using the repository-wide
    ``score >= threshold`` decision rule; the counts are derived from the same comparison
    so the matrix and the rates can never disagree.
    """
    genuine_array = np.asarray(genuine, dtype=np.float64)
    negative_array = np.asarray(negatives, dtype=np.float64)
    far, frr = rates_at_threshold(genuine_array, negative_array, threshold)
    true_positive = int((genuine_array >= threshold).sum())
    false_negative = int((genuine_array < threshold).sum())
    false_positive = int((negative_array >= threshold).sum())
    true_negative = int((negative_array < threshold).sum())
    return {
        "threshold": float(threshold),
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "far": far,
        "frr": frr,
        "accuracy": (true_positive + true_negative)
        / max(1, true_positive + true_negative + false_positive + false_negative),
    }



def _decision_threshold(
    genuine: np.ndarray,
    negatives: np.ndarray,
    *,
    fixed: float | None,
    target_far: float | None,
) -> tuple[float, str, float | None]:
    """Pick the operating point and return ``(threshold, source, achieved_far)``.

    Precedence mirrors :mod:`training.evaluate`: an explicit ``--threshold`` wins, then a
    ``--target-far`` budget on the combined impostors, then the benchmark's own equal-error
    threshold. ``achieved_far`` is only filled for the target-FAR branch.
    """
    if fixed is not None:
        return float(fixed), "fixed", None
    if target_far is not None:
        threshold, achieved_far = threshold_at_far(negatives, target_far)
        return float(threshold), "target_far", achieved_far
    threshold = float(eer_threshold(genuine, negatives)[0])
    return threshold, "test_eer", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--already-preprocessed",
        action="store_true",
        help=(
            "Treat the manifest images as Stage 2.2 grayscale files (default 384x192) "
            "instead of raw crops."
        ),
    )
    parser.add_argument(
        "--references",
        type=int,
        default=2,
        help="Genuine signatures enrolled per signer (the manifest ships three).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Apply this fixed score threshold instead of deriving one from the benchmark.",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help=(
            "Use the largest threshold whose FAR on the combined impostors is at most "
            "this value, which also minimises FRR subject to the budget."
        ),
    )
    parser.add_argument("--report", type=Path, help="Also write the JSON report here.")
    args = parser.parse_args()
    if args.references < 1:
        raise ValueError("--references must be at least 1")
    if args.target_far is not None and not 0.0 <= args.target_far <= 1.0:
        raise ValueError("--target-far must be in [0, 1]")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.threshold is not None and args.target_far is not None:
        raise ValueError("--threshold and --target-far are mutually exclusive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    rows = read_ultimate_manifest(args.manifest, args.image_root)
    records, by_folder = build_records(rows)
    print(
        f"{len(records)} images from {len(by_folder)} signers; embedding on {args.device} "
        f"(already_preprocessed={args.already_preprocessed})"
    )
    embeddings = embed_records(
        model,
        records,
        already_preprocessed=args.already_preprocessed,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    enrolled = reference_indices(rows, by_folder, args.references)
    scores = score_queries(embeddings, rows, enrolled)
    genuine = np.asarray(scores[GENUINE], dtype=np.float64)
    skilled = np.asarray(scores["skilled"], dtype=np.float64)
    random_scores = np.asarray(scores[RANDOM], dtype=np.float64)
    negatives = np.concatenate([skilled, random_scores])

    threshold, threshold_source, achieved_far = _decision_threshold(
        genuine, negatives, fixed=args.threshold, target_far=args.target_far
    )
    overall = confusion(genuine, negatives, threshold)
    overall["far_skilled"] = rates_at_threshold(genuine, skilled, threshold)[0]
    overall["far_random"] = rates_at_threshold(genuine, random_scores, threshold)[0]
    overall["eer_skilled"] = eer_threshold(genuine, skilled)[1]
    overall["eer_random"] = eer_threshold(genuine, random_scores)[1]
    overall["eer_combined"] = eer_threshold(genuine, negatives)[1]

    report: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "references": args.references,
        "already_preprocessed": bool(args.already_preprocessed),
        "device": str(args.device),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "queries": {
            "genuine": int(genuine.size),
            "skilled": int(skilled.size),
            "random": int(random_scores.size),
        },
        "overall": overall,
    }
    if achieved_far is not None:
        report["target_far"] = args.target_far
        report["target_far_achieved"] = achieved_far

    # Per-dataset breakdown: the same operating point on each source dataset, so a
    # regression in one script cannot hide behind another dataset in the pooled number.
    by_dataset: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {GENUINE: [], "skilled": [], RANDOM: []}
    )
    for index, query_kind, score in iter_query_scores(embeddings, rows, enrolled):
        by_dataset[rows[index]["source_dataset"]][query_kind].append(score)

    datasets: dict[str, object] = {}
    for name in sorted(by_dataset):
        buckets = by_dataset[name]
        dataset_genuine = np.asarray(buckets[GENUINE], dtype=np.float64)
        dataset_skilled = np.asarray(buckets["skilled"], dtype=np.float64)
        dataset_random = np.asarray(buckets[RANDOM], dtype=np.float64)
        block = confusion(
            dataset_genuine,
            np.concatenate([dataset_skilled, dataset_random]),
            threshold,
        )
        if dataset_skilled.size:
            block["far_skilled"] = rates_at_threshold(
                dataset_genuine, dataset_skilled, threshold
            )[0]
            block["eer_skilled"] = eer_threshold(dataset_genuine, dataset_skilled)[1]
        if dataset_random.size:
            block["far_random"] = rates_at_threshold(
                dataset_genuine, dataset_random, threshold
            )[0]
            block["eer_random"] = eer_threshold(dataset_genuine, dataset_random)[1]
        datasets[name] = block
    report["per_dataset"] = datasets

    text = json.dumps(report, indent=2)
    print(text)
    if args.report is not None:
        destination = Path(args.report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


