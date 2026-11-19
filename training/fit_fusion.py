"""Fit a score-level fusion config on the validation split.

The fusion config combines the deep cosine similarity of the ConvNeXt-Tiny Siamese
embedding with the nine handcrafted geometric/structural similarity features from
:mod:`scoring.handcrafted_features`. Two rules are fitted and compared:

``weighted``
    ``Final = w1 * S_deep + w2 * mean(H_hand)``, with ``w1`` grid-searched on the
    validation split.
``logistic``
    A logistic-regression classifier over ``[S_deep, *H_hand]``, stored as plain JSON
    coefficients so the config stays portable.

The fit runs on the **validation** split only; the test split is never touched here.
``--cv-folds`` (default 5) additionally measures the fused score out-of-fold with
signer-grouped folds, so the stored threshold is not the optimistic in-sample one.

Usage:
    python -m training.fit_fusion --checkpoint checkpoints/cedar_v1/best_model.pt \\
        --manifest data/processed/manifest.csv --already-preprocessed \\
        --output checkpoints/fusion/fusion.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from scoring.fusion import (
    HAND_NORMALIZATION_KEY,
    FusionConfig,
    FusionScorer,
    HandNormalizer,
    PairMatrix,
    build_pair_matrix,
    enroll_and_build_queries,
    fit_logistic_fusion,
    fit_weighted_fusion,
    save_fusion_config,
)
from scoring.handcrafted_features import FEATURE_NAMES, describe_records
from scoring.metrics import eer_threshold, threshold_at_far
from training.stage3_metric_learning import (
    SiameseSignatureModel,
    dataset_labels,
    embed_records,
    read_split,
)


def _fit_config(
    fusion_type: str, deep: np.ndarray, hand: np.ndarray, labels: np.ndarray
) -> tuple[FusionConfig, dict[str, object]]:
    """Fit one fusion rule on the supplied rows."""
    if fusion_type == "weighted":
        return fit_weighted_fusion(deep, hand, labels)
    if fusion_type == "logistic":
        return fit_logistic_fusion(deep, hand, labels)
    raise ValueError(f"unsupported fusion type: {fusion_type}")


def _query_scores_from_rows(
    row_scores: np.ndarray, query_ids: np.ndarray, kinds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average contiguous per-pair rows back to per-query scores.

    :func:`build_pair_matrix` emits all reference rows of one query consecutively, so the
    block boundaries can be recovered from the first occurrence of each query id. The
    per-query average is the unit the evaluation protocol thresholds, so the fitted
    threshold must live in that space.
    """
    _, first_index = np.unique(query_ids, return_index=True)
    boundaries = np.append(first_index[1:], len(query_ids))
    counts = boundaries - first_index
    totals = np.add.reduceat(row_scores, first_index)
    return totals / counts, kinds[first_index]


def _to_query_level(matrix: PairMatrix) -> PairMatrix:
    """Collapse per-reference rows to the per-query unit the evaluator thresholds.

    :func:`training.evaluate` thresholds the *average* score over a signer's references,
    but :func:`scoring.fusion.fit_weighted_fusion` grid-searches its weight on raw
    per-reference rows. Those are different distributions (averaging shrinks variance),
    so the chosen weight can be optimal for a unit that is never reported. Aggregating
    first makes the fitting objective identical to the measured one.

    For the ``weighted`` rule this does not change any score -- averaging is linear, so
    ``mean(w*deep + (1-w)*hand) == w*mean(deep) + (1-w)*mean(hand)`` -- it only moves the
    objective to the query level. The ``logistic`` rule is left at pair level, because
    :class:`scoring.fusion.FusionScorer` applies its coefficients per pair.
    """
    _, first_index = np.unique(matrix.query_ids, return_index=True)
    boundaries = np.append(first_index[1:], len(matrix.query_ids))
    counts = (boundaries - first_index).reshape(-1, 1)
    return PairMatrix(
        deep=np.add.reduceat(matrix.deep, first_index) / counts.ravel(),
        hand=np.add.reduceat(matrix.hand, first_index, axis=0) / counts,
        # Labels, signers and kinds are constant within a query, so the first row is
        # representative and avoids an integer division.
        labels=matrix.labels[first_index],
        signers=matrix.signers[first_index],
        kinds=matrix.kinds[first_index],
        query_ids=matrix.query_ids[first_index],
        datasets=None if matrix.datasets is None else matrix.datasets[first_index],
    )


def _cross_validated_threshold(
    matrix: PairMatrix,
    fusion_type: str,
    folds: int,
    target_far: float | None = None,
) -> tuple[float, float]:
    """Return ``(threshold, out_of_fold_eer)`` from signer-grouped folds.

    The split is grouped by signer, so a held-out fold contains identities the fold's
    rule never saw. That makes the measured threshold a fair estimate instead of the
    optimistic in-sample one.

    The threshold is always calibrated against **skilled forgeries only**, at the query
    level: that is the negative class the reported FAR / FRR use, and it is far harder
    than the random-impostor pairs that dominate the raw pair count. When ``target_far``
    is given, the threshold is the largest one meeting that FAR budget -- which also
    minimises FRR subject to the budget -- instead of the equal-error point, and the
    second return value is then the achieved FAR rather than the EER.
    """
    signers = np.unique(matrix.signers)
    if folds < 2:
        raise ValueError("cross-validation needs at least two folds")
    if len(signers) < folds:
        raise ValueError(
            f"cannot build {folds} signer-grouped folds from {len(signers)} signers"
        )

    oof_rows = np.full(len(matrix), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for train_index, test_index in splitter.split(
        matrix.deep, matrix.labels, groups=matrix.signers
    ):
        train = matrix.subset(train_index)
        if len(np.unique(train.labels)) < 2:
            raise ValueError("a fitting fold contains only genuine or only negative pairs")
        config, _ = _fit_config(fusion_type, train.deep, train.hand, train.labels)
        scorer = FusionScorer(config)
        oof_rows[test_index] = scorer.score_rows(
            matrix.deep[test_index], matrix.hand[test_index], FEATURE_NAMES
        )

    if np.isnan(oof_rows).any():
        raise RuntimeError("out-of-fold scoring left rows unscored")

    query_scores, query_kinds = _query_scores_from_rows(
        oof_rows, matrix.query_ids, matrix.kinds
    )
    genuine = query_scores[query_kinds == "genuine"]
    skilled = query_scores[query_kinds == "skilled"]
    if genuine.size == 0 or skilled.size == 0:
        raise ValueError("cross-validation needs both genuine and skilled queries")
    if target_far is not None:
        threshold, achieved_far = threshold_at_far(skilled, target_far)
        return float(threshold), float(achieved_far)
    threshold, eer = eer_threshold(genuine, skilled)
    return float(threshold), float(eer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--already-preprocessed", action="store_true")
    parser.add_argument("--references", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fusion-type",
        choices=("weighted", "logistic", "both"),
        default="both",
        help="Which fusion rule(s) to fit; 'both' keeps the lower out-of-fold EER.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Signer-grouped folds used to measure the out-of-fold threshold.",
    )
    parser.add_argument(
        "--max-random-queries",
        type=int,
        default=None,
        help="Deterministic cap on random query pairs (default: use every pair).",
    )
    parser.add_argument(
        "--fit-unit",
        choices=("pair", "query"),
        default="query",
        help=(
            "Unit the weighted rule is optimised on. 'query' (default) matches the unit "
            "training.evaluate actually thresholds; 'pair' reproduces the old behaviour."
        ),
    )
    parser.add_argument(
        "--normalize-hand",
        choices=("none", "dataset"),
        default="none",
        help=(
            "Z-score the handcrafted features within each dataset before fusing. Off by "
            "default because it needs the dataset label at scoring time; see "
            "scoring.fusion.HandNormalizer."
        ),
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help=(
            "Calibrate the stored threshold to this FAR on skilled forgeries instead of "
            "the equal-error point (e.g. 0.03). The threshold returned is the largest one "
            "meeting the budget, which also minimises FRR subject to the budget."
        ),
    )
    args = parser.parse_args()
    if args.target_far is not None and not 0.0 <= args.target_far <= 1.0:
        raise ValueError("--target-far must be in [0, 1]")

    root = (args.image_root or args.manifest.parent).resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    records = read_split(args.manifest, root, {"val", "validation"})
    print(f"Read {len(records)} validation records")

    embeddings = embed_records(
        model,
        records,
        already_preprocessed=args.already_preprocessed,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(f"Embedded {len(embeddings)} records")

    descriptors = describe_records(
        records, already_preprocessed=args.already_preprocessed
    )
    print(f"Described {len(descriptors)} records")

    enrollment = enroll_and_build_queries(
        records,
        args.references,
        max_random_queries=args.max_random_queries,
    )
    print(f"Built {len(enrollment.queries)} queries")

    # The dataset label is derived from the processed layout (BHSig260/..., cedar/...).
    # It is only required for the opt-in per-dataset standardisation, so resolving it is
    # only fatal when that option is on.
    record_datasets = (
        dataset_labels(records, root) if args.normalize_hand == "dataset" else None
    )
    if record_datasets is not None:
        print(f"Dataset labels: {sorted(set(record_datasets))}")

    matrix = build_pair_matrix(
        embeddings, descriptors, enrollment, datasets=record_datasets
    )
    print(
        f"Built {len(matrix)} fitting pairs across "
        f"{len(enrollment.references)} signers "
        f"({int(matrix.labels.sum())} genuine)"
    )

    # Random (impostor) pairs are far easier to reject than skilled forgeries, so if they
    # dominate the negatives the fitted weight and threshold are calibrated for the easy
    # case while the reported FAR is measured on skilled forgeries.
    negative_rows = int((matrix.labels == 0).sum())
    random_rows = int((matrix.kinds == "random").sum())
    if negative_rows and random_rows / negative_rows > 0.5:
        print(
            f"warning: {random_rows}/{negative_rows} "
            f"({random_rows / negative_rows:.1%}) of the fitting negatives are random "
            "impostor pairs, which are much easier than skilled forgeries. Pass "
            "--max-random-queries 0 to fit the rule and threshold on skilled forgeries only."
        )

    normalizer: HandNormalizer | None = None
    if record_datasets is not None:
        # Statistics are estimated on the raw fitting rows and then applied to those same
        # rows. The goal is to remove the per-script level shift so one global weight and
        # one global threshold become meaningful, not to whiten away the signal.
        normalizer = HandNormalizer.fit(matrix.hand, matrix.datasets)
        matrix = matrix.with_hand(normalizer.transform(matrix.hand, matrix.datasets))
        print(f"Standardised handcrafted features per dataset: {sorted(normalizer.mean)}")

    types = ["weighted", "logistic"] if args.fusion_type == "both" else [args.fusion_type]
    results: dict[str, tuple[FusionConfig, dict[str, object], float | None]] = {}
    for fusion_type in types:
        # The weighted rule is optimised in the unit the evaluator thresholds. The
        # logistic rule keeps its per-reference rows, because FusionScorer applies its
        # coefficients per pair and changing that would need a scorer-side change too.
        if args.fit_unit == "query" and fusion_type == "weighted":
            fit_matrix = _to_query_level(matrix)
        else:
            if args.fit_unit == "query" and fusion_type == "logistic":
                print(
                    "note: --fit-unit query does not apply to the logistic rule; its "
                    "coefficients are applied per reference pair by FusionScorer"
                )
            fit_matrix = matrix

        config, metadata = _fit_config(
            fusion_type, fit_matrix.deep, fit_matrix.hand, fit_matrix.labels
        )
        print(f"Fitted {fusion_type} fusion on {len(fit_matrix)} {args.fit_unit}-level rows")
        try:
            threshold, out_of_fold = _cross_validated_threshold(
                fit_matrix, fusion_type, args.cv_folds, args.target_far
            )
            config = replace(config, threshold=threshold)
            label = "FAR" if args.target_far is not None else "EER"
            print(
                f"{fusion_type}: out-of-fold {label} {out_of_fold:.4f} "
                f"at threshold {threshold:.4f}"
            )
        except ValueError as error:
            print(f"{fusion_type}: cross-validation skipped ({error})")
            out_of_fold = None
        results[fusion_type] = (config, metadata, out_of_fold)

    if len(results) > 1:
        best_type = min(
            results,
            key=lambda name: results[name][2] if results[name][2] is not None else 1.0,
        )
        criterion = "FAR" if args.target_far is not None else "EER"
        print(f"Selected {best_type} fusion (lowest out-of-fold {criterion})")
    else:
        best_type = types[0]

    config, _, out_of_fold = results[best_type]
    # Record how the config was produced so ``training.evaluate`` can detect a
    # preprocessing mismatch (``--already-preprocessed``) instead of silently
    # computing handcrafted features on different pixels than during fitting.
    provenance: dict[str, object] = {
        **config.metadata,
        "already_preprocessed": bool(args.already_preprocessed),
        "references": args.references,
        "fusion_type": best_type,
        "cv_folds": args.cv_folds,
        "fit_unit": args.fit_unit,
        "signers": len(enrollment.references),
        "fit_pairs": len(matrix),
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
    }
    if args.target_far is not None:
        provenance["target_far"] = args.target_far
        provenance["out_of_fold_far"] = out_of_fold
    else:
        provenance["out_of_fold_eer"] = out_of_fold
    if normalizer is not None:
        provenance[HAND_NORMALIZATION_KEY] = normalizer.to_metadata()
    config = replace(config, metadata=provenance)
    saved = save_fusion_config(config, args.output)
    print(f"Saved {best_type} fusion config to {saved}")
    print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    main()

