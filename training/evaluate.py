"""Evaluate a trained checkpoint with FAR / FRR / EER on signer-disjoint splits.

Protocol (per signer): the first ``--references`` genuine signatures are enrolled as
references. A query's score is its mean cosine similarity to those references.
  * positives:        the signer's remaining genuine signatures
  * skilled negatives: forgeries of that signer
  * random negatives:  genuine signatures of other signers
The decision threshold is chosen at the EER on the validation signers and then
applied unchanged to the test signers, so the test numbers are not tuned on test data.

When ``--fusion-config`` is supplied, the same protocol is repeated with the fused
score from :mod:`scoring.fusion`. The deep-only baseline block is still reported
unchanged, so one run shows both results side by side. Fused decisions use the
threshold stored in the fusion config when the fitter recorded one, falling back to
the validation EER threshold otherwise.

Usage:
    python -m training.evaluate --checkpoint checkpoints/cedar_v1/best_model.pt \
        --manifest data/processed/manifest.csv --already-preprocessed
    python -m training.evaluate ... --fusion-config checkpoints/fusion/fusion.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from scoring.fusion import (
    HAND_NORMALIZATION_KEY,
    Enrollment,
    FusionScorer,
    HandNormalizer,
    enroll_and_build_queries,
    load_fusion_config,
    score_queries,
)
from scoring.handcrafted_features import describe_records
from scoring.metrics import eer_threshold, rates_at_threshold, threshold_at_far
from training.stage3_metric_learning import (
    SiameseSignatureModel,
    dataset_labels,
    embed_records,
    read_split,
)


def _evaluate_split(
    records,
    embeddings: np.ndarray,
    descriptors,
    references: int,
    scorer: FusionScorer | None,
    max_random_queries: int | None,
    datasets: Sequence[str] | None = None,
    hand_normalizer: HandNormalizer | None = None,
    dataset_of: Mapping[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Run the reference protocol and return deep (and fused) scores by query kind."""
    enrollment: Enrollment = enroll_and_build_queries(
        records,
        references,
        max_random_queries=max_random_queries,
        dataset_of=dataset_of,
    )
    return score_queries(
        embeddings,
        descriptors or [],
        enrollment,
        scorer,
        datasets=datasets,
        hand_normalizer=hand_normalizer,
    )


def _metrics(scores: dict[str, np.ndarray], prefix: str, threshold: float) -> dict[str, object]:
    """Return the FAR / FRR / EER block for one score family and split."""
    genuine = scores[f"{prefix}_genuine"]
    skilled = scores[f"{prefix}_skilled"]
    random = scores[f"{prefix}_random"]
    far_skilled, frr = rates_at_threshold(genuine, skilled, threshold)
    far_random, _ = rates_at_threshold(genuine, random, threshold)
    _, eer_skilled = eer_threshold(genuine, skilled)
    _, eer_random = eer_threshold(genuine, random)
    return {
        "far_skilled_at_threshold": far_skilled,
        "far_random_at_threshold": far_random,
        "frr_at_threshold": frr,
        "eer_skilled": eer_skilled,
        "eer_random": eer_random,
        "genuine_queries": int(len(genuine)),
        "skilled_forgeries": int(len(skilled)),
        "random_pairs": int(len(random)),
    }


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
    parser.add_argument(
        "--fusion-config",
        type=Path,
        help="Fusion config written by training.fit_fusion; enables fused scoring.",
    )
    parser.add_argument(
        "--fusion-report",
        type=Path,
        help="Optional path to also write the printed JSON report to disk.",
    )
    parser.add_argument(
        "--max-random-queries",
        type=int,
        default=None,
        help="Deterministic cap on random query pairs (default: use every pair).",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help=(
            "Pick the decision threshold as the largest one whose FAR on the validation "
            "skilled forgeries is at most this value (e.g. 0.03). Because FAR falls and "
            "FRR rises with the threshold, this also minimises FRR subject to the FAR "
            "budget. Default: choose the threshold at the equal-error point."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Use this fixed score threshold on test data instead of deriving one from validation.",
    )
    parser.add_argument(
        "--random-negatives",
        choices=("within-dataset", "all"),
        default="within-dataset",
        help=(
            "Which signers the random (impostor) negatives come from. 'within-dataset' "
            "(default) uses only other signers of the same dataset, so the random-forgery "
            "rate measures within-script impostors. 'all' uses every other signer, which "
            "mixes in cross-script pairs that are trivially separable and makes the rate "
            "look far better than it is (the previous behaviour)."
        ),
    )
    args = parser.parse_args()
    if args.target_far is not None and not 0.0 <= args.target_far <= 1.0:
        raise ValueError("--target-far must be in [0, 1]")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.threshold is not None and args.target_far is not None:
        raise ValueError("--threshold and --target-far are mutually exclusive")

    root = (args.image_root or args.manifest.parent).resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    scorer: FusionScorer | None = None
    config = None
    if args.fusion_config is not None:
        config = load_fusion_config(args.fusion_config)
        scorer = FusionScorer(config)
        fitted_preprocessed = bool(config.metadata.get("already_preprocessed", True))
        if fitted_preprocessed != bool(args.already_preprocessed):
            print(
                "warning: --already-preprocessed differs from the fusion config metadata; "
                "handcrafted features may be computed on different pixels than during fitting"
            )

    # A fusion config fitted with --normalize-hand dataset carries the per-dataset
    # standardisation statistics. Rebuilding the normalizer from the config guarantees the
    # test features are transformed with exactly the numbers used at fitting time.
    hand_normalizer: HandNormalizer | None = None
    if config is not None:
        hand_normalizer = HandNormalizer.from_metadata(
            config.metadata.get(HAND_NORMALIZATION_KEY)
        )
        if hand_normalizer is not None:
            print(
                "applying per-dataset handcrafted standardisation from the fusion config "
                f"({len(hand_normalizer.mean)} datasets)"
            )

    deep: dict[str, dict[str, np.ndarray]] = {}
    fused: dict[str, dict[str, np.ndarray]] = {}
    for name, splits in (("val", {"val", "validation"}), ("test", {"test"})):
        records = read_split(args.manifest, root, splits)
        embeddings = embed_records(
            model,
            records,
            already_preprocessed=args.already_preprocessed,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        # Descriptors are only needed when fusion is active; skip the cost otherwise.
        descriptors = (
            describe_records(records, already_preprocessed=args.already_preprocessed)
            if scorer is not None
            else []
        )
        # Dataset labels are needed for the per-dataset normalisation and for restricting
        # random negatives to the same dataset.
        needs_datasets = (
            hand_normalizer is not None or args.random_negatives == "within-dataset"
        )
        datasets = dataset_labels(records, root) if needs_datasets else None
        dataset_of = (
            {record.signer_id: label for record, label in zip(records, datasets)}
            if datasets is not None and args.random_negatives == "within-dataset"
            else None
        )
        scores = _evaluate_split(
            records,
            embeddings,
            descriptors,
            args.references,
            scorer,
            args.max_random_queries,
            datasets=datasets,
            hand_normalizer=hand_normalizer,
            dataset_of=dataset_of,
        )
        deep[name] = {key: value for key, value in scores.items() if key.startswith("deep_")}
        if scorer is not None:
            fused[name] = {
                key: value for key, value in scores.items() if key.startswith("fused_")
            }

    # Threshold from validation signers; "skilled" is the harder and more relevant case.
    # With --target-far the threshold is the largest one whose FAR on the validation
    # skilled forgeries meets the budget, which also minimises FRR subject to it.
    validation_genuine = deep["val"]["deep_genuine"]
    validation_skilled = deep["val"]["deep_skilled"]
    _, validation_eer = eer_threshold(validation_genuine, validation_skilled)
    if args.threshold is not None:
        threshold = float(args.threshold)
        validation_far = None
        threshold_source = "fixed"
    elif args.target_far is not None:
        threshold, validation_far = threshold_at_far(validation_skilled, args.target_far)
        threshold_source = "validation_target_far"
    else:
        threshold = float(eer_threshold(validation_genuine, validation_skilled)[0])
        validation_far = None
        threshold_source = "validation_eer"
    report: dict[str, object] = {
        "threshold_from_validation": threshold,
        "threshold_source": threshold_source,
        "validation_eer_skilled": validation_eer,
        "random_negative_scope": args.random_negatives,
        "test": _metrics(deep["test"], "deep", threshold),
    }
    if validation_far is not None:
        report["target_far"] = args.target_far
        report["validation_far_skilled_at_threshold"] = validation_far

    if scorer is not None and config is not None:
        fused_genuine = fused["val"]["fused_genuine"]
        fused_skilled = fused["val"]["fused_skilled"]
        _, fused_val_eer = eer_threshold(fused_genuine, fused_skilled)
        if args.threshold is not None:
            fused_threshold = float(args.threshold)
            fused_val_far = None
        elif args.target_far is not None:
            fused_threshold, fused_val_far = threshold_at_far(
                fused_skilled, args.target_far
            )
        else:
            fused_threshold = float(eer_threshold(fused_genuine, fused_skilled)[0])
            fused_val_far = None
        # Prefer the threshold the fitter measured out-of-fold; fall back to the
        # validation split when the config carries no calibrated threshold.
        if args.threshold is not None:
            applied = float(args.threshold)
            threshold_source = "fixed"
        elif config.threshold is not None:
            applied = float(config.threshold)
            threshold_source = "fusion_config_out_of_fold"
        else:
            applied = fused_threshold
            threshold_source = (
                "validation_target_far" if args.target_far is not None else "validation_eer"
            )
        fusion_report: dict[str, object] = {
            "config": str(Path(args.fusion_config).expanduser().resolve()),  # type: ignore[union-attr]
            "type": config.type,
            "features": list(config.features),
            "dropped_features": list(config.dropped_features),
            "threshold_from_validation": fused_threshold,
            "threshold_used": applied,
            "threshold_source": threshold_source,
            "validation_eer_skilled": fused_val_eer,
            "test": _metrics(fused["test"], "fused", applied),
        }
        if fused_val_far is not None:
            fusion_report["target_far"] = args.target_far
            fusion_report["validation_far_skilled_at_threshold"] = fused_val_far
        report["fusion"] = fusion_report

    text = json.dumps(report, indent=2)
    print(text)
    if args.fusion_report is not None:
        destination = Path(args.fusion_report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
