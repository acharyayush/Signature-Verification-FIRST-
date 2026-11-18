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

import numpy as np
import torch

from scoring.fusion import (
    Enrollment,
    FusionScorer,
    enroll_and_build_queries,
    load_fusion_config,
    score_queries,
)
from scoring.handcrafted_features import describe_records
from scoring.metrics import eer_threshold, rates_at_threshold
from training.stage3_metric_learning import (
    SiameseSignatureModel,
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
) -> dict[str, np.ndarray]:
    """Run the reference protocol and return deep (and fused) scores by query kind."""
    enrollment: Enrollment = enroll_and_build_queries(
        records, references, max_random_queries=max_random_queries
    )
    return score_queries(embeddings, descriptors or [], enrollment, scorer)


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
    args = parser.parse_args()

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
        scores = _evaluate_split(
            records,
            embeddings,
            descriptors,
            args.references,
            scorer,
            args.max_random_queries,
        )
        deep[name] = {key: value for key, value in scores.items() if key.startswith("deep_")}
        if scorer is not None:
            fused[name] = {
                key: value for key, value in scores.items() if key.startswith("fused_")
            }

    # Threshold from validation signers; "skilled" is the harder and more relevant case.
    threshold, val_eer = eer_threshold(
        deep["val"]["deep_genuine"], deep["val"]["deep_skilled"]
    )
    report: dict[str, object] = {
        "threshold_from_validation": threshold,
        "validation_eer_skilled": val_eer,
        "test": _metrics(deep["test"], "deep", threshold),
    }

    if scorer is not None and config is not None:
        fused_threshold, fused_val_eer = eer_threshold(
            fused["val"]["fused_genuine"], fused["val"]["fused_skilled"]
        )
        # Prefer the threshold the fitter measured out-of-fold; fall back to the
        # validation EER when the config carries no calibrated threshold.
        if config.threshold is not None:
            applied = float(config.threshold)
            threshold_source = "fusion_config_out_of_fold"
        else:
            applied = fused_threshold
            threshold_source = "validation_eer"
        report["fusion"] = {
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

    text = json.dumps(report, indent=2)
    print(text)
    if args.fusion_report is not None:
        destination = Path(args.fusion_report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
