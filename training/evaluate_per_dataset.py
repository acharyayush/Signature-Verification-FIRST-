"""Per-dataset FAR / FRR / EER breakdown for a trained checkpoint.

:mod:`training.evaluate` pools every signer in the manifest into a single score
distribution. Adding a new script (Persian, Thai) therefore changes the *mixture*, and
the headline FAR / FRR / EER stop being comparable with an earlier, smaller manifest: a
dataset that got worse and a dataset that was always hard produce the same pooled
number. This script runs the *same* protocol on each dataset slice separately, so a
regression in one dataset cannot hide behind another.

Datasets are identified by the first path component under ``--image-root`` (e.g.
``BHSig260``, ``cedar``, ``persian``, ``thai``), which is the layout
:mod:`preprocessing.preprocess_dataset` writes. ``--group-depth 2`` groups by the first
two components instead, which separates ``BHSig260/BHSig100_Bengali`` from
``BHSig260/BHSig160_Hindi``.

Every block reuses :func:`training.evaluate._metrics`,
:func:`scoring.fusion.score_queries` and :func:`training.stage3_metric_learning.read_split`,
so the per-dataset numbers come from exactly the same code as the pooled ones and the two
cannot drift apart.

Note: inside a dataset slice the "random" negatives are genuine signatures of other
signers *of that dataset*, i.e. within-script impostors. That is the harder and more
meaningful random-forgery number. The pooled run mixes in cross-script pairs, which are
almost trivially separable and make random FAR look artificially good.

Usage:
    python -m training.evaluate_per_dataset \\
        --checkpoint checkpoints/stage3/best_model.pt \\
        --manifest data/processed/manifest.csv --already-preprocessed \\
        --references 5 --fusion-config checkpoints/fusion/fusion.json \\
        --report checkpoints/fusion/per_dataset.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scoring.fusion import (
    HAND_NORMALIZATION_KEY,
    FusionScorer,
    HandNormalizer,
    enroll_and_build_queries,
    load_fusion_config,
    score_queries,
)
from scoring.handcrafted_features import describe_records
from scoring.metrics import eer_threshold, threshold_at_far
from training.evaluate import _metrics
from training.stage3_metric_learning import (
    SiameseSignatureModel,
    dataset_labels,
    embed_records,
    read_split,
)


def _slice_scores(
    records,
    embeddings: np.ndarray,
    descriptors,
    keys: list[str],
    indices: np.ndarray,
    references: int,
    scorer: FusionScorer | None,
    max_random_queries: int | None,
    hand_normalizer: HandNormalizer | None,
) -> dict[str, np.ndarray]:
    """Run the shared verification protocol on one slice of the records."""
    subset_records = [records[int(index)] for index in indices]
    enrollment = enroll_and_build_queries(
        subset_records, references, max_random_queries=max_random_queries
    )
    subset_descriptors = (
        [descriptors[int(index)] for index in indices] if descriptors else []
    )
    subset_keys = [keys[int(index)] for index in indices]
    return score_queries(
        embeddings[indices],
        subset_descriptors,
        enrollment,
        scorer,
        datasets=subset_keys,
        hand_normalizer=hand_normalizer,
    )


def _collect(
    records,
    embeddings: np.ndarray,
    descriptors,
    keys: list[str],
    references: int,
    scorer: FusionScorer | None,
    max_random_queries: int | None,
    hand_normalizer: HandNormalizer | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, set[str]],
    dict[str, int],
]:
    """Score the pooled split and every dataset slice inside it."""
    pooled = _slice_scores(
        records,
        embeddings,
        descriptors,
        keys,
        np.arange(len(records)),
        references,
        scorer,
        max_random_queries,
        hand_normalizer,
    )

    signers: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for index, key in enumerate(keys):
        signers[key].add(records[index].signer_id)
        counts[key] += 1

    per_dataset: dict[str, dict[str, np.ndarray]] = {}
    for key in sorted(counts):
        indices = np.flatnonzero(np.asarray(keys) == key)
        try:
            per_dataset[key] = _slice_scores(
                records, embeddings, descriptors, keys, indices, references, scorer,
                max_random_queries, hand_normalizer,
            )
        except ValueError as error:
            # Too few signers or too few genuine images per signer in this slice.
            print(f"  {key}: skipped ({error})")
    return pooled, per_dataset, signers, counts


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
        "--group-depth",
        type=int,
        default=1,
        help="Path components under --image-root that identify one dataset.",
    )
    parser.add_argument("--fusion-config", type=Path)
    parser.add_argument("--max-random-queries", type=int, default=None)
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help=(
            "Take the shared operating point as the largest threshold whose FAR on the "
            "pooled validation skilled forgeries is at most this value (e.g. 0.03), "
            "instead of the equal-error point."
        ),
    )
    parser.add_argument("--report", type=Path, help="Also write the JSON report here.")
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

    scorer: FusionScorer | None = None
    hand_normalizer: HandNormalizer | None = None
    if args.fusion_config is not None:
        config = load_fusion_config(args.fusion_config)
        scorer = FusionScorer(config)
        # A config fitted with --normalize-hand dataset carries its statistics, so the
        # test features are transformed with the same numbers used at fitting time.
        hand_normalizer = HandNormalizer.from_metadata(
            config.metadata.get(HAND_NORMALIZATION_KEY)
        )
        if hand_normalizer is not None:
            print(f"per-dataset handcrafted standardisation: {sorted(hand_normalizer.mean)}")

    collected: dict[str, tuple] = {}
    for name, wanted in (("val", {"val", "validation"}), ("test", {"test"})):
        records = read_split(args.manifest, root, wanted)
        keys = dataset_labels(records, root, args.group_depth)
        print(f"{name}: {len(records)} records in {len(set(keys))} datasets: {sorted(set(keys))}")
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
        collected[name] = _collect(
            records, embeddings, descriptors, keys,
            args.references, scorer, args.max_random_queries, hand_normalizer,
        )

    # One operating point from the pooled validation split, derived the same way
    # ``training.evaluate`` derives it, so every slice is compared at the same
    # threshold and only the EER columns are threshold-free.
    val_pooled = collected["val"][0]
    validation_genuine = val_pooled["deep_genuine"]
    validation_skilled = val_pooled["deep_skilled"]
    _, validation_eer = eer_threshold(validation_genuine, validation_skilled)
    if args.target_far is not None:
        threshold, validation_far = threshold_at_far(validation_skilled, args.target_far)
        threshold_source = "validation_target_far"
    else:
        threshold = float(eer_threshold(validation_genuine, validation_skilled)[0])
        validation_far = None
        threshold_source = "validation_eer"

    test_pooled, test_per_dataset, test_signers, test_counts = collected["test"]
    val_per_dataset = collected["val"][1]
    report: dict[str, object] = {
        "threshold_from_validation": threshold,
        "threshold_source": threshold_source,
        "validation_eer_skilled": validation_eer,
        "references": args.references,
        "group_depth": args.group_depth,
        "datasets": {},
    }
    if validation_far is not None:
        report["target_far"] = args.target_far
        report["validation_far_skilled_at_threshold"] = validation_far
    datasets: dict[str, object] = {}
    for key in sorted(test_per_dataset):
        entry: dict[str, object] = {
            "signers_in_test": len(test_signers[key]),
            "records_in_test": test_counts[key],
            "test": _metrics(test_per_dataset[key], "deep", threshold),
        }
        if key in val_per_dataset:
            _, entry["validation_eer_skilled"] = eer_threshold(
                val_per_dataset[key]["deep_genuine"], val_per_dataset[key]["deep_skilled"]
            )
        if scorer is not None:
            entry["test_fused"] = _metrics(test_per_dataset[key], "fused", threshold)
        datasets[key] = entry
    report["datasets"] = datasets
    report["pooled_test"] = _metrics(test_pooled, "deep", threshold)
    if scorer is not None:
        report["pooled_test_fused"] = _metrics(test_pooled, "fused", threshold)

    text = json.dumps(report, indent=2)
    print(text)
    if args.report is not None:
        destination = Path(args.report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
