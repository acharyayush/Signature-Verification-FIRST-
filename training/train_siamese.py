"""Train the Stage 3 ConvNeXt-Tiny signature embedding model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing.augmentation import STRENGTHS, augmentation_config
from preprocessing.signature_preprocessing import DEFAULT_TARGET_SIZE
from scoring.fusion import enroll_and_build_queries, score_queries
from scoring.metrics import eer_threshold
from training.stage3_metric_learning import (
    NegativePool,
    PKBatchSampler,
    SignatureDataset,
    SiameseSignatureModel,
    dataset_labels,
    embed_records,
    load_manifest,
    triplet_loss,
)


def _device_from_argument(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_epoch(
    model: SiameseSignatureModel,
    loader: DataLoader,
    device: torch.device,
    *,
    mining: Literal["semi-hard", "batch-hard"],
    margin: float,
    optimizer: torch.optim.Optimizer | None = None,
    negative_pool: NegativePool = "split",
    forgery_weight: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {"loss": 0.0}
    batch_count = 0

    for images, signer_ids, is_forgery in loader:
        images = images.to(device)
        signer_ids = signer_ids.to(device)
        is_forgery = is_forgery.to(device=device, dtype=torch.bool)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            embeddings = model(images)
            loss, metrics = triplet_loss(
                embeddings,
                signer_ids,
                is_forgery,
                margin=margin,
                mining=mining,
                negative_pool=negative_pool,
                forgery_weight=forgery_weight,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=5.0,
                )
                optimizer.step()

        totals["loss"] += loss.detach().item()
        # ``get`` keeps this robust when ``triplet_loss`` starts reporting a new metric.
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value.item()
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("data loader produced no batches")
    return {name: value / batch_count for name, value in totals.items()}


def _validation_eer(
    model: SiameseSignatureModel,
    validation_records,
    *,
    references: int,
    already_preprocessed: bool,
    ink: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> float | None:
    """Return the signer-disjoint validation EER on skilled forgeries, or ``None``.

    This is the metric that is actually reported, so using it to pick the checkpoint stops
    the triplet-loss scale from deciding the run. It reuses the shared protocol
    (``embed_records`` + ``enroll_and_build_queries`` + ``score_queries``), so the number
    logged here is computed exactly like the number in ``training.evaluate``.

    Returns ``None`` when the validation split cannot be enrolled (too few genuine images
    per signer), so the caller can fall back to the loss.
    """
    embeddings = embed_records(
        model,
        validation_records,
        already_preprocessed=already_preprocessed,
        device=device,
        batch_size=batch_size,
        workers=workers,
    )
    try:
        enrollment = enroll_and_build_queries(validation_records, references)
    except ValueError:
        return None
    scores = score_queries(embeddings, [], enrollment, None)
    genuine = scores["deep_genuine"]
    skilled = scores["deep_skilled"]
    if genuine.size == 0 or skilled.size == 0:
        return None
    return float(eer_threshold(genuine, skilled)[1])


def _set_seed(seed: int, device: torch.device) -> None:
    # This must be set before CUDA work starts. It makes cuBLAS reductions
    # deterministic on supported CUDA versions.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    elif device.type == "mps":
        warnings.warn(
            "MPS does not guarantee deterministic training; use --device cpu "
            "or a deterministic CUDA setup for repeatable results.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        torch.use_deterministic_algorithms(True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/stage3"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--ink", choices=("auto", "light", "dark"), default="auto")
    parser.add_argument(
        "--already-preprocessed",
        action="store_true",
        help="Input files are already 256x512 Stage 2 output images",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--signers-per-batch", type=int, default=12)
    parser.add_argument("--genuine-per-signer", type=int, default=4)
    parser.add_argument("--forgeries-per-signer", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument(
        "--negative-pool",
        choices=("split", "union"),
        default="split",
        help=(
            "How the two negative kinds are mined. 'split' (default) mines the "
            "cross-signer genuine pool and the same-signer forgery pool separately and "
            "adds the losses, so more forgeries add signal. 'union' merges them and takes "
            "the single closest negative, which lets forgeries crowd out the cross-signer "
            "negatives that teach identity separation (the previous behaviour)."
        ),
    )
    parser.add_argument(
        "--forgery-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the forgery term relative to the identity term under "
            "--negative-pool split. The terms are combined as "
            "(identity + w * forgery) / (1 + w), so the loss magnitude stays comparable "
            "to an identity-only loss."
        ),
    )
    parser.add_argument(
        "--select-metric",
        choices=("eer", "loss"),
        default="eer",
        help=(
            "What the best checkpoint is chosen by. 'eer' (default) uses the signer-"
            "disjoint validation EER on skilled forgeries -- the metric that is reported "
            "-- which stops the triplet loss scale from deciding the run. 'loss' is the "
            "previous behaviour."
        ),
    )
    parser.add_argument(
        "--eval-references",
        type=int,
        default=2,
        help="Enrolled references per signer when computing the validation EER.",
    )
    parser.add_argument(
        "--balance-datasets",
        action="store_true",
        help=(
            "Sample signers so every source dataset contributes an equal share of each "
            "batch, instead of sampling signers uniformly. Without this, a dataset with "
            "more signers dominates the gradient. Requires the manifest paths to sit "
            "under --image-root."
        ),
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Augment the training split only (rotation/tilt, shear, mild perspective, "
            "elastic warp, pen stroke width, ink contrast, blur, noise). Validation "
            "and test data are never augmented, because the checkpoint is selected by the "
            "validation EER. Each augmented image is re-whitened and re-fitted through the "
            "same crop + resize as inference, so the representation contract is unchanged."
        ),
    )
    parser.add_argument(
        "--augment-strength",
        choices=STRENGTHS,
        default="standard",
        help=(
            "Augmentation preset used with --augment. 'standard' tilts up to 20 degrees; "
            "'light' up to 10; 'strong' up to 26. BHSig260 (Indic script) automatically "
            "gets a milder elastic warp, which would otherwise destroy meaningful strokes."
        ),
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase1_epochs < 0 or args.phase2_epochs < 0:
        raise ValueError("epoch counts cannot be negative")
    if args.phase1_epochs + args.phase2_epochs == 0:
        raise ValueError("at least one training phase must have a positive epoch count")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.margin <= 0:
        raise ValueError("margin must be positive")
    if args.forgery_loss_weight < 0:
        raise ValueError("forgery-loss-weight cannot be negative")
    if args.eval_references < 1:
        raise ValueError("eval-references must be at least 1")

    device = _device_from_argument(args.device)
    _set_seed(args.seed, device)
    train_records, validation_records = load_manifest(args.manifest, args.image_root)
    all_signers = sorted(
        {record.signer_id for record in train_records + validation_records}
    )
    signer_to_index = {signer: index for index, signer in enumerate(all_signers)}

    # Optional dataset-balanced signer sampling. Signers are otherwise sampled uniformly,
    # so a dataset that holds more signers contributes proportionally more of every batch
    # and its script dominates the gradient.
    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root
        else Path(args.manifest).expanduser().resolve().parent
    )
    signer_groups: dict[str, str] = {}
    if args.balance_datasets:
        labels = dataset_labels(train_records + validation_records, image_root)
        signer_groups = {
            record.signer_id: label
            for record, label in zip(train_records + validation_records, labels)
        }
        per_dataset: dict[str, int] = {}
        for label in signer_groups.values():
            per_dataset[label] = per_dataset.get(label, 0) + 1
        print(
            f"Dataset-balanced sampling over {len(per_dataset)} datasets: "
            f"{sorted(per_dataset.items())}"
        )

    # Training-only augmentation. The config is resolved per source dataset so an Indic
    # script (BHSig260) gets a milder elastic warp than a Latin one (CEDAR), which would
    # otherwise destroy the connected strokes that discriminate writers.
    train_record_datasets: list[str] | None = None
    augment_configs: dict[str, AugmentationConfig] = {}
    if args.augment:
        try:
            train_record_datasets = dataset_labels(train_records, image_root)
            augment_configs = {
                label: augmentation_config(args.augment_strength, dataset=label)
                for label in sorted(set(train_record_datasets))
            }
        except ValueError as error:
            # Labels come from the processed layout. When it cannot be resolved the
            # augmentation still runs, with one config for every record.
            train_record_datasets = None
            augment_configs = {}
            print(f"warning: per-dataset augmentation configs unavailable ({error}); "
                  "using one config for all records")
        print(
            f"Augmentation: strength={args.augment_strength}, "
            f"datasets={sorted(augment_configs) or ['<all>']}"
        )

    train_dataset = SignatureDataset(
        train_records,
        signer_to_index,
        already_preprocessed=args.already_preprocessed,
        ink=args.ink,
        augment=args.augment,
        augment_config=augmentation_config(args.augment_strength),
        augment_seed=args.seed,
        record_datasets=train_record_datasets,
        dataset_configs=augment_configs,
    )
    # Validation is never augmented: the checkpoint is selected by the validation EER.
    validation_dataset = SignatureDataset(
        validation_records,
        signer_to_index,
        already_preprocessed=args.already_preprocessed,
        ink=args.ink,
    )

    train_sampler = PKBatchSampler(
        train_records,
        signers_per_batch=args.signers_per_batch,
        genuine_per_signer=args.genuine_per_signer,
        forgeries_per_signer=args.forgeries_per_signer,
        seed=args.seed,
        signer_groups=signer_groups,
    )
    validation_eligible_signers = {
        record.signer_id
        for record in validation_records
        if not record.is_forgery
    }
    genuine_counts = {
        signer: sum(
            record.signer_id == signer and not record.is_forgery
            for record in validation_records
        )
        for signer in validation_eligible_signers
    }
    eligible_validation_signers = sum(count >= 2 for count in genuine_counts.values())
    if eligible_validation_signers < 2:
        raise ValueError(
            "validation needs at least two signers with two genuine images each"
        )
    validation_batch_signers = min(args.signers_per_batch, eligible_validation_signers)
    validation_sampler = PKBatchSampler(
        validation_records,
        signers_per_batch=validation_batch_signers,
        genuine_per_signer=2,
        forgeries_per_signer=args.forgeries_per_signer,
        seed=args.seed,
        signer_groups=signer_groups,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Using device: {device}")
    model = SiameseSignatureModel(pretrained=not args.no_pretrained).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    # The checkpoint is chosen by the metric that gets reported, not by the training
    # objective: the triplet loss scale depends on the batch composition (how many
    # forgeries, which signers, which script mix), so selecting on it makes a data-mix
    # change look like a quality change.
    best_selection_score = math.inf
    best_selection_metric = "validation_loss"

    phases: list[tuple[str, int, float, Literal["semi-hard", "batch-hard"]]] = [
        ("warmup", args.phase1_epochs, 1e-3, "semi-hard"),
        ("fine_tune", args.phase2_epochs, 1e-4, "batch-hard"),
    ]
    global_epoch = 0
    eer_unavailable = False
    for phase_name, epoch_count, learning_rate, mining in phases:
        if epoch_count == 0:
            continue
        if phase_name == "warmup":
            model.freeze_backbone()
        else:
            model.unfreeze_last_stages(count=2)

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        for phase_epoch in range(epoch_count):
            train_sampler.set_epoch(global_epoch)
            # Vary the augmentation draw per epoch; the per-sample seed makes it
            # reproducible for a fixed (--seed, epoch, index).
            train_dataset.set_epoch(global_epoch)
            train_metrics = _run_epoch(
                model,
                train_loader,
                device,
                mining=mining,
                margin=args.margin,
                optimizer=optimizer,
                negative_pool=args.negative_pool,
                forgery_weight=args.forgery_loss_weight,
            )
            # Validate with batch-hard in every phase: semi-hard and batch-hard
            # losses are not comparable, which would skew best-checkpoint selection.
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                mining="batch-hard",
                margin=args.margin,
                negative_pool=args.negative_pool,
                forgery_weight=args.forgery_loss_weight,
            )
            global_epoch += 1
            record: dict[str, object] = {
                "phase": phase_name,
                "epoch": phase_epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
            }

            selection_score = validation_metrics["loss"]
            selection_metric = "validation_loss"
            if args.select_metric == "eer":
                validation_eer = _validation_eer(
                    model,
                    validation_records,
                    references=args.eval_references,
                    already_preprocessed=args.already_preprocessed,
                    ink=args.ink,
                    device=device,
                    batch_size=64,
                    workers=args.workers,
                )
                if validation_eer is not None:
                    record["validation_eer_skilled"] = validation_eer
                    selection_score = validation_eer
                    selection_metric = "validation_eer_skilled"
                elif not eer_unavailable:
                    # Say it once instead of silently selecting on the loss: a quiet
                    # fallback is exactly how the forgery-count confound went unnoticed.
                    eer_unavailable = True
                    print(
                        "warning: validation EER is unavailable (a validation signer has "
                        f"at most {args.eval_references} genuine images); "
                        "selecting on the triplet loss instead. Lower --eval-references."
                    )
            print(json.dumps(record))

            if selection_score < best_selection_score:
                best_selection_score = selection_score
                best_selection_metric = selection_metric
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "embedding_dim": model.embedding_dim,
                        # Selection provenance: which metric chose this checkpoint and
                        # what it scored, plus the epoch's validation loss for reference.
                        "selection_metric": best_selection_metric,
                        "selection_score": best_selection_score,
                        "validation_loss": validation_metrics["loss"],
                        "epoch": global_epoch,
                        "phase": phase_name,
                        "seed": args.seed,
                        "negative_pool": args.negative_pool,
                        "forgery_loss_weight": args.forgery_loss_weight,
                        "preprocessing": {
                            "already_preprocessed": args.already_preprocessed,
                            "ink": args.ink,
                            "target_size": list(DEFAULT_TARGET_SIZE),
                            # Stage 2.2 default: true grayscale ink preserved, paper
                            # background forced to pure white.
                            "background_mode": "whiten",
                            "augment": args.augment,
                            "augment_strength": args.augment_strength if args.augment else None,
                        },
                    },
                    checkpoint_path,
                )

    print(f"Best checkpoint: {checkpoint_path} ({best_selection_metric}={best_selection_score:.4f})")


if __name__ == "__main__":
    main()
