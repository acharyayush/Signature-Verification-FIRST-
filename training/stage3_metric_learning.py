"""Stage 3 ConvNeXt-Tiny embeddings and signer-balanced triplet learning."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping, Sequence

import numpy as np
import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from preprocessing.augmentation import AugmentationConfig, augment_signature
from preprocessing.signature_preprocessing import (
    load_preprocessed_image,
    to_model_input,
)

MiningMode = Literal["semi-hard", "batch-hard"]
# ``split`` mines the identity and forgery negative pools separately; ``union`` merges them
# into one pool and takes the single closest negative (the previous behaviour).
NegativePool = Literal["split", "union"]


@dataclass(frozen=True)
class SignatureRecord:
    """One manifest example and the signer whose identity it represents."""

    image_path: Path
    signer_id: str
    is_forgery: bool


def _parse_bool(value: str, *, row_number: int) -> bool:
    """Parse common CSV boolean spellings and report invalid values with row context."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(
        f"manifest row {row_number}: is_forgery must be true/false or 1/0"
    )


def load_manifest(
    manifest_path: str | Path,
    image_root: str | Path | None = None,
) -> tuple[list[SignatureRecord], list[SignatureRecord]]:
    """Load train/validation rows from a signer-disjoint CSV manifest.

    Required columns: ``image_path,signer_id,is_forgery,split``. Relative image
    paths are resolved from ``image_root`` or the manifest's directory. Forgeries
    must use the ID of the signer they imitate.
    """
    manifest = Path(manifest_path).expanduser().resolve()
    # Relative paths are rooted at image_root when provided, otherwise beside the CSV.
    root = Path(image_root).expanduser().resolve() if image_root else manifest.parent
    required = {"image_path", "signer_id", "is_forgery", "split"}
    train: list[SignatureRecord] = []
    validation: list[SignatureRecord] = []
    test_signers: set[str] = set()

    with manifest.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing required columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            image_path = row["image_path"].strip()
            signer_id = row["signer_id"].strip()
            split = row["split"].strip().lower()
            if not image_path or not signer_id:
                raise ValueError(
                    f"manifest row {row_number}: image_path and signer_id cannot be empty"
                )
            if split not in {"train", "val", "validation", "test"}:
                raise ValueError(
                    f"manifest row {row_number}: split must be train, val, or test"
                )

            path = Path(image_path).expanduser()
            if not path.is_absolute():
                path = root / path
            record = SignatureRecord(
                image_path=path.resolve(),
                signer_id=signer_id,
                is_forgery=_parse_bool(row["is_forgery"], row_number=row_number),
            )
            if split == "train":
                train.append(record)
            elif split in {"val", "validation"}:
                validation.append(record)
            else:
                test_signers.add(signer_id)

    if not train or not validation:
        raise ValueError("manifest must contain at least one train row and one val row")

    train_signers = {record.signer_id for record in train}
    validation_signers = {record.signer_id for record in validation}
    # Keep every signer confined to one split so validation does not leak identities.
    overlap = (train_signers & validation_signers) | (train_signers & test_signers) | (
        validation_signers & test_signers
    )
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"signers must be disjoint across train, val, and test: {examples}")
    return train, validation


def read_split(
    manifest: str | Path,
    root: str | Path,
    splits: set[str],
) -> list[SignatureRecord]:
    """Read manifest rows whose ``split`` column is in ``splits``.

    :func:`load_manifest` returns only the train and validation rows it needs for
    training. Evaluation and fusion fitting also need the test rows, so this helper
    reads an arbitrary split selection. It is the single split reader used by
    ``training.evaluate`` and ``training.fit_fusion``, which keeps both scripts
    scoring exactly the same records.

    Args:
        manifest: CSV manifest with ``image_path,signer_id,is_forgery,split``.
        root: Directory used to resolve relative image paths.
        splits: Lower-case split names to keep, e.g. ``{"val", "validation"}``.

    Raises:
        ValueError: If no row matches ``splits``.
    """
    manifest_path = Path(manifest).expanduser().resolve()
    image_root = Path(root).expanduser().resolve()
    wanted = {name.strip().lower() for name in splits}
    records: list[SignatureRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"].strip().lower() not in wanted:
                continue
            path = Path(row["image_path"].strip()).expanduser()
            records.append(
                SignatureRecord(
                    image_path=(path if path.is_absolute() else image_root / path).resolve(),
                    signer_id=row["signer_id"].strip(),
                    is_forgery=row["is_forgery"].strip().lower() in {"1", "true", "yes"},
                )
            )
    if not records:
        raise ValueError(f"no manifest rows for splits {sorted(wanted)}")
    return records


def dataset_labels(
    records: Sequence[SignatureRecord],
    root: str | Path,
    depth: int = 1,
) -> list[str]:
    """Return the source-dataset label of every record, derived from its path.

    The manifest schema is ``image_path,signer_id,is_forgery,split``: it carries no
    dataset column, so the dataset has to be recovered from the layout that
    :mod:`preprocessing.preprocess_dataset` writes. For the processed layout
    ``BHSig260/BHSig100_Bengali/Genuine/x.png`` a ``depth`` of 1 yields ``BHSig260``
    and a ``depth`` of 2 yields ``BHSig260/BHSig100_Bengali``.

    This is the single place that maps a record to a dataset, so the per-dataset
    breakdown, the per-dataset fusion threshold and the per-dataset handcrafted
    normalisation all agree on what "a dataset" is.

    Args:
        records: Manifest records with an absolute, resolved ``image_path``.
        root: Directory the manifest paths resolve against (``--image-root``).
        depth: Number of leading path components that identify a dataset.

    Raises:
        ValueError: If a record's path is not under ``root``. Guessing a label would
            silently create bogus datasets, so the caller is asked to fix
            ``--image-root`` instead.
    """
    image_root = Path(root).expanduser().resolve()
    labels: list[str] = []
    for record in records:
        path = Path(record.image_path)
        try:
            relative = path.relative_to(image_root)
        except ValueError as error:
            raise ValueError(
                f"image path {path} is not under --image-root {image_root}; "
                "pass the directory the manifest paths resolve against"
            ) from error
        parts = relative.parts[:-1]  # drop the file name
        labels.append("/".join(parts[:depth]) if parts else "unknown")
    return labels


class SignatureDataset(Dataset[tuple[Tensor, int, bool]]):
    """Load signatures, preprocess them, and return model-ready tensors and labels.

    Augmentation is opt-in and must only be enabled for the **training** split. The
    checkpoint is selected by the validation EER, so augmenting the validation set would
    corrupt the selector. Augmentation is applied here, after the shared loader and before
    ``to_model_input``, which keeps the handcrafted-feature branch (which reads the same
    loader) free of augmented pixels.
    """

    def __init__(
        self,
        records: Sequence[SignatureRecord],
        signer_to_index: dict[str, int],
        *,
        already_preprocessed: bool = False,
        ink: Literal["auto", "light", "dark"] = "auto",
        augment: bool = False,
        augment_config: AugmentationConfig | None = None,
        augment_seed: int = 0,
        record_datasets: Sequence[str] | None = None,
        dataset_configs: Mapping[str, AugmentationConfig] | None = None,
    ) -> None:
        self.records = list(records)
        self.signer_to_index = signer_to_index
        self.already_preprocessed = already_preprocessed
        self.ink = ink
        self.augment = augment
        self.augment_config = augment_config or AugmentationConfig()
        self.augment_seed = augment_seed
        # Optional per-record source dataset and per-dataset config, so a mixed-script
        # manifest can use a different augmentation for each script.
        self.record_datasets = list(record_datasets) if record_datasets is not None else None
        if self.record_datasets is not None and len(self.record_datasets) != len(self.records):
            raise ValueError("record_datasets must have one label per record")
        self.dataset_configs = dict(dataset_configs or {})
        # Bumped by ``set_epoch`` so the augmentation differs each epoch while staying
        # reproducible for a fixed (seed, epoch, index).
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation draw for the next epoch."""
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[Tensor, int, bool]:
        record = self.records[index]
        # One shared loading path: the handcrafted feature extractor reads the same
        # pixels through ``load_preprocessed_image``, so the two modalities cannot drift.
        preprocessed = load_preprocessed_image(
            record.image_path,
            already_preprocessed=self.already_preprocessed,
            ink=self.ink,
        )
        if self.augment:
            config = self.augment_config
            if self.record_datasets is not None:
                # A mixed-script manifest resolves its config per source dataset.
                config = self.dataset_configs.get(self.record_datasets[index], config)
            # Seeding from (seed, epoch, index) makes a run reproducible byte for byte
            # while the augmentation still varies across epochs and samples.
            rng = np.random.default_rng([self.augment_seed, self.epoch, index])
            preprocessed = augment_signature(preprocessed, config, rng)
        # The model accepts three channels; preprocessing repeats grayscale for RGB.
        tensor = torch.from_numpy(to_model_input(preprocessed))
        return tensor, self.signer_to_index[record.signer_id], record.is_forgery


class PKBatchSampler(Sampler[list[int]]):
    """Build batches with P signers, K genuine samples, and up to F forgeries each.

    P and K provide genuine positive pairs across distinct identities. Forgeries
    are sampled only as optional negatives associated with their imitated signer.
    """

    def __init__(
        self,
        records: Sequence[SignatureRecord],
        *,
        signers_per_batch: int,
        genuine_per_signer: int,
        forgeries_per_signer: int = 0,
        seed: int = 42,
        signer_groups: Mapping[str, str] | None = None,
    ) -> None:
        if signers_per_batch < 2:
            raise ValueError("signers_per_batch must be at least 2")
        if genuine_per_signer < 2:
            raise ValueError("genuine_per_signer must be at least 2")
        if forgeries_per_signer < 0:
            raise ValueError("forgeries_per_signer cannot be negative")

        self.signers_per_batch = signers_per_batch
        self.genuine_per_signer = genuine_per_signer
        self.forgeries_per_signer = forgeries_per_signer
        self.seed = seed
        self.epoch = 0
        # Keep genuine and forged record indices separate for each signer.
        self.by_signer: dict[str, dict[bool, list[int]]] = {}
        for index, record in enumerate(records):
            groups = self.by_signer.setdefault(record.signer_id, {False: [], True: []})
            groups[record.is_forgery].append(index)

        self.eligible_signers = [
            signer
            for signer, groups in self.by_signer.items()
            if len(groups[False]) >= genuine_per_signer
        ]
        # Each batch needs enough genuine samples per signer to form positives.
        if len(self.eligible_signers) < signers_per_batch:
            raise ValueError(
                f"need at least {signers_per_batch} signers with "
                f"{genuine_per_signer} genuine images; found {len(self.eligible_signers)}"
            )

        # Optional dataset grouping, used to stop one dataset from dominating a batch.
        self.signer_groups = dict(signer_groups or {})
        self.grouped_signers: dict[str, list[str]] = {}
        for signer in self.eligible_signers:
            group = self.signer_groups.get(signer)
            if group is not None:
                self.grouped_signers.setdefault(group, []).append(signer)

    def __len__(self) -> int:
        # Number of batches per epoch; signers are sampled anew for each batch.
        return max(1, (len(self.eligible_signers) + self.signers_per_batch - 1) // self.signers_per_batch)

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic sampling sequence for the next epoch."""
        self.epoch = epoch

    def _sample_signers(self, rng: random.Random) -> list[str]:
        """Pick ``signers_per_batch`` signers, optionally balanced across datasets.

        Without ``signer_groups`` signers are sampled uniformly, so a dataset that holds
        more signers contributes proportionally more of every batch and its script
        dominates the gradient. With groups, each dataset contributes an equal quota as far
        as its size allows, and any shortfall is topped up uniformly.
        """
        if not self.grouped_signers:
            return rng.sample(self.eligible_signers, self.signers_per_batch)

        groups = list(self.grouped_signers)
        rng.shuffle(groups)
        quota = math.ceil(self.signers_per_batch / len(groups))
        chosen: list[str] = []
        for group in groups:
            candidates = self.grouped_signers[group]
            take = min(quota, len(candidates), self.signers_per_batch - len(chosen))
            if take > 0:
                chosen.extend(rng.sample(candidates, take))
            if len(chosen) == self.signers_per_batch:
                break
        if len(chosen) < self.signers_per_batch:
            taken = set(chosen)
            remaining = [signer for signer in self.eligible_signers if signer not in taken]
            chosen.extend(rng.sample(remaining, self.signers_per_batch - len(chosen)))
        return chosen

    def __iter__(self) -> Iterator[list[int]]:
        # A local seeded RNG makes each epoch reproducible without changing global RNG.
        rng = random.Random(self.seed + self.epoch)
        for _ in range(len(self)):
            signer_batch = self._sample_signers(rng)
            indices: list[int] = []
            for signer in signer_batch:
                groups = self.by_signer[signer]
                indices.extend(rng.sample(groups[False], self.genuine_per_signer))
                # Not every signer has forgeries; use all available up to the cap.
                forgery_count = min(self.forgeries_per_signer, len(groups[True]))
                if forgery_count:
                    indices.extend(rng.sample(groups[True], forgery_count))
            yield indices


class SiameseSignatureModel(nn.Module):
    """Map each signature to a normalized embedding using a shared encoder.

    ConvNeXt extracts visual features. The projection head maps them to the
    embedding space, and L2 normalization makes cosine and Euclidean distances
    useful for comparing signatures.
    """

    def __init__(self, *, pretrained: bool = True, embedding_dim: int = 512) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            # LayerNorm acts per sample, independent of PK batch composition.
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch and return one unit-length embedding per image."""
        return F.normalize(self.projection(self.backbone(images)), p=2, dim=1)

    def compare(self, reference: Tensor, query: Tensor) -> Tensor:
        """Return pairwise cosine similarities; these are not calibrated probabilities."""
        reference_embedding = self(reference)
        query_embedding = self(query)
        return F.cosine_similarity(reference_embedding, query_embedding, dim=1)

    def freeze_backbone(self) -> None:
        """Stop gradient updates to the pretrained feature extractor."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_stages(self, count: int = 2) -> None:
        """Freeze the backbone, then enable gradients only in its last stages."""
        stages = getattr(self.backbone, "stages", None)
        if stages is None or count < 1:
            raise ValueError("backbone must expose stages and count must be positive")
        self.freeze_backbone()
        for stage in stages[-count:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True


def _select_negative(
    anchor_distances: Tensor,
    mask: Tensor,
    hardest_positive: Tensor,
    margin: float,
    mining: MiningMode,
    *,
    allow_empty: bool = False,
) -> Tensor:
    """Pick one negative distance per anchor from the candidates in ``mask``.

    ``batch-hard`` takes the closest candidate. ``semi-hard`` prefers the closest
    candidate that is farther than the hardest positive but still inside the margin, and
    falls back to the closest candidate when none qualifies. Rows with no candidate yield
    ``inf`` unless ``allow_empty`` is false, in which case they raise.
    """
    empty_rows = mask.sum(dim=1) == 0
    if not allow_empty and bool(empty_rows.any()):
        raise ValueError("every anchor must have at least one candidate negative")
    distances = anchor_distances.masked_fill(~mask, float("inf"))
    hardest_negative = distances.min(dim=1).values
    if mining != "semi-hard":
        return hardest_negative
    semi_hard_mask = (
        mask
        & (anchor_distances > hardest_positive[:, None])
        & (anchor_distances < (hardest_positive + margin)[:, None])
    )
    semi_hard_distances = anchor_distances.masked_fill(~semi_hard_mask, float("inf"))
    return torch.where(
        semi_hard_mask.any(dim=1),
        semi_hard_distances.min(dim=1).values,
        hardest_negative,
    )


def triplet_loss(
    embeddings: Tensor,
    signer_ids: Tensor,
    is_forgery: Tensor,
    *,
    margin: float = 0.3,
    mining: MiningMode = "batch-hard",
    negative_pool: NegativePool = "split",
    forgery_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute triplet loss for genuine anchors and return detached diagnostics.

    Positives are other genuine samples from the same signer. Forged samples never act as
    anchors or positives.

    Negatives come from two pools that teach different things:

    ``identity``
        genuine samples from *other* signers. This is what makes different writers
        separable, and it is the pool the random-forgery FAR depends on.
    ``forgery``
        forgeries imitating the anchor's own signer. This is what makes skilled forgeries
        rejectable.

    ``negative_pool="split"`` (default) mines each pool separately and adds the two losses,
    so more forgeries *add* signal. ``negative_pool="union"`` is the previous behaviour:
    the pools are merged and a single closest negative is mined. Because a forgery imitates
    the anchor's signer it is nearly always closer than any other signer's genuine
    signature, so under ``"union"`` the identity pool is crowded out and stops contributing
    gradient as soon as forgeries are present -- and the more forgeries per signer, the
    harder the closest one (an order statistic), so the loss becomes dominated by forgery
    rejection at the expense of identity separation.

    Args:
        embeddings: ``(N, D)`` embeddings for the batch.
        signer_ids: ``(N,)`` signer index per sample.
        is_forgery: ``(N,)`` bool, ``True`` for skilled forgeries.
        margin: Triplet margin.
        mining: ``"batch-hard"`` or ``"semi-hard"``, applied within each pool.
        negative_pool: ``"split"`` (default) or ``"union"`` (legacy).
        forgery_weight: Weight of the forgery term. The terms are combined as
            ``(identity + w * forgery) / (1 + w)`` so the loss magnitude stays comparable
            to an identity-only loss and the learning rate does not need retuning.

    Returns:
        The scalar loss and detached diagnostics: ``mean_positive``, ``mean_negative``
        (the identity, or the union, selected negative), ``mean_identity_negative``,
        ``mean_forgery_negative``, ``forgery_coverage`` and ``active_fraction``.
    """
    if embeddings.ndim != 2 or signer_ids.ndim != 1 or is_forgery.ndim != 1:
        raise ValueError("expected 2D embeddings and 1D signer/forgery labels")
    if embeddings.shape[0] != signer_ids.shape[0] or embeddings.shape[0] != is_forgery.shape[0]:
        raise ValueError("embeddings and labels must have the same batch size")
    if mining not in {"semi-hard", "batch-hard"}:
        raise ValueError(f"unsupported mining mode: {mining}")
    if negative_pool not in {"split", "union"}:
        raise ValueError(f"unsupported negative pool: {negative_pool}")
    if forgery_weight < 0:
        raise ValueError("forgery_weight must be non-negative")

    # Pairwise distances are used to select examples for each genuine anchor.
    distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    same_signer = signer_ids[:, None].eq(signer_ids[None, :])
    genuine = ~is_forgery.bool()
    not_self = ~torch.eye(len(signer_ids), dtype=torch.bool, device=signer_ids.device)
    # Positives exclude the anchor itself and must be genuine.
    positives = same_signer & genuine[None, :] & not_self
    # The two negative pools. Keeping them separate is what stops forgeries from hiding
    # the cross-signer negatives that teach identity separation.
    identity_pool = (~same_signer) & genuine[None, :]
    forgery_pool = same_signer & is_forgery.bool()[None, :]
    anchors = torch.where(genuine)[0]

    positive_mask = positives[anchors]
    identity_mask = identity_pool[anchors]
    forgery_mask = forgery_pool[anchors]
    if torch.any(positive_mask.sum(dim=1) == 0):
        raise ValueError("each genuine anchor needs another genuine sample from its signer")
    if torch.any(identity_mask.sum(dim=1) == 0):
        raise ValueError(
            "each genuine anchor needs a genuine sample from another signer; "
            "increase --signers-per-batch"
        )

    anchor_distances = distances[anchors]
    # Batch-hard loss takes the farthest valid positive and closest valid negative.
    positive_distances = anchor_distances.masked_fill(~positive_mask, float("-inf"))
    hardest_positive = positive_distances.max(dim=1).values

    identity_negative = _select_negative(
        anchor_distances, identity_mask, hardest_positive, margin, mining
    )
    identity_per_anchor = F.relu(hardest_positive - identity_negative + margin)

    has_forgery = forgery_mask.any(dim=1)
    if bool(has_forgery.any()):
        forgery_negative = _select_negative(
            anchor_distances, forgery_mask, hardest_positive, margin, mining, allow_empty=True
        )
        forgery_per_anchor = F.relu(hardest_positive - forgery_negative + margin)
        mean_forgery_negative = forgery_negative[has_forgery].mean().detach()
    else:
        forgery_negative = torch.full_like(identity_negative, float("inf"))
        forgery_per_anchor = torch.zeros_like(identity_per_anchor)
        mean_forgery_negative = torch.zeros((), dtype=identity_per_anchor.dtype)

    if negative_pool == "union":
        # Legacy behaviour: one merged pool, so the closest of the two kinds wins and the
        # identity pool stops contributing as soon as a closer forgery exists.
        union_mask = identity_mask | forgery_mask
        selected = _select_negative(
            anchor_distances, union_mask, hardest_positive, margin, mining
        )
        per_anchor_loss = F.relu(hardest_positive - selected + margin)
        return per_anchor_loss.mean(), {
            "mean_positive": hardest_positive.mean().detach(),
            "mean_negative": selected.mean().detach(),
            "mean_identity_negative": identity_negative.mean().detach(),
            "mean_forgery_negative": mean_forgery_negative,
            "forgery_coverage": has_forgery.float().mean().detach(),
            "active_fraction": (per_anchor_loss > 0).float().mean().detach(),
        }

    identity_loss = identity_per_anchor.mean()
    forgery_loss = (
        forgery_per_anchor[has_forgery].mean()
        if bool(has_forgery.any())
        else torch.zeros((), dtype=identity_loss.dtype)
    )
    # Averaging by (1 + weight) keeps the loss magnitude comparable to an identity-only
    # loss, so the learning rate and gradient clipping do not need retuning.
    loss = (identity_loss + forgery_weight * forgery_loss) / (1.0 + forgery_weight)

    active = identity_per_anchor > 0
    if bool(has_forgery.any()):
        active = active.clone()
        active[has_forgery] |= forgery_per_anchor[has_forgery] > 0
    # Metrics are detached so logging them does not retain the autograd graph.
    return loss, {
        "mean_positive": hardest_positive.mean().detach(),
        "mean_negative": identity_negative.mean().detach(),
        "mean_identity_negative": identity_negative.mean().detach(),
        "mean_forgery_negative": mean_forgery_negative,
        "forgery_coverage": has_forgery.float().mean().detach(),
        "active_fraction": active.float().mean().detach(),
    }


@torch.no_grad()
def embed_records(
    model: nn.Module,
    records: Sequence[SignatureRecord],
    *,
    already_preprocessed: bool = False,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    workers: int = 0,
) -> np.ndarray:
    """Embed ``records`` in order and return an ``(N, embedding_dim)`` float array.

    The output row order matches ``records``, so callers can index embeddings by
    manifest position. This is the embedding routine shared by ``training.evaluate``
    and ``training.fit_fusion``; keeping one implementation guarantees the deep score
    in a fusion fit is computed from the same forward pass as the reported baseline.

    Args:
        model: A trained Siamese embedding model in evaluation mode.
        records: Manifest records to embed.
        already_preprocessed: Forwarded to :class:`SignatureDataset`.
        device: Device used for the forward pass.
        batch_size: DataLoader batch size.
        workers: DataLoader worker count.
    """
    dataset = SignatureDataset(
        records,
        {record.signer_id: 0 for record in records},
        already_preprocessed=already_preprocessed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model.eval()
    chunks = [model(images.to(device)).cpu().numpy() for images, _, _ in loader]
    return np.concatenate(chunks)
