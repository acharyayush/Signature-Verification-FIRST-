"""Evaluate the "Ultimate Test Signs" benchmark: FAR / FRR and a confusion matrix.

The benchmark is built by :mod:`scripts.build_ultimate_test_signs` and described by
``Ultimate Test Signs/manifest_ultimate.csv``. The builder stores Stage 2.2 preprocessed
``384 x 192`` grayscale PNGs, and evaluation reads them as they are: this script does no
preprocessing, so the images must already be preprocessed (the builder guarantees it).
Every signer folder holds three genuine signatures, three skilled forgeries of that
signer, and three genuine signatures of a *different* signer of the same dataset::

    Signer_001/Genuine/  3 genuine signatures   -> positive pairs
    Signer_001/Forged/   3 skilled forgeries    -> skilled impostor pairs (negatives)
    Signer_001/Random/   3 other-signer genuine -> random impostor pairs  (negatives)

Protocol (pairwise). There is no enrolment and no averaging over references: every
comparison is a single pair, scored by the cosine similarity of its two embeddings. A
signer with three genuine, three skilled, and three random signatures contributes nine
pairs::

    genuine: every unordered pair of the genuine signatures     -> C(3, 2) = 3 pairs
    skilled: the i-th genuine against the i-th skilled forgery  -> 3 pairs
    random:  the i-th genuine against the i-th random impostor  -> 3 pairs

Both impostor folders are sorted by file name and paired positionally with the sorted
genuine signatures, so the pairing is deterministic and every genuine signature takes part.
Genuine pairs are the positives and both impostor kinds are the negatives. A pair is
accepted when ``score >= threshold``, because a higher score means "more likely the same
signer"::

    true positives   (TP): genuine pair accepted
    false negatives  (FN): genuine pair rejected
    true negatives   (TN): impostor pair rejected
    false positives  (FP): impostor pair accepted

with ``FAR = FP / (FP + TN)`` and ``FRR = FN / (FN + TP)``. Skilled-forgery and random FAR
are also reported separately, because a learned forgery and an unrelated signer are very
different threats.

Operating point: by default a fixed threshold of ``0.823`` is applied
(:data:`DEFAULT_THRESHOLD`, a temporary stand-in for a calibrated value). Pass
``--threshold`` to apply a different fixed cut (for example the deep threshold
:mod:`training.evaluate` derived on the validation signers), or ``--target-far`` to take the
largest threshold whose combined-impostor FAR meets a budget.

Score-level fusion (optional). With ``--fusion-config`` every pair is also scored by the
fused rule from :mod:`scoring.fusion` (the deep cosine plus the nine handcrafted features)
and a ``fusion`` block with its own FAR / FRR / EER is added next to the deep baseline, both
pooled and per dataset. The fused operating point is taken from ``--fused-threshold``, then
the threshold the fitter stored in the config, then ``--threshold``, then
:data:`DEFAULT_THRESHOLD`; the deep threshold is never silently reused unless nothing else is
available, because the two live in different score spaces.

Usage:
    python -m training.evaluate_ultimate \\
        --checkpoint best_model_first.pt \\
        --manifest "Ultimate Test Signs/manifest_ultimate.csv"
    python -m training.evaluate_ultimate ... --threshold 0.823 \\
        --report ultimate_report.json
    python -m training.evaluate_ultimate ... --fusion-config fusion.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scoring.fusion import (
    HAND_NORMALIZATION_KEY,
    FusionScorer,
    HandNormalizer,
    load_fusion_config,
)
from scoring.handcrafted_features import describe_records, pair_features
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

    ``signer_id`` is the signer *folder* the row lives in, so every pair is scored within the
    signer folder it belongs to. ``is_forgery`` is only set for the explicit forgeries; the
    genuine/forged/random distinction needed for the confusion matrix is carried by the row
    itself, not by this record flag.
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
# Temporary fixed operating point used when neither ``--threshold`` nor ``--target-far`` is
# given. 0.823 is a stand-in for a calibrated value; restore the equal-error default once
# the real threshold is available.
DEFAULT_THRESHOLD = 0.823
_REQUIRED_COLUMNS = {"folder", "class", "source_dataset", "source_signer", "image_path"}


def _folder_signatures(
    rows: list[dict[str, str]],
    indices: list[int],
    kind: str,
) -> list[int]:
    """Return the folder's ``kind`` rows, ordered by file name for a deterministic pairing."""
    return sorted(
        (index for index in indices if rows[index]["class"] == kind),
        key=lambda index: rows[index]["image_path"],
    )


def pair_indices(
    rows: list[dict[str, str]],
    by_folder: dict[str, list[int]],
) -> dict[str, list[tuple[int, int]]]:
    """Build the comparison pairs every signer contributes, grouped by pair kind.

    Per folder, with the genuine/forged/random signatures each ordered by file name:

    * ``genuine``: every unordered pair of the genuine signatures, ``C(n, 2)`` pairs.
    * ``skilled``: the ``i``-th genuine against the ``i``-th skilled forgery.
    * ``random``:  the ``i``-th genuine against the ``i``-th random impostor.

    A folder holding three of each class therefore yields ``3 + 3 + 3 = 9`` pairs, which is
    the protocol the benchmark is built around. Every pair takes part, so nothing is dropped
    silently.

    Raises:
        ValueError: If a folder has fewer than two genuine signatures, or its forged/random
            counts differ from its genuine count (the positional pairing would drop pairs).
    """
    pairs: dict[str, list[tuple[int, int]]] = {GENUINE: [], "skilled": [], RANDOM: []}
    for folder, indices in by_folder.items():
        genuine = _folder_signatures(rows, indices, GENUINE)
        if len(genuine) < 2:
            raise ValueError(
                f"{folder} has {len(genuine)} genuine signatures; at least two are needed "
                "for a genuine pair"
            )
        for first in range(len(genuine)):
            for second in range(first + 1, len(genuine)):
                pairs[GENUINE].append((genuine[first], genuine[second]))
        for kind, impostor_class in (("skilled", FORGED), (RANDOM, RANDOM)):
            impostors = _folder_signatures(rows, indices, impostor_class)
            if len(impostors) != len(genuine):
                raise ValueError(
                    f"{folder} has {len(genuine)} genuine but {len(impostors)} {kind} "
                    "signatures; the positional pairing needs equal counts"
                )
            pairs[kind].extend(
                (genuine[position], impostors[position]) for position in range(len(genuine))
            )
    return pairs


def iter_pair_scores(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    pairs: dict[str, list[tuple[int, int]]],
):
    """Yield ``(row_index, pair_kind, score)`` for every comparison pair.

    ``row_index`` is the first (genuine) row of the pair, so a caller can attribute the pair
    to that row's ``source_dataset``. Embeddings are L2-normalised, so their dot product is
    the cosine similarity. ``pair_kind`` is ``genuine``, ``skilled``, or ``random``.
    """
    for kind, index_pairs in pairs.items():
        for first, second in index_pairs:
            yield first, kind, float(embeddings[first] @ embeddings[second])


def score_pairs(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    pairs: dict[str, list[tuple[int, int]]],
) -> dict[str, list[float]]:
    """Collect :func:`iter_pair_scores` into one score list per pair kind."""
    scores: dict[str, list[float]] = {GENUINE: [], "skilled": [], RANDOM: []}
    for _, pair_kind, score in iter_pair_scores(embeddings, rows, pairs):
        scores[pair_kind].append(score)
    return scores


def iter_fused_pair_scores(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    pairs: dict[str, list[tuple[int, int]]],
    descriptors: list,
    scorer: FusionScorer,
    hand_normalizer: HandNormalizer | None = None,
):
    """Yield ``(first, second, pair_kind, fused_score)`` for every comparison pair.

    Mirrors :func:`iter_pair_scores` but replaces the cosine score with the fused rule from
    :mod:`scoring.fusion`: the pair's nine handcrafted features are combined with its deep
    cosine score. When a ``hand_normalizer`` is present the features are standardised with
    the *query* row's dataset, exactly as :func:`scoring.fusion.score_queries` does, so the
    fused score is computed in the space the config was fitted in.
    """
    for kind, index_pairs in pairs.items():
        for first, second in index_pairs:
            deep = float(embeddings[first] @ embeddings[second])
            features = pair_features(descriptors[first], descriptors[second])
            if hand_normalizer is not None:
                features = hand_normalizer.transform_features(
                    features, rows[second]["source_dataset"]
                )
            yield first, second, kind, float(scorer.score(deep, features))


def fusion_decision_threshold(
    *,
    fused_threshold: float | None,
    config_threshold: float | None,
    deep_threshold: float | None,
) -> tuple[float, str]:
    """Pick the fused operating point and return ``(threshold, source)``.

    Precedence: an explicit ``--fused-threshold``, then the threshold the fitter stored in
    the config, then an explicit ``--threshold``, then the temporary fixed
    :data:`DEFAULT_THRESHOLD`. The fused score lives in a different space from the deep
    cosine, so a threshold is only borrowed across the two when nothing else is available.
    """
    if fused_threshold is not None:
        return float(fused_threshold), "fused_threshold"
    if config_threshold is not None:
        return float(config_threshold), "fusion_config"
    if deep_threshold is not None:
        return float(deep_threshold), "threshold"
    return DEFAULT_THRESHOLD, "default"


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

    Precedence: an explicit ``--threshold`` wins, then a ``--target-far`` budget on the
    combined impostors, then the temporary fixed :data:`DEFAULT_THRESHOLD`. ``achieved_far``
    is only filled for the target-FAR branch.
    """
    if fixed is not None:
        return float(fixed), "fixed", None
    if target_far is not None:
        threshold, achieved_far = threshold_at_far(negatives, target_far)
        return float(threshold), "target_far", achieved_far
    return DEFAULT_THRESHOLD, "default", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Apply this fixed score threshold instead of the temporary default 0.823.",
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
    parser.add_argument(
        "--fusion-config",
        type=Path,
        help="Fusion config written by training.fit_fusion; adds a fused score block.",
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
    parser.add_argument(
        "--fusion-report",
        type=Path,
        help="Optional path to also write the full JSON report (deep + fusion) to disk.",
    )
    parser.add_argument("--report", type=Path, help="Also write the JSON report here.")
    args = parser.parse_args()
    if args.target_far is not None and not 0.0 <= args.target_far <= 1.0:
        raise ValueError("--target-far must be in [0, 1]")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.threshold is not None and args.target_far is not None:
        raise ValueError("--threshold and --target-far are mutually exclusive")
    if args.fused_threshold is not None and not 0.0 <= args.fused_threshold <= 1.0:
        raise ValueError("--fused-threshold must be in [0, 1]")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    scorer: FusionScorer | None = None
    fusion_config = None
    hand_normalizer: HandNormalizer | None = None
    if args.fusion_config is not None:
        fusion_config = load_fusion_config(args.fusion_config)
        scorer = FusionScorer(fusion_config)
        hand_normalizer = HandNormalizer.from_metadata(
            fusion_config.metadata.get(HAND_NORMALIZATION_KEY)
        )
        fitted_checkpoint = fusion_config.metadata.get("checkpoint")
        if fitted_checkpoint and Path(str(fitted_checkpoint)).name != Path(
            args.checkpoint
        ).name:
            print(
                "warning: the fusion config was fitted for a different checkpoint "
                f"({Path(str(fitted_checkpoint)).name}); the fused weights are applied "
                "as-is and may not transfer"
            )

    rows = read_ultimate_manifest(args.manifest, args.image_root)
    records, by_folder = build_records(rows)
    pairs = pair_indices(rows, by_folder)
    pair_count = sum(len(group) for group in pairs.values())
    print(
        f"{len(records)} images from {len(by_folder)} signers; {pair_count} comparison "
        f"pairs ({pair_count // len(by_folder)} per signer); embedding on {args.device} "
        "(preprocessed images; no preprocessing applied)"
    )
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

    fused_pair_list: list[tuple[int, int, str, float]] = []
    fused_genuine: np.ndarray | None = None
    fused_skilled: np.ndarray | None = None
    fused_random: np.ndarray | None = None
    if scorer is not None:
        descriptors = describe_records(records, already_preprocessed=True)
        fused_pair_list = list(
            iter_fused_pair_scores(
                embeddings, rows, pairs, descriptors, scorer, hand_normalizer
            )
        )
        fused_by_kind: dict[str, list[float]] = {GENUINE: [], "skilled": [], RANDOM: []}
        for _, _, pair_kind, score in fused_pair_list:
            fused_by_kind[pair_kind].append(score)
        fused_genuine = np.asarray(fused_by_kind[GENUINE], dtype=np.float64)
        fused_skilled = np.asarray(fused_by_kind["skilled"], dtype=np.float64)
        fused_random = np.asarray(fused_by_kind[RANDOM], dtype=np.float64)

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
        "device": str(args.device),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "pairs": {
            "genuine": int(genuine.size),
            "skilled": int(skilled.size),
            "random": int(random_scores.size),
            "per_signer": pair_count // len(by_folder),
        },
        "overall": overall,
    }
    if achieved_far is not None:
        report["target_far"] = args.target_far
        report["target_far_achieved"] = achieved_far

    fused_threshold_value: float | None = None
    if scorer is not None and fusion_config is not None:
        fused_threshold_value, fused_threshold_source = fusion_decision_threshold(
            fused_threshold=args.fused_threshold,
            config_threshold=fusion_config.threshold,
            deep_threshold=args.threshold,
        )
        fused_negatives = np.concatenate([fused_skilled, fused_random])
        fused_overall = confusion(fused_genuine, fused_negatives, fused_threshold_value)
        fused_overall["far_skilled"] = rates_at_threshold(
            fused_genuine, fused_skilled, fused_threshold_value
        )[0]
        fused_overall["far_random"] = rates_at_threshold(
            fused_genuine, fused_random, fused_threshold_value
        )[0]
        fused_overall["eer_skilled"] = eer_threshold(fused_genuine, fused_skilled)[1]
        fused_overall["eer_random"] = eer_threshold(fused_genuine, fused_random)[1]
        fused_overall["eer_combined"] = eer_threshold(fused_genuine, fused_negatives)[1]
        report["fusion"] = {
            "config": str(Path(args.fusion_config).expanduser().resolve()),
            "type": fusion_config.type,
            "features": list(fusion_config.features),
            "dropped_features": list(fusion_config.dropped_features),
            "hand_normalized": hand_normalizer is not None,
            "threshold": fused_threshold_value,
            "threshold_source": fused_threshold_source,
            "overall": fused_overall,
        }

    # Per-dataset breakdown: the same operating point on each source dataset, so a
    # regression in one script cannot hide behind another dataset in the pooled number.
    by_dataset: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {GENUINE: [], "skilled": [], RANDOM: []}
    )
    for index, pair_kind, score in iter_pair_scores(embeddings, rows, pairs):
        by_dataset[rows[index]["source_dataset"]][pair_kind].append(score)

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

    # The same fused operating point on each source dataset, next to the deep block.
    if scorer is not None and fused_threshold_value is not None:
        fused_by_dataset: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {GENUINE: [], "skilled": [], RANDOM: []}
        )
        for index, _, pair_kind, score in fused_pair_list:
            fused_by_dataset[rows[index]["source_dataset"]][pair_kind].append(score)
        for name in sorted(datasets):
            block = datasets[name]
            buckets = fused_by_dataset.get(name)
            if not buckets:
                continue
            dataset_genuine = np.asarray(buckets[GENUINE], dtype=np.float64)
            dataset_skilled = np.asarray(buckets["skilled"], dtype=np.float64)
            dataset_random = np.asarray(buckets[RANDOM], dtype=np.float64)
            fused_block = confusion(
                dataset_genuine,
                np.concatenate([dataset_skilled, dataset_random]),
                fused_threshold_value,
            )
            if dataset_skilled.size:
                fused_block["far_skilled"] = rates_at_threshold(
                    dataset_genuine, dataset_skilled, fused_threshold_value
                )[0]
                fused_block["eer_skilled"] = eer_threshold(
                    dataset_genuine, dataset_skilled
                )[1]
            if dataset_random.size:
                fused_block["far_random"] = rates_at_threshold(
                    dataset_genuine, dataset_random, fused_threshold_value
                )[0]
                fused_block["eer_random"] = eer_threshold(
                    dataset_genuine, dataset_random
                )[1]
            block["fusion"] = fused_block  # type: ignore[index]

    text = json.dumps(report, indent=2)
    print(text)
    if args.report is not None:
        destination = Path(args.report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    if args.fusion_report is not None:
        destination = Path(args.fusion_report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


