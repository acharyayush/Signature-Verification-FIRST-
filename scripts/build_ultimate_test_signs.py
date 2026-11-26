"""Build the "Ultimate Test Signs" evaluation set from the raw datasets.

For a fixed number of *test-split* signers (from the per-dataset manifests in
``data/manifests/``) this script applies the Stage 2.2 preprocessing to the raw
signature images and writes the resulting ``384 x 192`` PNGs into a self-contained
folder tree. Strokes are grayscale by default; pass ``--binarize`` for pure black on
white (``0``/``255``).

    Ultimate Test Signs/
        Signer_001/
            Genuine/  3 genuine signatures of signer 001
            Forged/   3 skilled forgeries imitating signer 001
            Random/   3 genuine signatures of a *different* signer (negative)
        Signer_002/
        ...

Only signers whose manifest ``split`` column is ``test`` are eligible, so the set
is disjoint from the training and validation signers. The selection is
dataset-balanced and deterministic (``SEED``), and every written file is recorded
in ``Ultimate Test Signs/manifest_ultimate.csv``.

Preprocessing is :func:`preprocessing.signature_preprocessing.preprocess_signature`
-- the same function training and inference use -- so the benchmark cannot drift from
the model. The evaluation reads the stored images directly, so run it with
``python -m training.evaluate_ultimate ...`` (there is no preprocessing step at evaluation
time).

"Random" negatives are drawn from another test signer of the **same dataset**
(the repository's default ``--random-negatives within-dataset`` scope), and the
random signer is guaranteed to differ from the target signer.

Usage:
    python scripts/build_ultimate_test_signs.py
    python scripts/build_ultimate_test_signs.py --num-signers 50 --per-class 3
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2

from preprocessing.signature_preprocessing import preprocess_signature

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
DEFAULT_OUT = REPO / "Ultimate Test Signs"

# Raw datasets with a manifest and raw images available locally.
DATASETS = ["bhsig260", "cedar", "sigresize"]
SEED = 42


def load_test_pool() -> dict[str, dict]:
    """Map each test signer to its dataset and its genuine/forgery file paths."""
    pool: dict[str, dict] = {}
    for name in DATASETS:
        manifest = DATA / "manifests" / f"{name}.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != "test":
                    continue
                entry = pool.setdefault(
                    row["signer_id"], {"dataset": name, "genuine": [], "forged": []}
                )
                bucket = "forged" if row["is_forgery"] == "true" else "genuine"
                entry[bucket].append(DATA / row["image_path"])
    for entry in pool.values():
        entry["genuine"].sort()
        entry["forged"].sort()
    return pool


def select_signers(pool: dict[str, dict], num: int, rng: random.Random) -> list[str]:
    """Pick ``num`` signers, keeping every dataset represented.

    Every signer of the two smaller datasets is kept and the remainder is filled
    from BHSig260, so a 50-signer set does not silently drop CEDAR/SigResize.
    """
    by_dataset: dict[str, list[str]] = {name: [] for name in DATASETS}
    for signer, entry in pool.items():
        by_dataset[entry["dataset"]].append(signer)

    selected: list[str] = []
    for name in DATASETS:
        selected.extend(sorted(by_dataset[name]))
    if len(selected) < num:
        raise SystemExit(
            f"only {len(selected)} test signers available, cannot build {num}"
        )

    # Keep the smaller datasets whole; trim the largest (BHSig260) first.
    overflow = len(selected) - num
    for name in sorted(DATASETS, key=lambda n: len(by_dataset[n]), reverse=True):
        if overflow <= 0:
            break
        drop_pool = [s for s in selected if pool[s]["dataset"] == name]
        rng.shuffle(drop_pool)
        drop = set(drop_pool[:overflow])
        selected = [s for s in selected if s not in drop]
        overflow -= len(drop)
    return sorted(selected)


def preprocess_files(
    paths: list[Path], dest: Path, count: int, *, binarize: bool = False
) -> list[Path]:
    """Preprocess ``count`` raw signatures into ``dest`` as Stage 2.2 PNGs.

    Every chosen file is read as a color image, passed through
    :func:`preprocessing.signature_preprocessing.preprocess_signature` (the same step
    training and inference use), and written as a ``384 x 192`` PNG. With ``binarize`` the
    strokes are pure black on white (``0``/``255``); otherwise the grayscale strokes are
    kept. The returned paths are the files that were written.

    Raises:
        SystemExit: If there are too few files, an image cannot be read, or it lacks ink.
    """
    dest.mkdir(parents=True, exist_ok=True)
    chosen = paths[:count]
    if len(chosen) < count:
        raise SystemExit(f"not enough files for {dest}: need {count}, have {len(paths)}")
    written = []
    for src in chosen:
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"could not read raw signature {src}")
        try:
            processed = preprocess_signature(image, binarize=binarize)
        except ValueError as error:
            raise SystemExit(f"could not preprocess {src}: {error}") from error
        target = dest / src.with_suffix(".png").name
        if not cv2.imwrite(str(target), processed):
            raise SystemExit(f"could not write preprocessed signature {target}")
        written.append(target)
    return written


def build(num: int, per_class: int, out_dir: Path, *, binarize: bool = False) -> None:
    rng = random.Random(SEED)
    pool = load_test_pool()
    selected = select_signers(pool, num, rng)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rows: list[dict[str, str]] = []
    for index, signer in enumerate(selected, start=1):
        entry = pool[signer]
        signer_dir = out_dir / f"Signer_{index:03d}"

        genuine = preprocess_files(
            entry["genuine"], signer_dir / "Genuine", per_class, binarize=binarize
        )
        forged = preprocess_files(
            entry["forged"], signer_dir / "Forged", per_class, binarize=binarize
        )

        # Random negatives: genuine signatures of a different signer in the same dataset.
        candidates = [
            s
            for s in selected
            if s != signer and pool[s]["dataset"] == entry["dataset"]
        ]
        if not candidates:
            raise SystemExit(f"no random-negative signer for {signer}")
        random_signer = rng.choice(candidates)
        random_files = preprocess_files(
            pool[random_signer]["genuine"],
            signer_dir / "Random",
            per_class,
            binarize=binarize,
        )

        for kind, files in (
            ("genuine", genuine),
            ("forged", forged),
            ("random", random_files),
        ):
            source_signer = signer if kind != "random" else random_signer
            for target in files:
                rows.append(
                    {
                        "folder": signer_dir.name,
                        "class": kind,
                        "source_dataset": entry["dataset"],
                        "source_signer": source_signer,
                        "image_path": target.relative_to(out_dir).as_posix(),
                    }
                )

    with (out_dir / "manifest_ultimate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "folder",
                "class",
                "source_dataset",
                "source_signer",
                "image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Built {len(selected)} signers under {out_dir}")
    style = "binarized" if binarize else "grayscale"
    print(f"Wrote {len(rows)} {style} rows to {out_dir / 'manifest_ultimate.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-signers", type=int, default=50)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Write pure black-on-white (0/255) PNGs instead of grayscale strokes.",
    )
    args = parser.parse_args()
    build(args.num_signers, args.per_class, args.output, binarize=args.binarize)


if __name__ == "__main__":
    main()
