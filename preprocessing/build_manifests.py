"""Build per-dataset manifests and merge them into ``data/manifest.csv``.

Per-dataset manifests live in ``data/manifests/<name>.csv`` with image paths relative
to ``data/``. Each has its own signer-disjoint split. This script writes one file per
supported raw dataset and then merges every file in that folder into
``data/manifest.csv``.

Supported raw layouts under ``data/raw/``:

* ``BHSig260/<lang>/Genuine/<B|H>-S-<n>-G-<k>.tif`` and ``<lang>/Forged/<..>-F-<k>.tif``
* ``CEDAR/full_org/original_<signer>_<k>.png`` and ``full_forg/forgeries_<signer>_<k>.png``
* ``SigResize/Genuine/<signer>/<k>.tif`` and ``SigResize/Forgery/<signer>/<k>.tif``
* ``Thai/Genuine/<signer>/<file>.png`` and ``Thai/Forgery/<signer>/<file>.png``

Thai ships every scan twice, once plain and once with a ``test_`` filename prefix, and
the two copies are byte-identical. The ``test_`` copies are dropped so the same scan is
not counted twice within a signer.

Usage:
    python -m preprocessing.build_manifests
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

DATA = Path("data")
FIELDS = ["image_path", "signer_id", "is_forgery", "split"]
NAME = re.compile(r"^([BH])-S-(\d+)-([GF])-\d+\.tif$", re.IGNORECASE)
LANGUAGE = {"B": "bn", "H": "hi"}
CEDAR_NAME = re.compile(r"^(?:original|forgeries)_(\d+)_\d+\.png$", re.IGNORECASE)
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
DUPLICATE_PREFIX = "test_"


def split_signers(signers: list[str], seed: int = 42) -> dict[str, str]:
    """Assign whole signers to train/val/test (about 70/15/15)."""
    shuffled = sorted(signers)
    random.Random(seed).shuffle(shuffled)
    n_val = n_test = round(len(shuffled) * 0.15)
    labels = ["test"] * n_test + ["val"] * n_val
    labels += ["train"] * (len(shuffled) - len(labels))
    return dict(zip(shuffled, labels))


def _image_files(folder: Path, *, skip_prefix: str | None = None) -> list[Path]:
    """Image files directly inside ``folder``, skipping metadata and duplicate copies."""
    files = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if skip_prefix is not None and path.name.startswith(skip_prefix):
            continue
        files.append(path)
    return files


def _manifest_rows(entries: list[tuple[Path, str, bool]]) -> list[dict[str, str]]:
    """Attach a signer-disjoint split to ``(path, signer_id, is_forgery)`` entries."""
    splits = split_signers(sorted({signer for _, signer, _ in entries}))
    return [
        {
            "image_path": path.relative_to(DATA).as_posix(),
            "signer_id": signer,
            "is_forgery": str(is_forgery).lower(),
            "split": splits[signer],
        }
        for path, signer, is_forgery in entries
    ]


def build_bhsig260() -> None:
    root = DATA / "raw" / "BHSig260"
    rows: list[dict[str, str]] = []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dataset_rows: list[tuple[Path, str, bool]] = []
        for path in sorted(dataset_dir.rglob("*.tif")):
            match = NAME.match(path.name)
            if not match:
                raise ValueError(f"unexpected file name: {path}")
            language, number, kind = match.groups()
            is_forgery = kind.upper() == "F"
            # The folder must agree with the filename, or labels would be wrong.
            if is_forgery != (path.parent.name.lower() == "forged"):
                raise ValueError(f"folder and file name disagree: {path}")
            dataset_rows.append(
                (path, f"bhsig_{LANGUAGE[language.upper()]}_{int(number):03d}", is_forgery)
            )
        # Split per language so Bengali and Hindi each get their own signer-disjoint split.
        rows += _manifest_rows(dataset_rows)
    write(DATA / "manifests" / "bhsig260.csv", rows)


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def build_cedar() -> None:
    root = DATA / "raw" / "CEDAR"
    entries: list[tuple[Path, str, bool]] = []
    for folder, is_forgery in (("full_org", False), ("full_forg", True)):
        for path in _image_files(root / folder):
            match = CEDAR_NAME.match(path.name)
            if not match:
                raise ValueError(f"unexpected CEDAR file name: {path}")
            entries.append((path, f"cedar_{int(match.group(1)):03d}", is_forgery))
    write(DATA / "manifests" / "cedar.csv", _manifest_rows(entries))


def build_sigresize() -> None:
    root = DATA / "raw" / "SigResize"
    entries: list[tuple[Path, str, bool]] = []
    for folder, is_forgery in (("Genuine", False), ("Forgery", True)):
        for signer_dir in sorted(p for p in (root / folder).iterdir() if p.is_dir()):
            if not signer_dir.name.isdigit():
                raise ValueError(f"unexpected SigResize signer folder: {signer_dir}")
            signer = f"sigresize_{int(signer_dir.name):03d}"
            for path in _image_files(signer_dir):
                entries.append((path, signer, is_forgery))
    write(DATA / "manifests" / "sigresize.csv", _manifest_rows(entries))


def build_thai() -> None:
    root = DATA / "raw" / "Thai"
    entries: list[tuple[Path, str, bool]] = []
    for folder, is_forgery in (("Genuine", False), ("Forgery", True)):
        for signer_dir in sorted(p for p in (root / folder).iterdir() if p.is_dir()):
            if not signer_dir.name.isdigit():
                raise ValueError(f"unexpected Thai signer folder: {signer_dir}")
            signer = f"thai_{signer_dir.name}"
            for path in _image_files(signer_dir, skip_prefix=DUPLICATE_PREFIX):
                entries.append((path, signer, is_forgery))
    write(DATA / "manifests" / "thai.csv", _manifest_rows(entries))


def merge() -> None:
    merged: list[dict[str, str]] = []
    for path in sorted((DATA / "manifests").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            merged += list(csv.DictReader(handle))
    write(DATA / "manifest.csv", merged)


if __name__ == "__main__":
    build_bhsig260()
    build_cedar()
    build_sigresize()
    build_thai()
    merge()
