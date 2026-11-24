"""Demo: apply every augmentation to 10 sample signatures and save a contact sheet.

Usage (from the repository root):
    python -m preprocessing.demo_augmentations

Picks 10 different signatures from ``data/raw`` (CEDAR, BHSig260 Bengali and Hindi,
SigResize), preprocesses each with the current Stage 2.2 path (pure-white background, true
grayscale ink), then applies each of the four training augmentations individually plus two
full random draws. One PNG per ``(operation, source)`` is written to
``augmentation_samples/<operation>/`` and a master contact sheet to
``augmentation_samples/contact_sheet.png``, so the effect of every transform can be inspected
by eye.

Every transform goes through the same helpers the training dataset uses, and each result is
re-whitened and re-fitted through the inference crop + resize, so what is shown here is
exactly the Stage 2.2 image training would see -- not an intermediate.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from preprocessing.augmentation import (
    _apply_ink_contrast,
    _apply_rotation,
    _apply_stroke_width,
    augment_signature,
    augmentation_config,
)
from preprocessing.signature_preprocessing import (
    DEFAULT_TARGET_SIZE,
    aspect_aware_resize,
    fit_to_canvas,
    preprocess_signature,
    whiten_background,
)

OUTPUT_DIR = Path("augmentation_samples")
RAW_ROOT = Path("data/raw")

# Contact-sheet geometry.
CELL_W, CELL_H = 256, 128
LABEL_W = 200
HEADER_H = 36
GRID = (210, 210, 210)  # light grey separators, BGR


class _FixedGenerator:
    """Deterministic stand-in for ``numpy.random.Generator``.

    Used to force a specific branch (for example ink thickening vs thinning, or a darker vs
    lighter ink factor) so the demo shows one direction at a time instead of a random one.
    """

    def __init__(self, random_value: float = 1.0, uniform_value: float = 0.0) -> None:
        self.random_value = random_value
        self.uniform_value = uniform_value

    def random(self) -> float:
        return self.random_value

    def uniform(self, low: float = 0.0, high: float = 1.0, size=None):
        value = float(np.clip(self.uniform_value, low, high))
        return np.full(size, value) if size is not None else value

    def normal(self, loc: float = 0.0, scale: float = 1.0, size=None):
        return np.zeros(size) if size is not None else float(loc)


def _refit(image: np.ndarray) -> np.ndarray:
    """Whiten the background and re-run the shared inference crop + resize + whiten."""
    gray = whiten_background(image)
    try:
        return fit_to_canvas(gray, DEFAULT_TARGET_SIZE)
    except ValueError:  # a transform removed every ink pixel
        return aspect_aware_resize(image, DEFAULT_TARGET_SIZE, pad_value=255)


# ``prefix``, folder, glob pattern, how many to take. Ten sources in total.
_SOURCE_POOLS = (
    ("cedar", RAW_ROOT / "CEDAR" / "full_org", "*.png", 3),
    ("bhsig_bn", RAW_ROOT / "BHSig260" / "BHSig100_Bengali" / "Genuine", "*.tif", 2),
    ("bhsig_hi", RAW_ROOT / "BHSig260" / "BHSig160_Hindi" / "Genuine", "*.tif", 2),
    ("sigresize", RAW_ROOT / "SigResize" / "Genuine", "*/*.tif", 3),
)


def _pick_sources(seed: int = 7) -> list[tuple[str, Path, str]]:
    """Return ``(label, path, dataset)`` for 10 different signatures across the datasets."""
    rng = random.Random(seed)
    chosen: list[tuple[str, Path, str]] = []
    for prefix, folder, pattern, count in _SOURCE_POOLS:
        files = sorted(folder.glob(pattern))
        if len(files) < count:
            raise FileNotFoundError(
                f"not enough signatures under {folder}: found {len(files)}, need {count}"
            )
        for path in rng.sample(files, count):
            chosen.append((f"{prefix}_{path.stem}", path, prefix))
    return chosen


def _operations(image: np.ndarray, index: int) -> dict[str, np.ndarray]:
    """The four training augmentations applied individually, plus two full random draws.

    Each entry is the Stage 2.2 image training would see, so the sheet shows the real effect
    rather than an intermediate transform.
    """
    return {
        "original": _refit(image),
        "rotate_-20deg": _refit(_apply_rotation(image, -20.0, fill=255)),
        "rotate_+10deg": _refit(_apply_rotation(image, 10.0, fill=255)),
        "rotate_+20deg": _refit(_apply_rotation(image, 20.0, fill=255)),
        "stroke_thicker": _refit(
            _apply_stroke_width(image, 3, _FixedGenerator(random_value=0.0))
        ),
        "stroke_thinner": _refit(
            _apply_stroke_width(image, 3, _FixedGenerator(random_value=0.9))
        ),
        "ink_darker": _refit(
            _apply_ink_contrast(image, 0.25, _FixedGenerator(uniform_value=0.25))
        ),
        "ink_lighter": _refit(
            _apply_ink_contrast(image, 0.25, _FixedGenerator(uniform_value=-0.25))
        ),
        "blur": _refit(cv2.GaussianBlur(image, (0, 0), 1.2)),
        # The two full draws go through ``augment_signature``, which re-fits internally.
        "full_random_standard": augment_signature(
            image, augmentation_config("standard"), np.random.default_rng([42, 0, index])
        ),
        "full_random_strong": augment_signature(
            image, augmentation_config("strong"), np.random.default_rng([42, 0, index])
        ),
    }


def _label(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)


def _contact_sheet(
    results: dict[str, dict[str, np.ndarray]],
    operation_names: list[str],
    source_labels: list[str],
) -> np.ndarray:
    """Rows = operations, columns = sources, with a label gutter on the left."""
    rows, cols = len(operation_names), len(source_labels)
    height = HEADER_H + rows * CELL_H
    width = LABEL_W + cols * CELL_W
    sheet = np.full((height, width, 3), 255, np.uint8)

    for column, source in enumerate(source_labels):
        _label(sheet, source[:26], LABEL_W + column * CELL_W + 6, 24)
    for row, name in enumerate(operation_names):
        _label(sheet, name, 6, HEADER_H + row * CELL_H + CELL_H // 2)

    for row, name in enumerate(operation_names):
        for column, source in enumerate(source_labels):
            small = cv2.resize(
                results[source][name], (CELL_W, CELL_H), interpolation=cv2.INTER_AREA
            )
            top, left = HEADER_H + row * CELL_H, LABEL_W + column * CELL_W
            sheet[top : top + CELL_H, left : left + CELL_W] = cv2.cvtColor(
                small, cv2.COLOR_GRAY2BGR
            )

    for row in range(rows + 1):
        y = HEADER_H + row * CELL_H
        cv2.line(sheet, (LABEL_W, y), (width, y), GRID, 1)
    for column in range(cols + 1):
        x = LABEL_W + column * CELL_W
        cv2.line(sheet, (x, HEADER_H), (x, height), GRID, 1)
    cv2.line(sheet, (LABEL_W, 0), (LABEL_W, height), GRID, 1)
    cv2.line(sheet, (0, HEADER_H), (width, HEADER_H), GRID, 1)
    return sheet


def _write_readme(
    output_dir: Path,
    sources: list[tuple[str, Path, str]],
    operation_names: list[str],
    sheet_shape: tuple[int, int, int],
) -> None:
    lines = [
        "# Augmentation samples",
        "",
        "Generated by `python -m preprocessing.demo_augmentations`.",
        "",
        f"Contact sheet: `contact_sheet.png` ({sheet_shape[1]}x{sheet_shape[0]}). Rows are",
        "operations, columns are the 10 source signatures.",
        "",
        "Every sub-directory holds one PNG per source. Each image is already re-whitened and",
        "re-fitted through the inference crop + resize, so it is exactly the Stage 2.2 image",
        "training would see.",
        "",
        "## Operations",
        "",
    ]
    lines += [f"- `{name}/`" for name in operation_names]
    lines += ["", "## Sources", ""]
    lines += [f"- `{label}` ({dataset}) <- `{path}`" for label, path, dataset in sources]
    lines += [
        "",
        "## The four training augmentations",
        "",
        "One transform per real-world cause of variation, and nothing else. Rotation and ink",
        "contrast fire on every sample; stroke width and blur fire with a probability, so a",
        "sample sees 2-4 of them. Training samples one random draw per sample per epoch.",
        "",
        "| operation | fires | cause it models |",
        "| --- | --- | --- |",
        "| `rotate_*` | always | the cheque signature is tilted by hand |",
        "| `ink_darker` / `ink_lighter` | always (random sign) | pen pressure / ink darkness |",
        "| `stroke_thicker` / `stroke_thinner` | p=0.5 | a different pen |",
        "| `blur` | p=0.15 | a soft scan or photo |",
        "| `full_random_standard` / `full_random_strong` | the real draws | all four combined |",
    ]
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def save_augmentation_samples(
    sources: list[tuple[str, Path, str]] | None = None,
    output_dir: str | Path = OUTPUT_DIR,
) -> Path:
    """Write every augmentation for every source, plus the contact sheet and README."""
    sources = list(sources) if sources is not None else _pick_sources()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, np.ndarray]] = {}
    for index, (label, path, dataset) in enumerate(sources, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read signature image: {path}")
        operations = _operations(preprocess_signature(image), index - 1)
        results[label] = operations
        for name, array in operations.items():
            folder = out / name
            folder.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(folder / f"{label}.png"), array):
                raise OSError(f"could not write {folder / label}.png")
        print(f"{index}/{len(sources)} {label}: {len(operations)} outputs")

    operation_names = list(next(iter(results.values())).keys())
    sheet = _contact_sheet(results, operation_names, list(results))
    if not cv2.imwrite(str(out / "contact_sheet.png"), sheet):
        raise OSError("could not write contact_sheet.png")
    _write_readme(out, sources, operation_names, sheet.shape)
    print(
        f"wrote contact_sheet.png ({sheet.shape[1]}x{sheet.shape[0]}) and "
        f"{len(operation_names)} operation folders to {out}"
    )
    return out


if __name__ == "__main__":
    save_augmentation_samples()
