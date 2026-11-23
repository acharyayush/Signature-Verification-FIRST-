# Kaggle: training and evaluation

`train_and_evaluate.ipynb` is a ready-to-run Kaggle notebook that trains the Stage 3 encoder
and evaluates it. Import it with **File → Import Notebook** (or upload it as a dataset and
open it), attach your preprocessed dataset, and run the cells top to bottom.

Everything the notebook runs is a plain CLI command, so the same steps work outside a
notebook. The cells are:

| Cell | What it does |
| --- | --- |
| 1 | clone the repo, `pip install timm opencv-python scikit-learn` |
| 2 | check the GPU |
| 3 | set `PROCESSED` to the attached dataset and list it |
| 4 | verify row/split/signer counts and that every image path resolves |
| 5 | **train with `--augment --augment-strength standard --balance-datasets`** |
| 6 | optional no-augmentation baseline (ablation) |
| 7 | fit the fusion config on validation only |
| 8 | evaluate deep vs fused on the test split |
| 9 | per-dataset breakdown (`--group-depth 1`) |
| 10 | optional: filter the manifest to a subset of datasets |
| 11 | tarball the artifacts and re-run the unit tests |

## 1. Build the dataset locally (once)

Stage 2.2 is CPU work and only needs to happen once. Do it on your machine, not on Kaggle.
The output must use the **current** preprocessing: pure-white background, true grayscale ink
preserved, `256 x 512` canvas.

```bash
# raw -> data/manifests/*.csv (per dataset) ; then merge the ones you want
python -m preprocessing.build_manifests

# CEDAR + BHSig260 + SigResize only
python -m preprocessing.preprocess_dataset \
  --manifest data/manifest_three.csv \
  --image-root data \
  --output-dir data/grayscale_processed \
  --grayscale
```

`--grayscale` keeps the signature's true grayscale. Without it the output is binarised to
`0/255`, which throws away the pen-pressure information the model uses.

Sanity check:

```bash
python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("data/grayscale_processed/manifest.csv")))
print("rows   ", len(rows))
print("splits ", collections.Counter(r["split"] for r in rows))
print("datasets", collections.Counter(r["image_path"].split("/")[0] for r in rows))
PY
```

## 2. Package and upload it (once)

The processed manifest stores paths **relative to itself**, so the upload must be a *flat*
archive: `manifest.csv` at the root and the image folders beside it.

```bash
python - <<'PY'
import zipfile
from pathlib import Path

src = Path("data/grayscale_processed").resolve()
out = Path("data/grayscale-upload.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(src / "manifest.csv", "manifest.csv")
    for path in sorted(src.rglob("*.png")):
        zf.write(path, path.relative_to(src).as_posix())
print("wrote", out, round(out.stat().st_size / 2**20, 1), "MiB")
PY
```

Upload `data/grayscale-upload.zip` as a **private** Kaggle dataset (e.g.
`signature-verification-processed`). Kaggle extracts archives on upload, so it mounts at
`/kaggle/input/signature-verification-processed/` with `manifest.csv` and the dataset folders
at the top level. No copy step is needed: `--image-root` defaults to the manifest's directory,
so `--manifest /kaggle/input/.../manifest.csv` resolves every image.

## 3. Notes

- **Augmentation set.** Tilt (up to 20 deg), shear, occasional perspective, elastic, pen
  stroke width, ink contrast, blur and noise. Gamma is off because it duplicates ink
  contrast; most transforms fire with a probability, so a sample sees ~5 of them.
- **Augmentation is training-only.** Validation is never augmented, because the checkpoint is
  selected by the validation EER. Each augmented image is re-whitened and re-fitted through the
  same crop + resize as inference.
- **Per-dataset augmentation** is resolved from the path layout: BHSig260 (Indic script) gets a
  milder elastic warp than CEDAR, because deforming connected strokes destroys the structure
  that discriminates writers.
- **`--balance-datasets`** gives every source dataset an equal share of each batch, so BHSig260
  (260 signers) does not crowd out CEDAR (55).
- **Ablation.** Run Cell 5 and Cell 6, then compare `eer_skilled` in Cell 8 and the per-dataset
  numbers in Cell 9. The two runs differ only by `--augment`.
- **CUDA OOM?** Drop `--signers-per-batch` to 8.
- `data/` and `checkpoints/` are gitignored, so neither the PNGs nor the trained weights are
  ever committed.
