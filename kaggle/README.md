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
preserved, `192 x 384` canvas.

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

- **Augmentation set.** Exactly four transforms: rotation (tilt, up to 20 deg), pen stroke
  width, ink contrast and blur. Rotation and ink contrast fire on every sample; stroke
  width (p=0.5) and blur (p=0.15) are occasional.
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

## 4. Genuine-random review (`genuine_random_review.ipynb`)

`genuine_random_review.ipynb` isolates the **genuine-random** comparisons of an Ultimate Test
Signs run — a genuine signature against a genuine signature of a *different* signer — and
shows the pairs that were **accepted** (false positives) with their images side by side. It
does not load a model or touch the GPU; it only reads the per-pair CSV and the images.

| Cell | What it does |
| --- | --- |
| 1 | clone the repo |
| 2 | set `PAIRS_CSV`, `IMAGE_ROOT`, `MANIFEST`, `DECISION`, `SHOW` |
| 3 | run `scripts.analyze_genuine_random` (filter, evaluate, write CSV + HTML) |
| 4 | summary metrics as a table |
| 5 | per-signer breakdown |
| 6 | per-dataset breakdown |
| 7 | render the HTML report inline |
| 8 | table of accepted pairs only |

**Inputs you attach**

| Input | What |
| --- | --- |
| per-pair CSV | `ultimate_pairs_grayscale_fusion_thr0.80.csv` (or any CSV from `scripts.export_ultimate_pairs`) |
| Ultimate Test Signs | the `Ultimate Test Signs/` folder with `manifest_ultimate.csv` and the `Signer_*` image folders |

The script writes two files into `OUTPUT_DIR`:

- `genuine_random_pairs.csv` — the 150 genuine-random rows plus a `metric,value` summary block,
  with `signer`, `dataset`, and `accepted` columns added.
- `genuine_random_report.html` — a self-contained page with the reviewed pairs' images embedded
  as base64. It works offline and renders inline in the notebook.

`DECISION` selects what "accepted" means: `deep` (the cosine threshold, default), `fused`
(the fusion threshold), `either` (either rule), or `both` (both rules). `SHOW` selects which
pairs the HTML lists: `accepted` (default) or `all`.

Outside the notebook the same review is one command:

```bash
python -m scripts.analyze_genuine_random \
  --csv ultimate_pairs_grayscale_fusion_thr0.80.csv \
  --image-root "Ultimate Test Signs" \
  --manifest "Ultimate Test Signs/manifest_ultimate.csv" \
  --output-dir genuine_random_review \
  --decision deep --show accepted
```

On the current `ultimate_pairs_grayscale_fusion_thr0.80.csv` run that reports 10 accepted
genuine-random pairs at the deep threshold (`FAR_random = 0.0667`) and 7 at the fused
threshold (`FAR_random_fused = 0.0467`), over 50 signers and 150 genuine-random pairs.

## 5. Random false-accept review (`random_false_accepts_review.ipynb`)

`genuine_random_review.ipynb` works from a per-pair CSV. This notebook instead **replays the
enrolment protocol** on the test split, so it needs the checkpoint, the preprocessed dataset,
and optionally the raw dataset and the fusion config. It logs every **random** (impostor)
query whose score reaches the threshold — the FAR_random false accepts — and writes:

| Output | What |
| --- | --- |
| `random_false_accepts.csv` | one row per false accept: `query_id`, `query_signer_id`, `query_dataset`, `ref_signer_id`, `score`, `per_ref_scores`, plus image metadata (width, height, aspect ratio, ink density, ink bbox) for the query and each reference |
| `random_false_accepts_summary.json` | FAR, per-dataset counts, per-reference pattern counts, and the hub analysis |
| `contact_sheets/*.png` | the top-scoring false accepts as side-by-side sheets: the query, the five references, and each reference's score |

The CSV's `pattern` column classifies each false accept as `single_reference` (one reference
lifted the mean over the threshold), `all_refs_similar` (every reference looked convincing),
or `mixed`. The hub analysis reports `top_1_share` / `top_5_share` for query signers and
reference signers, so a few dominant "hub" signers are easy to spot.

`DECISION` selects the score space: `deep` (the cosine threshold, default) or `fused` (the
fusion config's calibrated `threshold`). `SCORING_DEPTH` sets which signers count as
impostors: `1` matches `training.evaluate` (same top-level dataset), `2` keeps impostors
within one script. `DATASET_DEPTH` controls the report labels (`2` separates `bhsig_hi` from
`bhsig_bn`).

Outside the notebook the same review is one command:

```bash
python -m scripts.analyze_random_false_accepts \
  --checkpoint best_model.pt \
  --manifest data/grayscale_processed/manifest.csv \
  --image-root data/grayscale_processed \
  --raw-root data/raw \
  --fusion-config fusion.json \
  --output-dir random_false_accepts \
  --decision fused --top 20
```

On the local test split (64 signers, 35,206 random pairs) this reports **1,629** false
accepts at the deep threshold (`FAR_random = 0.0463`) and **249** at the fused threshold
(`FAR_random = 0.0071`). The deep false accepts split into 1,251 `mixed`, 375
`all_refs_similar`, and 3 `single_reference`; the fused ones split into 142
`all_refs_similar` and 107 `mixed`. At the fused operating point the top query signer
accounts for 18.9% of the false accepts, so the failures are spread across many signers
rather than concentrated in a few hubs.
