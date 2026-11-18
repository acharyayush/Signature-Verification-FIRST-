# Offline Signature Verification

## What this repository does (start here)

This is an **offline signature verification (OSV)** project for bank cheques. Given a
signature cropped from a scanned cheque, it answers one question: *does this signature
match the account holder's known reference signatures, or is it likely a forgery?*

It does **not** classify signatures into identity classes. Instead it *learns an
embedding space*: a shared-weight neural network maps every signature to a
512-dimensional vector, and signatures written by the same person end up close
together while signatures from different people (and forgeries) land farther apart.
Verification is then just a similarity comparison (cosine similarity) between the
cheque signature and one or more stored reference signatures.

### Key facts for a first-time reader

- **No images or trained weights are included.** This repository is code only. You
  supply your own signature dataset and describe it with a small CSV *manifest*.
- **The pipeline has four stages, and only part of it lives here.** Signature
  detection and artifact cleaning are done *upstream* (for example with YOLO) and are
  outside this repo. What is implemented here is preprocessing (stage 2.2), the
  metric-learning model and training (stage 3), and evaluation (stage 4).
- **This is a decision-support / research tool, not a production fraud system.** A
  similarity score is not a calibrated probability, and the decision thresholds must be
  calibrated on representative, signer-disjoint data before any operational use.

### The pipeline at a glance

| Stage | Responsibility | Status |
| --- | --- | --- |
| 1. Signature extraction | Detect and crop the signature from the cheque (e.g. YOLO). | Upstream, outside this repo |
| 2.1 Artifact cleaning | Remove stamps, printed lines, and noise from the crop. | Upstream, outside this repo |
| 2.2 Image preprocessing | Grayscale normalization, tight ink crop, aspect-aware resize to `384 × 192`. | Implemented in [`preprocessing/`](./preprocessing) |
| 3. Deep metric learning | Train a ConvNeXt-Tiny Siamese encoder that outputs L2-normalized 512-D embeddings. | Implemented in [`training/`](./training) |
| 4. Scoring & verification | Compare embeddings with cosine similarity; optionally fuse handcrafted geometric features; evaluate FAR / FRR / EER. | Implemented in [`training/evaluate.py`](./training/evaluate.py) and [`scoring/`](./scoring) |

### Repository layout

```text
Signature-Verification/
├── MAIN_README.md        Full pipeline guide (all stages, dataset manifest, training)
├── README.md             This file: project overview + Stage 3 details
├── requirements.txt      PyTorch, torchvision, timm, OpenCV, scikit-learn, Albumentations
├── docs/
│   ├── offline_signature_verification_architecture.md   Full system design & evaluation plan
│   └── score_fusion.md              Handcrafted features + score-level fusion design
├── scoring/
│   ├── __init__.py
│   ├── handcrafted_features.py      Ink mask, Zhang-Suen thinning, 9 geometric features
│   ├── fusion.py                    Weighted / logistic fusion, enrollment protocol, pair matrix
│   └── metrics.py                   Shared FAR / FRR / EER helpers
├── preprocessing/
│   ├── signature_preprocessing.py   Stage 2.2: grayscale normalize + aspect-aware resize
│   ├── build_manifests.py           Build the BHSig260 manifest and merge per-dataset manifests
│   ├── preprocess_dataset.py        One-time: raw dataset -> 192x384 PNGs + processed manifest
│   └── demo_outputs.py              Demo: preprocess the sample JPEGs
├── training/
│   ├── stage3_metric_learning.py    Model, dataset, PK batch sampler, triplet loss
│   ├── train_siamese.py             Training CLI (two-phase warm-up + fine-tune)
│   ├── evaluate.py                  FAR / FRR / EER on signer-disjoint splits
│   └── fit_fusion.py                Fit a fusion config on the validation split
├── tests/
│   ├── test_stage3_metric_learning.py   Focused tests for sampling and triplet mining
│   ├── test_handcrafted_features.py     Tests for ink mask, thinning, and pair features
│   └── test_score_fusion.py             Tests for fusion scoring and EER helpers
├── sample_images/         Two sample signature JPEGs used by the demo
└── preprocessed_outputs/  Example Stage 2.2 outputs (grayscale and repeated-RGB PNGs)
```

### How it works, step by step

1. **Preprocess (stage 2.2).** `preprocess_signature` converts a crop to grayscale
   (dark ink on white, keeping grayscale stroke intensity), flattens uneven
   illumination, crops tightly to the ink with a small margin, and resizes without
   stretching onto a `384 × 192` white canvas. `to_model_input` then replicates the
   single channel to 3 channels and applies ImageNet normalization, producing the
   `3 × 192 × 384` tensor the model expects. Empty crops raise `ValueError` so they can
   be routed to human review.
2. **Train (stage 3).** `python -m training.train_siamese` reads a signer-disjoint CSV
   manifest and trains the encoder in two phases: first the ConvNeXt backbone is frozen
   and only the projection head is trained with semi-hard triplet mining; then the last
   two backbone stages are unfrozen and training continues with batch-hard mining.
   Batches are signer-balanced (`P` signers, `K` genuine signatures each, plus up to `F`
   skilled forgeries of that signer). Skilled forgeries are only ever used as *negatives*.
3. **Score (stage 4).** At inference, the cheque signature and the stored references are
   embedded and compared by cosine similarity. `SiameseSignatureModel.compare` returns
   these pairwise similarities. Multiple references are aggregated (for example, averaged).
4. **Evaluate.** `python -m training.evaluate` enrolls the first N genuine signatures of
   each signer as references and reports FAR, FRR, and EER separately for genuine
   queries, random forgeries, and skilled forgeries. The threshold is chosen at the EER
   on the validation signers and applied unchanged to the test signers.


### Quick start

```bash
# 1. Create the environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2. Try the preprocessing demo on the bundled sample images
python -m preprocessing.demo_outputs        # writes preprocessed_outputs/

# 3. Run the focused unit tests
python -m unittest discover -s tests -t .

# 4. Train (requires your own signer-disjoint manifest; see the Manifest section below)
python -m training.train_siamese \
  --manifest data/manifest.csv \
  --output-dir checkpoints/stage3
```

### Important limitations

- The repository ships **no signature data and no pretrained checkpoint**; the manifest
  is required before training or evaluation will run.
- Public datasets such as **BHSig260** are prepared with
  `preprocessing/build_manifests.py` and `preprocessing/preprocess_dataset.py`; training
  and inference must use the **same** preprocessing settings or you will get train/serve skew.
- Evaluation must be **signer-disjoint** (never split by image), and thresholds must be
  chosen on validation data, never tuned on the test set.

For the full design rationale, the manifest format, the training options, and the
evaluation protocol, see [`MAIN_README.md`](./MAIN_README.md) and
[`docs/offline_signature_verification_architecture.md`](./docs/offline_signature_verification_architecture.md).


## Stage 3: ConvNeXt-Tiny metric learning

Stage 3 is implemented as a shared-weight ConvNeXt-Tiny encoder with a normalized
512-dimensional projection head, signer-balanced batches, and semi-hard then
batch-hard triplet mining. Training data is described by a CSV manifest; the
repository does not include signature images.

### Environment

Use Python 3.11 and an isolated virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv/bin/python` as the workspace interpreter. `--device
auto` selects CUDA when available and otherwise CPU. You can explicitly pass
`--device mps` on Apple Silicon, but the MPS forward pass crashed with the
installed PyTorch build during validation here; use CPU unless MPS works on
your macOS/PyTorch combination. The first pretrained run may download
ConvNeXt weights.

### Manifest

Create a CSV with the required columns:

```csv
image_path,signer_id,is_forgery,split
signatures/signer_001/genuine_01.png,signer_001,false,train
signatures/signer_001/genuine_02.png,signer_001,false,train
signatures/signer_002/forged_01.png,signer_002,true,train
signatures/validation/signer_101/genuine_01.png,signer_101,false,val
```

Relative image paths are resolved from the manifest's directory (or
`--image-root`). `signer_id` on a skilled forgery must identify the signer being
imitated. Keep each signer in exactly one split; optional `test` rows are not
used for training or validation. Include at least two distinct validation
signers with two genuine images each, and enough training signers with the
requested number of genuine images for each batch.

By default, images are read as Stage 2.1 outputs and passed through
`preprocess_signature` using the same Stage 2.2 path as inference. Stage 2.2
preserves grayscale stroke intensity so pressure and ink flow are not lost and
whitens the estimated paper background to 255. For already preprocessed 192x384
grayscale images, pass `--already-preprocessed`. The preprocessing assumes dark
signatures on a white or whitish background. The legacy `--ink` option remains
accepted for compatibility but is ignored.

### Train

```bash
python -m training.train_siamese \
  --manifest data/manifest.csv \
  --output-dir checkpoints/stage3
```

Defaults follow the plan: 5 frozen-backbone warm-up epochs with semi-hard
mining, then 20 epochs with the final two backbone stages unfrozen and
batch-hard mining. The default batch uses 12 signers, 4 genuine samples per
signer, and up to 2 skilled forgeries per signer. Adjust batch size and epoch
counts for available data and hardware. Checkpoints are selected by
signer-disjoint validation loss (always computed with batch-hard mining so
warm-up and fine-tune losses are comparable) and written to `checkpoints/stage3/best_model.pt`.

Run the focused checks with:

```bash
python -m unittest discover -s tests -t .
```

## Stage 4: Score-level fusion (optional)

The deep cosine score alone is already a strong baseline. On skilled forgeries it can
still be improved by fusing in cheap, interpretable handcrafted evidence computed from
the same preprocessed `192 × 384` grayscale image. The fusion module lives in
[`scoring/`](./scoring) and is entirely optional: without `--fusion-config` the deep
baseline numbers are unchanged.

### What is fused

Nine handcrafted features per `(reference, query)` pair, all in `[0, 1]`:

| Feature | Meaning |
| --- | --- |
| `sim_ink_aspect_similarity` | Symmetric ratio of tight ink bounding-box aspect ratios |
| `sim_stroke_width_similarity` | Symmetric ratio of mean stroke width (ink area / skeleton length) |
| `x_endpoints_ratio` | Symmetric ratio of skeleton endpoint density |
| `x_junctions_ratio` | Symmetric ratio of skeleton junction density |
| `x_slant_diff` | Normalised angular difference of dominant slant (distance, not similarity) |
| `sim_density_similarity` | Symmetric ratio of ink density inside the bounding box |
| `h_proj` | `1 - 0.5 * L1` between L1-normalised horizontal ink profiles |
| `v_proj` | `1 - 0.5 * L1` between L1-normalised vertical ink profiles |
| `sim_hu_similarity` | `1 / (1 + mean |Δ log-Hu|)` over the seven Hu moments |

Skeleton topology uses `cv2.ximgproc.thinning` when the OpenCV contrib build is present
and otherwise falls back to a pure-NumPy Zhang–Suen thinning implementation, so no extra
dependency (`opencv-contrib-python`, `scikit-image`) is required. See
[`docs/score_fusion.md`](./docs/score_fusion.md) for the full design.

### Two fusion rules

* **weighted** — `Final = w1 * S_deep + w2 * mean(H_hand)` with `w1 + w2 = 1`. The deep
  weight is grid-searched on the validation split; ties are broken toward `w1 = 0.5` so
  the deep score is not silently discarded.
* **logistic** — a logistic regression over `[S_deep, *H_hand]`, stored as plain JSON
  coefficients (no pickled estimator). Near-duplicate features (`|r| >= 0.95`) are
  dropped deterministically before fitting.

Fusion is computed **per reference pair and then averaged**, matching the protocol the
deep baseline uses.

### Fit a fusion config

```bash
python -m training.fit_fusion \
  --checkpoint checkpoints/stage3/best_model.pt \
  --manifest data/processed/manifest.csv \
  --already-preprocessed \
  --output checkpoints/fusion/fusion.json
```

The fitter runs on the **validation split only**. `--cv-folds` (default 5) measures the
fused score out-of-fold with signer-grouped folds, so the stored threshold is not the
optimistic in-sample one. `--fusion-type both` fits both rules and keeps the one with
the lower out-of-fold EER. The test split is never touched during fitting.

### Evaluate with fusion

```bash
python -m training.evaluate \
  --checkpoint checkpoints/stage3/best_model.pt \
  --manifest data/processed/manifest.csv \
  --already-preprocessed \
  --fusion-config checkpoints/fusion/fusion.json
```

The report prints the deep baseline block unchanged and a `fusion` block next to it, so
one run shows both results side by side. Fused decisions use the threshold stored in the
config when present; otherwise they fall back to the validation EER threshold.
