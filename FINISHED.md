# FINISHED — Stage 4 (handcrafted features + score-level fusion)

Status as of this commit: **complete and verified**. The optional Stage 4 fusion path is
implemented, tested (99 tests green), documented, and has been exercised end-to-end
against a real BHSig260/CEDAR manifest with a real (random-init) checkpoint.

This file records what was added, what the pipeline looks like now, how to run and test
it, the exact Kaggle commands, and what should happen next.

---

## 1. What was added in this change set

### New modules

| File | Purpose |
| --- | --- |
| `scoring/__init__.py` | Public surface of the scoring package |
| `scoring/handcrafted_features.py` | Ink mask, Zhang–Suen thinning (with `cv2.ximgproc` fast path), and the 9 geometric features |
| `scoring/fusion.py` | Fusion config (JSON round-trip), `WeightedScorer`, `LogisticScorer`, `FusionScorer`, enrollment protocol, pair matrix, EER helpers |
| `scoring/metrics.py` | Shared FAR / FRR / EER helpers used by both the fusion fitter and the evaluator |
| `training/fit_fusion.py` | Fits a fusion config on the validation split with signer-grouped cross-validation |

### New tests

| File | Tests |
| --- | --- |
| `tests/test_handcrafted_features.py` | 15 |
| `tests/test_score_fusion.py` | 70 |

### New docs

| File | Purpose |
| --- | --- |
| `docs/score_fusion.md` | Design doc for the features and the two fusion rules |
| `README.md` (Stage 4 section) | User-facing summary + fit/evaluate commands |

### Bug fixed

`training/fit_fusion.py` built its metadata dict but never persisted
`already_preprocessed`, so `training/evaluate.py`'s train/serve-skew guard read
`config.metadata.get("already_preprocessed", True)` and silently defaulted to `True`.
The fitter now records the full provenance:

```python
config = replace(config, metadata={
    **config.metadata,
    "already_preprocessed": bool(args.already_preprocessed),
    "references": args.references,
    "fusion_type": best_type,
    "cv_folds": args.cv_folds,
    "out_of_fold_eer": out_of_fold_eer,
    "signers": len(enrollment.references),
    "fit_pairs": len(matrix),
    "checkpoint": str(args.checkpoint.resolve()),
    "manifest": str(args.manifest.resolve()),
})
```

`evaluate.py:126` now warns when `--already-preprocessed` disagrees with the config
instead of silently scoring different pixels than the fitter saw.

### Doc accuracy fixes

Four claims in `docs/score_fusion.md` did not match the code and were corrected:

1. **Slant** — was described as second-order moments; the code uses a doubled-angle
   circular mean over Sobel gradients (`_dominant_slant`).
2. **`binarize_ink`** — the fixed `INK_THRESHOLD = 200` cut applies only to already-binary
   images; other grayscale input falls back to Otsu. A blank crop raises `ValueError`.
3. **Hu moments** — signed log space `sign(h) * log10(|h| + 1e-12)`, not plain log.
4. **Skeletonisation** — prefers `cv2.ximgproc.thinning`, falls back to pure-NumPy
   Zhang–Suen. The same wording was corrected in `README.md`.

---

## 2. Current pipeline

| Stage | Responsibility | Where |
| --- | --- | --- |
| 1. Signature extraction | Detect/crop the signature from the cheque (YOLO etc.) | Upstream, **not in this repo** |
| 2.1 Artifact cleaning | Remove stamps, printed lines, noise | Upstream, **not in this repo** |
| 2.2 Image preprocessing | Grayscale normalize, tight ink crop, aspect-aware resize to `384 × 192` | `preprocessing/signature_preprocessing.py` |
| 3. Deep metric learning | ConvNeXt-Tiny Siamese encoder → L2-normalized 512-D embedding | `training/stage3_metric_learning.py`, `training/train_siamese.py` |
| 4. Scoring | Cosine similarity over references; optional 9-feature fusion; FAR/FRR/EER | `training/evaluate.py`, `scoring/` |

### Stage 4 in detail

```
preprocessed 192x384 grayscale  ──┬─► ConvNeXt-Tiny encoder ──► 512-D embedding ──► cosine sim ──┐
                                  │                                                             ├─► fusion ──► score ──► threshold ──► accept/reject
                                  └─► describe_image() ──► 9 features ──► pair features ────────┘
```

* **Features** (all in `[0, 1]`; eight similarities, `x_slant_diff` is a distance):
  `sim_ink_aspect_similarity`, `sim_stroke_width_similarity`, `x_endpoints_ratio`,
  `x_junctions_ratio`, `x_slant_diff`, `sim_density_similarity`, `h_proj`, `v_proj`,
  `sim_hu_similarity`.
* **Rules**: `weighted` (`w1*S_deep + w2*mean(H_hand)`, `w1+w2=1`) and `logistic`
  (plain-JSON coefficients, no pickled estimator). Near-duplicates (`|r| >= 0.95`) are
  dropped deterministically before fitting.
* **Protocol**: fusion is computed **per reference pair and then averaged**, matching the
  deep baseline's protocol, so the comparison is apples-to-apples.
* **Threshold**: chosen by signer-grouped cross-validation on the **validation split only**
  and stored in the config. The test split is never touched during fitting.
* **Optionality**: without `--fusion-config`, `evaluate.py` takes the deep-only path and
  the baseline numbers are unchanged (descriptors are not even computed).

---

## 3. Verification performed

### Unit tests — 99/99 green

```bash
python -m unittest discover -s tests -t .
# Ran 99 tests in 4.060s
# OK
```

| Module | Tests |
| --- | --- |
| `tests/test_handcrafted_features.py` | 15 |
| `tests/test_score_fusion.py` | 70 |
| `tests/test_stage3_metric_learning.py` | 14 |

All changed modules also compile cleanly:

```bash
python -m py_compile scoring/handcrafted_features.py scoring/fusion.py \
  training/fit_fusion.py training/evaluate.py
```

### End-to-end smoke run (real data, random-init checkpoint)

To prove the plumbing (not the model quality), a 6-signer subset of the real
`data/processed/manifest.csv` was used with a randomly initialised ConvNeXt-Tiny
checkpoint:

```bash
python -m training.fit_fusion \
  --checkpoint /tmp/fusion_e2e/random_model.pt \
  --manifest /tmp/fusion_e2e/subset_manifest.csv \
  --image-root data/processed \
  --already-preprocessed --references 5 --device cpu --batch-size 32 \
  --output /tmp/fusion_e2e/fusion.json --cv-folds 3 --max-random-queries 40
```

Observed fitter output:

```
Read 324 validation records
Embedded 324 records
200/324 signatures described
Described 324 records
Built 334 queries
Built 1670 fitting pairs across 6 signers (570 genuine)
Fitted weighted fusion on all pairs
weighted: out-of-fold EER 0.3775 at threshold 0.9841
Fitted logistic fusion on all pairs
logistic: out-of-fold EER 0.2991 at threshold 0.6288
Selected logistic fusion (lowest out-of-fold EER)
Saved logistic fusion config to /tmp/fusion_e2e/fusion.json
```

Then:

```bash
python -m training.evaluate \
  --checkpoint /tmp/fusion_e2e/random_model.pt \
  --manifest /tmp/fusion_e2e/subset_manifest.csv \
  --image-root data/processed \
  --already-preprocessed --references 5 --device cpu --batch-size 32 \
  --fusion-config /tmp/fusion_e2e/fusion.json \
  --fusion-report /tmp/fusion_e2e/report.json --max-random-queries 40
```

Report shape (numbers are meaningless with a random checkpoint — only the structure
matters):

```json
{
  "threshold_from_validation": 0.9983,
  "validation_eer_skilled": 0.4018,
  "test": { "eer_skilled": 0.5, "eer_random": 0.5526, "genuine_queries": 38,
            "skilled_forgeries": 60, "random_pairs": 38 },
  "fusion": {
    "type": "logistic",
    "threshold_from_validation": 0.5549,
    "threshold_used": 0.6288,
    "threshold_source": "fusion_config_out_of_fold",
    "validation_eer_skilled": 0.1297,
    "test": { "far_skilled_at_threshold": 0.0, "far_random_at_threshold": 0.0,
              "frr_at_threshold": 0.6316, "eer_skilled": 0.2947,
              "eer_random": 0.3421 }
  }
}
```

What this proves: the fitter selects a rule from out-of-fold EER, the saved threshold is
preferred over the validation-optimistic one (`threshold_source:
fusion_config_out_of_fold`), and the fused block is emitted alongside the untouched deep
baseline. It does **not** prove recognition quality — that requires a trained checkpoint.

---

## 4. How to test (local)

```bash
# 1. Environment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2. Full suite (fast, CPU only, ~4 s)
python -m unittest discover -s tests -t .

# 3. Per-module
python -m unittest tests.test_handcrafted_features -v
python -m unittest tests.test_score_fusion -v
python -m unittest tests.test_stage3_metric_learning -v

# 4. Preprocessing demo (writes preprocessed_outputs/)
python -m preprocessing.demo_outputs

# 5. Sanity-check the handcrafted features on real images
python -c "
import glob, time
from scoring.handcrafted_features import describe_image
import cv2
for p in sorted(glob.glob('data/processed/BHSig260/*/Genuine/*.png'))[:5]:
    t = time.perf_counter()
    d = describe_image(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    print(f'{time.perf_counter()-t:.3f}s  ink={d.ink_area} skel={d.skeleton_length} aspect={d.aspect:.2f}')
"
```

Expected: `Ran 99 tests ... OK`, and ~5 ms per image for the feature pass.

### Quick smoke run without a GPU

```bash
python -m training.train_siamese \
  --manifest data/processed/manifest.csv \
  --already-preprocessed \
  --output-dir checkpoints/stage3-smoke \
  --device cpu --no-pretrained \
  --phase1-epochs 1 --phase2-epochs 0
```

This validates the training CLI and checkpoint writing; it does not produce a usable model.


---

## 5. Kaggle: commands and setup

Kaggle gives you a free GPU (T4 x2 or P100), which is what this pipeline needs. The
`data/` and `checkpoints/` folders are gitignored, so on Kaggle you clone the code and
build the data yourself.

### 5.1 One-time setup in the notebook

```python
# Cell 1 — clone the repo and install only what Kaggle is missing.
# Kaggle images already ship torch/torchvision; reinstalling them wastes ~2 GB and time.
!git clone https://github.com/acharyayush/Signature-Verification-FIRST-.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q --no-input timm opencv-python scikit-learn albumentations

# Cell 2 — verify the checkout and the GPU
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print(sys.version)
```

### 5.2 Add the dataset

In the notebook UI use **Add Data** and attach a BHSig260 dataset (e.g.
`bhsig260-bengali-hindi`). Then stage it into the layout `build_manifests.py` expects:

```bash
# Cell 3 — data/raw/BHSig260/<dataset_dir>/*.tif  with Genuine/ and Forged/ subfolders
mkdir -p data/raw/BHSig260 data/manifests
cp -r /kaggle/input/*/BHSig260-Bengali data/raw/BHSig260/ 2>/dev/null || true
cp -r /kaggle/input/*/BHSig260-Hindi   data/raw/BHSig260/ 2>/dev/null || true
ls data/raw/BHSig260
find data/raw/BHSig260 -name '*.tif' | head -3
find data/raw/BHSig260 -name '*.tif' | wc -l
```

The filenames must look like `B-S-1-G-01.tif` (or `H-S-1-F-01.tif`) and every forgery
must sit in a folder literally named `Forged`/`forged` — `build_manifests.py` raises
`ValueError` if the folder and the filename disagree, which is what keeps labels honest.

### 5.3 Build the manifest and preprocess

```bash
# Cell 4 — per-dataset manifests + merged data/manifest.csv, then 192x384 PNGs
python -m preprocessing.build_manifests
python -m preprocessing.preprocess_dataset \
  --manifest data/manifest.csv \
  --output-dir data/processed

# sanity check
head -3 data/processed/manifest.csv
python -c "
import csv, collections
rows = list(csv.DictReader(open('data/processed/manifest.csv')))
print('rows', len(rows))
print('splits', collections.Counter(r['split'] for r in rows))
print('signers', len({r['signer_id'] for r in rows}))
"
```

### 5.4 Train the Stage 3 encoder

```bash
# Cell 5 — ~25 epochs on a T4. Reduce epochs for a first pass if you are time-limited.
python -m training.train_siamese \
  --manifest data/processed/manifest.csv \
  --already-preprocessed \
  --output-dir checkpoints/stage3 \
  --device cuda \
  --phase1-epochs 5 \
  --phase2-epochs 20 \
  --signers-per-batch 12 \
  --genuine-per-signer 4 \
  --forgeries-per-signer 2 \
  --workers 2 \
  --seed 42
```

Best weights land in `checkpoints/stage3/best_model.pt`. If you hit CUDA OOM, lower
`--signers-per-batch` to 8 (that is 8 signers × 6 images = 48 images per batch).

### 5.5 Fit the fusion config on validation

```bash
# Cell 6 — fits both rules, keeps the lower out-of-fold EER. Validation split only.
python -m training.fit_fusion \
  --checkpoint checkpoints/stage3/best_model.pt \
  --manifest data/processed/manifest.csv \
  --already-preprocessed \
  --references 5 \
  --device cuda \
  --batch-size 32 \
  --fusion-type both \
  --cv-folds 5 \
  --output checkpoints/fusion/fusion.json
```

### 5.6 Evaluate deep vs fused on the test split

```bash
# Cell 7 — one run prints the deep baseline block and the fusion block side by side.
python -m training.evaluate \
  --checkpoint checkpoints/stage3/best_model.pt \
  --manifest data/processed/manifest.csv \
  --already-preprocessed \
  --references 5 \
  --device cuda \
  --batch-size 32 \
  --fusion-config checkpoints/fusion/fusion.json \
  --fusion-report checkpoints/fusion/report.json
```

### 5.7 Persist the artifacts

`/kaggle/working` is what gets saved as the notebook output. `checkpoints/` is gitignored,
so copy what you want to keep into the working directory root or into a tarball:

```bash
# Cell 8
tar -czf /kaggle/working/artifacts.tar.gz \
  checkpoints/stage3/best_model.pt \
  checkpoints/fusion/fusion.json \
  checkpoints/fusion/report.json
ls -lh /kaggle/working/artifacts.tar.gz
```

### 5.8 Run the tests on Kaggle too

```bash
# Cell 9 — confirms the repo is intact in the Kaggle image
python -m unittest discover -s tests -t .
```

### 5.9 Kaggle gotchas

* **Do not `pip install -r requirements.txt`** on Kaggle — it will try to reinstall
  torch/torchvision and can break the preinstalled CUDA build. Install only the extras.
* **`data/` is gitignored.** After a session restart you must re-stage the dataset and
  re-run Cells 3–4 (or commit the generated `data/processed/manifest.csv` into a dataset
  of your own and attach it).
* **`build_manifests.py` needs `data/manifests/` to exist** — hence the `mkdir -p` in
  Cell 3. It also runs at import time (it is a script, not an argparse CLI).
* **Session limits.** 12 h on GPU notebooks. Preprocessing ~13k images is roughly
  10–20 min; 25 training epochs is the long pole. Save checkpoints as you go.
* **Determinism.** `--seed 42` is threaded through; keep it fixed across runs you intend
  to compare.



---

## 6. Next steps

In priority order:

1. **Train a real checkpoint on a GPU.** Everything so far used a random-init model, so
   there are no meaningful quality numbers yet. Run Cell 5 on Kaggle with the full
   manifest and keep `checkpoints/stage3/best_model.pt`.
2. **Fit fusion on the real checkpoint** (Cell 6) and record the out-of-fold EER for both
   rules. If `logistic` and `weighted` come out close, prefer `weighted` — it is easier to
   explain to a reviewer and has fewer free parameters.
3. **Report deep vs fused on the test split** (Cell 7) and keep the JSON. The decision to
   ship fusion should be based on the **skilled-forgery** EER/FAR, not on the average.
4. **Tune the operating threshold for the business cost.** EER is a summary statistic; a
   cheque workflow usually cares about a fixed FAR (e.g. 0.1%) with the lowest achievable
   FRR. Re-derive the threshold from the validation ROC at that FAR and store it in the
   config.
5. **Per-dataset breakdown.** BHSig260 (Bengali/Hindi) and CEDAR differ in pen, paper and
   scan quality. Report FAR/FRR/EER per dataset so a regression in one does not hide
   behind the other.
6. **Calibrate to a probability if a downstream system needs one.** Cosine similarity is
   not a calibrated probability; either fit a Platt/isotonic mapping on validation or keep
   the threshold-only decision and document it.
7. **Add a regression test for the fusion metadata guard.** A test that asserts
   `fit_fusion` persists `already_preprocessed` and that `evaluate` warns on a mismatch
   would have caught the bug fixed here.
8. **Wire up the upstream stages.** Stages 1 and 2.1 (detection + artifact cleaning) are
   still outside this repo; the fusion path assumes a clean crop.

### Known limitations to keep in the write-up

* The fusion features are hand-designed and were only smoke-tested; their real value on
  skilled forgeries is unproven until a trained checkpoint is evaluated.
* `sim_stroke_width_similarity` is ink area over skeleton length — a proxy that is
  sensitive to binarisation and to scan resolution.
* Slant is estimated globally per image; a signature with two conflicting stroke
  directions will average them into one number.
* Thresholds are signer-disjoint-validated but not session- or scanner-disjoint; a change
  of scanner can shift scores.
* The evaluator's "random forgery" pairs are synthesised from other signers, so they
  measure impostor separation, not real-world random forgeries.

---

## 7. Reproducing this exact verification

```bash
git clone https://github.com/acharyayush/Signature-Verification-FIRST-.git
cd Signature-Verification-FIRST-
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t .     # expect: Ran 99 tests ... OK
```

With a trained checkpoint present:

```bash
python -m training.fit_fusion \
  --checkpoint checkpoints/stage3/best_model.pt \
  --manifest data/processed/manifest.csv \
  --already-preprocessed \
  --output checkpoints/fusion/fusion.json

python -m training.evaluate \
  --checkpoint checkpoints/stage3/best_model.pt \
  --manifest data/processed/manifest.csv \
  --already-preprocessed \
  --fusion-config checkpoints/fusion/fusion.json \
  --fusion-report checkpoints/fusion/report.json
```


---

## 6. What is next

Ordered by value, not by effort.

### 6.1 Train a real checkpoint (blocking everything else)

Everything downstream needs a trained model. The smoke run used a random-init network, so
no quality number in this repo is meaningful yet. Run Cell 5 on Kaggle with the default
25 epochs and record `best_model.pt` + the validation loss curve.

### 6.2 Report the real deep-vs-fused comparison

Once the checkpoint exists, run Cells 6–7 and put the four numbers in the README table:
`eer_skilled` and `eer_random` for the deep baseline and for the fused scorer. The fusion
path only earns its place if `eer_skilled` drops without `eer_random` collapsing.

### 6.3 Calibrate the operating threshold

The fused threshold is currently the EER point from signer-grouped CV. For a cheque
workflow the operating point should be chosen from a cost model: a false accept on a
cheque is far more expensive than a false reject routed to a human. Add a small
`--target-far` option to `training/evaluate.py` that reports FRR at a fixed FAR, so the
threshold can be set to a business constraint rather than the EER point.

### 6.4 Per-signer threshold analysis

EER hides a lot. Some signers are much easier than others. Add a per-signer FAR/FRR
breakdown to the report so a few hard signers cannot be averaged away. This is also the
cheapest way to find preprocessing failures (a signer with FAR ≈ 1.0 is usually a data
problem, not a model problem).

### 6.5 Close the loop on the upstream stages

Stages 1 and 2.1 (detection, artifact cleaning) are outside this repo. Until they are
wired in, the pipeline is only valid on already-cropped, already-cleaned signatures. The
next integration step is a thin adapter that takes a YOLO crop and hands it to
`preprocess_signature`, so the same code path runs in production and in evaluation.

### 6.6 Optional: score calibration

Cosine similarity is not a probability. If a downstream system needs a probability, fit a
Platt/isotonic calibrator on validation scores and store it next to the fusion config.
This is deliberately out of scope for now — the fusion threshold is already fitted on
validation, and adding a calibrator without a cost model would just add a second
unvalidated mapping.

### 6.7 Optional: more features

The nine features are cheap and interpretable. If 6.2 shows the fused model is limited by
the handcrafted branch, the next candidates are curvature/loop statistics and
stroke-width variance along the skeleton. Do not add features before 6.2 — without a
measured gap there is no way to know whether the feature helps.

---

## 7. Definition of done for this change set

- [x] `scoring/` module with 9 handcrafted features and two fusion rules
- [x] `training/fit_fusion.py` fitting on validation with signer-grouped CV
- [x] `training/evaluate.py` emitting deep + fused blocks in one run
- [x] Fusion config serialized as plain JSON (no pickled estimators)
- [x] Train/serve-skew guard wired to real metadata (the bug fixed above)
- [x] 99 unit tests green, all changed modules compile
- [x] Design doc + README section, both matching the code
- [x] End-to-end run on real BHSig260/CEDAR manifest data

Not done, and explicitly out of scope here: trained checkpoint, real quality numbers,
threshold calibration, upstream stages 1–2.1.

---

## 8. Limitations to keep in mind

* The repository ships **no data and no weights**. `data/` and `checkpoints/` are
  gitignored; you must supply a manifest.
* A similarity score is **not** a calibrated probability. Do not use these numbers as
  fraud probabilities without calibration on representative data.
* Evaluation must stay **signer-disjoint**. Splitting by image leaks identities and
  inflates every metric.
* Thresholds must be chosen on validation only. The fitter enforces this; do not
  hand-tune on the test split afterwards.
* The handcrafted branch assumes **dark ink on a white/whitish background**. A dark-mode
  scan or a heavy stamp will produce a bad ink mask and a misleading feature vector.
