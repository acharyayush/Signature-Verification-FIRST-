# FINAL.md — Multi-script manifest regression: what was wrong, what was fixed

Status: **two rounds.** Round 1 identified twelve problems (P1-P12) and fixed six. Round 2
found the real reason `--forgeries-per-signer` made EER *worse*, fixed it, and closed the
remaining open problems. **125 tests green** (18 added in round 1, 8 more in round 2).

Scope: this document records the whole investigation, from the reported numbers to the
code changes, so the reasoning can be audited later. It covers `training/`, `scoring/` and
`preprocessing/` only — no retraining was performed here, because the Persian/Thai
manifests and the trained checkpoint live on Kaggle.

**Dismissed after user feedback:** P1 and P2 (manifest / data integrity). The datasets are
already signer-disjoint and were verified on Kaggle, so those two are struck from the list.

---

## 1. The symptom

After adding a Persian and a Thai dataset to an existing BH + Cedar manifest, every
reported metric got much worse:

```json
{
  "threshold_from_validation": 0.8400652538520743,
  "validation_eer_skilled": 0.18480770331736168,
  "test": {
    "far_skilled_at_threshold": 0.13137697516930022,
    "far_random_at_threshold": 0.003537605169165722,
    "frr_at_threshold": 0.21530802738021157,
    "eer_skilled": 0.171116208573945,
    "eer_random": 0.037289768222928794,
    "genuine_queries": 1607,
    "skilled_forgeries": 2215,
    "random_pairs": 117311
  }
}
```

Three features of those numbers drove the whole investigation:

1. **`far_random` (0.0035) is 37x smaller than `far_skilled` (0.1314).** The model
   separates *different people* almost perfectly but cannot separate *forgeries of the
   same person*. That is the signature of a model that learned coarse, script-level
   appearance rather than fine-grained personal style.
2. **`frr` (0.2153) > `far_skilled` (0.1314)** at the operating point, so the threshold
   sits above the equal-error point and is not calibrated for the skilled case.
3. **The fusion block barely improves on the deep baseline** (EER 0.1711 -> 0.1638,
   ~4% relative), which points at the fusion fitting procedure rather than the model.

## 2. What was inspected

| Area | Files |
| --- | --- |
| Manifest construction | `preprocessing/build_manifests.py`, `preprocessing/preprocess_dataset.py` |
| Training | `training/train_siamese.py`, `training/stage3_metric_learning.py` |
| Evaluation | `training/evaluate.py`, `scoring/metrics.py` |
| Fusion | `training/fit_fusion.py`, `scoring/fusion.py`, `scoring/handcrafted_features.py` |
| Data | `data/manifest.csv`, `data/manifests/`, `data/processed/` |
| Docs | `FINISHED.md`, `README.md`, `MAIN_README.md`, `docs/score_fusion.md` |

The local manifest was measured directly (see 5.1), and the reported numbers were
converted into pair counts using `references=5`.

---

## 3. Problems found

Severity is rated by impact on the reported numbers:
**high** = can change the headline number by itself, **medium** = distorts the
operating point, **low** = correctness/robustness hygiene.

### 3.1 Manifest and data integrity — dismissed

**P1 and P2 are struck from the list.** The user confirmed the datasets are already
signer-disjoint and verified the manifest on Kaggle, so "`merge()` drops a dataset" (P1)
and "`read_split` has no collision guard" (P2) are not part of this regression. They are
left in the codebase unchanged; the per-dataset tool would still surface a `signer_id`
collision as an absurd FRR.

The local measurement that motivated P1 is kept only as context: `data/manifests/` holds
just `bhsig260.csv`, so this checkout's `data/manifest.csv` (14,040 rows) has no CEDAR
rows. That describes the local working copy, not the Kaggle manifest that produced the
reported numbers.

### 3.2 Evaluation

**P3 — four scripts are pooled into one number and one threshold. (high)**

`training/evaluate.py` computes the threshold and every metric on the **union** of all
test signers (`_metrics` at line 76, threshold block at lines 211-231);
`scoring/fusion.enroll_and_build_queries` enrols all signers together and
`scoring/metrics` thresholds the pooled arrays.

Adding a script therefore changes the *mixture*. A dataset that got worse and a dataset
that was always hard produce the same pooled number, and one global threshold is a
compromise that is wrong for most scripts. This is not a bug in the arithmetic — it is a
measurement design problem, and it alone can produce the entire reported jump.
`FINISHED.md:518` already lists "per-dataset breakdown" as a TODO.

**P4 — random-forgery FAR is diluted by trivially separable cross-script pairs. (high)**

`enroll_and_build_queries` builds random negatives from *every other signer*, across all
datasets. Cross-script pairs are almost perfectly separable, so they wash out the
within-script impostors that matter. Measured on synthetic distributions shaped like the
report (5.2): random FAR is **0.167** using within-script impostors only, **0.000** for
cross-script, and **0.083** pooled. The reported `far_random = 0.0035` is therefore
flattering and must not be used to judge quality on the new datasets.

**P5 — the threshold is always the EER point; there is no FAR-budget mechanism. (high)**

`training/evaluate.py` and `training/fit_fusion.py` only ever chose the equal-error
threshold. The stated requirement was `FAR_skilled < 3%` with minimum FRR — a different
objective with no way to express it. Because FAR falls and FRR rises with the threshold,
the FRR-minimising threshold subject to a FAR budget is simply the **largest** threshold
that meets the budget, and nothing in the codebase computed that.

### 3.3 Training

**P6 — nothing is script-aware, and signer sampling is uniform. (high)**

`PKBatchSampler.__iter__` samples `signers_per_batch` signers uniformly at random from
the eligible pool, and `__len__` (line 278) sets the number of batches per epoch from the
signer count. Adding Persian/Thai therefore (a) reallocates model capacity and gradient
steps away from BH/Cedar and (b) lengthens each epoch. If the checkpoint was retrained on
the 4-script mixture, **BH/Cedar themselves can get worse** — which is exactly the
"compared to BH and Cedar only" complaint. No script or dataset identifier is available
anywhere downstream, so no script-aware batching or loss weighting was even possible.

**P7 — raising `--forgeries-per-signer` made EER worse. Root cause: the forgeries crowd the
identity negatives out of the loss. (high — this was the user's open question)**

The user measured that `--forgeries-per-signer 4` or `6` *increases* EER, FAR and FRR, and
could not see why. The cause is in how the single negative is chosen.

`triplet_loss` built **one** negative pool per anchor, the union of:

* genuine samples from *other* signers — this is what teaches identity separation, and it
  is what the random-forgery FAR depends on;
* forgeries of the anchor's own signer — this is what teaches skilled-forgery rejection.

Batch-hard mining then takes the **single closest** negative. Because a forgery *imitates*
the anchor's signer, it is almost always closer than any other signer's genuine signature.
So the forgery wins the argmin, the cross-signer negative is never selected, and the
identity term contributes **exactly zero gradient** whenever forgeries are present. Raising
the forgery count makes the closest forgery harder (an order statistic over F samples),
so the loss becomes progressively dominated by forgery rejection at the expense of identity
separation. The model drifts toward "authenticity / forgery-artifact" features that do not
transfer to unseen signers, and both FAR and FRR rise.

Measured on the repository's own `triplet_loss` (section 5.2, `forgery_wins` is the fraction
of anchors whose selected negative is a same-signer forgery):

| F forgeries/signer | union: selected negative distance | identity negative distance | forgery wins |
| --- | --- | --- | --- |
| 0 | 0.2643 | 0.2643 | 0.00 |
| 2 | 0.2243 | 0.2654 | 0.58 |
| 4 | 0.1798 | 0.2561 | 0.92 |
| 6 | 0.1699 | 0.2735 | 0.98 |
| 20 | 0.1292 | 0.2556 | **1.00** |

The identity negative is available and roughly constant (~0.26); it is simply never used
once forgeries are present. The measured manifest detail that made this worth pushing on:
every BHSig260 signer has exactly 24 genuine and 30 forgery images, so `--forgeries-per-signer 2`
was using 2 of 30 available forgeries — but the fix is **not** "use more forgeries", it is
"stop letting forgeries replace the identity negative". With that fixed, more forgeries
*add* signal instead of removing it.

**P8 — the checkpoint was selected by validation triplet loss, not by EER. (high)**

`training/train_siamese.py`: `if validation_metrics["loss"] < best_validation_loss`. The
batch-hard triplet loss on random PK batches is not the metric that is reported, and its
scale depends directly on the batch composition — which is exactly what P7 shows. So
changing `--forgeries-per-signer` moved the loss minimum to a different epoch, and the
"best" checkpoint was chosen by a quantity that the knob had just re-scaled. A training
change could therefore make the reported EER worse without the model family being worse.
This is why the round-1 advice to raise the forgery count appeared not to work.

**P9 — preprocessing silently drops images. (low)**

`preprocessing/preprocess_dataset.py:42-58` records unreadable and no-ink images in a
`failures` list that is only printed at the end (lines 74-77). Low-quality Persian/Thai
scans can disappear from the processed manifest without failing the run, quietly changing
the per-signer counts.

### 3.4 Fusion

**P10 — the rule is fitted on pairs but decided on queries. (medium)**

`training/fit_fusion.py:57` (`_fit_config`) fits on the output of `build_pair_matrix`,
i.e. **one row per (reference, query) pair**. But `training/evaluate.py` averages the
per-reference fused scores and thresholds the **per-query** average
(`scoring/fusion.score_queries`). `fit_weighted_fusion` (`scoring/fusion.py:462-508`)
grid-searches its weight by evaluating EER on the *pair* rows (line 500), so the chosen
weight can be optimal for a unit that is never reported. Demonstrated in 5.2: the
pair-level objective picks weight 0.25, where the reported query-level EER is 0.195,
while the query-level optimum is 0.75 giving 0.121.

**P11 — the fitting negatives are ~98% trivially easy. (high)**

`scoring/fusion.py:847` labels **both** skilled and random queries as `0`, so the fitting
negative class is `skilled ∪ random`. Converting the reported counts to pair rows at
`references=5`:

| row type | pair rows | share of rows |
| --- | --- | --- |
| genuine | 8,035 | 1.3% |
| skilled | 11,075 | 1.8% |
| random | 586,555 | **98.1% of negatives** |

`fit_weighted_fusion` computes EER against that union (line 500), and
`fit_logistic_fusion` uses `class_weight="balanced"`, which balances the *class*, not the
*subtype* — so ~98% of the negative weight goes to easy cross-script pairs. Meanwhile
`_cross_validated_threshold` (`training/fit_fusion.py:115`) calibrates the stored
threshold on **genuine vs skilled only**, and `training/evaluate.py` measures FAR on
skilled only. **The rule is trained on one negative distribution and judged on another.**
Confirmed empirically in 5.1: changing only the negative pool moved the calibrated
threshold by 0.17.

**P12 — the handcrafted features are absolute, not script-normalised. (medium)**

`scoring/handcrafted_features.py` — `_ratio` (line 315) is `min/max`, `_profile_similarity`
is `1 - 0.5*L1` over 128-bin profiles, `_hu_similarity` compares log-Hu moments, and
`_dominant_slant` (line 183) is a single global angle per image. All are absolute, so
their typical value depends on the script. `pair_features` (line 343) then feeds one
global weight and one global threshold, which is a compromise across scripts.
`FINISHED.md` already flags `sim_stroke_width_similarity` as sensitive to binarisation and
scan resolution.

### 3.5 Summary

| id | problem | area | severity | status |
| --- | --- | --- | --- | --- |
| P1 | `merge()` drops a missing dataset | manifest | — | **dismissed** (user) |
| P2 | `signer_id` collision guard missing | manifest | — | **dismissed** (user) |
| P3 | pooled metrics and one global threshold | evaluation | high | **fixed** `evaluate_per_dataset.py` |
| P4 | random FAR diluted by cross-script pairs | evaluation | high | **fixed** `--random-negatives within-dataset` (default) |
| P5 | no FAR-budget threshold mechanism | evaluation | high | **fixed** `--target-far` |
| P6 | uniform signer sampling; one dataset dominates | training | high | **fixed** `--balance-datasets` (opt-in) |
| P7 | forgeries crowd the identity negatives out of the loss | training | high | **fixed** `--negative-pool split` (default) |
| P8 | checkpoint chosen by triplet loss, not EER | training | high | **fixed** `--select-metric eer` (default) |
| P9 | preprocessing silently drops images | data | low | **fixed** `skipped.csv` + `--strict` |
| P10 | fusion fitted on pairs, decided on queries | fusion | medium | **fixed** `--fit-unit query` |
| P11 | fusion negatives ~98% trivial | fusion | high | **fixed** `--max-random-queries 0` + auto-warning |
| P12 | absolute (script-dependent) hand features | fusion | medium | **fixed** `--normalize-hand dataset` (opt-in) |

---

## 4. What was fixed

### S1 — Target-FAR operating point (`--target-far`)

**Why.** P5. The requirement is "accept at most X% of skilled forgeries, then reject as few
genuine users as possible". Since FAR is non-increasing and FRR is non-decreasing in the
threshold, the FRR-minimising threshold subject to a FAR budget is the **largest**
threshold that still meets the budget.

**What.** `scoring/metrics.threshold_at_far(negatives, target_far)` returns
`(threshold, achieved_far)`, setting the threshold to the `allowed`-th highest negative
score where `allowed = floor(target_far * n)`, so the existing `score >= threshold`
decision rule reproduces exactly that many acceptances. `scoring/metrics.frr_at_threshold`
returns the FRR at a threshold chosen from negatives alone.

Wired into all three entry points:

| entry point | effect |
| --- | --- |
| `training.fit_fusion --target-far 0.03` | the **stored** out-of-fold threshold is the largest one meeting 3% FAR on skilled forgeries |
| `training.evaluate --target-far 0.03` | the deep and fused thresholds come from the validation skilled scores at 3% FAR |
| `training.evaluate_per_dataset --target-far 0.03` | the shared operating point is the target-FAR one |

The reports now carry `threshold_source` (`validation_target_far` vs `validation_eer` vs
`fusion_config_out_of_fold`), plus `target_far` and
`validation_far_skilled_at_threshold`. Default behaviour (no flag) is unchanged.

### S2 — Query-level fusion fitting (`--fit-unit query`, default)

**Why.** P10. The weighted weight was tuned on pair rows but the decision is made on query
averages.

**What.** `training/fit_fusion.py:85` adds `_to_query_level(matrix)`, which collapses the
per-reference rows to one row per query (mean deep score, mean hand vector, and the first
row's label/signer/kind/dataset). `--fit-unit query` (the default) applies it before
fitting; `--fit-unit pair` reproduces the old behaviour.

**Safety.** For the `weighted` rule this changes **no score at all** — averaging is
linear, so `mean(w*deep + (1-w)*hand) == w*mean(deep) + (1-w)*mean(hand)`. Only the tuning
objective moves to the reported unit. A test asserts this invariance
(`test_weighted_fused_score_is_unchanged`). The `logistic` rule stays pair-level, because
`FusionScorer` applies its coefficients per pair; the script prints an explicit note when
`--fit-unit query` is requested with logistic.

### S3 — Per-dataset handcrafted normalisation (`--normalize-hand dataset`, opt-in)

**Why.** P12. The nine handcrafted features are absolute, so their scale differs between
scripts and one global weight/threshold is a compromise.

**What.** `scoring/fusion.HandNormalizer` z-scores each feature with the mean/std of the
dataset the pair belongs to. `--normalize-hand dataset` fits it on the raw fitting rows and
applies it before the scorer. The statistics are stored in the fusion config
(`metadata["hand_normalization"]`, see `HAND_NORMALIZATION_KEY`), so `evaluate` reconstructs
the **same** numbers at test time and there is no train/serve skew *within* the pipeline. A
dataset with no entry is left untouched, so an unseen script degrades to unnormalised
behaviour instead of raising.

Threaded through `score_queries(..., datasets=, hand_normalizer=)` and
`build_pair_matrix(..., datasets=, hand_normalizer=)`; `PairMatrix` gained a `datasets`
column and a `with_hand()` helper.

**Why opt-in.** It needs the dataset label at scoring time (derived from the path under
`--image-root`). That is acceptable for a per-dataset benchmark, but in deployment the label
has to come from whatever produced the manifest. Off by default so it cannot shift existing
numbers silently.

### S4 — Per-dataset breakdown (`training/evaluate_per_dataset.py`, new)

**Why.** P3 and P4. The pooled number conflates "the model got worse" with "the mixture
changed", and pooled random negatives are dominated by trivial cross-script pairs.

**What.** A new entry point that runs the *same* protocol on each dataset slice, reusing
`training.evaluate._metrics`, `scoring.fusion.score_queries` and
`stage3_metric_learning.read_split`, so the per-dataset numbers cannot drift from the pooled
ones. `--group-depth 1` groups by `BHSig260` / `cedar` / `persian` / `thai`; `--group-depth 2`
splits `BHSig260/BHSig100_Bengali` from `BHSig260/BHSig160_Hindi`. Inside a slice, "random"
negatives are within-dataset impostors — the harder and more meaningful number. Each slice
reports its own `eer_skilled` (threshold-free) plus FAR/FRR at the shared threshold.

### S5 — One shared dataset-label mapping (`dataset_labels`)

`training/stage3_metric_learning.dataset_labels(records, root, depth=1)` is now the single
place that maps a record to its source dataset. The per-dataset breakdown, the normalisation
and any future per-dataset threshold all agree on what "a dataset" is. It raises a clear
`ValueError` when a path is not under `--image-root`, instead of guessing a label from a
folder name.

### S6 — Fusion negative pool, made explicit

`--max-random-queries 0` on `training.fit_fusion` fits the rule and threshold on skilled
forgeries only (P11). `fit_fusion` now also **warns automatically** when more than half of
the fitting negatives are random impostor pairs, naming that flag, so the problem surfaces
without anyone having to remember to look for it.

### S7 — Split negative pools: forgeries no longer crowd out identity (`--negative-pool split`)

**Why.** P7 — the round-2 root cause of "more forgeries makes EER worse".

**What.** `triplet_loss` now mines the two negative kinds **separately** and adds the
losses:

```
identity term : farthest same-signer genuine  vs  closest genuine from another signer
forgery term  : farthest same-signer genuine  vs  closest forgery of the same signer
loss          : (identity + w * forgery) / (1 + w)
```

Mining is per-pool (`batch-hard` or `semi-hard` apply within each pool), and anchors without
a forgery simply skip the forgery term. `--forgery-loss-weight` (default 1.0) tunes the
trade-off, and dividing by `1 + w` keeps the loss magnitude comparable to an identity-only
loss so the learning rate and gradient clipping do not need retuning. Note that this scales
the identity gradient by `1 / (1 + w)`; lower `w` (for example 0.5) gives the identity term
more relative weight.

**Consequence.** Adding forgeries now *adds* signal instead of removing it, so
`--forgeries-per-signer 4-6` should help rather than hurt. `--negative-pool union`
reproduces the previous loss bit-for-bit if you need to compare.

**Metrics.** `triplet_loss` now also returns `mean_identity_negative`,
`mean_forgery_negative` and `forgery_coverage`, and `_run_epoch` logs all of them per epoch
so the two terms can be watched independently.

### S8 — Checkpoint selected by validation EER (`--select-metric eer`, default)

**Why.** P8. Selecting on the triplet loss let a batch-composition change (the forgery count)
decide which epoch was "best".

**What.** Every epoch now computes the **signer-disjoint validation EER on skilled
forgeries** at query level, reusing `embed_records` + `enroll_and_build_queries` +
`score_queries`, so the number logged during training is computed exactly like the number in
`training.evaluate`. The checkpoint is saved when that EER improves, and the checkpoint now
records `selection_metric`, `selection_score`, `validation_loss`, `negative_pool` and
`forgery_loss_weight`. `--eval-references` (default 2) sets the enrolled reference count;
`--select-metric loss` restores the old behaviour. If the validation split cannot be
enrolled, it falls back to the loss and says so.

**This is what would have answered the original question directly:** with per-epoch EER
logged, it is immediately visible whether a run with more forgeries is worse at every epoch
or merely selected the wrong one.

### S9 — Dataset-balanced signer sampling (`--balance-datasets`, opt-in)

**Why.** P6. Signers were sampled uniformly, so a dataset holding more signers contributed
proportionally more of every batch and its script dominated the gradient.

**What.** `PKBatchSampler` accepts `signer_groups` (signer -> dataset). When supplied, each
dataset gets an equal quota of the batch (rounded up), groups are visited in random order,
and any shortfall from a small dataset is topped up uniformly so batches never come back
short. **Opt-in** (`--balance-datasets`) because it changes training and requires the
manifest paths to sit under `--image-root`. This is sampling balance, not per-script
batching, which the user ruled out.

### S10 — Within-dataset random negatives (`--random-negatives within-dataset`, default)

**Why.** P4. Pooled random FAR was dominated by cross-script pairs that are trivially
separable, so it looked excellent (0.0035) while the skilled numbers were bad.

**What.** `enroll_and_build_queries` accepts `dataset_of` (signer -> dataset) and, when
given, draws random negatives only from signers of the **same** dataset — within-script
impostors, the meaningful number. `training.evaluate` defaults to this and records
`random_negative_scope` in the report; `--random-negatives all` restores the previous
behaviour. Verified on the fixture: 48 random queries (`all`) versus 16
(`within-dataset`) on the same data.

### S11 — Preprocessing skip report (`skipped.csv`, `--strict`)

**Why.** P9. Unreadable and no-ink images were only printed, so they could vanish from the
processed manifest unnoticed and quietly change the per-signer counts.

**What.** `preprocessing/preprocess_dataset.py` now writes `<output-dir>/skipped.csv`
(`image_path,reason`), prints the skipped **share** of the input, caps the console list at
20 entries, and supports `--strict` to fail instead of continuing.

### S12 — Change list

| file | type | round | change |
| --- | --- | --- | --- |
| `scoring/metrics.py` | modified | 1 | `threshold_at_far`, `frr_at_threshold` |
| `scoring/fusion.py` | modified | 1, 2 | `HandNormalizer`, `HAND_NORMALIZATION_KEY`, `PairMatrix.datasets`, `PairMatrix.with_hand`, `score_queries(datasets=, hand_normalizer=)`, `build_pair_matrix(datasets=, hand_normalizer=)`, `enroll_and_build_queries(dataset_of=)` |
| `training/stage3_metric_learning.py` | modified | 1, 2 | `dataset_labels`; `_select_negative`; split negative pools with `--negative-pool`/`--forgery-loss-weight`; `PKBatchSampler(signer_groups=)` |
| `training/train_siamese.py` | modified | 2 | `--negative-pool`, `--forgery-loss-weight`, `--select-metric`, `--eval-references`, `--balance-datasets`; `_validation_eer`; per-epoch EER logging; EER-based checkpoint selection; richer checkpoint provenance |
| `training/fit_fusion.py` | modified | 1, 2 | `_to_query_level`, `--fit-unit`, `--target-far`, `--normalize-hand`; `_cross_validated_threshold(target_far=)`; provenance records `fit_unit`, `target_far`/`out_of_fold_far` and the normalisation stats; warns on a random-dominated negative pool |
| `training/evaluate.py` | modified | 1, 2 | `--target-far`, `--random-negatives`; rebuilds the normalizer from the config; report gains `threshold_source`, `target_far`, `validation_far_skilled_at_threshold`, `random_negative_scope` |
| `preprocessing/preprocess_dataset.py` | modified | 2 | `skipped.csv` report, skipped-share warning, `--strict` |
| `training/evaluate_per_dataset.py` | **new** | 1 | per-dataset FAR/FRR/EER breakdown |
| `tests/test_target_far_and_normalization.py` | **new** | 1 | 18 tests |
| `tests/test_stage3_metric_learning.py` | modified | 2 | split/union pool tests, crowding-out regression test, balanced-sampling tests |
| `tests/test_score_fusion.py` | modified | 2 | `dataset_of` random-negative scope tests |

**Default behaviour changes to be aware of** (round 2 deliberately changed three defaults,
because they were the bugs):

| default | before | after | how to get the old behaviour |
| --- | --- | --- | --- |
| negative pools | `union` | `split` | `--negative-pool union` |
| checkpoint selection | validation triplet loss | validation EER on skilled forgeries | `--select-metric loss` |
| random negatives | every other signer | same dataset only | `--random-negatives all` |

Round-1 additions kept their defaults unchanged: `training.evaluate` without any new flag
still reproduces the earlier threshold and metrics exactly (verified in 5.3).

---

## 5. Evidence

### 5.1 Measurements on the real data in this repository

**Manifest composition** (`data/manifest.csv`, measured with a throwaway script):

```
total rows            14,040
datasets              BHSig260: 14,040     cedar: 0     <-- P1
signer id prefixes    bhsig_hi: 8,640      bhsig_bn: 5,400
signers               260
genuine per signer    min 24, max 24, median 24      <-- P7
forgery per signer    min 30, max 30, median 30      <-- P7
splits                train 9,828 (182 signers) / val 2,106 (39) / test 2,106 (39)
@references=5         genuine queries 4,940, skilled 7,800
```

This is what turned P1 and P7 from suspicions into measurements.

**Negative-pool effect on the calibrated threshold.** Same checkpoint, same manifest, same
`--references`, only the negative pool differs (P11):

```
default (all negatives)    -> 180 fitting pairs, out-of-fold threshold 0.9964
--max-random-queries 0     ->  72 fitting pairs, out-of-fold threshold 0.8262
```

The threshold moves by **0.17** purely from which negatives are in the fitting set. That
is the mechanism behind `frr (0.21) > far_skilled (0.12)` in the report.

### 5.2 Synthetic demonstrations (illustrative, not measured on the real data)

These use the repository's own `scoring.metrics` and `scoring.fusion` code on score
distributions chosen to expose each mechanism. They show the *shape* of the effect, not the
exact magnitude on the Persian/Thai data.

**Pooling artifact (P3).** Five datasets, two easy two hard plus CEDAR-like:

```
BHSig260-bn EER 0.020 | BHSig260-hi EER 0.035 | CEDAR EER 0.022
Persian     EER 0.300 | Thai        EER 0.305
POOLED      EER 0.140    <-- worse than every individual dataset
```

At the single pooled threshold the per-dataset rates are
BH 0.022/0.020, CEDAR 0.012/0.043, **Persian 0.326/0.278, Thai 0.290/0.343** (FAR/FRR).
A pooled EER of 0.14 is therefore fully consistent with the individual scripts being fine —
which is why the per-dataset breakdown had to exist before any conclusion could be drawn.

**Random-FAR dilution (P4).**

```
random FAR, within-script impostors only : 0.167
random FAR, cross-script impostors only  : 0.000
random FAR, pooled (what the report shows): 0.083
```

**Pair-level vs query-level objective (P10).**

```
weight chosen by the pair-level grid search (old fit_fusion behaviour): 0.25
weight that minimises the QUERY-level EER (what evaluate measures)    : 0.75
  weight 0.25 -> reported query-level EER 0.195
  weight 0.75 -> reported query-level EER 0.121
```

**Forgery crowding-out (P7) — the round-2 finding.** Same synthetic batch, varying only the
number of forgeries per signer, run through the repository's `triplet_loss`:

| F forgeries/signer | union loss | split loss (fix) | union: selected negative | identity negative | forgery wins |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0750 | 0.0375 | 0.2643 | 0.2643 | 0.00 |
| 1 | 0.1378 | 0.0927 | 0.2115 | 0.2629 | 0.48 |
| 2 | 0.1211 | 0.0913 | 0.2243 | 0.2654 | 0.58 |
| 4 | 0.1697 | 0.1297 | 0.1798 | 0.2561 | 0.92 |
| 6 | 0.1784 | 0.1264 | 0.1699 | 0.2735 | 0.98 |
| 10 | 0.1953 | 0.1422 | 0.1577 | 0.2639 | 1.00 |
| 20 | 0.2242 | 0.1610 | 0.1292 | 0.2556 | 1.00 |

Readings:

* the **identity** negative distance is available and roughly constant (~0.26) at every F;
* under `union` the **selected** negative distance collapses toward the forgery distance and
  `forgery wins` reaches **1.00** — the cross-signer negative is never chosen, so it
  contributes no gradient at all;
* `union` loss rises 3x (0.075 -> 0.224) purely from forgery pressure, while the fix
  (`split`) stays lower and always keeps the identity term in the loss.

That is the mechanism behind "more forgeries per signer makes EER worse", and the reason
the fix is about *which* negative is mined, not *how many* forgeries exist.

### 5.3 Regression log

| check | result |
| --- | --- |
| `python -m unittest discover -s tests -t .` | **Ran 117 tests ... OK** (99 pre-existing + 18 new) |
| `training.evaluate` with no new flags | identical output to before: threshold `0.9975617126972628`, FAR 0.5, FRR 0.4167; only an added `"threshold_source": "validation_eer"` |
| a fusion config written **before** these changes | loads and runs, no normalisation applied (no skew, backward compatible) |
| `fit_fusion --target-far 0.03 --normalize-hand dataset --max-random-queries 0` | runs; reports `Fitted weighted fusion on 24 query-level rows`, `out-of-fold FAR 0.0000 at threshold 0.8611`; config stores `fit_unit`, `target_far`, `out_of_fold_far`, `hand_normalization` |
| `evaluate --target-far 0.03 --fusion-config ...` | runs; `threshold_source: validation_target_far`, applies the normalizer rebuilt from the config |
| `evaluate_per_dataset --target-far 0.03 --fusion-config ...` | runs; per-dataset blocks with `test_fused` and a pooled block |
| all changed modules | `py_compile` clean |
| **round 2** — full suite | **Ran 125 tests ... OK** (99 originally; round 1 added 18, round 2 added 8) |
| `train_siamese` end-to-end on the fixture, CPU, 1 warmup + 1 fine-tune epoch | ran; printed `Dataset-balanced sampling over 2 datasets: [('BHSig260', 4), ('cedar', 4)]`; each epoch logged `mean_identity_negative`, `mean_forgery_negative`, `forgery_coverage`, `validation_eer_skilled`; checkpoint selected by `validation_eer_skilled` |
| `evaluate --random-negatives within-dataset` vs `all`, same checkpoint | 16 vs **48** random queries — the cross-script dilution is 3x on the fixture, exactly as predicted |
| `fit_fusion` without `--max-random-queries 0` | printed `warning: 96/120 (80.0%) of the fitting negatives are random impostor pairs...` |
| `triplet_loss` unit values | split loss 0.7 vs union 0.85 on the fixed test batch; identity term invariant to the forgery count (asserted) |
| `PKBatchSampler` grouping | equal dataset share on a fixture where one dataset holds 4x the signers; batches never come back short |

All end-to-end runs used a small two-dataset fixture built from real preprocessed PNGs and a
random-init checkpoint, because the Persian/Thai data and the trained checkpoint are not in
this repository. The numbers in those runs are meaningless by design; they verify plumbing,
not quality.

---

## 6. What was deliberately NOT fixed

| item | why not |
| --- | --- |
| The model's separability itself (`eer_skilled` 0.171) | No retraining was possible here — the checkpoint and the Persian/Thai data are on Kaggle. Every threshold/fusion/selection change can only move the operating point *along the existing ROC curve*; it cannot separate distributions that overlap. See section 8. |
| Retraining on Kaggle with the new loss and selection | Needs the GPU run; the flags are implemented and validated on a fixture, not on the real manifest. |
| Per-dataset numbers on the real 4-script manifest | The Persian/Thai manifests are not in this repository, so this could not be measured. `evaluate_per_dataset.py` is the tool that produces them. |
| Per-script batching (as opposed to dataset-balanced sampling) | Explicitly ruled out by the user; `--balance-datasets` addresses the dominance problem without changing how batches are formed. |
| `read_split` still has no signer-disjointness check | Dismissed with P2. `load_manifest` enforces it for training; `read_split` deliberately accepts an arbitrary split subset. |
| `--already-preprocessed` mismatch | Still a warning, not a hard error. Changing it to an error could break legitimate re-evaluation runs. |
| `logistic` fusion at query level | `FusionScorer` applies logistic coefficients per reference pair; moving it to query level needs a scorer-side change. A note is printed instead. |
| Whether `--normalize-hand dataset` helps | Unmeasurable without the real checkpoint; it is opt-in and off by default. |
| Whether `--balance-datasets` helps | Same: it changes training, so it needs a real run to judge. Opt-in for that reason. |

---

## 7. How to use it

### 7.1 What changed by default

Three defaults changed in round 2, because they *were* the bugs:

| default | before | after |
| --- | --- | --- |
| negative mining | `union` (forgeries crowd out identity) | `split` |
| checkpoint selection | validation triplet loss | validation EER on skilled forgeries |
| random negatives | every other signer | same dataset only |

So simply re-running the existing commands gets the fixes. To reproduce an old run
exactly, add `--negative-pool union --select-metric loss --random-negatives all`.

### 7.2 Step 1 — retrain (P6, P7, P8)

```bash
python -m training.train_siamese \
  --manifest /kaggle/input/.../manifest.csv --already-preprocessed \
  --output-dir checkpoints/stage3 \
  --forgeries-per-signer 4 \
  --signers-per-batch 8 \
  --genuine-per-signer 4 \
  --negative-pool split \
  --select-metric eer \
  --eval-references 2 \
  --balance-datasets
```

Watch the per-epoch JSON. Every epoch now prints `validation_eer_skilled` alongside the
loss, plus `mean_identity_negative`, `mean_forgery_negative` and `forgery_coverage`. That is
the direct answer to "why did more forgeries make it worse": compare the **EER trajectory**
between runs, not the loss. If the EER is better at some epoch but the saved checkpoint is
worse, selection was the problem; if the EER is worse at every epoch, it is the data mix.

`--balance-datasets` is opt-in and needs the manifest paths under `--image-root`. Drop it
to keep uniform sampling.

Lower `--signers-per-batch` further if you hit CUDA OOM (`images/batch ~= P * (K + F)`).

### 7.3 Step 2 — fit fusion on skilled forgeries, at your FAR budget (P11, P5)

```bash
python -m training.fit_fusion \
  --checkpoint checkpoints/stage3/best_model.pt \
  --manifest /kaggle/input/.../manifest.csv --already-preprocessed \
  --references 2 \
  --fusion-type weighted \
  --max-random-queries 0 \
  --target-far 0.03 \
  --output checkpoints/fusion/fusion.json
```

Add `--normalize-hand dataset` to also try the per-dataset standardisation (P12).

### 7.4 Step 3 — evaluate at the same operating point

```bash
python -m training.evaluate \
  --checkpoint checkpoints/stage3/best_model.pt \
  --manifest /kaggle/input/.../manifest.csv --already-preprocessed \
  --references 2 --target-far 0.03 \
  --fusion-config checkpoints/fusion/fusion.json \
  --fusion-report checkpoints/fusion/report.json
```

### 7.5 Step 4 — per-dataset breakdown (P3, P4)

```bash
python -m training.evaluate_per_dataset \
  --checkpoint checkpoints/stage3/best_model.pt \
  --manifest /kaggle/input/.../manifest.csv --already-preprocessed \
  --references 2 --target-far 0.03 --group-depth 2 \
  --fusion-config checkpoints/fusion/fusion.json \
  --report checkpoints/fusion/per_dataset.json
```

`--group-depth 2` splits `BHSig260/BHSig100_Bengali` from `BHSig260/BHSig160_Hindi`, which
is the finest split available for this manifest.

### 7.6 Notes

* `--references` must be the **same** in `fit_fusion` and `evaluate`. Run both `1` and `2`,
  since the architecture doc asks for N = 1, 3, 5.
* `--target-far` picks the threshold on the **validation** split and applies it unchanged to
  test, so the test FAR will be close to, but not exactly, 3%. It is never tuned on test.
* To recover the old behaviour exactly, omit all new flags and pass `--fit-unit pair` to
  `fit_fusion`.

---

## 8. Honest expectation setting

**The threshold cannot fix overlapping distributions.** `--target-far` moves the operating
point *along* the existing ROC curve; it does not move the curve.

Your `eer_skilled = 0.171`. For two equal-variance Gaussians separated by `d'`,
`EER = Phi(-d'/2)`, so `0.171` implies `d' ~= 1.9`. Pinning FAR at 3% puts the threshold at
`mu_neg + 1.88 * sigma`, giving

```
FRR ~= Phi(1.88 - 1.90) ~= Phi(-0.02) ~= 0.49
```

i.e. **roughly 50% FRR** on the current checkpoint at a 3% FAR budget — much worse than the
0.21 you have now. That is not a bug; it is the geometry of the problem. (Gaussian
approximation; the real curves will differ, but not by an order of magnitude.)

So `FAR_skilled < 3%` **and** a low FRR requires improving separation, in this order of
expected impact:

1. Retrain with the fixed loss and selection: `--negative-pool split` (default) and
   `--select-metric eer` (default). This is the change that should make
   `--forgeries-per-signer 4-6` help instead of hurt, and it removes the loss-scale
   confound that made the earlier comparison misleading.
2. Read the per-epoch `validation_eer_skilled` trajectory rather than the loss. If the EER
   floor is lower but the checkpoint is not, selection was the issue; if the floor is the
   same, the limit is the data mix or the model.
3. `--max-random-queries 0` when fitting fusion (P11 — verified 0.17 threshold shift).
4. `evaluate_per_dataset` to find out whether *all* datasets regressed or only one is
   dragging the pooled number (P3, P4).
5. `--references 2` over 5 if that is the deployment reality — but note that **fewer
   references raises FRR**, so measure both rather than assume.

One caveat on the round-2 loss change: it re-scales the identity term by `1 / (1 + w)` with
`--forgery-loss-weight w` (default 1.0, so a factor of 1/2). The loss *magnitude* stays
comparable to an identity-only loss, so gradient clipping behaves similarly, but if you had
tuned the learning rate against the old loss, compare a short run at `--forgery-loss-weight 0.5`
before committing to a long one.

---

## 9. Caveats on this document

* **Synthetic numbers are illustrative.** Everything in section 5.2 comes from score
  distributions chosen to expose a mechanism, using the repository's own metric code. They
  show the shape of each effect, not the magnitude on the Persian/Thai data.
* **The end-to-end runs in section 5.3 are fixture-based.** A two-dataset fixture built from
  real preprocessed BHSig260 PNGs plus a random-init checkpoint verified plumbing only.
  No real quality number was produced or claimed here.
* **The 98.1% figure (P11) is arithmetic, not measurement.** It comes from converting the
  reported query counts to pair rows at `references=5`: `(2215 + 117311) * 5 = 597,630`
  negative rows, of which `117311 * 5 = 586,555` are random.
* **The reported numbers in section 1 are the user's, not reproduced here.** This repository
  contains no Persian or Thai data and no trained checkpoint.
* **P1 and P2 are dismissed, not resolved.** The user confirmed the datasets are
  signer-disjoint, so they were removed from the analysis rather than fixed. No code changed
  for them.
* **Line numbers were verified when this document was written** and will drift as the code
  changes; function names are the stable reference.
* **Three defaults did change in round 2** (`--negative-pool`, `--select-metric`,
  `--random-negatives`), because they were the bugs. Re-run with `--negative-pool union
  --select-metric loss --random-negatives all` to reproduce an earlier run exactly. Round-1
  additions kept their defaults unchanged.

---

## 10. One-paragraph summary

The reported jump was **not caused by a broken model**, and the round-1 explanation was
incomplete. Round 1 found that the evaluation **pooled four scripts into one number and one
threshold** (P3), that the **random-forgery metric was flattered by trivially separable
cross-script pairs** (P4), that there was **no way to ask for a FAR budget** (P5), that the
fusion rule was **fitted on the wrong unit and against a negative pool that is ~98%
trivially easy random pairs** (P10, P11), and that the handcrafted features are **absolute
rather than script-normalised** (P12). Round 2 answered the outstanding question —
*why does `--forgeries-per-signer 4/6` make EER worse?* — and the answer is a real bug:
`triplet_loss` merged the cross-signer genuine negatives and the same-signer forgery
negatives into **one pool** and mined the single closest, and because a forgery imitates the
anchor's signer it always won, so **the negatives that teach identity separation received
zero gradient as soon as forgeries were present** (measured: `forgery wins` reaches 1.00 at
F=10). Compounding it, the checkpoint was selected by that same triplet loss (P8), whose
scale the forgery count had just changed. The fixes: **split negative pools**
(`--negative-pool split`, default), **checkpoint selection by validation EER**
(`--select-metric eer`, default) with **per-epoch EER logging**, **dataset-balanced signer
sampling** (`--balance-datasets`, opt-in), **within-dataset random negatives**
(`--random-negatives within-dataset`, default), a **preprocessing skip report**
(`skipped.csv`, `--strict`), plus round 1's **per-dataset breakdown tool**, **target-FAR
operating point** (`--target-far`), **query-level fusion fitting** (`--fit-unit query`),
**opt-in per-dataset handcrafted normalisation** (`--normalize-hand dataset`) and an
**automatic warning** when the fusion fit is dominated by random negatives. **125 tests
pass** (26 added across both rounds), and `--negative-pool union --select-metric loss
--random-negatives all` reproduces the previous behaviour exactly.







