# Handcrafted features and score-level fusion

This document describes the optional Stage 4 fusion path in this repository: the nine
handcrafted features computed from each preprocessed signature, and the two fusion rules
that combine them with the ConvNeXt-Tiny deep cosine score.

The fusion path is **optional**. Running `training/evaluate.py` without
`--fusion-config` reproduces the deep baseline exactly; the fusion code is only reached
when a fitted config is supplied.

## Why fuse at all

The deep Siamese embedding captures global appearance but is trained on identity
discrimination. It is already strong on random forgeries. On **skilled** forgeries the
writer deliberately imitates the reference, so global appearance is close and a
single similarity score can saturate. Cheap geometric evidence — stroke width, slant,
ink density, skeleton topology, projection profiles — is complementary and cheap to
compute. Fusing both at the score level (rather than the feature level) keeps each
modality's representation intact and lets the fusion be fitted and audited separately.

## Feature contract

Every feature is computed from the same `uint8` grayscale image the CNN consumes:
the Stage 2.2 output of `preprocessing.signature_preprocessing.preprocess_signature`
(or an already preprocessed `256 x 512` grayscale file). Reading through the shared
loader is what prevents train/serve skew between the CNN and the handcrafted branch.

All nine values are bounded to `[0, 1]`. Eight are similarities where larger means
"more alike"; `x_slant_diff` is a normalised distance where smaller means "more alike".

| Feature | Direction | Description |
| --- | --- | --- |
| `sim_ink_aspect_similarity` | similarity | Symmetric ratio of tight ink bounding-box aspect ratios |
| `sim_stroke_width_similarity` | similarity | Symmetric ratio of mean stroke width (`ink_area / skeleton_length`) |
| `x_endpoints_ratio` | similarity | Symmetric ratio of skeleton endpoint density |
| `x_junctions_ratio` | similarity | Symmetric ratio of skeleton junction density |
| `x_slant_diff` | distance | Normalised angular difference of dominant slant, in `[0, 1]` |
| `sim_density_similarity` | similarity | Symmetric ratio of ink density inside the bounding box |
| `h_proj` | similarity | `1 - 0.5 * L1` between L1-normalised horizontal ink profiles |
| `v_proj` | similarity | `1 - 0.5 * L1` between L1-normalised vertical ink profiles |
| `sim_hu_similarity` | similarity | `1 / (1 + mean |Δ log-Hu|)` over the seven Hu moments |

The `_ratio` helper is symmetric: `min(a, b) / max(a, b)`, returning `1.0` when both
values are zero. Profile similarity is `1 - 0.5 * L1` between L1-normalised vectors,
which is bounded to `[0, 1]`. Slant difference is wrapped modulo π and divided by π/2
so it lands in `[0, 1]`.

### Skeleton and topology

`binarize_ink` produces a boolean ink mask. For an already-binary image (the Stage 2.2
output, whose only values are 0 and 255) it applies the fixed cut
`INK_THRESHOLD = 200`; for any other grayscale image it falls back to Otsu's method.
Because Stage 2.2 output is dark ink on a whitened background, the fixed cut is safe
there and avoids re-estimating Otsu per image. An image with no ink raises
`ValueError` so a blank crop is routed to review instead of being scored.

Skeletonisation uses `cv2.ximgproc.thinning` when the OpenCV contrib build happens to be
installed, and otherwise falls back to a **pure-NumPy Zhang–Suen thinning**
implementation. No `opencv-contrib-python` or `scikit-image` dependency is added, so the
module runs on the plain `opencv-python` wheel that `requirements.txt` already pins.
Endpoints and junctions are counted from the 8-neighbourhood of the thinned skeleton and
normalised by skeleton length.

### Slant

Dominant slant is estimated from a doubled-angle circular mean over Sobel gradients,
restricted to a dilated ink region so paper texture and background gradients do not
vote. The gradient is perpendicular to the stroke, so the mean is rotated by 90° to
describe the stroke itself. The estimate is wrapped modulo π; only differences between
images are used, so the absolute convention does not matter. The comparison is the
wrapped angular difference, so a slant of `+5°` and `-5°` do not count as a large
mismatch.

### Hu moments

The seven Hu moments are computed on the ink mask cropped to its tight bounding box,
then converted to *signed* log space as `sign(h) * log10(|h| + 1e-12)` so a moment
changing sign is not hidden by the magnitude compression. `sim_hu_similarity` compares
the log-Hu vectors as `1 / (1 + mean |Δ|)`, which is bounded and well-behaved when a
moment is near zero.
