# Model Card — See-Sense

## Model details
- **Task:** 10-class image classification on 32×32 RGB images (CIFAR-10 label set).
- **Architecture (both models):** `CifarCNN`, a VGG-style network: 3 blocks of
  `[conv3×3 → (BatchNorm) → ReLU] × 2 → MaxPool 2×2 → Dropout2d`, widths 32/64/128,
  then global average pool → Dropout → `Linear(128, 10)`. 6 conv layers in total.
- **Two served models (selectable in the UI):**

| | Baseline (`b0c1d2f1`) | Improved (`eb17638c`) |
|---|---|---|
| BatchNorm | no | yes |
| Dropout (2d / fc) | 0 / 0 | 0.05–0.15 / 0.3 |
| Augmentation | none | flip, ±10° rotation, ±4 px shift, color jitter 0.2, 8×8 cutout p=0.5 |
| Weight decay | 0 | 5e-4 |
| Label smoothing | 0 | 0.1 |
| LR schedule | constant 0.01 | one-cycle, peak 0.1 |
| Optimizer | SGD + Nesterov momentum 0.9 | same |
| Epochs / batch / seed | 30 / 128 / 0 | 30 / 128 / 0 |
| Parameters | 288,298 | 288,746 |

- **Explainability:** Grad-CAM on the last conv layer (128 × 8 × 8), upsampled
  bilinearly and blended (jet colormap) over the image. Because the head is
  GAP → Linear, Grad-CAM here equals class activation mapping (CAM).
- **Serving:** FastAPI on Render (CPU, 1 thread). Optional flip test-time
  augmentation (on by default) averages the logits of the image and its mirror.
- **Author:** Luke Hoyle, CST-435, Grand Canyon University, October 2026.

## Intended use
An **educational demo** for an undergraduate AI-literacy course: show a visitor
what a CNN predicts, how confident it is, and which pixels drove the decision,
so they leave understanding that a CNN does not see the way a human does.

**Out of scope:** any decision about a person (identity, demographics, emotion,
behavior), medical, security, or moderation use, and any image outside the 10
CIFAR-10 categories. The model always answers with one of the 10 labels, even
for a selfie or a laptop.

## Training data
- **CIFAR-10** (Krizhevsky, 2009): 60,000 32×32 color images, 10 balanced
  classes. 45,000 train / 5,000 stratified validation (seed 0) / 10,000 test.
- The test set was used once per model, after training, for the numbers below.
- Known dataset issues: tiny resolution, objects centered and filling the frame,
  2000s-era web photos, some near-duplicate images, a small rate of label noise,
  and strong context cues (ships on water, planes in sky, frogs on green).

## Metrics (CIFAR-10 test set, 10,000 images, flip TTA)

| Metric | Baseline | Improved |
|---|---|---|
| Top-1 accuracy (TTA) | **80.25%** | **87.45%** |
| Top-1 accuracy (single view) | 78.03% | 86.61% |
| Macro F1 | 0.7996 | 0.8733 |
| Best val accuracy (epoch) | 78.50% (26) | 85.86% (30) |
| Train accuracy at that epoch | 96.97% | 83.22%* |

\* The improved model's train accuracy is measured on augmented images with
dropout active, so it sits below validation accuracy. The baseline's 18-point
train/val gap is classic overfitting; regularization closes it.

**Weakest class for both models: cat** (recall 0.540 baseline, 0.704 improved),
mostly confused with dog. Per-class precision/recall, confusion matrices, and
training curves are in the `runs` table and the Compare Models tab.

## Privacy
Uploads are processed in memory and discarded. Only the SHA-256 hash, width,
height, format, byte size, predicted label, and top-5 probabilities are stored
in Supabase. Raw pixels, filenames, EXIF data (including GPS), IP addresses and
user agents are never logged.

## Limitations and risks
- **Not calibrated for out-of-distribution input:** a photo of a person still
  gets a confident CIFAR-10 label. Confidence is not correctness.
- **Center crop:** only the central square of an upload is classified; an object
  at the edge is cropped away (the full-size overlay dims those margins).
- **Shortcut features:** Grad-CAM shows the models sometimes key on background
  (water, sky, grass) instead of the object.
- **32×32 input:** fine detail is destroyed before the model ever sees it.
- **Distribution shift:** modern phone photos differ from CIFAR-10's 2009 web
  images; real-world accuracy will be lower than the test-set number.
