# See-Sense: Engineering Report

**Author:** Luke Hoyle · CST-435 Topic 3 · **GitHub:** https://github.com/lukehoyle2004/CST-435-Repo/tree/main/Deep-Learning/Project-Templates/Topic_3_see-sense
**Live:** [Streamlit UI](https://see-sense-luke.streamlit.app) · [Render API](https://see-sense-api.onrender.com/docs) · [Supabase](https://siwzvvvbdjrbbymetrek.supabase.co) (project `siwzvvvbdjrbbymetrek`)

## 1. Problem

See-Sense classifies an uploaded photo into one of the ten CIFAR-10 classes and shows a Grad-CAM heatmap of the pixels behind the decision. The goal is teaching: visitors see that a CNN reads a 32×32 thumbnail, must pick one of ten labels, and often keys on background.

## 2. Convolution, stride, padding and receptive field

A convolution slides a $K\times K$ kernel over a $C_{in}$-channel input; each output is a dot product plus a bias:

$$Y_{c_{out}}[i,j] = b_{c_{out}} + \sum_{c=1}^{C_{in}}\sum_{u=0}^{K-1}\sum_{v=0}^{K-1} W_{c_{out},c}[u,v]\,X_c[\,Si+u-P,\ Sj+v-P\,]$$

For input size $N$, padding $P$ and stride $S$, the output size is

$$O = \left\lfloor \frac{N + 2P - K}{S} \right\rfloor + 1$$

| Layer | $N$ | $K$ | $P$ | $S$ | $O$ |
|---|---|---|---|---|---|
| 3×3 conv, "same" | 32 | 3 | 1 | 1 | 32 |
| 3×3 conv, stride 2 | 32 | 3 | 1 | 2 | 16 |
| 2×2 max-pool | 32 | 2 | 0 | 2 | 16 |

Padding of 1 keeps every conv size-preserving, so only the pools shrink the map (32 → 16 → 8 → 4).

**Receptive field.** With jump $j_l$ (distance in input pixels between neighboring outputs) the recurrence is

$$r_l = r_{l-1} + (K_l - 1)\,j_{l-1}, \qquad j_l = j_{l-1}\,S_l, \qquad r_0 = j_0 = 1$$

**Worked example, three stacked 3×3 kernels (stride 1):** $r = 1 \to 3 \to 5 \to 7$. Three 3×3 layers see the same 7×7 patch as one 7×7 kernel but use $3\cdot 9C^2 = 27C^2$ weights instead of $49C^2$, and add two extra ReLUs.

**See-Sense network** (each stage = conv, conv, pool):

| After | conv | conv | pool | jump |
|---|---|---|---|---|
| Stage 1 | 3 | 5 | 6 | 2 |
| Stage 2 | 10 | 14 | 16 | 4 |
| Stage 3 | 24 | **32** | 36 | 8 |

The Grad-CAM layer (stage 3, last conv, 128×8×8) has a 32-pixel receptive field: each cell can in principle see the whole image. This is what gives the model capacity to combine object and context, and it is also why it can learn background shortcuts.

## 3. Baseline vs. improved CNN

Both runs use the same skeleton (~288k parameters), 30 epochs, batch 128, seed 0, the same 45k/5k/10k split and the same laptop CPU. Each run is a row in the Supabase `runs` table; the headline numbers come from `select * from model_comparison;`.

| Model | Run | Train acc | Val acc | Test acc | Test (flip TTA) | Macro F1 |
|---|---|---|---|---|---|---|
| Baseline | `b0c1d2f1` | 97.0% | 78.5% | 78.0% | 80.25% | 0.800 |
| Improved | `eb17638c` | 83.2% | 85.9% | 86.6% | **87.45%** | 0.873 |

The improved run adds augmentation (flip, ±10° rotation, 4-px translate, color jitter, cutout), BatchNorm, Dropout2d plus FC dropout, weight decay 5e-4, label smoothing 0.1 and a one-cycle LR schedule.

The baseline memorizes (an 18.5-point train/val gap). The weakest classes gained most: cat F1 0.61 → 0.76, bird 0.72 → 0.82. Cat↔dog is still the top confusion.

**Leave-one-out ablations.** Six more runs each remove one technique from the improved recipe, with the same seed and budget; all are rows in `runs`, queried through the `ablation_comparison` view.

| Removed | Test (TTA) | Contribution (pts) | Train − val |
|---|---|---|---|
| one-cycle LR (→ constant 0.01) | 81.59% | **+5.86** | −4.55 |
| weight decay | 86.07% | +1.38 | −3.48 |
| BatchNorm | 86.09% | +1.36 | −3.64 |
| label smoothing | 87.99% | −0.54 | −3.21 |
| augmentation | 88.98% | −1.53 | +8.19 |
| dropout | 89.55% | −2.10 | −0.43 |

The optimization techniques carried the gain: the one-cycle schedule is worth almost 6 points, and weight decay and BatchNorm about 1.4 each. Augmentation, dropout and label smoothing *cost* accuracy at a 30-epoch budget. With all six on, the model underfits (train 83.2% below validation 85.9%), so extra regularization removes more signal than noise. Augmentation still reduces overfitting (removing it opens an 8-point gap); it simply needs longer training to pay off. The contributions sum to 4.4 points, less than the 7.2-point total, because the techniques interact.

## 4. Grad-CAM

For class score $y^c$ and feature maps $A^k$ ($k = 1..128$, each 8×8, $Z = 64$):

$$\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}, \qquad L^c = \mathrm{ReLU}\Big(\sum_k \alpha_k^c A^k\Big)$$

The weight $\alpha_k^c$ is the global-average-pooled gradient: how much map $k$ pushes class $c$ up. The ReLU keeps only positive evidence. $L^c$ is upsampled from 8×8 to the input size and overlaid. A forward hook captures $A^k$ and a backward hook captures its gradient.

The Compare Models tab shows 12 held-out test images, 6 the improved model gets right and 6 it gets wrong, with both models' overlays.

**Failure 1, dataset failure (#7068, bird → airplane, 92%).** The image is an ostrich: a large dark body on long legs. The heatmap sits squarely on the body, so the model looked at the right object. CIFAR-10 "bird" is mostly small perched or flying birds, and a big dark fuselage-shaped body is closer to its airplanes. The baseline makes the same mistake (90%). The training data never taught this kind of bird.

**Failure 2, model failure (#6401, deer → horse, 45%).** The heat covers the neck and torso and skips the head, where antlers and ear shape separate deer from horse. The baseline got this image right, so the information is learnable; the improved model learned to weight body shape over head detail. Low confidence confirms it is guessing between two similar body plans.

## 5. Ethical consideration

**Deployment contexts.** (1) A classroom demo, where students may upload photos of themselves, friends or their homes. (2) Reuse as a wildlife or campus-camera tagger, where a confident wrong label (deer called horse) could trigger the wrong response.

**Technical safeguards.** (1) Only a SHA-256 hash, size and format are stored; image bytes, filenames and EXIF never leave memory. (2) The UI always shows the full probability bar and the heatmap, so a 45% guess is visibly a guess.

**Governance safeguards.** (1) Row-level security: the public anon key can only read hashes and labels; writes need the server-side service key. (2) The model card states intended use (education only) and out-of-scope uses (people, surveillance, safety decisions).

**What we deliberately did NOT log:** image pixels or thumbnails, filenames, EXIF (GPS, camera, time), IP addresses, user agents and session IDs.

**Christian-worldview reflection.** Responsible disclosure means telling users plainly what the system does and does not do. Proverbs 12:22 says the Lord delights in those who act faithfully, and a demo that hid its 13% error rate behind a confident label would not be faithful. Luke 16:10 says whoever can be trusted with little can be trusted with much: handling a stranger's photo with care, by keeping nothing we do not need, is the small version of that trust.

## 6. Limitations

The ten-class, 32×32 setting fails on out-of-distribution photos by design. Each ablation is a single seed, so differences under about 0.5 points are within noise, and the no-schedule run also lowers the peak LR. The ablations point to a better recipe than the served one (no dropout, 89.55%); the next step is training it longer, where augmentation should start to help.

## References

Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images.* University of Toronto.
Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV.*
Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR.*
