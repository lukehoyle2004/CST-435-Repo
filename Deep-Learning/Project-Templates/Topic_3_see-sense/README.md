# See-Sense — a CIFAR-10 Vision-Classification Service with Grad-CAM Explanations

| | Link |
|---|---|
| **Streamlit UI** | https://see-sense-luke.streamlit.app |
| **Render API** (docs at `/docs`) | https://see-sense-api.onrender.com |
| **Supabase (Postgres)** | https://siwzvvvbdjrbbymetrek.supabase.co (project ref `siwzvvvbdjrbbymetrek`) |
| **GitHub** | https://github.com/lukehoyle2004/CST-435-Repo/tree/main/Deep-Learning/Project-Templates/Topic_3_see-sense |
| **Video walkthrough (Luke Hoyle)** | https://youtu.be/FbRoQ2bi20s |
| **Engineering report** | [docs/ENGINEERING_REPORT.md](docs/ENGINEERING_REPORT.md) |

CST-435 Topic 3, Product 3. Author: Luke Hoyle.

## Product pitch

A teaching team wants an image-recognition demo for an undergraduate AI-literacy
course. Upload a photo and See-Sense names it, shows its probability for each
of the ten classes, and paints a Grad-CAM heatmap over the pixels
that drove the decision. A public Recent Classifications wall shows what others
asked (anonymized to a hash), and every classification links back to the
training run that produced it. Visitors should leave knowing that a CNN does
not see the way they do: it sees a 32×32 thumbnail, must choose one of ten
labels, and often keys on water, sky or grass rather than the object.

**Why CIFAR-10 instead of Fashion-MNIST?** Color photos of everyday objects are
what visitors will upload, so the demo's failures are realistic and teachable.
Fashion-MNIST is grayscale, centered product shots on a black background; a phone
photo of a shoe is wildly out of distribution and the lesson would be lost.
CIFAR-10 also makes the gap between models visible (80% vs 87%); on
Fashion-MNIST both would score ~90%+.

## Architecture

```
 Browser
    │
    ▼
┌──────────────────────────┐  multipart POST /predict   ┌──────────────────────────────┐
│ Streamlit Community Cloud│ ─────────────────────────► │ FastAPI on Render.com         │
│ ui/app.py (thin client)  │ ◄───────────────────────── │ api/main.py                   │
│ no torch, no PIL         │  JSON: label, probs, top-5,│ baseline.pt + improved.pt      │
│ 5 tabs                   │  base64 Grad-CAM PNGs      │ preprocessing + Grad-CAM hook  │
└────────────┬─────────────┘                            └───────────────┬──────────────┘
             │ anon key, SELECT only (RLS)                              │ service-role key
             │ runs, predictions ⨝ runs                                 │ INSERT predictions,
             ▼                                                          ▼ image_metadata
          ┌─────────────────────────────────────────────────────────────────┐
          │ Supabase Postgres: runs · predictions · image_metadata          │
          │ view model_comparison · no image bytes anywhere                  │
          └─────────────────────────────────────────────────────────────────┘
```

Training happens offline (laptop CPU) in the companion project
`AIT-204-CNN/cifar10_project`; `scripts/export_models.py` copies the weights
into `models/` and `db/seed_runs.py` writes one `runs` row per training run.

## Results (CIFAR-10 test set, 10,000 images)

| Model | Run | Techniques | Test acc (TTA) | Single view | Macro F1 |
|---|---|---|---|---|---|
| Baseline | `b0c1d2f1` | none | 80.25% | 78.03% | 0.7996 |
| Improved | `eb17638c` | augmentation, BatchNorm, dropout, weight decay, label smoothing, one-cycle LR | **87.45%** | 86.61% | 0.8733 |

Both runs: same architecture skeleton, 30 epochs, batch 128, seed 0, the same
45k/5k/10k split, ~288k parameters. The headline numbers come from SQL:
`select * from model_comparison;`. The baseline overfits (97% train vs 78.5%
val); the improved model's regularizers close that gap.

### Ablations: which technique contributed most

Leave-one-out: six more runs, each the improved recipe with ONE technique removed,
same seed (0), 30 epochs, batch 128 and split. Every run is a row in `runs`; the
table comes from `select * from ablation_comparison;` and is charted in the
Compare Models tab.

| Technique removed | Run | Test acc (TTA) | Contribution (pts) | Train − val gap |
|---|---|---|---|---|
| one-cycle LR → constant 0.01 | `eabe8c9b` | 81.59% | **+5.86** | −4.55 |
| weight decay | `05412b86` | 86.07% | +1.38 | −3.48 |
| BatchNorm | `37ffc726` | 86.09% | +1.36 | −3.64 |
| label smoothing | `289eb4fd` | 87.99% | −0.54 | −3.21 |
| augmentation | `fa11ca8c` | 88.98% | −1.53 | +8.19 |
| dropout | `59b34c8f` | 89.55% | −2.10 | −0.43 |

Contribution = improved accuracy (87.45%) minus the ablated run's accuracy.

- **The one-cycle schedule contributed most** (+5.9 pts), followed by weight
  decay and BatchNorm (about +1.4 each). These make optimization work inside a
  30-epoch budget.
- **Augmentation, dropout and label smoothing cost accuracy at this budget.**
  With all six on, the model underfits (train 83.2% < val 85.9%), so extra
  regularization takes away more than it gives. Augmentation still does its job,
  since removing it opens an 8-point overfitting gap, but 30 epochs is too short
  for it to pay off in accuracy.
- The contributions sum to +4.4 pts, not the full +7.2 over the baseline: the
  techniques interact, which is why each is measured against the full recipe.
- Caveats: one seed per run, so differences under about 0.5 pts are within noise.
  The no-schedule run uses the baseline's constant 0.01, so it measures the
  schedule including its higher peak LR.
- The served "improved" model is the planned recipe. The ablation points to a
  better one (no dropout, 89.55%), which is the next thing to train.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Multipart image upload → label, 10 probabilities, top-5, Grad-CAM PNGs (base64); logs hash to `predictions` + `image_metadata` |
| `GET` | `/runs`, `/runs/{run_id}` | Training runs from Supabase |
| `GET` | `/predictions` | Recent predictions joined to `runs` |
| `GET` | `/gallery` | Both models on 12 held-out test images (6 right, 6 wrong for the improved model) |
| `GET` | `/classes`, `/models` | Label list; served models and their run ids |
| `GET` | `/healthz`, `/version` | Liveness + DB ping; git SHA + torch version |

## Streamlit tabs

**Concepts** (convolution, output size, receptive field, Grad-CAM derivation in
LaTeX) · **Try It** (upload, top-5, Grad-CAM side by side) · **Compare Models**
(metrics, curves, confusion matrices, per-class precision/recall, held-out
gallery) · **Recent Classifications** (anon-key Supabase query, sortable table,
class histogram, drill-down to the run) · **Model Card**.

## Dataset card (CIFAR-10)

| | |
|---|---|
| Source | Krizhevsky (2009), University of Toronto; loaded via `torchvision.datasets.CIFAR10` |
| Size | 60,000 RGB images, 32×32, 10 balanced classes (6,000 each) |
| Split used | 45,000 train / 5,000 validation (seed 0) / 10,000 official test |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Known issues | Tiny resolution; objects centered and filling the frame; strong background cues (sky for planes and birds, water for ships); narrow subtypes (few large or flightless birds); a small number of label errors in the test set |
| People | Not a people dataset; See-Sense should not be used to classify people |
| License | Free for research and education |

## Data persisted / not persisted

| Persisted (Supabase) | Never persisted |
|---|---|
| SHA-256 of the upload | Image bytes or thumbnails |
| Width, height, format, byte size | Filename |
| Predicted label, confidence, top-5 | EXIF metadata (GPS, camera, time) |
| Run id that served it, TTA flag, latency | IP address, user agent, session id |
| Training runs: hyperparameters, metrics, curves | |

**Why no raw images:** people upload photos of their homes, kids and faces. The
demo needs none of that to work, a hash is enough to count repeat uploads, and
data we never store cannot leak, be subpoenaed, or be repurposed to train a
face model. The anon key can read the wall because the wall contains nothing
but hashes and labels.

## Run locally

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                   # 35 pass; live Supabase test skips without .env

copy .env.example .env                      # SUPABASE_URL + SUPABASE_SERVICE_KEY
copy ui\.streamlit\secrets.toml.example ui\.streamlit\secrets.toml
uvicorn api.main:app --reload --port 8000   # terminal 1
streamlit run ui/app.py                     # terminal 2
```

## Deploy

1. **Supabase:** SQL Editor → run `db/migrations/001_init.sql` → `python -m db.seed_runs`.
2. **Render:** New → Blueprint → this repo (`render.yaml` sets `rootDir`); add
   `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`.
3. **Streamlit Cloud:** main file `Deep-Learning/Project-Templates/Topic_3_see-sense/ui/app.py`,
   Python 3.11; paste `API_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` into Secrets.

## Tests

`tests/` covers the multipart `/predict` smoke test, Grad-CAM overlay dimensions
= input dimensions, a regression test on a fixed CIFAR-10 reference image for
both models, a live Supabase round-trip (`/predict` → `predictions` row linked
to a `runs` row), and a Supabase-outage path (`/predict` still answers with
`logged: false`; DB endpoints return 503). CI: `.github/workflows/see-sense-tests.yml`.

## Individual contribution (Luke Hoyle)

- Trained the improved CNN and the matched baseline; added the `--baseline` mode
- Added leave-one-out ablation flags and ran the six ablation runs overnight
- Wrote the SQL migration (`runs`, `predictions`, `image_metadata`, `model_comparison`, RLS)
- Built the FastAPI service, Grad-CAM overlay rendering, and Supabase logging
- Built the five-tab Streamlit client and the Compare Models gallery
- Wrote the pytest suite, CI workflow, model card, and this README
- Deployed all three clouds

### Task list

- [x] Train matched baseline and improved CNNs (same seed, epochs, batch, split)
- [x] Export weights and seed both runs into Supabase `runs`
- [x] Run six leave-one-out ablations (same seed and budget), persist them to `runs`, and add the `ablation_comparison` view
- [x] Write the SQL schema, `model_comparison` view, and RLS policies
- [x] Build the FastAPI `/predict` service with Grad-CAM and hash-only logging
- [x] Build the five Streamlit tabs, including the held-out Grad-CAM gallery and class histogram
- [x] Write the pytest suite (upload, Grad-CAM shape, frozen regression, Supabase round-trip, outage path) and CI
- [x] Deploy Supabase, Render, and Streamlit Community Cloud
- [x] Write the README, model card, dataset card, and engineering report
- [x] Record the video walkthrough
