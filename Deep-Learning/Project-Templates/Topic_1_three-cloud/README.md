# Regress-It — Three-Cloud Reference Template

> A working, forkable template for a three-cloud architecture:
> **Streamlit (UI) ⇄ FastAPI (Model API) ⇄ Supabase (Data)**. Fork it, wire up
> your own accounts, and reuse the exact same pattern for the other templates in
> this repository.

**Name:** Luke Hoyle
**Section:** CST-435 (AIT-204 assignment, Topic 1 — Three-Cloud Product)

## Live deployment URLs

| Tier | Platform | URL |
|------|----------|-----|
|**GitHub Repository:** | https://github.com/lukehoyle2004/CST-435-Repo/tree/main/Deep-Learning/Project-Templates/Topic_1_three-cloud |
| **UI** | Streamlit Community Cloud | https://cst-435-repo-bfgbqrhpiwe8tkbkqfz5vw.streamlit.app/ |
| **API** | Render.com | https://regress-it-api-ci19.onrender.com/docs |
| **Data** | Supabase | https://ftsrfgtwmzqgkdqdndni.supabase.co (project ref: `ftsrfgtwmzqgkdqdndni`) |
| **Migration file:** | `db/migrations/001_init.sql` (in this repo) |

---

## What it does

Regress-It is an interactive teaching demo for 1-D linear regression. You pick a
learning rate, batch size, and epoch count; the API trains `y = w·x + b` with
PyTorch mini-batch SGD on synthetic data, reports held-out **MSE / MAE / R²**,
and persists every run. The UI lets you visualise convergence, make predictions,
and browse run history.

## Architecture

```
┌──────────────────────┐   HTTPS/JSON    ┌──────────────────────┐   service-role   ┌──────────────────┐
│  Streamlit Cloud     │ ──────────────► │  FastAPI on Render   │ ───────────────► │  Supabase        │
│  (ui/app.py)         │                 │  (api/main.py)       │   full access    │  Postgres        │
│  thin client, no ML  │                 │  PyTorch training    │                  │  datasets/runs/  │
│                      │ ◄────anon key,  │                      │                  │  predictions     │
│                      │   read-only ────┼─────────────────────┼──────────────────►│  (RLS: anon can  │
└──────────────────────┘   SELECT runs   └──────────────────────┘                  │   only SELECT)   │
                                                                                    └──────────────────┘
```

- **UI never touches the model or writes SQL.** It calls the API over HTTPS and
  performs one read-only `SELECT` on `runs` with the publishable (anon-equivalent) key.
- **API owns the model and all writes**, using the Supabase **secret** (service-role-equivalent) key.
- **Supabase is the single source of truth** for datasets, runs, and predictions.

See [`TUTORIAL.md`](./TUTORIAL.md) for the full step-by-step build and deploy guide.

## Project structure

```
three-cloud/
├── TUTORIAL.md               # Full build + deploy guide (start here)
├── README.md                 # This file
├── MODEL_CARD.md             # Model details, intended use, limitations
├── shared/                   # Code shared by both tiers
│   ├── schemas.py            # Pydantic API contract
│   └── data.py               # Synthetic linear data generator
├── api/                      # FastAPI tier (deploys to Render)
│   ├── main.py               # Endpoints
│   ├── training.py           # PyTorch linear regression
│   ├── db.py                 # Supabase (service-role) data access
│   ├── configs/default.yaml  # Default hyperparameters
│   └── requirements.txt
├── ui/                       # Streamlit tier (deploys to Streamlit Cloud)
│   ├── app.py                # 5-tab thin client
│   ├── requirements.txt      # No torch
│   └── .streamlit/secrets.toml.example
├── db/                       # Database tier (Supabase)
│   ├── migrations/001_init.sql
│   └── seed.py
├── tests/                    # pytest suite
├── render.yaml               # Render blueprint
├── requirements-dev.txt      # Both tiers + pytest (local dev)
└── .env.example
```

## Quickstart (local)

```bash
cd Deep-Learning/Project-Templates/Topic_1_three-cloud

# 1. Install everything (both tiers + test tools)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# 2. Run the tests (6 pass; the live-Supabase test skips without creds)
pytest -q

# 3. Configure secrets
cp .env.example .env                                   # API: SUPABASE_URL + SECRET key
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml

# 4. Run the API
uvicorn api.main:app --reload --port 8000

# 5. In another terminal, run the UI
streamlit run ui/app.py
```

To deploy to the three clouds, follow **Part E** of [`TUTORIAL.md`](./TUTORIAL.md):
apply `db/migrations/001_init.sql` in the Supabase SQL Editor → deploy the API
from `render.yaml` on Render → deploy the UI on Streamlit Community Cloud.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/datasets` | Create a synthetic dataset |
| `POST` | `/train` | Train a model, persist the run, return metrics |
| `GET`  | `/runs/{run_id}` | Fetch one run |
| `GET`  | `/runs` | List recent runs |
| `POST` | `/predict` | Predict `ŷ` for an `x` using a fitted run |
| `GET`  | `/healthz` | Liveness / DB ping |
| `GET`  | `/version` | Build SHA + framework versions |

---

## Engineering Report

### Decision Justifications

**Learning rate.** I ran six training jobs on the same dataset. Anything from 0.0015 to 0.025 converged fine, landing at a held-out R² between 0.982 and 0.987 within 100 epochs. I picked 0.02 as the default. It converges faster than 0.0015 without the wobble that starts showing up as you push the rate
higher. Then I tried 1.5. The gradient updates overshot every single step and the weights blew up to infinity within a few epochs. I had to cap MSE/MAE/R² at a sentinel value (1e9) just so the run wouldn't crash the whole app trying
to serialize infinity as JSON.

![alt text](image-2.png)

![alt text](image-1.png)

**Stopping criterion.** I used a flat 100 epochs instead of stopping early based on loss improvement. For a model with two parameters and a clean synthetic dataset, 100 epochs is way more than enough to converge unless the run is actively diverging, so a fixed budget kept things simple. The downside:
a bad learning rate still burns all 100 epochs before it returns. Checking for NaN/Inf loss and bailing out early would fix that, and it's the obvious next thing to add.

**Validation split.** 80/20 train/test, seeded so it's reproducible. With 500 points and a model this simple (just a slope and an intercept), 20% held out is plenty to get a stable R² without starving the training set.

**Why the run history table matters.** Every run's hyperparameters and metrics get written to Supabase and show up as a plain table in the Run History tab. I don't have to rerun anything or squint at overlapping loss curves to compare configs. Sort by lr or r2 and the converging runs (R² ~0.98)
and the one diverging run (pinned at the sentinel) are obvious at a glance.

![alt text](image-3.png)

### Worldview Reflection

A model that reports R² of 0.98 can look a lot more trustworthy than it actually is to someone who can't check the assumptions behind it. Honesty, in a Christian sense, isn't just not lying. It means making sure the other person actually understands what they're relying on. Proverbs 11:1 calls a false balance an abomination, and handing a client a clean dashboard without
explaining that it's trained on 500 synthetic points from a known linear function, with zero guarantee it holds up outside that range, is basically a false balance. Technically true, but built to look better than it's earned.

Stewardship is the other half of it. A data scientist is trusted with a client's decisions and their money, and stewarding that well means telling people the model's weak points before they ask, not waiting to get caught.
Concretely: this demo only handles one input feature, it has no way to notice when new data falls outside what it trained on, and the lr=1.5 run shows how a normal-looking setting can silently produce garbage with no clear error.
That's exactly why the run history table and health checks exist. Building that protection into the product itself, instead of counting on the client to know what to ask, is what stewardship actually looks like here.

---

## Reusing this pattern

The three-cloud split and the file layout stay identical for every product. Swap
only the model in `api/training.py`, the Pydantic contract in `shared/schemas.py`,
the tables in `db/migrations/`, and the UI tabs — the UI stays a thin client and
Supabase stays the single source of truth. Two worked examples
(`income-insight`, `see-sense`) live alongside this one; see the
final section of [`TUTORIAL.md`](./TUTORIAL.md).

## Checklist

- [x] Three live URLs listed at the top of this README
- [x] `datasets`, `runs`, `predictions` tables in Supabase with RLS
- [x] 6+ API endpoints
- [x] 5 Streamlit tabs (Concepts, Train, Predict, Run History, Model Card)
- [x] PyTorch training with held-out MSE/MAE/R²
- [x] pytest suite passing
- [x] `MODEL_CARD.md` completed
- [X] Screenshots inserted in place of the two placeholders above
