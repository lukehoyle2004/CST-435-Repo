# FleetDeck: the FineTuneFleet Dashboard

The bank's support lead will not read an OpenAPI page or a sweep table. **FleetDeck is their
one browser link.** It shows the deployed fine-tuned router and the zero-shot baseline side by
side on any message, the full experiment leaderboard with confidence intervals, and what
routing a month of messages would cost here compared with a frontier LLM API.

| | Link |
|---|---|
| **FleetDeck (Streamlit Community Cloud)** | https://fleetdeck-luke.streamlit.app |
| **GitHub (this interface)** | https://github.com/lukehoyle2004/CST-435-Repo/tree/main/Deep-Learning/Project-Templates/Topic_6B_fleet-deck |
| **Service it targets** | Product 6 "FineTuneFleet": [`Topic_6_fine-it`](../Topic_6_fine-it) in the same repository |
| **Render API it calls** | https://finetunefleet-api.onrender.com (docs at `/docs`) |
| **Supabase project reference** | `rtlxnyiyoyxvmqhvhcdf` (project `finetunefleet`) |

CST-435 / AIT-204 Topic 6, Product 6B (individual). Author: Luke Hoyle. The Streamlit
application lives in [`ui/`](ui).

## Tabs

| Tab | What it does | Source |
|---|---|---|
| **Concepts** | Typeset objectives for masked LM, causal LM, span corruption and contrastive InfoNCE (with the mutual-information bound); a task-suitability matrix; an honest list of when zero-shot prompting beats fine-tuning | static (`ui/content.py`) |
| **Try It** | One message through **both** routers, one after the other, side by side: queue, confidence, needs-review flag, all eight probabilities, server model time, end-to-end wait, model load time, cold start, and the server-side log id | `POST /predict`, `POST /predict_zero_shot` |
| **Leaderboard** | Every experiment, read with the anon key from the `ftf_leaderboard` view, sortable by several metrics. Each metric shows its 95% interval beside it, a chart plots the intervals, and runs whose interval overlaps the best run's are called **not meaningfully different** | anon-key read of Supabase |
| **Cost & Latency** | Sends a **fixed 16-message batch** to the live service, then prices the measured throughput on three Render plans against Claude Haiku, Sonnet and Opus 5.5 at a chosen monthly volume. Every assumption is shown on screen; all logged traffic comes from `ftf_latency` | `POST /predict_batch` + anon-key read |
| **Model Card** | Intended use, data, metrics with intervals, limitations, ethics | static |

Every tab has a "Read me first" panel written for a non-technical reader.

## Two clocks, labelled separately

Every prediction shows **server model time** (`latency_ms`, measured by the API on Render and
logged to Supabase) and **end-to-end wait** (measured by FleetDeck from request to response).
A caption breaks the difference down into model load time (`load_ms`) and network, proxy and
logging. Streamlit runs Python on Streamlit Cloud's server, so "end-to-end" means the wait
between Streamlit Cloud and Render.

## Failing honestly

`ui/client.py` classifies every failed call, and the page responds differently to each:

| Situation | How it is recognized | What the user sees |
|---|---|---|
| **Cold start** | Render's own 502/503/504 page (HTML, not the API's JSON) | a progress panel that polls `/healthz` for up to 2 minutes, and "Cold start: the service is starting up" |
| **Timeout** | no answer within 120 s | "No answer in time: waking up or busy" |
| **Outage** | connection refused or DNS failure, or no wake-up within 2 minutes | "Service unreachable / unavailable. Nothing below is a live result" |
| **Bad input** | 422 from the API | the API's own reason |
| **Service error** | other API error (e.g. the model failed to load) | the API's message |

A failed call **removes the previous answer** instead of leaving it on screen to be mistaken for
a new one. Successful results are labelled with the time they arrived. Nothing is cached across
failures: the next click re-checks the service, so the page **recovers without a reload**. If
Supabase cannot be read, the leaderboard shows nothing rather than old numbers.

## Tier boundary and secrets

**This tier runs no model, holds no weights and writes no SQL.** Every prediction crosses the
network to the FastAPI service on Render, and the API writes every log row server-side with the
service-role key. FleetDeck holds only:

* `API_URL`: the Render service;
* `SUPABASE_URL` and `SUPABASE_ANON_KEY`: the **anon** key. Row-level security limits it to
  `SELECT` on the leaderboard and the latency aggregates. Postgres refuses it the corpus and every
  prediction row, and it cannot write.

Secrets are read from Streamlit's secrets manager (locally `ui/.streamlit/secrets.toml`, which is
git-ignored; see `secrets.toml.example`). `ui/requirements.txt` contains no torch, transformers or
safetensors. `tests/test_boundary.py` checks all of this against the code: no model imports, no
weight files, no insert/update/delete/rpc, reads limited to the two views, and no service-role key
or JWT in the source.

## Run locally

```bash
py -3.12 -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

```bash
.venv/Scripts/python -m streamlit run ui/app.py
```

Copy `ui/.streamlit/secrets.toml.example` to `ui/.streamlit/secrets.toml` and paste in the anon key first.

## Deploy (Streamlit Community Cloud)

1. *Create app* → this repository, branch `main`, main file
   `Deep-Learning/Project-Templates/Topic_6B_fleet-deck/ui/app.py`.
2. *Advanced settings → Secrets*: paste the three lines from `secrets.toml.example`, with the real
   anon key.
3. Deploy, then open each tab once to check it.

## Tests and CI

```bash
.venv/Scripts/python -m pytest -q
```

28 tests. They run the real Streamlit script headlessly (`streamlit.testing.v1.AppTest`) against
a fake API transport and a read-only fake Supabase that use the production response shapes.
Covered:
* all five tabs and both clocks;
* the fixed cost batch and its assumptions;
* the leaderboard's intervals, sorting and "not meaningfully different" call-outs;
* cold start vs outage, a stale answer removed after a failure, and recovery without a reload;
* the client's failure classification;
* the tier boundary.

The [FleetDeck tests](../../../.github/workflows/fleetdeck-tests.yml) workflow runs them on every
push.

## Deployment reflection

**Cheaper, faster, good enough? With the numbers this dashboard shows.**
* **Good enough:** yes, on the evidence available. The fine-tuned router scores 0.965 macro-F1
  [0.958, 0.972] against 0.669 for the zero-shot model, and catches 97% of fraud reports.
* **Faster:** yes. 155 ms of model time per message on the live 0.1-CPU server, against 4.4 s
  for the zero-shot model on the same hardware.
* **Cheaper:** only past a volume threshold. An always-on Render Starter instance is $7 a month
  at any volume. The cheapest frontier option, Claude Haiku 5.5, is about $2.60 per 100,000
  messages, so below roughly 270,000 messages a month it is the cheaper choice, before even
  counting the engineering time the fine-tuned pipeline took.

FleetDeck shows that crossover on purpose, with a volume slider, rather than one flattering
volume.

**A presentation choice I'll defend: the interval goes beside the number, not in a tooltip.**
Every leaderboard metric reads like "0.954 [0.946, 0.962]", and a sentence under the table names
the runs that are *not meaningfully different* from the best. A tooltip would make the table
cleaner, and almost nobody would hover over it. The support lead's real question is "should we
retrain with the other learning rate?" The honest answer, that the top two runs cannot be
told apart on macro-F1, only shows up if the uncertainty is on screen. For the same reason,
Try It shows the cold-start load time as its own number instead of folding it into the
latency.

**Worldview reflection: the number I was tempted to leave off this dashboard.**

> **Draft for Luke to revise into his own words before submitting.**

Next to Opus at $3,050 a month, the $7 instance looks like an obvious win. The row I was
tempted to leave off is **Claude Haiku 5.5 at $26 a month**. It shows that for a small bank
(under about 270,000 messages a month) the API is cheaper than what I built. It stays in, with
the volume slider on screen. Proverbs 11:1 says "a false balance is an
abomination to the Lord, but a just weight is his delight". A dashboard is a set of scales.
Choosing which rows appear is choosing the weights, and a stakeholder who decides on rigged
weights has been deceived even if every number shown was true. Showing the crossover costs my
product an easy argument. It gives the bank what it needs to choose well, and that is what
the dashboard is for.
