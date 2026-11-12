# TrioDeck: the TrioFormer Comparison Interface

The learning-and-development team from Product 5 will not read an OpenAPI page. **TrioDeck is
their one browser link**: a Streamlit app that runs every TrioFormer endpoint, shows BERT,
GPT-2 and T5 side by side on the same paragraph with their measured latencies, and charts the
benchmarks view over time.

| | Link |
|---|---|
| **TrioDeck (Streamlit Community Cloud)** | https://triodeck-luke.streamlit.app |
| **GitHub (this interface)** | https://github.com/lukehoyle2004/CST-435-Repo/tree/main/Deep-Learning/Project-Templates/Topic_5B_trio-deck |
| **Team repository / service it targets** | Product 5 "TrioFormer": [`Topic_5_former-it`](../Topic_5_former-it) in the same repository |
| **Render API it calls** | https://trioformer-api.onrender.com (docs at `/docs`) |
| **Supabase project reference** | `jurmqdwomeiilrlebvuq` (project `trioformer`) |

CST-435 / AIT-204 Topic 5, Product 5B (individual). Author: Luke Hoyle. The Streamlit
application lives in [`ui/`](ui).

## Tabs

| Tab | What it does | Source |
|---|---|---|
| **Concepts** | Typeset derivation of scaled dot-product attention and of multi-head attention as a concatenation of parallel heads, with a step-by-step shape table; sinusoidal vs learned vs rotary positional encodings, and why rotary dominates modern LLMs | static (`ui/content.py`) |
| **My Encoder Block** | Runs the hand-written encoder from Product 5 and draws every head's attention heatmap; a **positional-encoding toggle** shows the claim from Concepts: without it, "a b a" and "a a b" get the same answer | `POST /encoder_attention` |
| **BERT — Fill Mask** | Ranked candidates for one `[MASK]` (or a hidden word) | `POST /bert_fillmask` |
| **GPT-2 — Continue Text** | Greedy or sampled continuation | `POST /gpt2_continue` |
| **T5 — Summarize** | Summary of a paragraph | `POST /t5_summarize` |
| **Compare** | One paragraph through all three archetypes, side by side, each with its server-side and end-to-end time | the three endpoints above |
| **Latency History** | Median server-side latency per model per hour, from the `trio_benchmarks_hourly` materialized view, plus all-time per-model numbers from `trio_benchmarks` | anon-key read of Supabase |

Every tab has a "When to use this archetype" (or "Read me first") panel written for a
non-technical reader.

## Two clocks, labelled separately

Every model result shows:

* **Server-side model time (`latency_ms`)**: measured by the API on Render, tokenize → model
  → decode. This is the number logged to Supabase.
* **End-to-end wait (this app ↔ Render)**: measured by TrioDeck, from sending the request to
  having the response.
* **Checkpoint load (`load_ms`)** and **network + proxy** (end-to-end minus the server's
  `total_ms`) explain the difference between the two.

Streamlit runs Python on Streamlit Cloud's server, not in the browser, so the end-to-end figure
is the wait between Streamlit Cloud and Render. The browser then adds only the time to redraw
the page over Streamlit's websocket. The label says exactly which hop it measures.

## Failing honestly

`ui/client.py` classifies every failed call, and the page responds differently to each:

| Situation | How it is recognized | What the user sees |
|---|---|---|
| **Cold start** | Render's own 502/503/504 page (HTML, not the API's JSON) | "Cold start: the service is starting up", plus a progress panel that polls `/healthz` for up to 90 s |
| **Timeout** | no answer within 300 s | "No answer in time: waking up or the small free server is busy" |
| **Outage** | connection refused or DNS failure, or no wake-up within 90 s | "Service unreachable / unavailable. Nothing below is a live result" |
| **Bad input** | 422 from the API | the API's own reason, e.g. "put [MASK] in the text" |
| **Service error** | other API error (e.g. a model failed to load) | the API's message |

A failed call **removes** the previous answer from the screen instead of leaving it there to be
mistaken for a new one. Successful results are labelled with the time they were received.
Nothing is cached across failures, so the next click re-checks the service and the page
recovers **without a reload**. If the API cannot refresh the benchmark views, Latency History
says so and shows the views' own "refreshed at" time. If Supabase cannot be read at all, it
shows nothing rather than old numbers.

## Tier boundary and secrets

**This tier runs no model and writes no SQL.** Every inference crosses the network to the FastAPI
service on Render, and every write to Supabase happens server-side, inside the API, with the
service-role key. TrioDeck holds only:

* `API_URL`: the Render service,
* `SUPABASE_URL` and `SUPABASE_ANON_KEY`: the **anon** key, which row-level security limits to
  `SELECT` on the two aggregate views (`trio_benchmarks`, `trio_benchmarks_hourly`). It cannot
  read the hashes, the model outputs, or anything else, and cannot write.

The secrets are read from Streamlit's secrets manager (locally `ui/.streamlit/secrets.toml`, which
is git-ignored; see `secrets.toml.example`). The service-role key never appears in this tier.
The code backs these claims up: `ui/requirements.txt` contains no torch, transformers or model
library, and `tests/test_boundary.py` fails the build if the UI imports a model library, calls a
Supabase write (`insert`/`update`/`upsert`/`delete`/`rpc`), reads any table other than the two
views, mentions the service-role key, or commits a key or a `secrets.toml`.

## Run locally

```bash
py -3.12 -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

Copy `ui/.streamlit/secrets.toml.example` to `ui/.streamlit/secrets.toml` and paste the anon key, then:

```bash
.venv/Scripts/python -m streamlit run ui/app.py
```

## Deploy (Streamlit Community Cloud)

New app → repository `lukehoyle2004/CST-435-Repo`, branch `main`, main file
`Deep-Learning/Project-Templates/Topic_5B_trio-deck/ui/app.py`, Python 3.12. Under **Advanced
settings → Secrets**, paste the three lines from `secrets.toml.example` with the real anon key.

## Tests

```bash
.venv/Scripts/python -m pytest -q
```

27 tests run the real Streamlit script headlessly (`streamlit.testing.v1.AppTest`) against a fake
TrioFormer API and a read-only fake Supabase. They cover all seven tabs, both latency clocks, the
positional-encoding toggle, cold start, outage, recovery without a reload, Supabase being
unreadable, the client's failure classification, and the tier-boundary checks.
[TrioDeck tests](../../../.github/workflows/triodeck-tests.yml) runs them on every push.

## Deployment reflection

**Measurement.** Render's free instance sleeps after 15 idle minutes. Waking it took 93 s
before `/healthz` answered, and the first call to each model then paid a checkpoint load on
top (19.3 s for BERT). Through the deployed Compare tab, BERT's server-side `latency_ms` was
4,939 ms but the app waited 24,442 ms; GPT-2 took 124,331 ms against 127,673 ms; T5 took
9,622 ms against 13,773 ms. The gap is not the model. Network transit and Render's proxy
account for only 80–144 ms. The rest is loading checkpoints, because only one model fits in
512 MB, plus any queueing behind the model lock. Even the server time is not purely the
model's: the same 20-token GPT-2 call took 4.4 s at 21:33 and 124 s at 21:51 the same day,
as Render throttled the shared 0.1-CPU instance.

**Architecture.** Splitting the tiers made the interface harder to build. It has to tell a
cold start from an outage, it cannot run the three models in parallel on one small server,
and it needs three deployments and their secrets kept in step. In return, the interface holds
no model and no write key. It can be public with only an anon key that reads two aggregate
views, and any classmate's interface can call the same logged service.

**Worldview.** The number easiest to hide is GPT-2's 124,331 ms. A demo shown only at the
fast hour would make the free-tier stack look cheaper than it is, and the team would build
its recommendation on a false weight. "A false balance is an abomination to the LORD, but a
just weight is his delight" (Proverbs 11:1, ESV). A dashboard is a set of scales, so TrioDeck
shows the slow reading next to the fast one, each labeled with what it measures.
