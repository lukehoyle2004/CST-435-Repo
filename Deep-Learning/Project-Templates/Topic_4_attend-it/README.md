# Seq-Sight — a Sequence-Modeling Service with Attention Explanations

| | Link |
|---|---|
| **Streamlit UI** | https://seq-sight-luke.streamlit.app |
| **Render API** (docs at `/docs`) | https://seq-sight-api-lp8s.onrender.com |
| **Supabase (Postgres)** | https://owovmircxxpypggamqsy.supabase.co (project `see-seq`, ref `owovmircxxpypggamqsy`) |
| **GitHub** | https://github.com/lukehoyle2004/CST-435-Repo/tree/main/Deep-Learning/Project-Templates/Topic_4_attend-it |
| **Video walkthrough (Luke Hoyle)** | _add link_ |
| **Engineering report** | [docs/ENGINEERING_REPORT.md](docs/ENGINEERING_REPORT.md): attention derivation, matched comparison, interpretation, ethics |
| **Model card** | [MODEL_CARD.md](MODEL_CARD.md) |

CST-435 Topic 4, Product 4. Author: Luke Hoyle. (The folder keeps the template's name,
`Topic_4_attend-it`; the product is Seq-Sight.)

## Product pitch

A streaming service gets thousands of written reviews a week and wants the unhappy ones in
front of a person fast. Seq-Sight reads a review and says **positive**, **negative**, or
**uncertain**. It shows *which words it weighed* as a heatmap, and every answer is kept in a
public log you can reopen later, heatmap included. The text itself is never stored.

Under the hood are two classifiers trained under identical conditions on 25,000 IMDB
reviews: a plain LSTM, and the same LSTM with a hand-written scaled-dot-product attention
block. The app lets you watch them disagree. Attention is more accurate (87.65% vs 86.64%),
better calibrated, and much better on long reviews. Its Failure Gallery also shows it
being 100% sure and wrong about sarcasm, which is why the service says "uncertain" instead
of guessing when confidence drops below 75%.

## Architecture

```
 Browser
    │
    ▼
┌───────────────────────────┐  POST /predict {text, model}  ┌───────────────────────────────┐
│ Streamlit Community Cloud │ ─────────────────────────────► │ FastAPI on Render.com          │
│ ui/app.py (thin client)   │ ◄───────────────────────────── │ api/main.py                    │
│ no torch, no tokenizer    │  JSON: label, probabilities,  │ lstm.pt + attention.pt + vocab │
│ 6 tabs                    │  tokens, attention[], flags   │ shared/text.py tokenizer       │
└─────────────┬─────────────┘                               └───────────────┬───────────────┘
              │ anon key, SELECT only (RLS)                                 │ service-role key
              │ seq_runs, seq_predictions ⨝ seq_runs                        │ INSERT seq_predictions
              ▼                                                             ▼ (hash + attention)
          ┌──────────────────────────────────────────────────────────────────────┐
          │ Supabase Postgres: seq_runs · seq_predictions · view seq_model_comparison │
          │ no review text anywhere                                               │
          └──────────────────────────────────────────────────────────────────────┘
```

Training runs offline on a laptop CPU: `scripts/train.py --kind {lstm,attention}`, then
`scripts/analyze.py` for the comparison and gallery. The exported weights live in `models/`,
and `db/seed_runs.py` writes one `seq_runs` row per run. Tables are prefixed `seq_` so they
can share a Supabase project with See-Sense without touching its tables.

## Results (IMDB test set, 25,000 reviews)

Matched runs: same seed (0), split, vocabulary, batch order, 6 epochs, batch 64, Adam 2e-3,
max 256 tokens and CPU. The only difference is the attention block (+8,320 parameters).

| | Plain LSTM | LSTM + attention |
|---|---|---|
| Test accuracy (95% CI) | 86.64% (86.22–87.06) | **87.65% (87.24–88.05)** |
| Macro-F1 | 0.866 | **0.876** |
| Calibration error (ECE) | 0.061 | **0.043** |
| Wrong at ≥ 90% confidence | 1,419 | **976** |
| Accuracy on reviews > 256 tokens | 81.42% | **83.42%** |
| Validation accuracy after epoch 1 | 70.9% | **86.3%** |
| Best epoch | 6 / 6 | 4 / 6 (then overfits) |

McNemar's test on the 2,450 reviews where they disagree: p = 4.0 × 10⁻⁷.

**Two concrete differences, each tied to the architecture:**
1. **Long-range dependencies.** The plain LSTM classifies from its last hidden state, so a
   cue at word 40 must survive 200+ gated updates. Attention gives every state a one-step
   path to the output. The gap is largest on the longest reviews (+2.0 points above 256
   tokens, versus +0.5 below 100).
2. **Optimization and calibration.** The same short path carries gradients straight to early
   words: attention reached 86% validation accuracy in one epoch, against 71% for the LSTM.
   Pooling over all states also makes it less swayable by one late phrase: 31% fewer
   confident mistakes and a lower ECE.
3. **Parallelism, honestly.** Attention scores are computed for all positions at once. Here
   they sit on top of an LSTM, though, so training time was the same (107 vs 103 minutes);
   the speed-up needs attention to *replace* the recurrence.

Full analysis: [docs/ENGINEERING_REPORT.md §4](docs/ENGINEERING_REPORT.md#4-lstm-vs-attention-matched-comparison).

## Streamlit tabs

- **Concepts:** derivation of scaled-dot-product attention with masking, the worked numeric
  example, a √d_k saturation table, and attention compared with the recurrent step.
- **Try It:** type a review and run either model or both side by side. Shows the attention
  heatmap over the words, the top words, the probabilities and the guardrail flags. Below
  that, six held-out IMDB test reviews that can be run live.
- **Baseline vs. Attention:** reads `seq_runs` from Supabase. Headline numbers with 95%
  intervals, accuracy by length, training curves, the disagreement table, per-class F1,
  accuracy vs. abstention, and reviews only one model got right.
- **Recent Predictions:** queries `seq_predictions` with the anon key. Filters, counts, and a
  **drilldown** into any past prediction that re-draws its stored attention vector. It shows
  the words for public IMDB examples and the positions for typed text, which was never stored.
- **Failure Gallery:** the attention model's 12 most confident mistakes, with heatmaps and
  the three patterns behind them.
- **Model Card:** this repo's [MODEL_CARD.md](MODEL_CARD.md).

**Cold start:** Render's free tier sleeps. The two database tabs render immediately while a
banner wakes the API (up to a minute). If the API or Supabase is down, the tabs say so
instead of crashing.

## API

| Method | Path | |
|---|---|---|
| POST | `/predict` | `{text, model: "attention"\|"lstm"}` returns the label, probabilities, tokens, out-of-vocabulary marks, `attention[]` (one weight per token), top tokens, `verdict`, `abstained`, `flags`, `text_sha256`, `prediction_id` and `logged` |
| POST | `/predict_example` | `{test_index, model}`: a public IMDB test review from the gallery |
| GET | `/gallery` | the examples, Failure Gallery, and one-model-right reviews (precomputed) |
| GET | `/predictions`, `/predictions/{id}` | the prediction log; the detail view includes the attention vector |
| GET | `/runs`, `/runs/{run_id}` | the training runs |
| GET | `/models`, `/classes`, `/healthz`, `/version` | metadata; `/healthz` stays 200 when the database is down |

If Supabase is unreachable, `/predict` still answers with `logged: false`, and the database
endpoints return 503.

## Sequence representation

`shared/text.py`, used by training, the API and the tests:
1. **Normalize:** lowercase, `<br />` → space, unescape HTML.
2. **Tokenize:** words, with contractions kept whole (`don't`), plus `!` and `?`.
3. **Vocabulary:** from the training split only; `<pad>`=0, `<unk>`=1, top 20,000 words
   seen at least twice. Its SHA-256 is checked at API start-up.
4. **Truncate** to the first 256 tokens.
5. **Right-pad** each batch and run the LSTM on `pack_padded_sequence`.
6. **Mask:** a boolean padding mask sets padded scores to −∞ before the softmax.

Details and tests: [report §2](docs/ENGINEERING_REPORT.md#2-sequence-representation).

## Dataset card (IMDB Large Movie Review Dataset)

| | |
|---|---|
| Source | Maas et al. (2011), Stanford AI Lab; downloaded as the `stanfordnlp/imdb` parquet files from Hugging Face (`scripts/download_imdb.py`) |
| Size | 50,000 labeled English movie reviews: 25,000 train and 25,000 test, each split 50/50 positive/negative. There is also an unlabeled split, which we do not use. |
| Labels | from the star rating: ≤ 4 of 10 = negative, ≥ 7 of 10 = positive. **5–6 star (neutral) reviews are excluded.** |
| Split used | official train → 22,500 train / 2,500 validation (seeded shuffle, seed 0); official test (25,000) for every reported number |
| Collection | at most 30 reviews per movie; train and test cover disjoint movies; reviews from IMDB up to 2011 |
| Length | median 174 tokens (test set, our tokenizer); 28.2% of test reviews exceed 256 tokens and are truncated |
| Known issues | no neutral class; English only; movies only; dated vocabulary and references; some labels disagree with the text (see the Failure Gallery); sarcasm common; HTML artifacts (handled in normalize) |
| People | reviews are written by IMDB users and may name actors and directors; no user ids are included |
| License | released by the authors for research use; cite Maas et al. (2011) |
| Not committed | the parquet files live in `data/` (gitignored); `models/` holds only weights, vocabulary and derived statistics, plus the first 256 tokens of the 25 test reviews shown in the gallery |

## Data persisted / not persisted

| Persisted (Supabase `seq_predictions`) | Never persisted |
|---|---|
| SHA-256 hash of the submitted text | the review text, or any of its words |
| token counts: scored, before truncation, unknown | the token list |
| predicted label, confidence, both probabilities | IP address, user agent, cookies, session id |
| **the attention vector** (one number per scored token) | any account or user identity (there are no accounts) |
| guardrail result: abstained, flag codes | |
| model, serving run id, latency, timestamp | |
| for gallery examples only: the public IMDB test index | |

`seq_runs` holds training metadata and metrics only.

### Privacy statement

Seq-Sight never stores what you type. The API tokenizes your review in memory, scores it,
and writes only a SHA-256 hash, counts, the answer and the attention numbers. The text and
the tokens are discarded when the request ends. The attention vector is numbers by
position, so on its own it does not reveal the words. The public Recent Predictions tab can
read these rows because they contain nothing that identifies you or your review; the
public (anon) key can only read, and writes need the server's secret key.

Two honest caveats:
- **Hashes of short texts can be guessed.** If you type a very short, common phrase ("great
  movie"), someone could hash candidate phrases and find a match. Do not enter anything
  private.
- **Render keeps short-lived request logs** that may record request metadata. The app
  itself logs no text.

## Run locally

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt                 # CPU-only torch
pytest -q                                           # live Supabase test skips without .env

python -m scripts.download_imdb                     # only needed to retrain (≈ 41 MB)
python -m scripts.train --kind lstm --threads 6     # ≈ 17 min/epoch on a laptop CPU
python -m scripts.train --kind attention --threads 6
python -m scripts.analyze                           # runs.json + gallery.json

copy .env.example .env                              # SUPABASE_URL + SUPABASE_SERVICE_KEY
copy ui\.streamlit\secrets.toml.example ui\.streamlit\secrets.toml
uvicorn api.main:app --reload --port 8000           # terminal 1
cd ui && streamlit run app.py                       # terminal 2 (reads ui/.streamlit/secrets.toml)
```

## Deploy

1. **Supabase:** in the SQL Editor run `db/migrations/001_init.sql`, then run
   `python -m db.seed_runs` (needs `.env`). It can share See-Sense's project, since
   every object is prefixed `seq_`.
2. **Render:** New → Blueprint → this repo, with `render.yaml` (it sets `rootDir`, Python
   3.11.9, and one torch thread). Add `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`.
3. **Streamlit Community Cloud:** main file
   `Deep-Learning/Project-Templates/Topic_4_attend-it/ui/app.py`, Python 3.11. Paste
   `API_URL`, `SUPABASE_URL` and `SUPABASE_ANON_KEY` into Secrets.
4. Put the three URLs in the table at the top of this README.

## Tests

`pytest -q` runs:
- **Hand-written attention** (`test_attention.py`): the worked example, a match against
  PyTorch with masks, zero weight on padding, √d_k variance, and saturation.
- **Sequence representation** (`test_text.py`, `test_imdb.py`, `test_model.py`): the
  tokenizer, vocabulary, padding and masks; packed LSTM invariance to padding; matched
  parameter counts; a toy task where attention must find the deciding token.
- **Input length and attention shape** (`test_api.py`):
  - rejects empty, untokenizable and over-20,000-character input
  - truncates to 256 tokens
  - returns one weight per token, summing to 1
- **Guardrails** (`test_api.py`), and logging of the hash plus attention, never the text.
- **Regression** (`test_regression.py`): both exported models must reproduce
  `tests/fixtures/reference.json` (tokens, probabilities and attention to 1e-5).
- **Supabase round trip** (`test_supabase_roundtrip.py`): `/predict` → a `seq_predictions`
  row with its attention, linked to `seq_runs`. It runs only with a `.env`.
- **Outage path:** `/predict` still answers, the database endpoints return 503, and
  `/healthz` stays up.
- **Database contract** (`test_schema.py`): `runs.json` and logged rows match the
  migration's columns, RLS is on, and every object is `seq_`-prefixed.
- **Streamlit** (`test_ui_app.py`, `test_ui_viz.py`): the real `ui/app.py` runs headlessly
  against the API, exercising all six tabs, Try It, drilldowns and the cold-start outage;
  the heatmap HTML escapes user text.

CI: `.github/workflows/seq-sight-tests.yml` on every push that touches this folder.

## Individual contribution (Luke Hoyle)

- Wrote the tokenizer, vocabulary, padding masks and IMDB loader in `shared/`, with tests
- Hand-wrote scaled-dot-product attention and the matched LSTM / LSTM + attention models
- Trained both models under matched conditions and wrote the comparison and Failure Gallery analysis
- Wrote the Supabase migration (`seq_runs`, `seq_predictions` with attention vectors, the view, RLS)
- Built the FastAPI service with guardrails and hash-only logging
- Built the six-tab Streamlit client with heatmaps and the Recent Predictions drilldown
- Wrote the pytest suite, CI, model card, dataset card, report and this README
- Deployed all three clouds

### Task list

- [x] Tokenization, vocabulary, padding and masks in `shared/`, unit-tested
- [x] Hand-written scaled-dot-product attention, unit-tested against a worked example and PyTorch
- [x] Train the plain LSTM and the LSTM + attention model with matched seed and budget
- [x] Comparison analysis (length buckets, calibration, McNemar, abstention) and Failure Gallery
- [x] SQL schema with attention vectors, comparison view and RLS; seed both runs
- [x] FastAPI `/predict` with attention, guardrails and hash-only logging
- [x] Six Streamlit tabs, including the Recent Predictions drilldown heatmap
- [x] pytest suite (input length, attention shape, regression, Supabase round trip, outage, UI) and CI
- [x] Deploy Supabase, Render and Streamlit Community Cloud; add the URLs above
- [x] README, model card, dataset card, privacy statement, engineering report
- [ ] Record the video walkthrough
