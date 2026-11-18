# Cost comparison: self-hosted router vs frontier LLM APIs

Measured 2026-11-18T21:23:29Z on `https://finetunefleet-api.onrender.com` (commit `1a5b1a5`), Render free instance (0.1 CPU, 512 MB). Prices as of 2026-11-18; every input is in [`docs/cost_assumptions.json`](../cost_assumptions.json).

## Measured on the live server

| | Fine-tuned DistilBERT | Zero-shot flan-t5-small |
|---|---:|---:|
| Server model time per message, batch of 16 (p50 of 3 rounds) | 155 ms | 4,357 ms |
| Single call, warm: server model time p50 | 192 ms | 5,034 ms |
| Single call, warm: end-to-end p50 (laptop ↔ Render) | 1,664 ms | 6,304 ms |
| Model load when switching to it (one model resident) | 399 ms | 14,799 ms |
| Test macro-F1 [95% CI] | **0.965** [0.958, 0.972] | 0.669 [0.649, 0.686] |

## Monthly cost by volume

Frontier calls: an instruction listing the eight queues (~220 tokens) plus the message (measured mean 12.6 DistilBERT tokens, a ±30% proxy for the API's tokenizer), 6 output tokens, plus each model's billed reasoning tokens. Self-hosted capacity = seconds per month ÷ measured seconds per message, scaled by the plan's CPU share (assumes CPU-bound).

| Option | per 1,000 msgs | 10k / month | 100k / month | 1M / month | 10M / month |
|---|---:|---:|---:|---:|---:|
| Self-hosted, Render Free (0.1 CPU; capacity ≈ 17.0M msgs/month) | $0.0000 at capacity | $0 | $0 | $0 | $0 |
| Self-hosted, Render Starter (0.5 CPU; capacity ≈ 85.0M msgs/month) | $0.0007 at capacity | $7 | $7 | $7 | $7 |
| Self-hosted, Render Standard (1.0 CPU; capacity ≈ 170.0M msgs/month) | $0.0025 at capacity | $25 | $25 | $25 | $25 |
| Claude Haiku 5.5 API | $0.026 | $0 | $3 | $26 | $263 |
| Claude Sonnet 5.5 API | $0.525 | $5 | $53 | $525 | $5,252 |
| Claude Opus 5.5 API | $3.050 | $31 | $305 | $3,050 | $30,504 |

Frontier quality was **not** measured (sending the test set to a paid API was out of scope); the comparison is cost and speed only for those rows. Engineering time to build and operate the self-hosted pipeline is not included in its column, and it is the real cost at low volume.
