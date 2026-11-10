# Cost comparison (generated 2026-11-10)

Latency source: **benchmark script summary (database not reachable)** (`docs/benchmarks/latency_local.json`). Assumptions and prices: `docs/cost_assumptions.json` -- A frontier model needs an instruction (prompt_overhead_tokens) and, on current models, spends some billed reasoning tokens even at low effort (reasoning_tokens_per_call, an estimate). Token counts from our tokenizers are used as a proxy for the frontier tokenizer (roughly +-30%).

## Cost per 1,000 calls (USD)

| archetype | mean server ms | tokens in / out | claude-haiku-5-5 | claude-sonnet-5-5 | claude-opus-5-5 | Render starter (full use) | Render standard (full use) |
|---|---|---|---|---|---|---|---|
| encoder (fill-mask) | 381 | 94 / 1 | $0.091 | $1.819 | $3.637 | $0.001 | $0.004 |
| decoder (continue) | 1450 | 91 / 40 | $0.110 | $2.202 | $4.404 | $0.004 | $0.014 |
| encoder-decoder (summarize) | 1490 | 104 / 43 | $0.113 | $2.257 | $4.515 | $0.004 | $0.014 |

## Break-even volume (calls per month)

Below this volume the pay-per-call API is cheaper than keeping the Render instance running.

| archetype | starter vs claude-haiku-5-5 | starter vs claude-sonnet-5-5 | starter vs claude-opus-5-5 | standard vs claude-haiku-5-5 | standard vs claude-sonnet-5-5 | standard vs claude-opus-5-5 | capacity at measured latency |
|---|---|---|---|---|---|---|---|
| encoder (fill-mask) | 76,979 | 3,849 | 1,924 | 274,927 | 13,746 | 6,873 | 6,903,073 |
| decoder (continue) | 63,579 | 3,179 | 1,589 | 227,066 | 11,353 | 5,677 | 1,812,164 |
| encoder-decoder (summarize) | 62,020 | 3,101 | 1,551 | 221,500 | 11,075 | 5,538 | 1,763,877 |

A break-even volume larger than the capacity column means the instance can never serve enough calls one at a time to beat that API on price.
