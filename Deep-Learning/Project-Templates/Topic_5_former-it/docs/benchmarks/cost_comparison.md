# Cost comparison (generated 2026-11-10)

Latency source: **Supabase trio_benchmarks view** (`docs/benchmarks/latency_render.json`). Assumptions and prices: `docs/cost_assumptions.json` -- A frontier model needs an instruction (prompt_overhead_tokens) and, on current models, spends some billed reasoning tokens even at low effort (reasoning_tokens_per_call, an estimate). Token counts from our tokenizers are used as a proxy for the frontier tokenizer (roughly +-30%).

## Cost per 1,000 calls (USD)

| archetype | mean server ms | tokens in / out | claude-haiku-5-5 | claude-sonnet-5-5 | claude-opus-5-5 | Render starter (full use) | Render standard (full use) |
|---|---|---|---|---|---|---|---|
| encoder (fill-mask) | 3676 | 91 / 1 | $0.091 | $1.812 | $3.625 | $0.010 | $0.035 |
| decoder (continue) | 11600 | 88 / 40 | $0.110 | $2.196 | $4.392 | $0.031 | $0.110 |
| encoder-decoder (summarize) | 12481 | 100 / 42 | $0.112 | $2.246 | $4.492 | $0.033 | $0.119 |

## Break-even volume (calls per month)

Below this volume the pay-per-call API is cheaper than keeping the Render instance running.

| archetype | starter vs claude-haiku-5-5 | starter vs claude-sonnet-5-5 | starter vs claude-opus-5-5 | standard vs claude-haiku-5-5 | standard vs claude-sonnet-5-5 | standard vs claude-opus-5-5 | capacity at measured latency |
|---|---|---|---|---|---|---|---|
| encoder (fill-mask) | 77,246 | 3,862 | 1,931 | 275,877 | 13,794 | 6,897 | 714,869 |
| decoder (continue) | 63,758 | 3,188 | 1,594 | 227,707 | 11,385 | 5,693 | 226,546 |
| encoder-decoder (summarize) | 62,339 | 3,117 | 1,558 | 222,638 | 11,132 | 5,566 | 210,568 |

A break-even volume larger than the capacity column means the instance can never serve enough calls one at a time to beat that API on price.
