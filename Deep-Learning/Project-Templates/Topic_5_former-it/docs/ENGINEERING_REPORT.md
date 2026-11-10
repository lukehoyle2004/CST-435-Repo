# TrioFormer: Engineering Report

Luke Hoyle · CST-435 / AIT-204 Topic 5, Product 5 · Halo Group D2 (sole member)

> **Before submitting:** the latency table in §1 currently holds the **local dry run**
> (`docs/benchmarks/latency_local.json`). Replace it with the rows of the Supabase
> `trio_benchmarks` view after running `scripts/benchmark_latency.py` against the Render
> service, and rerun `scripts/cost_compare.py`.

## 1. Engineering report: three archetypes, one service

TrioFormer sends the same paragraph to three pretrained transformers: DistilBERT (encoder),
DistilGPT-2 (decoder) and T5-small (encoder-decoder). It also serves an encoder I wrote from
scratch. Each call is timed on the server and logged to Supabase before the response leaves.

**(a) What each is natively good at.** The three share the same building block: multi-head
self-attention plus an MLP, wrapped in residual connections. What separates them is the
attention mask and which way the information flows. BERT's encoder lets every token attend
to every other token in both directions, so its representation of a word uses the whole
sentence. That is ideal for understanding tasks. Asked to fill "make downtown streets
[MASK] for commuters" it ranks *safer* first (0.82). It produces no new text, though, and its
knowledge is shallow: for "The capital of France is [MASK]." it prefers *marseille*. GPT-2's
decoder applies a causal mask, so each position sees only what came before it. That makes it a
natural generator of fluent continuations, and also a confident inventor of facts: continuing
a paragraph about a Tuesday council vote, it reported a vote "in November." T5 combines the two.
A bidirectional encoder reads the source once, and a causal decoder writes a new sequence while
cross-attending to that encoding. Its summaries stay close to the source, often copying whole
clauses. That makes them faithful but mostly extractive.

**(b) Compute profile.** The architecture predicts the cost, and the logged latencies confirm it.

| Model (warm, median) | Tokens in → out | Server latency | Cold checkpoint load |
|---|---|---|---|
| encoder (DistilBERT) | ~94 → 1 | ~0.27 s | 0.2–1.1 s |
| decoder (DistilGPT-2) | ~91 → 40 | ~1.3 s | ~0.14–0.23 s |
| encoder-decoder (T5-small) | ~104 → ~43 | ~1.4–1.6 s | ~0.2–0.25 s |
| hand-written encoder | 5 words → label | ~1.3 ms | always resident |

The encoder runs **one** forward pass, whatever the answer is, so its cost grows only with
input length. The decoder runs one forward pass **per generated token**. Even with key/value
caching, forty tokens cost roughly five times BERT's single pass. T5 pays for an encoder pass,
then a decoder pass per token for each of two beams. On a 0.1-CPU, 512 MB instance a second
cost appears that the table separates out: only one checkpoint fits in memory, so an
interface that compares all three models swaps checkpoints on almost every call. The view
records these as `cold_start` rows with their own `load_ms`. In the interleaved benchmark,
the swaps added roughly 0.15–0.4 s per call on top of the model compute. The cost comparison
(`docs/benchmarks/cost_comparison.md`) puts this in money. At list prices, a frontier API costs
about $0.09–0.11 per thousand of these calls on Claude Haiku 5.5 and $3.6–4.5 on Claude Opus
5.5. A $7-per-month always-on instance breaks even against Opus at roughly 1,550–1,900 calls
a month, and against Haiku only at about 62,000–77,000. The honest summary is that small
self-hosted models win on cost only at volume, never on quality.

**(c) When I would choose each.** An **encoder** for a help desk that routes thousands of
tickets a day by topic: one cheap pass per ticket, a bounded label, no invented text. A
**decoder** for an autocomplete feature in an internal writing tool, where fluency matters, a
person reviews every suggestion, and the user controls the output length. An
**encoder-decoder** for turning long meeting transcripts into short summaries that must stay
anchored to what was actually said, with a link back to the source for anyone who acts on them.

## 2. Reflection: a lowered bar and new responsibilities

This product is a thin layer: about a thousand lines of glue around three models that took
research labs years and large compute budgets to build. One student shipped it in a week on a
free tier. The bar to *ship* a transformer-powered tool is now low enough that almost anyone
can do it. The bar to ship one *responsibly* has not dropped, and that leaves three
responsibilities with whoever ships it.

**Auditability.** Someone must be able to see what the service actually said, and how fast and
at what cost. For that reason `trio_inferences` records, for every call, the model's output, the
server-side latency and load time, token counts, the source of the call, and the time. It is
written *before* the response is sent, and an outage is flagged (`logged: false`) instead of
hidden. **Provenance.** Every row records `checkpoint@commit` and the serving git SHA, and the
checkpoints are pinned to exact Hub commits, so a silent upstream change cannot alter what was
served. **Disclosure.** The documentation shows the models' failures (Marseille, an invented
November vote), not only their wins.

The table also leaves things out on purpose. It never records the submitted paragraph, only its
SHA-256 hash and length. It stores no IP address, account, cookie or user agent. For BERT it keeps the
candidate words, not the filled-in paragraph. One gap cannot be designed away. Generated
outputs can repeat the input, because T5 copies clauses, so the raw tables are readable only by
the server, and the public key sees nothing but per-model aggregates. That trade-off is
stated in the privacy statement, not buried.

## 3. Worldview reflection

"Everyone to whom much was given, of him much will be required" (Luke 12:48, ESV). I was given
a great deal I did not earn: openly released models, free hosting, free tooling. Capability
arrived first, and in this verse accountability is meant to follow it. **The responsibility I
accept** by shipping TrioFormer is to tell the truth about it. It logs every answer before
giving it, keeps no one's words, pins exactly which model spoke, and publishes the
embarrassing outputs and the real costs alongside the flattering ones. **The responsibility I
am deferring** is the content itself. These models are unfiltered. They can produce biased
or false text, and this service does not check outputs for harm or for truth. I am
handing that duty to the Product 5B interface and to whoever uses the demo, and I say so in the
model card instead of implying the models are safe. Being a good steward here means not
calling a small model wise just because it sounds fluent.
