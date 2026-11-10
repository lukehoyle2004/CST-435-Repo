# Model Card: TrioFormer

TrioFormer serves four transformers behind one API. Three are pretrained checkpoints used
exactly as published (no fine-tuning); one is written from scratch in this repository.

## Model details

| Name in the API | Checkpoint (pinned Hub commit) | Archetype | Params | Task served |
|---|---|---|---|---|
| `bert` | `distilbert-base-uncased` @ `12040accade4e8a0f71eabdb258fecc2e7e948be` | encoder | 67.0 M | fill one `[MASK]`, top-k candidates |
| `gpt2` | `distilgpt2` @ `2290a62682d06624634c1f46a6ad5be0f47f38aa` | decoder | 81.9 M | continue text (greedy by default, `no_repeat_ngram_size=3`) |
| `t5` | `t5-small` @ `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` | encoder-decoder | 60.5 M | `summarize:` with beam search (default 2 beams) |
| `encoder_block` | `src/transformer.py`, weights `models/encoder_block.pt` (sha256 `4807bf7f17…`) | encoder (hand-written) | 17.6 k | word-level palindrome detection + attention maps |

* **Libraries:** torch 2.4.1 (CPU), transformers 4.44.2, fp32 by default (`TRIO_DTYPE`).
* **Provenance:** `/version` reports every checkpoint's configured revision and the commit
  actually loaded. Every logged inference stores `checkpoint@commit` and the serving git SHA.
* **Why distilled / small checkpoints:** the deployment target is Render's free instance
  (512 MB RAM, 0.1 CPU). The full `bert-base` / `gpt2` checkpoints would not leave headroom
  for torch itself. Even so, only one pretrained model is kept in memory at a time.

### The hand-written encoder

Two pre-norm blocks (D = 32, 4 heads, MLP width 64), sinusoidal positional encoding, masked
mean-pooling, linear head. Words are relabelled by first appearance (`dog bites man bites dog`
→ `1 2 3 2 1`), so it needs no vocabulary of real words. Trained with AdamW + one-cycle LR for
4,000 steps (batch 128) on 40,000 synthetic sequences of length 2–12, half palindromes. 60%
of the negatives are near misses (a palindrome with one word changed).

| Held-out (4,000 sequences) | 95.7% accuracy |
|---|---|
| length ≤ 4 | 100% |
| length 8 | 93.4% |
| length 12 | 90.6% |

Its parameters can be copied one-for-one into `torch.nn.TransformerEncoderLayer(norm_first=True)`.
Under matched initialization both train to the same loss to within 1.2 × 10⁻⁷
(`docs/benchmarks/block_vs_torch.json`).

## Intended use

* Teaching and internal evaluation: showing engineering leadership what an encoder, a decoder
  and an encoder-decoder are each good at, and what each costs to serve on modest hardware.
* A backend for Product 5B interfaces (the Concepts / My Encoder Block / Compare tabs).
* **Not intended** for production writing, factual question answering, summarizing documents
  someone will act on without reading the source, or any decision about a person.

## Evaluation and observed behavior

The pretrained checkpoints are not re-evaluated on benchmark datasets here; their published
model cards cover that. What this service measures is **latency** (logged per call to
Supabase, aggregated by `trio_benchmarks`) and **regression**: `tests/fixtures/reference_outputs.json`
pins the answers to six fixed inputs so a silent dependency or checkpoint change fails CI.

Observed while building it (all reproducible through the API):

* `bert`: "I drank a cup of [MASK] this morning." → coffee 0.54, tea 0.14, water 0.08.
  "The capital of France is [MASK]." → **marseille** 0.14, nantes 0.09. Paris is not first.
  DistilBERT ranks words that fit the sentence's form, and that is different from knowing
  facts.
* `gpt2`: fluent but invents facts. Continuing a paragraph about a Tuesday council vote, it
  wrote that the council "voted to expand it in November".
* `t5`: summaries are mostly extractive. It copies whole clauses from the input and lowercases
  them, which is useful to know before logging its output (see Privacy).
* `encoder_block`: correct on the examples above, but its heads spread attention almost
  uniformly (mirror-position weight ≈ 1/n). It solves the task without showing a readable
  anti-diagonal, so the API reports that honestly instead of claiming one.

## Latency (how to read it)

`latency_ms` is model compute only. `load_ms` is checkpoint loading that happened during that
request (0 when warm). `total_ms` is the whole server-side request. On a one-model-resident
instance, a "compare all three" interface pays a cold swap on almost every call, and
`trio_benchmarks` splits warm and cold calls for exactly that reason. Local dry-run medians,
warm: BERT ≈ 0.27 s, GPT-2 ≈ 1.3 s for 40 tokens, T5 ≈ 1.4–1.6 s for ~43 tokens. The deployed
numbers are in the engineering report.

## Privacy

Input text is never stored: only its SHA-256 hash and size (`trio_inputs`). Model outputs are
stored (`trio_inferences`, truncated to 1,000 characters), except BERT's filled-in paragraph,
of which only the candidate words are kept. Because T5 and GPT-2 can repeat input words in
their output, the raw tables are readable only by the server's service role. The public anon
key sees only the per-model aggregates in `trio_benchmarks`.

## Limitations and risks

* Small distilled models: weaker than the full checkpoints and far weaker than frontier LLMs.
  Outputs can be wrong, biased, or invented, and are shown without any fact checking.
* English only; inputs beyond 512 tokens are truncated (BERT/T5) or trimmed to the most recent
  context (GPT-2).
* The pretrained models inherit the biases of their web-scale training data (BookCorpus,
  English Wikipedia, OpenWebText, C4). No mitigation is applied beyond not using them for
  decisions about people.
* Latency on a 0.1-CPU shared instance is noisy and includes cold swaps. Treat it as a
  relative comparison between archetypes on the same hardware, not as each model's best case.
* The toy encoder's 95.7% is on synthetic data, and accuracy falls with length.
