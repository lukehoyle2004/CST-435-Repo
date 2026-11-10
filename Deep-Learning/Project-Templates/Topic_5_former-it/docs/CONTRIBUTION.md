# Individual Contribution: Luke Hoyle

CST-435 / AIT-204 Topic 5, Product 5 "TrioFormer" (CLC). Halo Group D2 lists one member, so
every task below was planned, built, tested, deployed and documented by me. The commit history
of `Deep-Learning/Project-Templates/Topic_5_former-it` records the work phase by phase.

## Key points

- Hand-wrote scaled-dot-product attention, multi-head attention, sinusoidal positional encoding
  and a pre-norm transformer encoder block (`src/transformer.py`), typed and shape-annotated.
- Proved it correct: 19 unit tests (shape, attention sums to 1, gradient flow, permutation
  equivariance only without positional encoding) plus exact matches against `torch.nn`.
- Benchmarked it against `torch.nn.TransformerEncoderLayer` under matched initialization: the
  loss curves agree to 1.2 × 10⁻⁷ over 300 steps, with a single-step timing comparison.
- Trained a toy palindrome encoder from my own block (95.7% held-out accuracy) so the teaching
  endpoint shows real attention maps. I reported honestly that its heads are diffuse.
- Designed the Supabase data layer: hashed `trio_inputs`, `trio_inferences` with server-side
  latency and cold-start load time, a concurrently refreshed `trio_benchmarks` materialized
  view, one-transaction logging RPC, and RLS that exposes only aggregates to the anon key.
- Built the FastAPI service with the seven required endpoints, logging before every response,
  graceful Supabase-outage handling, a one-model-resident registry for the 512 MB free tier,
  and checkpoints pinned to exact Hub commits.
- Measured memory (fp32 vs bf16, and rejected int8 quantization with evidence) and latency
  (grouped vs interleaved), and built a frontier-LLM cost comparison with break-even volumes.
- Wrote 78 tests (end-to-end on the real checkpoints, outage paths, regression fixture) and the
  GitHub Actions workflow.
- Wrote the README, architecture diagram, archetype matrix, model card, privacy statement and
  engineering report with the ethics and worldview reflection.

## Every task identified in the project

1. Read the rubric (resolving its tracked changes) and the Halo assignment, and plan the phases.
2. Fork the Three-Cloud template; put the L&D product pitch at the top of the README.
3. Author `db/migrations/001_init.sql`: `trio_inputs`, `trio_inferences`, `trio_benchmarks`
   materialized view, logging and refresh functions, row-level security and grants.
4. Apply the migration to a new Supabase project.
5. Write `src/transformer.py`: scaled-dot-product attention, multi-head attention, sinusoidal
   positional encoding, pre-norm encoder block, encoder stack, weight transfer helpers.
6. Unit-test the block: output shape, attention sums to 1, gradient flow, permutation equivariance.
7. Benchmark the block against `torch.nn.TransformerEncoderLayer` under matched initialization
   (loss curves and single-step timing).
8. Train the toy encoder that backs `/encoder_attention` and export its weights.
9. Choose checkpoints that fit Render's free tier; measure memory and latency; pin revisions.
10. Implement `POST /encoder_attention`, `/bert_fillmask`, `/gpt2_continue`, `/t5_summarize`.
11. Implement `GET /benchmarks`, `/healthz` (model loader and data client) and `/version`.
12. Log input hash, output and server-side latency to Supabase before every model response.
13. Handle a Supabase outage and a failed checkpoint load gracefully.
14. Write the test suite (FakeDB, outage fixture, end-to-end endpoint tests, regression fixture).
15. Add the GitHub Actions workflow with cached checkpoints.
16. Deploy the FastAPI service to Render (`render.yaml`, build-time model prefetch).
17. Run the latency benchmark against the deployed service to populate the materialized view.
18. Build the frontier-LLM cost comparison.
19. Write the model card, archetype-recommendation matrix and architecture diagram.
20. Write the engineering report: archetype comparison, reflection, worldview reflection.
21. Record the 3–5 minute video walkthrough and link it in the README.
