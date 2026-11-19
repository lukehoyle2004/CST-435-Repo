# Individual Contribution: Luke Hoyle

CST-435 / AIT-204 Topic 6, Products 6 "FineTuneFleet" and 6B "FleetDeck" (CLC). Halo Group D2
lists one member, so every task below was planned, built, tested, deployed and documented by me.
The commit history of `Deep-Learning/Project-Templates/Topic_6_fine-it` and `Topic_6B_fleet-deck`
records the work phase by phase.

## Key points

- Chose the customer scenario (a digital bank's eight support queues) and wrote the 77-intent →
  8-queue routing contract as three testable rules.
- Built a reproducible loader for Banking77 at a pinned revision. It deduplicates across splits,
  splits by intent with stratification, and loads 13,023 labelled documents into Supabase.
- Fine-tuned DistilBERT with the Hugging Face `Trainer` across five seeded configurations, kept the best
  epoch on validation, and persisted every run with bootstrap 95% CIs to `ftf_experiments`.
- Built a fair zero-shot baseline: label-likelihood scoring, contextual calibration, and a prompt
  chosen on validation. It went from 0.02 to 0.67 macro-F1 before any comparison was made.
- Designed the Supabase schema: the corpus with split labels, experiments with intervals and a
  single deployed row, predictions joined to their experiment, two views, and RLS limiting the
  anon key to aggregates.
- Built the FastAPI service (eight endpoints). It logs every prediction before responding,
  degrades loudly during an outage, and can only serve hash-verified weights from a GitHub release.
- Fitted both models into Render's 512 MB. Diagnosed the out-of-memory crash, served
  flan-t5-small with float16 weights and float32 compute (100% agreement on the test set), and
  rejected bf16 and int8 with measurements.
- Wrote 56 API-side tests (behavioural, regression fixture, outage, migration, live Supabase,
  training reproducibility) and 28 dashboard tests, plus both CI workflows.
- Built FleetDeck, the Streamlit dashboard: concepts, a side-by-side comparison, a sortable
  leaderboard with CIs, and a live cost and latency comparison.
- Wrote the README, model card, dataset card, pre-training objectives comparison, engineering
  report, ethics and worldview reflection.

## Every task identified in the project

1. Read the Topic 6 / 6B rubrics; choose the scenario, dataset, model and baseline; plan the phases.
2. Write the queue contract (`shared/queues.py`) and pin dataset and checkpoint revisions.
3. Write `scripts/prepare_data.py`: clean, deduplicate within and across splits, stratify validation.
4. Author `db/migrations/001_init.sql`: documents, experiments, predictions, views, RLS, grants.
5. Create the Supabase project, apply the migration, and load the corpus and experiments.
6. Write `src/finetune.py` (Trainer, seeded, best-on-validation) and `configs/sweep.yaml`.
7. Write `src/zero_shot.py`: label-likelihood scoring, contextual calibration, prompt variants.
8. Write `src/metrics.py`: fast macro-F1, fraud recall, bootstrap CIs, paired delta.
9. Run prompt selection and the five-configuration sweep (~2 hours on CPU); commit trainer logs.
10. Write `docs/PRETRAINING_OBJECTIVES.md`: MLM, CLM, span corruption and InfoNCE derivations;
    the suitability matrix; when zero-shot wins.
11. Build the FastAPI service: `/predict`, `/predict_zero_shot`, `/predict_batch`, `/experiments`,
    `/metrics`, `/queues`, `/healthz`, `/version`.
12. Log every prediction, joined to its experiment, before responding; handle Supabase outages.
13. Publish the checkpoint as a GitHub release; write `scripts/fetch_model.py` with hash checking.
14. Deploy to Render with a Blueprint; diagnose the out-of-memory crash and fix memory use.
15. Verify the served zero-shot variant on the full test set (`eval_served_zero_shot.py`).
16. Measure the review policy on the test set and report what it misses.
17. Write the test suite and the FineTuneFleet CI workflow.
18. Benchmark the live service and build the frontier-LLM cost comparison.
19. Build FleetDeck (Product 6B) with its tests, CI and deployment.
20. Write the README, architecture diagram, model card, dataset card and privacy statement.
21. Write the engineering report, ethics section and worldview reflection.
22. Record the video walkthrough (FleetDeck demo plus the README) and link it in the README.
