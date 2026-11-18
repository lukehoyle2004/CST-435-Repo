# Model card: FineTuneFleet support-message router

## Model details

| | |
|---|---|
| Run | `ft-lr5e-5-ep3` (the row marked deployed in `ftf_experiments`) |
| Base model | `distilbert-base-uncased` @ `12040accade4e8a0f71eabdb258fecc2e7e948be`: 66M parameters, masked-language-model pre-training plus distillation from BERT |
| Head | linear layer over the `[CLS]` representation → 8 queues (`id2label` fixed by `shared/queues.py`) |
| Training | Hugging Face `Trainer`, end to end; AdamW at 5e-5, 3 epochs, batch 32, linear schedule with 6% warm-up, weight decay 0.01, seed 42; evaluated and saved every epoch, and the epoch with the best **validation** macro-F1 was kept |
| Compute | 38.8 minutes on a laptop CPU (no GPU) |
| Chosen from | 5 seeded configurations (see the sweep table in the README); selected on validation, and it is also the best on test |
| Weights | GitHub release [`finetunefleet-v1`](https://github.com/lukehoyle2004/CST-435-Repo/releases/tag/finetunefleet-v1); SHA-256 of `model.safetensors`: `4748e601f4aff5647641035037ac43ba05acd04aa6ffde4e958369010e6bc21d`. `scripts/fetch_model.py` refuses any other file. |
| Training logs | [`docs/experiments/trainer_logs/ft-lr5e-5-ep3.json`](docs/experiments/trainer_logs/ft-lr5e-5-ep3.json) (loss every 25 steps, eval every epoch) |

**Baseline served next to it:** `google/flan-t5-small` @ `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`, zero-shot. Each of the eight queue labels is scored by length-normalized log-likelihood after an instruction prompt, then calibrated with content-free inputs (Zhao et al., 2021). The prompt was chosen on validation ([`zero_shot_prompt_selection.json`](docs/experiments/zero_shot_prompt_selection.json)). To fit the 512 MB server, its weights are stored in float16 and computed in float32. On all 3,076 test messages this served variant agrees with the float32 baseline the leaderboard scored ([`zero_shot_served_variant.json`](docs/experiments/zero_shot_served_variant.json)).

## Intended use

Suggest which of a digital bank's eight support queues should handle an English customer
message: card services, transfers, top-ups, cash & ATM, fees & FX, payments & disputes,
security & fraud, account & identity. Messages it is unsure about are flagged for a person:
under 60% confident, or the fraud queue at least 10% likely without winning.

**Out of scope.** Any decision *about* the customer: credit, account closure, a fraud verdict,
or refunds. Also out of scope: messages in other languages, voice transcripts, and banks whose
products do not match the Banking77 taxonomy, unless the model is re-evaluated on that data.

## Training and evaluation data

Banking77 (CC-BY-4.0), its 77 intents mapped to 8 queues by written rules, deduplicated across
splits: 8,952 train, 995 validation, 3,076 test. Full details are in
[docs/DATASET_CARD.md](docs/DATASET_CARD.md).

## Metrics

Held-out test set, 3,076 messages. The 95% intervals are percentile bootstrap (2,000 resamples,
seed 1234); the gain is a *paired* bootstrap on the same resamples.

| | Fine-tuned (deployed) | Zero-shot flan-t5-small |
|---|---|---|
| **Macro-F1** | **0.965** [0.958, 0.972] | 0.669 [0.649, 0.686] |
| Accuracy | 0.968 [0.961, 0.974] | 0.690 [0.673, 0.706] |
| **Fraud-queue recall** | **0.971** [0.948, 0.991] | 0.604 [0.543, 0.661] |
| Gain in macro-F1 | **+0.296** [+0.279, +0.315] | n/a |
| Laptop CPU, ms per message (batched) | 46.8 | 605.7 |

**Per queue (fine-tuned, test):**

| Queue | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| card_services | 0.988 | 0.976 | 0.982 | 760 |
| cash_atm | 0.975 | 0.975 | 0.975 | 199 |
| fees_fx | 0.971 | 0.965 | 0.968 | 479 |
| account_identity | 0.954 | 0.980 | 0.967 | 400 |
| top_ups | 0.972 | 0.955 | 0.963 | 358 |
| security_fraud | 0.947 | 0.971 | 0.959 | 240 |
| transfers | 0.974 | 0.940 | 0.957 | 400 |
| payments_disputes | 0.922 | 0.979 | 0.950 | 240 |

**Review policy on test** ([`review_policy.json`](docs/experiments/review_policy.json)):
* 1.3% of messages are flagged. Flagged messages are routed correctly only 44% of the time,
  against 97.4% for the rest, so the flag does find the doubtful ones.
* It catches only **22 of the 100 errors**. 87% of all answers are more than 99% confident,
  and 12 of the errors were.
* **6 of the 240 fraud reports were misrouted without a flag.**

## Limitations

* **Clean benchmark, messy reality.** Banking77 messages are short, single-intent and written
  by crowd workers. Expect lower accuracy on real traffic, and measure it before trusting the
  router.
* **Confident mistakes.** Most errors come with high confidence, so the review rule is a safety
  net, not a guarantee. Raising the threshold flags more messages, but it cannot catch an error
  the model is 99.9% sure of.
* **Where it confuses queues.** Payments & disputes has the lowest precision (0.92). Other
  queues' "declined" and "failed" messages land there, because it owns declined *card
  payments*: four declined transfers and two failed top-ups among them. Transfers has the lowest
  recall (0.94); 8 transfer messages went to top-ups and 6 to fees & FX, where "sending money
  in" and "a charge on my transfer" overlap.
* **Fixed label set.** A ninth queue needs new labels and retraining.
* **No frontier comparison of quality.** The baseline is a 77M-parameter zero-shot model. A
  frontier LLM would very likely score higher; its *cost* is compared in the README, but its
  accuracy was not measured.

## Ethical considerations

* **Who is hurt by an error.** A misrouted fraud report delays help to someone whose money is
  being taken. That is why fraud recall is reported separately, why a 10% fraud probability alone
  sends a message to a person, and why the six silent fraud misses are stated here instead of
  being averaged away.
* **Privacy.** The service never stores a customer's message. It logs a SHA-256 hash, the length,
  token count, the scores and the queue. The training data contains no real customers.
* **Automation boundary.** The router decides *who answers*, never *what the answer is*. A person
  handles every message.
