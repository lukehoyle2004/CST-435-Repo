# FineTuneFleet: Engineering Report

CST-435 / AIT-204 Topic 6, Product 6. Author: Luke Hoyle.

## 1. Engineering report: when fine-tuning pays, and how to prove it

**The question.** A bank can route support messages three ways: by having a person read them,
by prompting a model that was never trained on the task, or by fine-tuning a small model on its
own labelled history. FineTuneFleet builds the third option and serves the second next to it,
so the comparison is made on the same messages, the same server and the same metric.

**Why an encoder.** Routing is single-label classification of short messages (median 13
word-pieces), on a CPU, in 512 MB. A masked-LM encoder (DistilBERT) sees the whole message in
both directions in a single forward pass, and its `[CLS]` features separate with a linear head.
A decoder would read left to right and need far more parameters to classify as well by prompting.
The derivations and the task-suitability matrix are in
[PRETRAINING_OBJECTIVES.md](PRETRAINING_OBJECTIVES.md).

**Evaluation design.**
* **The data.** One frozen test split of 3,076 messages, deduplicated against training data (25
  overlapping messages removed), used once per run.
* **Model selection.** The deployed run was chosen on a separate validation split, and the
  zero-shot prompt was chosen there too.
* **Metrics.** Macro-F1 is the headline because the queues are unbalanced. Fraud recall is
  reported on its own because that queue's errors cost the most.
* **Uncertainty.** Every number carries a 95% bootstrap interval. The fine-tuning gain is a
  *paired* bootstrap: both models are rescored on the same resampled messages, which removes the
  shared noise of "an easy resample" from the difference.

**Results.**
* **The gain.** Fine-tuned macro-F1 is 0.965 [0.958, 0.972], against 0.669 [0.649, 0.686]
  zero-shot. The gain is +0.296 [+0.279, +0.315].
* **The sweep.**
  * Training the whole encoder beats training only the head by 0.16.
  * The learning rate (2e-5 vs 5e-5) is *not* distinguishable on macro-F1, but it is on fraud
    recall (0.929 vs 0.971).
  * One epoch reaches 0.933 in a third of the time.
* **Data matters most.** Cutting the training set to 1,000 messages cost 0.16 macro-F1. It
  dropped fraud recall to 0.35, which is the strongest argument in the project for collecting
  labels.

**Making the baseline fair.** The first zero-shot run was meaningless: uncalibrated
flan-t5-small sent every validation message to one queue. Rather than report that and declare
victory, I fixed it in three ways:
* scored the labels by likelihood, so the model always names a real queue;
* applied contextual calibration, dividing out its bias on content-free inputs;
* chose among six prompt variants on validation.

That raised the baseline from 0.02 to 0.67. The fine-tuned model still wins by 0.30, and now the
number means something.

**Fitting a 512 MB server.** The first deploy ran out of memory on the first zero-shot call. The
fixes were:
* **One resident model.** The cost of switching is reported as `load_ms`, not hidden.
* **flan-t5-small with float16 weights and float32 compute.** It agrees with the evaluated
  float32 baseline on all 3,076 test messages.
* **Rejected alternatives, with numbers.** bfloat16 was 10× slower on this CPU. int8 dynamic
  quantization dropped agreement to 13–30%.

The rule I followed was that a cheaper serving variant may only replace the evaluated model if
it is shown to give the same answers.

**Cost.** The live measurements and the comparison with Claude Haiku, Sonnet and Opus 5.5 are in
[benchmarks/cost_comparison.md](benchmarks/cost_comparison.md) and the README.
* **The pattern.** The self-hosted router costs nothing on the free tier and a fixed few dollars
  a month on a paid one. A frontier API is billed per message, so it is cheaper only at low
  volume.
* **What is left out.** The fine-tuned model's real cost is the engineering it took: labels, a
  training pipeline, a host, monitoring. That is why the README's "when zero-shot wins" list
  exists.
* **What was not measured.** I did not measure a frontier model's accuracy, and the report says
  so rather than guessing.

## 2. Ethics: stewardship of a fine-tuned model

Fine-tuning moves decisions that a prompt would leave visible into a set of weights that no one
can read. That puts five duties on whoever ships the model.

1. **Own the labels.** The model learns the 77 → 8 mapping in `shared/queues.py`, including
   choices someone made: a fee on a cash withdrawal goes to fees, not cash; a payment the customer
   does not recognise goes to fraud. Those rules are written down, tested, and repeated in the
   dataset card, because a fine-tuned model applies them silently and at scale.
2. **Report the expensive error separately.** Average accuracy hides the customer whose card is
   being used by someone else. Fraud recall has its own column and interval. The review rule
   sends any message with a 10% fraud probability to a person. The model card states what
   that rule still misses: 6 of 240 fraud reports routed elsewhere without a flag, and 12 errors
   made with over 99% confidence.
3. **Keep customers' words out of the log.** Predictions are logged by SHA-256, with the queue
   and scores, so the router can be audited without anyone reading customer messages after the
   fact. Row-level security keeps even those rows away from the public key.
4. **Make provenance unforgeable.** Each prediction row is joined to the experiment that produced
   its model. That experiment row holds the weights' hash, and the server refuses weights with
   any other hash. "Which model said that, and how good was it?" always has an answer.
5. **Do not overclaim.** Banking77 is clean, short and synthetic. Real messages are not. The model
   card says to expect lower accuracy and to measure it before trusting the router, and the
   router only decides *who answers*, never what the answer is.

## 3. Worldview reflection

"Moreover, it is required in stewards that a man be found faithful" (1 Corinthians 4:2).
A fine-tuned model is a thing entrusted. The pre-trained weights were given to me by people
who spent years building them. The 13,000 labelled questions were written by crowd workers I
will never meet. A bank that deploys this router entrusts it with customers who are often
anxious, sometimes because someone has just taken their money. Faithfulness here is not about
the model being impressive. It means telling the truth about it to the people who rely on it.

**The number I was tempted to omit.** The review rule sounds like a safety net: anything
uncertain or possibly fraudulent goes to a person. When I measured it, it caught only 22 of the
100 test errors, and **6 of the 240 customers reporting fraud were routed to the wrong team
with no flag at all.** The headline numbers (0.965 macro-F1, 0.971 fraud recall) are true, and
they would have looked better alone. I put the six into the model card and this report anyway,
because a steward who reports only the flattering numbers is managing his reputation, not the
thing entrusted to him. The same reasoning made me rebuild the zero-shot baseline after its
first run sent everything to one queue. Beating a broken competitor would have made my result
look larger and meant less.

**What I accept and what I defer.** I accept responsibility for the routing decision, the
labels it learned, the privacy of the log, and an honest account of its errors. I defer the
response itself to the people on each queue. The router chooses who answers. It must never
become the reason no one does.
