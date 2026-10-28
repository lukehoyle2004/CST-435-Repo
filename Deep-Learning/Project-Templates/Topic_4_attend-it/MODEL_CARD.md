# Model Card — Seq-Sight

## Model details

- **Task:** binary sentiment classification of English movie reviews (negative / positive).
- **Two matched models**, sharing one encoder: Embedding(20,000 × 128, `<pad>` → zero
  vector) → dropout 0.3 → single-layer unidirectional LSTM(128) over a **packed** sequence.
  - **Plain LSTM** (`lstm`): last hidden state → dropout → Linear(128, 2). 2,692,354 parameters.
  - **LSTM + attention** (`attention`): hand-written scaled-dot-product attention pooling.
    Keys $k_t = W_k h_t + b_k$ ($d_k=64$), one learned query $q$, values $h_t$, padding mask;
    weights $\alpha = \mathrm{softmax}(q k^\top/\sqrt{d_k} + M)$ and context $c=\sum_t \alpha_t h_t$ →
    dropout → Linear(128, 2). 2,700,674 parameters. Returns one attention weight per token.
- **Training:** seed 0, Adam (lr 2e-3), batch 64, 6 epochs, gradient clipping 1.0,
  cross-entropy; best-validation epoch kept (LSTM epoch 6, attention epoch 4). CPU only.
- **Input:** up to 20,000 characters, lowercased and tokenized into words plus `!` and `?`;
  only the **first 256 tokens** are read.
- **Owner:** Luke Hoyle, CST-435 (Grand Canyon University). Code: this folder.

## Intended use

- **Primary:** a teaching and portfolio demo of sequence modeling and attention as an
  explanation aid: compare a recurrent baseline with attention, inspect per-word weights,
  and see where both fail.
- **Acceptable with a person in the loop:** first-pass triage of English movie or TV
  reviews, where a human reads anything flagged negative or "uncertain" before acting.
- **Out of scope:**
  - automatic decisions about a person (moderation bans, refunds, hiring, grading)
  - non-English text, and domains far from film reviews (product, medical, financial or
    political text)
  - neutral or mixed-sentiment detection
  - treating the attention heatmap as proof of why the model decided

## Training data

The IMDB Large Movie Review Dataset (Maas et al., 2011): 22,500 training and 2,500
validation reviews from the official training split, and the official 25,000-review test
split for every reported number. Labels come from the star rating (≤ 4 negative, ≥ 7
positive); **neutral 5–6 star reviews are not in the dataset**. The full dataset card is in
the [README](README.md#dataset-card-imdb-large-movie-review-dataset).

## Metrics (IMDB test set, 25,000 reviews)

| | Plain LSTM | LSTM + attention |
|---|---|---|
| Accuracy (95% Wilson interval) | 86.64% (86.22–87.06) | 87.65% (87.24–88.05) |
| Macro-F1 | 0.866 | 0.876 |
| F1 negative / positive | 0.870 / 0.863 | 0.880 / 0.872 |
| Expected calibration error | 0.061 | 0.043 |
| Wrong at ≥ 90% confidence | 1,419 | 976 |
| Accuracy, reviews > 256 tokens (28% of test) | 81.42% | 83.42% |
| Accuracy on the 87.2% of reviews answered at ≥ 75% confidence | — | 91.74% |

Both models recall negatives better than positives (attention: 90.9% vs 84.4% recall).

## Guardrails (served by the API)

- **Abstain** and report "uncertain", falling back to the 50/50 base rate, when any of these
  hold:
  - confidence is below 75%
  - the review has fewer than 5 tokens
  - more than 20% of its words are outside the vocabulary
- **Flag** reviews that were truncated.
- The raw answer and both probabilities are still returned for transparency. The flags are
  logged with each prediction.

## Explanations

The attention vector shows where the model **pooled** LSTM states. It is not a faithful
causal explanation:
- each state $h_t$ already summarizes words $1..t$
- attention on IMDB is diffuse (mean normalized entropy 0.975; the top word gets a median
  3.8× its uniform share)

Use the heatmap to ask "what did it look at?", not "why is it right?".

## Privacy

Review text is never stored. Each prediction stores a SHA-256 hash, token counts, the
answer, the guardrail flags and the attention numbers. See the README's
[Data persisted / not persisted](README.md#data-persisted--not-persisted) section and
privacy statement.

## Limitations and risks

- **No neutral class:** "Not the best, not the worst." comes back negative at 100%.
- **Sarcasm and "so bad it's good" praise** are read literally. The Failure Gallery's 12
  worst mistakes are all about 100% confident.
- **Reads only the first 256 tokens,** so a verdict at the end of a long review is lost.
- **Vocabulary is frozen** from reviews up to 2011. New slang, names, and non-native or
  dialect phrasing more often become `<unk>`, so their writers get worse service.
- **One training seed per model.** The 1-point gap is significant on this test set
  (McNemar p = 4 × 10⁻⁷) but has not been checked across seeds.
- **Confidence is not correctness:** even after abstention, about 1 in 12 answered reviews
  is wrong.
