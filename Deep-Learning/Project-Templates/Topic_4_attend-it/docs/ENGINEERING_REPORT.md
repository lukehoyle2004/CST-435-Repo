# Seq-Sight: Engineering Report

CST-435 Topic 4, Product 4. Author: Luke Hoyle. Every number below comes from
`models/runs.json` (written by `scripts/analyze.py` on the full 25,000-review IMDB test
set) or from a unit test named next to it.

## 1. Problem

A small streaming service wants to triage written feedback: route clearly unhappy
reviews to a person and summarize the rest. They also want to know *why* the model
called a review negative, so it has to show which words it weighed. Seq-Sight trains two
matched sentiment classifiers on IMDB (Maas et al., 2011): a plain LSTM and the same LSTM
with a hand-written scaled-dot-product attention block. It serves both models, returns one
attention weight per word with every prediction, and stores that vector in Supabase, so any
past prediction can be inspected later. The review text itself is never stored.

## 2. Sequence representation

Variant B of the tutorial: variable-length text, packed sequences, padding masks. All of it
lives in `shared/text.py`, so training, the API and the tests encode text identically.

| Step | What happens | Why |
|---|---|---|
| normalize | lowercase, `<br />` → space, unescape HTML entities | IMDB reviews are scraped HTML |
| tokenize | regex `[a-z0-9]+(?:'[a-z]+)?\|[!?]`; curly apostrophes mapped to `'` | contractions stay whole (`don't`, `isn't`), so negation is one token the heatmap can land on; `!` and `?` carry tone |
| vocab | training split only; `<pad>`=0, `<unk>`=1; top 20,000 words with frequency ≥ 2, ties alphabetical | deterministic ids (its SHA-256 is stored with each run and checked at API start-up) |
| truncate | keep the first 256 tokens | 71.8% of test reviews fit; the other 28.2% are cut, and the API reports `truncated` |
| pad | right-pad each batch with 0 to its longest review | what `pack_padded_sequence` expects |
| pack | `pack_padded_sequence(enforce_sorted=False)` | the LSTM never runs over padding, so `h_T` is the state after the last *real* word |
| mask | boolean `[B, T]`, True on real tokens; scores at False positions set to −∞ | padding gets exactly zero attention |

Tests: `tests/test_text.py` (17 tests: tokenizer cases, vocab determinism and round trip,
padding and mask shapes, empty-input errors), `tests/test_imdb.py` (loader on parquet
fixtures, rejects the unlabeled split), `tests/test_model.py::test_padding_does_not_change_the_prediction`
(adding padding, or junk in the padding, changes neither the logits nor the attention).
Breaking the mask on purpose makes that test fail, so it really checks the mask.

## 3. Scaled-dot-product attention

### 3.1 Derivation

We want a summary of the hidden states $h_1..h_T$ that weights the informative ones.
Score each position by how well its **key** matches a **query**, turn the scores into a
probability distribution, and average the **values** with those probabilities:

$$s_t = \frac{q \cdot k_t}{\sqrt{d_k}} + M_t,\qquad
\alpha_t = \frac{e^{s_t}}{\sum_{j=1}^{T} e^{s_j}},\qquad
c = \sum_{t=1}^{T} \alpha_t v_t,\qquad
M_t = \begin{cases}0 & t \le \text{length}\\ -\infty & \text{padding}\end{cases}$$

In matrix form, for many queries at once:
$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\big(QK^\top/\sqrt{d_k} + M\big)V$
(Vaswani et al., 2017). The dot product is the similarity measure, the softmax makes the
weights positive and sum to 1, and $e^{-\infty}=0$ makes padding vanish exactly.

**In Seq-Sight** (`seqsight/attention.py`, written by hand rather than calling
`F.scaled_dot_product_attention`): values are the LSTM states $v_t = h_t \in \mathbb{R}^{128}$,
keys are $k_t = W_k h_t + b_k \in \mathbb{R}^{64}$, and there is **one learned query**
$q \in \mathbb{R}^{64}$, effectively the question "which positions decide the sentiment?".
The pooled $c \in \mathbb{R}^{128}$ goes to a linear layer. The block adds 8,320 parameters
($128\cdot64 + 64 + 64$).

### 3.2 Worked example

$d_k = 2$, $q = [1,1]$, keys $k_1=[1,1]$, $k_2=[1,-1]$, $k_3=[-1,0]$, scalar values $[10,20,30]$.

| token | $q\cdot k$ | $\div\sqrt2$ | $e^{s}$ | $\alpha = e^s / 5.6064$ | $\alpha v$ |
|---|---|---|---|---|---|
| 1 | 2 | 1.4142 | 4.1133 | **0.7337** | 7.337 |
| 2 | 0 | 0 | 1.0000 | 0.1784 | 3.567 |
| 3 | −1 | −0.7071 | 0.4931 | 0.0879 | 2.638 |

Output $c \approx 13.54$: mostly token 1, whose key points the same way as the query. This
exact case is `tests/test_attention.py::test_matches_hand_computed_example`, and the general
masked case is checked against PyTorch's reference implementation
(`test_matches_torch_reference_with_mask`).

### 3.3 Why divide by $\sqrt{d_k}$

If the entries of $q$ and $k$ are independent with mean 0 and variance 1, then
$q\cdot k = \sum_{i=1}^{d_k} q_i k_i$ is a sum of $d_k$ terms of variance 1:
$\mathrm{Var}(q\cdot k)=d_k$. The spread of the scores grows like $\sqrt{d_k}$. With
$d_k=256$ the standard deviation is 16, so gaps of tens of units between scores are
typical, and the softmax saturates:

| scores | softmax | $\partial\alpha_i/\partial s_i=\alpha_i(1-\alpha_i)$ |
|---|---|---|
| $[16, 0, -16]$ (unscaled, $d_k=256$) | $[1.000000, 0.000000, 0.000000]$ | $[0, 0, 0]$ |
| $[1, 0, -1]$ (divided by $\sqrt{256}=16$) | $[0.665, 0.245, 0.090]$ | $[0.223, 0.185, 0.082]$ |

A saturated softmax is one-hot and passes back almost no gradient, so the keys and query
stop learning. Dividing by $\sqrt{d_k}$ brings the score variance back to 1 at any width.
Tests: `test_sqrt_dk_scaling_keeps_scores_unit_variance` (measured variance ≈ 256 unscaled,
≈ 1 scaled) and `test_unscaled_scores_saturate_the_softmax` (max weight > 0.9 unscaled,
< 0.6 scaled).

### 3.4 Attention vs. the recurrent step

The LSTM step is $h_t = f(h_{t-1}, x_t)$ (Hochreiter & Schmidhuber, 1997).

| | Recurrent step (plain LSTM) | Attention pooling |
|---|---|---|
| Path from word $t$ to the output | $T-t$ recurrent steps | 1 step (weight $\alpha_t$) |
| What the classifier sees | fixed-size $h_T$ (128 numbers) | all $T$ states, re-weighted per review |
| Sequential operations | $O(T)$, in order | $O(1)$, all scores at once |
| Cost | $O(T\,d^2)$ | $O(T\,d\,d_k)$ with one query; $O(T^2 d)$ for full self-attention |
| Gradient to early words | multiplied through $T-t$ gates | one softmax away |
| Per-word signal to inspect | none | $\alpha_t$ (what the UI draws) |
| Word order | built in | none by itself; here it comes from the LSTM states |

## 4. LSTM vs. attention: matched comparison

### 4.1 Setup

Both runs share everything except the attention block: seed 0, the same 22,500 / 2,500
train/validation split of the official training set, the same vocabulary (SHA-256 checked),
the same batch order, 6 epochs, batch 64, Adam at lr 2e-3, gradient clipping at 1.0,
dropout 0.3, max 256 tokens, and the same laptop CPU (Intel, 6 threads each, run in
parallel). Each run keeps its best-validation epoch. Both rows are persisted to Supabase
`seq_runs` by `db/seed_runs.py`; `select * from seq_model_comparison;` reproduces the
headline numbers.

### 4.2 Results (25,000 test reviews)

| | Plain LSTM | LSTM + attention |
|---|---|---|
| Parameters | 2,692,354 | 2,700,674 (+8,320) |
| Best epoch | 6 of 6 | 4 of 6 |
| Validation accuracy | 87.92% | 89.88% |
| **Test accuracy** (95% Wilson interval) | 86.64% (86.22–87.06) | **87.65% (87.24–88.05)** |
| Macro-F1 | 0.8664 | 0.8764 |
| F1 negative / positive | 0.870 / 0.863 | 0.880 / 0.872 |
| Calibration error (ECE, 10 bins) | 0.0605 | **0.0431** |
| Wrong with ≥ 90% confidence | 1,419 | **976** |
| Training time (6 epochs) | 103 min | 107 min |

The two models disagree on 2,450 reviews: attention alone is right on 1,351, the LSTM alone
on 1,099. McNemar's test on those pairs gives p = 4.0 × 10⁻⁷, so the 1-point gap is not
chance *on this test set*. It is one seed per model, so it is not a claim about every seed.

### 4.3 Structural analysis

**Long-range dependencies.** Accuracy by raw review length:

| Raw length (tokens) | n | Plain LSTM | Attention | Gap |
|---|---|---|---|---|
| 1–100 | 3,106 | 88.89% | 89.44% | +0.55 |
| 101–200 | 11,684 | 88.78% | 89.63% | +0.85 |
| 201–256 | 3,158 | 88.19% | 88.03% | −0.16 |
| > 256 (cut to 256) | 7,052 | 81.42% | 83.42% | **+2.00** |

Both models drop on the longest reviews, because truncation removes the end, where reviewers
often give their verdict. The plain LSTM drops further: its classifier only sees $h_{256}$,
so a strong cue at word 40 has to survive 216 more gated updates. Attention reads $h_{40}$
directly, with a one-step path from that word to the output (§3.4). The 201–256 bucket is a
wash: those reviews are long, but the cue is usually near the end, which $h_T$ remembers well.

**Optimization.** The same short path helps gradients. After one epoch the attention model
was at 86.3% validation accuracy and the LSTM at 70.9%. The attention model reached its best
validation accuracy at epoch 4 and then overfit (training loss 0.17 → 0.10 while validation
loss rose 0.27 → 0.36). The LSTM was still improving at epoch 6, so with a larger budget its
gap would probably shrink somewhat. This is the honest reading of a matched 6-epoch budget.

**Calibration.** Attention is better calibrated (ECE 0.043 vs 0.061) and makes 31% fewer
≥ 90%-confident mistakes. Pooling over all states gives a less brittle summary than one
final state that one late phrase can swing.

**Parallelism.** The attention *scores* for all 256 positions are one matrix product, with
no dependence between positions. But in this design attention sits on top of an LSTM, so the
encoder is still sequential, and training time was the same (107 vs 103 minutes). The
parallelism advantage would only show if attention replaced the recurrence: a
self-attention encoder with position encodings, at $O(T^2 d)$ cost. That trade is the reason
Transformers exist, but we did not need it for 256-token reviews on a CPU.

## 5. Attention visualization and interpretation

Streamlit shows attention as a heatmap over the words, shaded relative to the top word. Try
It shows typed reviews and six held-out test reviews, the Failure Gallery shows 12 mistakes,
and Baseline vs. Attention shows 8 reviews where only one model was right. Every live
prediction stores its attention vector in `seq_predictions.attention` (a CHECK constraint
forces one weight per scored token), and the Recent Predictions drilldown re-draws it from
the database.

Illustrative inputs (top weights; "uniform" = 1/length):

| Review (abridged) | Prediction | Top weights | Reading |
|---|---|---|---|
| "widow hires a psychopath … **sloppy** film noir thriller which doesn't make much of its tension … 3/10" | negative, 100% (true negative) | sloppy .116, film .073, thriller .064 (uniform .043) | lands on the one judgment word |
| "I found this movie really hard to sit through … one of the **worst I've seen** … don't **bother**" | negative, 100% (true negative) | worst .092, i've .075, seen .064, bother .058 | the verdict phrase at the end |
| "a stunningly well made film with **exceptional** acting **directing** writing and **photography**" | positive, 99.9% (true positive) | directing .044, photography .038, film .038, exceptional .037 | spread over a list of praised crafts |
| "I expected a **dull** remake, but the performances are **wonderful** … well worth it" (typed) | negative, 65.4%; plain LSTM says positive, 78.5% | dull .146, wonderful .073, remake .061 | the early "dull" dominates; the "but" that reverses it is not weighted, and the guardrail abstains |
| "Not the best, not the worst." (typed) | negative, 100% | worst .426, best .237, not .102 | see below |

What the weights mean, given the architecture:

1. **Weights are over LSTM states, not words.** The LSTM is unidirectional, so $h_t$
   summarizes words $1..t$. Weight on "seen" in "worst I've seen" credits a state that
   already contains "worst". That is why weight often sits on, or just after, the sentiment
   word, and why it can drift late in a sentence.
2. **One content-based query.** The query asks the same question of every review, and the
   score depends on a state's *content*, not its position. Words that are strongly polar in
   training (worst, sloppy, wonderful) get high keys wherever they appear. Negation and irony
   only count if the LSTM folded them into the state. In "Not the best, not the worst" it did
   not: "worst" takes 43% of the weight and the model is 100% sure of a review that is neutral.
3. **Diffuse, not one-hot.** Over the test set the mean normalized entropy is 0.975
   (1 = uniform), and the top word gets a median 3.8× its uniform share. The model mostly
   averages many mildly informative states and up-weights a few. A heatmap is a fair picture
   of what it looked at, but it is not a proof of why it decided (Jain & Wallace, 2019); the
   UI says so.

**Failure Gallery.** The attention model's 12 most confident mistakes are all at ≈ 100%
confidence, and they fall into three groups:
- **Sarcasm and "so bad it's good":** "not citizen kane but it does deliver … badly acted
  and directed, poorly scripted, who cares?" is labeled positive. Attention goes to poorly,
  badly, acted and directed.
- **Probable label noise:** "definitely one of the best kung fu movies in the history of
  cinema" is labeled negative. Attention goes to best, kung, one and definitely.
- **Unusual word use:** "primary plot! primary direction! poor interpretation", where
  *primary* means *primitive*.

In each case attention did exactly its job: it found the strongly polar words. The error is
that polarity read literally is the wrong reading. Confidence is high because a few strong
states dominate the weighted average and push the logit far from zero.

## 6. Guardrails

The API (`seqsight/inference.py::guardrails`) returns the model's raw answer for
transparency, but marks a prediction **abstained** and shows the verdict **"uncertain"** in
any of these cases:
- confidence is below 75%
- the review has fewer than 5 tokens
- more than 20% of its words are outside the vocabulary

It also flags truncated reviews. The flags are stored with each prediction. The threshold
comes from the test set:

| Attention model answers when confidence ≥ | Reviews answered | Accuracy on those | Accuracy it would have had on the rest |
|---|---|---|---|
| 60% | 95.4% | 89.21% | 55.4% |
| 70% | 90.1% | 90.85% | 58.6% |
| **75%** | **87.2%** | **91.74%** | **59.9%** |
| 80% | 83.7% | 92.58% | 62.3% |
| 90% | 73.8% | 94.71% | 67.7% |

At 75% the service declines about one review in eight. On those reviews it would have been
right only 60% of the time, barely better than the 50/50 base rate it falls back to.

## 7. Ethical consideration

**Customer scenario.** A streaming service routes reviews scored negative to a support
queue and publishes a "what viewers loved" digest from those scored positive. Who is harmed
by a confidently wrong score:
- **A frustrated customer** whose sarcastic complaint ("masterpiece … 10/10, best two hours
  of my life wasted") is scored positive at 100%, so no one follows up.
- **A reviewer** whose mixed or neutral review is published as praise or flagged as an
  attack.
- **Writers of non-native or dialect English, or reviewers of non-Western films.** Their
  words are more often outside the 20,000-word vocabulary or used in unfamiliar ways (the
  "primary" failure), so they get systematically worse service.

**Guardrails.**
1. *Confidence threshold with a base-rate fallback* (the sentiment equivalent of a
   forecaster's climatology fallback). Below 75% confidence, on very short reviews, or when
   over a fifth of the words are unknown, the verdict is "uncertain" and the review goes to
   a person, never to an automatic action (§6).
2. *Stated uncertainty, not just a label.* The UI always shows both probabilities, the
   reported accuracy carries a 95% interval (87.24–88.05%), and the Baseline vs. Attention
   tab shows how accuracy rises when the model is allowed to abstain.
3. *Disclosed bias* (model card and dataset card):
   - IMDB contains **no neutral reviews**: 5- and 6-star ratings are excluded, so the model
     has never seen one and will call "not the best, not the worst" negative at 100%.
   - The data is English-only, about movies, and from 2011 or earlier.
   - The model reads only the first 256 tokens.
   - It misses sarcasm, as the Failure Gallery shows.
4. *Confidently-wrong cases are shown in public.* The Failure Gallery shows the model's worst
   mistakes next to its best answers, so no user is told that 100% means certain.
5. *Privacy.* Only a SHA-256 hash of the text, token counts, the answer, the guardrail flags
   and the attention numbers are stored; the text is not (README, "Data persisted / not
   persisted"). Hashes of very short, common texts can be guessed by brute force, and the
   privacy statement says so.

**What we deliberately did not do.** We did not store review text, even though it would make
the drilldown prettier. We did not hide the plain LSTM's answer when attention is shown. We
did not tune the threshold to make the demo look better than the test set supports.

**Christian-worldview reflection: stewardship of model uncertainty.** A model's uncertainty
is something we are entrusted with, not something to hide. In the parable of the talents
(Matthew 25:14–30) the servants are judged on how they handled what was put in their care.
The confidence score is what this model puts in our care. Spending it honestly means
abstaining when it is low and saying "uncertain" instead of rounding a 65% lean up to a
label. It also means publishing the 976 times the model was confidently wrong, not only its
87.65%. Proverbs 11:1 calls a false balance an abomination and a just weight a delight: a
probability shown as a certainty is a false balance, and the people it is weighed against
are the customer whose complaint is never read and the reviewer misjudged by a word list.
Humility (James 4:13–15: "you do not know what tomorrow will bring") fits a model trained
on 2011 movie reviews and asked to judge today's customers. Good stewardship here is
concrete: show the interval, honor the abstention, and keep a person in the loop for every
decision that touches someone.

## 8. Limitations

- One seed per model. The McNemar test covers test-set noise, not seed-to-seed variation.
- The LSTM was still improving at epoch 6, so a larger budget would probably narrow the gap.
- Truncating to the first 256 tokens hurts both models on 28% of reviews. Keeping the head
  *and* tail of each review would likely help more than any architecture change.
- Attention weights are a view of where the model pooled, not a faithful explanation.
- The model is binary with no neutral class.
- The vocabulary is frozen at training time.

## References

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR.*
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML.*
Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8).
Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *NAACL.*
Maas, A. L., et al. (2011). Learning word vectors for sentiment analysis. *ACL.*
McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions. *Psychometrika, 12*(2).
Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS.*
