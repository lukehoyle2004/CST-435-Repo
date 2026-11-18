# Pre-training objectives: what each one teaches, and which tasks it favors

FineTuneFleet starts from two checkpoints that were pre-trained with different objectives:
**DistilBERT** (masked language modeling, then distilled from BERT) is the model we fine-tune,
and **flan-t5-small** (span corruption, then instruction tuning) is the zero-shot baseline. This
note derives the four objectives that matter for modern transformers and says which downstream
tasks each one sets a model up for. That is the reason this product fine-tunes an *encoder*.

Notation: a training text is a token sequence $x = (x_1, \dots, x_T)$ from vocabulary $V$; the
model with parameters $\theta$ produces a distribution $p_\theta(\cdot \mid \text{context})$ over $V$.

## 1. Masked language modeling (MLM): BERT, DistilBERT, RoBERTa

Choose a random subset $M \subset \{1..T\}$ of about 15% of positions. Of those, 80% become
`[MASK]`, 10% become a random token and 10% stay unchanged. Call the corrupted sequence
$\tilde{x}$. The loss is cross-entropy on the masked positions only:

$$
\mathcal{L}_{\text{MLM}}(\theta) \;=\; -\,\mathbb{E}_{x,\,M}\Big[\sum_{i \in M} \log p_\theta\big(x_i \,\big|\, \tilde{x}\big)\Big]
$$

**Derivation of why it works.** Minimizing this cross-entropy is maximum likelihood for the
conditional $p(x_i \mid x_{\setminus M})$. The encoder sees $\tilde{x}$ with *bidirectional*
self-attention (no causal mask), so the representation $h_i$ at every position is trained to
summarize **both** sides of the context. The 80/10/10 split matters because if the model only ever
saw `[MASK]` at prediction sites, it could learn features that exist only next to `[MASK]`, and
`[MASK]` never appears at fine-tuning time. Random and unchanged tokens force it to model *every*
position.

**What it favors:** understanding a whole input at once: classification, token tagging (NER),
extractive QA, retrieval re-ranking. The pooled `[CLS]` vector (or a mean of $h_i$) already
separates meanings linearly, so a small head plus gentle fine-tuning reaches high accuracy.
**What it is bad at:** generating text. Nothing in $\mathcal{L}_{\text{MLM}}$ trains left-to-right
production, and only about 15% of positions give a training signal per pass, which makes it less
sample-efficient than causal LM.

*DistilBERT* adds a distillation term on top: the student matches the teacher's softened MLM
distribution, $\mathcal{L}_{\text{KD}} = -\sum_{v} p^{\text{teacher}}_T(v)\log p^{\text{student}}_T(v)$
at temperature $T$, plus a cosine loss that aligns hidden states. It keeps roughly 97% of BERT's
GLUE score with 40% fewer parameters and runs about 60% faster. That trade is exactly what a
free-tier CPU host needs.

## 2. Causal language modeling (CLM): GPT-2, GPT-3/4, Llama, Claude's pre-training stage

Factor the joint distribution left to right with the chain rule and maximize the likelihood of
every token:

$$
\log p_\theta(x) = \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}),
\qquad
\mathcal{L}_{\text{CLM}}(\theta) = -\,\mathbb{E}_x\Big[\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})\Big]
$$

**Derivation.** The chain rule is exact, so minimizing $\mathcal{L}_{\text{CLM}}$ is unbiased
maximum likelihood of the whole sequence. A causal mask ($A_{ij} = -\infty$ for $j > i$) lets one
forward pass compute all $T$ conditionals in parallel, so **every** position is a training signal.
That is one reason CLM scales so well with data.

**What it favors:** open-ended generation, continuation, dialogue, code, and anything that can be
*phrased* as text completion, including classification by prompting ("The team that should handle
this is ___"). At large scale this makes zero-shot and few-shot in-context learning possible.
**What it is weaker at:** at small scale, single-label classification. Each $h_t$ sees only the
left context, so the representation of an early word never sees what came after it.

## 3. Span corruption (denoising seq2seq): T5, flan-t5, BART-style

Choose random contiguous spans covering about 15% of tokens, with mean span length 3. Replace each
span with a unique sentinel $\langle s_k\rangle$ in the encoder input $\tilde{x}$. The decoder target
$y$ is the sentinels followed by the dropped spans:
$y = \langle s_1\rangle\, \text{span}_1\, \langle s_2\rangle\, \text{span}_2 \dots$

$$
\mathcal{L}_{\text{span}}(\theta) = -\,\mathbb{E}_{x}\Big[\sum_{t=1}^{|y|}\log p_\theta\big(y_t \,\big|\, y_{<t},\ \text{Enc}_\theta(\tilde{x})\big)\Big]
$$

**Derivation.** This is CLM on a short target, conditioned on a bidirectionally encoded source,
which makes it a hybrid. The encoder gets MLM-like bidirectional context; the decoder learns to
produce text. Dropping whole spans (not single tokens) makes the target short, so the objective is
cheaper per example than full-sequence CLM, and the model has to reconstruct multi-token units.

**What it favors:** text-to-text tasks where input and output differ: summarization, translation,
generative QA, and (after the **instruction tuning** that turns T5 into *flan*-T5) following a
written task description. That last property is what makes flan-t5-small usable as a **zero-shot
baseline** here: we write the routing task as an instruction and score the eight possible answers.

## 4. Contrastive objectives: SimCSE, Sentence-BERT, CLIP, E5

Learn an embedding $f_\theta$ such that a positive pair $(x, x^+)$ is closer than the $N-1$
in-batch negatives. With cosine similarity $s(a,b) = \frac{a^\top b}{\|a\|\|b\|}$ and temperature
$\tau$, the InfoNCE loss is

$$
\mathcal{L}_{\text{InfoNCE}} = -\,\mathbb{E}\left[\log \frac{\exp\big(s(f(x), f(x^+))/\tau\big)}{\sum_{j=1}^{N}\exp\big(s(f(x), f(x_j))/\tau\big)}\right]
$$

**Derivation.** This is an $N$-way softmax classification problem: "which of these $N$ candidates
is my positive?". Oord et al. show that minimizing it maximizes a lower bound on the mutual
information between $x$ and $x^+$:
$I(x; x^+) \ge \log N - \mathcal{L}_{\text{InfoNCE}}$. That is why bigger batches (larger $N$)
help. Positives come from augmentations (SimCSE uses dropout noise), paired data (question/answer,
image/caption), or labels.

**What it favors:** similarity search, semantic retrieval, clustering, deduplication, and
nearest-neighbor or few-shot classification (embed the message, compare to a handful of labeled
examples per queue). **What it is bad at:** generation, and fine-grained distinctions the positives
never contrasted.

## Task-suitability matrix

Ratings are for a model of comparable size, before task fine-tuning. ★★★ is a natural fit,
★★ works with adaptation, ★ is possible but fighting the objective.

| Downstream task | MLM (encoder) | CLM (decoder) | Span corruption (enc-dec) | Contrastive (bi-encoder) |
|---|:-:|:-:|:-:|:-:|
| Single-label classification (**this product**) | ★★★ | ★★ (★★★ at very large scale, by prompting) | ★★ | ★★ (k-NN / few-shot) |
| Token tagging (NER, PII detection) | ★★★ | ★ | ★★ | ★ |
| Extractive QA | ★★★ | ★★ | ★★ | ★ |
| Open-ended generation / dialogue | ★ | ★★★ | ★★ | ★ |
| Summarization / translation | ★ | ★★ | ★★★ | ★ |
| Zero-shot instruction following | ★ | ★★★ (if instruction-tuned) | ★★★ (if instruction-tuned) | ★ |
| Semantic search / dedup / clustering | ★★ | ★ | ★ | ★★★ |
| Latency on a CPU, per message | ★★★ (one encoder pass) | ★ (one pass per generated token) | ★★ | ★★★ |

**Why this product fine-tunes MLM-pre-trained DistilBERT.** Routing is single-label classification
of short messages (median 13 WordPiece tokens), on a CPU, under a 512 MB memory budget. MLM gives
bidirectional features that a linear head can separate. One forward pass per message keeps latency
in single-digit milliseconds. We have about 9,000 labeled examples, which is plenty for gentle
fine-tuning (learning rate $2\times10^{-5}$ to $5\times10^{-5}$, 1–3 epochs).

## When zero-shot prompting beats fine-tuning (an honest list)

Fine-tuning a small model is not always the right call. Prompting a large instruction-tuned model,
whether a frontier API or a large open model, wins when:

1. **There are no labels, or too few.** With fewer than about 50 examples per class, a fine-tuned
   small model overfits. A large model's instruction following does better. The sweep's
   `ft-lr5e-5-ep3-n1000` run shows how fast the small model's advantage builds as labels are added.
2. **The label set changes often.** Adding a ninth queue to a fine-tuned model means relabeling
   and retraining. For a prompted model it is one more line in the prompt.
3. **The task needs world knowledge or reasoning that is not in the labels**, such as "is this
   message about a sanctioned country?" A large model already knows things that 9,000 bank
   messages will never teach DistilBERT.
4. **Volume is low.** At a few hundred messages a day, the engineering cost of a training
   pipeline, a model host and monitoring exceeds a frontier API bill. The Cost & Latency numbers in
   the README put this crossover in dollars.
5. **The output must be generated text** (a drafted reply, an explanation), not one of N labels.

The fine-tuned small model wins in the opposite situation, which is this product's: a stable label
set, thousands of labeled examples, high volume, a latency budget in milliseconds, and a privacy
requirement that customer messages never leave the bank's own infrastructure. Here, a 77M-parameter
*zero-shot* baseline is not the competitor a bank would actually consider; the README's cost section
compares against frontier models too, so the comparison is not stacked against prompting.
