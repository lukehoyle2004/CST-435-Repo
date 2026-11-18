"""Static content for FleetDeck: the Concepts derivations, the model card, the
"Read me first" panels and the cost assumptions.

Kept apart from app.py so the words and the numbers that feed the cost
comparison can be reviewed (and tested) on their own. The metric values quoted
in the model card are copied from Product 6's committed sweep
(Topic_6_fine-it/docs/experiments/results.json); the live Leaderboard tab reads
the same rows from Supabase.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Read-me-first panels, for a non-technical reader (the bank's support lead).
# ---------------------------------------------------------------------------
SIDEBARS = {
    "concepts": (
        "**Why this page exists.** The router on the other tabs is a small model that was "
        "*taught* the bank's eight queues from about 9,000 labelled messages. The baseline next to "
        "it was never taught anything: it is asked, in plain English, which queue fits. This page "
        "explains why a model pre-trained by filling in blanked-out words (DistilBERT) is the right "
        "starting point for the first job, and an instruction-following model (flan-t5) for the second."
    ),
    "try": (
        "**How to read a result.** Each model returns a queue and how sure it is. *Needs review* "
        "means a person should look before it is routed: the model is under 60% sure, or it gives "
        "the fraud queue at least a 10% chance without picking it. The free server holds one model "
        "in memory at a time, so switching models costs a *cold start*: that time is shown "
        "separately, never hidden inside the latency."
    ),
    "leaderboard": (
        "**How to read the leaderboard.** Every row is one training run (or the zero-shot "
        "baseline), scored once on 3,076 test messages it never saw. The numbers in brackets are 95% "
        "intervals: if you collected another 3,076 messages, the score would likely land in that "
        "range. When two runs' ranges overlap, the table says they are *not meaningfully "
        "different*, and the cheaper one is a fine choice."
    ),
    "cost": (
        "**What this tab answers.** Is a small model we host ourselves cheaper, faster, and good "
        "enough compared with sending every message to a large commercial model? The button sends a "
        "fixed batch of 16 messages to the live service and uses the measured time. Every price "
        "and token count behind the dollar figures is listed under *Assumptions*, so you can "
        "disagree with them."
    ),
    "card": (
        "**What a model card is.** A one-page statement of what the model is for, what it was "
        "trained on, how well it works and where it fails, so nobody uses it for a job it was "
        "never tested on."
    ),
}

# ---------------------------------------------------------------------------
# Concepts: the four pre-training objectives (LaTeX) and the suitability matrix.
# ---------------------------------------------------------------------------
MLM = (r"\mathcal{L}_{\text{MLM}}(\theta) = -\,\mathbb{E}_{x,M}\Big[\sum_{i\in M}"
       r"\log p_\theta\big(x_i \,\big|\, \tilde{x}\big)\Big]")
MLM_WHY = (
    "About 15% of positions $M$ are corrupted (80% become `[MASK]`, 10% a random token, 10% stay). "
    "The encoder sees the corrupted text $\\tilde{x}$ with **bidirectional** attention, so every "
    "position's representation is trained to summarize both its left and its right context. That is "
    "exactly what classifying a whole message needs. DistilBERT adds a distillation term "
    "$-\\sum_v p^{\\text{teacher}}_T(v)\\log p^{\\text{student}}_T(v)$: 40% fewer parameters, about 97% of "
    "BERT's accuracy."
)
CLM = (r"\log p_\theta(x) = \sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}),\qquad "
       r"\mathcal{L}_{\text{CLM}} = -\,\mathbb{E}_x\Big[\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})\Big]")
CLM_WHY = (
    "The chain rule is exact, so this is unbiased maximum likelihood of the whole sequence, and a "
    "causal mask lets **every** position be a training signal in one pass. It makes models that "
    "*write*, and at very large scale it makes zero-shot prompting work. A small one is weak at "
    "classification: an early word never sees what follows it."
)
SPAN = (r"\mathcal{L}_{\text{span}} = -\,\mathbb{E}_x\Big[\sum_{t=1}^{|y|}"
        r"\log p_\theta\big(y_t \mid y_{<t},\ \mathrm{Enc}_\theta(\tilde{x})\big)\Big]")
SPAN_WHY = (
    "Contiguous spans (mean length 3) are replaced by sentinels; the decoder writes the missing "
    "spans. The encoder is bidirectional like MLM, and the decoder generates like CLM. After "
    "*instruction tuning* (flan-T5) the model follows a written task description, which is what lets "
    "it act as our zero-shot baseline: we write the routing question and score the eight answers."
)
INFONCE = (r"\mathcal{L}_{\text{InfoNCE}} = -\,\mathbb{E}\left[\log\frac{\exp\big(s(f(x),f(x^+))/\tau\big)}"
           r"{\sum_{j=1}^{N}\exp\big(s(f(x),f(x_j))/\tau\big)}\right],\qquad "
           r"I(x;x^+) \ge \log N - \mathcal{L}_{\text{InfoNCE}}")
INFONCE_WHY = (
    "An $N$-way classification of \"which candidate is my positive?\". Minimizing it raises a lower "
    "bound on mutual information, which is why bigger batches help. It makes embeddings for search, "
    "deduplication and nearest-neighbour classification, not text."
)

MATRIX_HEADER = ["Downstream task", "MLM (encoder)", "CLM (decoder)", "Span corruption (enc-dec)",
                 "Contrastive (bi-encoder)"]
MATRIX = [
    ("**Single-label classification (this product)**", "★★★", "★★ (★★★ by prompting at huge scale)",
     "★★", "★★ (k-NN)"),
    ("Token tagging (names, PII)", "★★★", "★", "★★", "★"),
    ("Extractive QA", "★★★", "★★", "★★", "★"),
    ("Open-ended generation, chat", "★", "★★★", "★★", "★"),
    ("Summarization, translation", "★", "★★", "★★★", "★"),
    ("Zero-shot instruction following", "★", "★★★ (instruction-tuned)", "★★★ (instruction-tuned)", "★"),
    ("Semantic search, dedup", "★★", "★", "★", "★★★"),
    ("CPU latency per message", "★★★ one pass", "★ one pass per token", "★★", "★★★"),
]

ZERO_SHOT_WINS = [
    "**No labels, or very few.** Our run trained on only 1,000 messages lost 0.16 macro-F1 and its "
    "fraud recall fell from 0.97 to 0.35. Below about 50 examples per queue, a large prompted model "
    "is the safer bet.",
    "**The queues change often.** A ninth queue means relabelling and retraining here; for a prompted "
    "model it is one more line in the prompt.",
    "**The decision needs world knowledge** that 9,000 bank messages cannot teach (for example, "
    "whether a merchant is in a sanctioned country).",
    "**Volume is low.** At a few hundred messages a day, a training pipeline and a model host cost "
    "more in engineering time than an API bill (see Cost & Latency).",
    "**The output must be written text** (a drafted reply), not one of eight labels.",
]

# ---------------------------------------------------------------------------
# Model card (numbers from Topic_6_fine-it/docs/experiments/results.json).
# ---------------------------------------------------------------------------
MODEL_CARD = """
### Model details
* **Name / run:** `ft-lr5e-5-ep3`, the FineTuneFleet support-message router (FineTuneFleet v1).
* **Architecture:** `distilbert-base-uncased` (66M parameters, masked-language-model pre-training plus
  distillation) with an 8-way classification head, fine-tuned end to end with the Hugging Face `Trainer`.
* **Training:** learning rate 5e-5, 3 epochs, batch 32, linear schedule with 6% warm-up, weight decay
  0.01, seed 42; the epoch with the best **validation** macro-F1 was kept. 39 minutes on a laptop CPU.
* **Selected from:** five seeded configurations, chosen on validation; it is also best on test.
* **Weights:** GitHub release `finetunefleet-v1`; the service refuses any file whose SHA-256 differs
  from the deployed row (`4748e601f4af…`).

### Intended use
Route English customer messages to a digital bank's support desk: one of eight queues (card
services, transfers, top-ups, cash & ATM, fees & FX, payments & disputes, security & fraud, account &
identity). It **suggests** a queue; low-confidence messages and possible fraud are flagged for a person.

**Out of scope:** deciding anything about a customer (credit, closure, fraud verdicts); messages
in other languages; channels other than written support messages; banks whose products differ from
the Banking77 taxonomy without re-evaluation.

### Data
[Banking77](https://huggingface.co/datasets/PolyAI/banking77) (PolyAI, CC-BY-4.0), 77 intents mapped to
8 queues by three written rules. After removing duplicates within and across splits: 8,952 train /
995 validation (stratified 10% of the original train) / 3,076 test. Median 13 word-pieces per message.

### Metrics (held-out test set, 3,076 messages; 95% bootstrap intervals)
| | Fine-tuned (deployed) | Zero-shot flan-t5-small |
|---|---|---|
| Macro-F1 | **0.965** [0.958, 0.972] | 0.669 [0.649, 0.686] |
| Accuracy | 0.968 | 0.690 |
| Fraud-queue recall | **0.971** [0.948, 0.991] | 0.604 [0.543, 0.661] |
| Gain in macro-F1 | +0.296 [+0.279, +0.315] (paired bootstrap) | – |

### Limitations
* Banking77 messages are short, clean, single-intent English questions. Real messages are longer,
  messier and sometimes about two things at once: expect lower accuracy, and measure it.
* Queues are not equally hard. The weakest are payments & disputes (F1 0.950; its precision is
  0.92, so other queues' messages land there) and transfers (recall 0.94); the best is card
  services (F1 0.982).
* Confidence is high almost everywhere (87% of test answers are above 0.99), so a wrong answer is
  often a *confident* one. The review rule flags 1.3% of messages and catches only 22 of the 100
  test errors; 6 of the 240 fraud reports were misrouted without being flagged. It is a safety
  net, not a guarantee.
* The zero-shot baseline is a 77M-parameter model; a frontier LLM would score higher (not measured here).

### Ethical considerations
A misrouted fraud report delays help to someone whose money is at risk, so fraud recall is reported
separately and a 10% fraud probability alone sends a message to a person. Customer messages are
never stored: the service logs a SHA-256 hash, the length and the model's answer.
"""

# ---------------------------------------------------------------------------
# Cost assumptions. Every number the Cost & Latency tab uses; shown on screen.
# ---------------------------------------------------------------------------
COST = {
    "as_of": "2026-11-18",
    "frontier": {
        "source": "Anthropic API list prices, standard tier, USD per million tokens "
                  "(Haiku: prompts up to 100K tokens)",
        "models": {
            "Claude Haiku 5.5": {"input": 0.10, "output": 0.50, "reasoning_tokens": 0,
                                 "note": "thinking turned off (allowed at effort high or below)"},
            "Claude Sonnet 5.5": {"input": 2.00, "output": 10.00, "reasoning_tokens": 0,
                                  "note": "thinking off with thinking type between_tools"},
            "Claude Opus 5.5": {"input": 4.00, "output": 20.00, "reasoning_tokens": 100,
                                "note": "thinking cannot be turned off; ~100 billed reasoning tokens "
                                        "at low effort (estimate)"},
        },
        "prompt_tokens": 220,
        "output_tokens": 6,
        "explain": "Each call sends an instruction that lists the eight queues with a one-line description "
                   "(about 220 tokens) plus the message, and gets back a queue name (about 6 tokens). "
                   "Message tokens are counted with DistilBERT's tokenizer as a stand-in for the "
                   "frontier model's (roughly ±30%).",
    },
    "self_hosted": {
        "source": "Render compute plans; verify monthly prices on render.com/pricing before quoting",
        "plans": {
            "Free (what runs now)": {"cpu": 0.1, "usd_per_month": 0,
                                     "note": "sleeps after 15 idle minutes; ~1 minute to wake"},
            "Starter": {"cpu": 0.5, "usd_per_month": 7, "note": "always on; same 512 MB"},
            "Standard": {"cpu": 1.0, "usd_per_month": 25, "note": "always on; 2 GB, both models resident"},
        },
        "measured_on": "Free (what runs now)",
        "explain": "Capacity = seconds in a month ÷ measured server seconds per message, scaled by the "
                   "plan's CPU share relative to the free plan (an assumption: the model is CPU-bound). "
                   "Engineering time to build and run the pipeline is not included.",
    },
}

FIXED_BATCH = [
    "My new card still hasn't arrived after two weeks.",
    "How do I activate the card you sent me?",
    "My transfer to my landlord is still pending since Monday.",
    "I sent money to the wrong account, can you reverse it?",
    "My top up didn't go through but the money left my account.",
    "Why was my top-up reverted?",
    "The ATM gave me less cash than I asked for.",
    "The cash machine kept my card.",
    "Why was I charged a fee for withdrawing cash?",
    "What exchange rate do you use when I pay in euros?",
    "A card payment I made yesterday shows up twice.",
    "I'd like a refund for an item I returned to the shop.",
    "Someone used my card without my permission.",
    "There's a transaction I don't recognise, I think my card was stolen.",
    "How do I verify my identity?",
    "Please close my account.",
]

SAMPLES = {
    "Card used without permission": "Someone used my card to buy things online and it wasn't me.",
    "Fee on a cash withdrawal": "Why did I get charged for taking cash out of the machine?",
    "Top-up failed": "I tried to top up with my debit card but it says declined.",
    "Ambiguous: refund or fraud?": "There's a payment I don't remember making and the shop won't refund it.",
    "Change PIN": "How can I change the PIN on my card?",
}
