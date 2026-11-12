"""Static teaching content for TrioDeck: the Concepts derivation and the archetype sidebars.

Kept apart from app.py so the words can be reviewed (and tested) on their own.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# "When to use this archetype" sidebars -- written for a non-technical reader.
# ---------------------------------------------------------------------------
SIDEBARS = {
    "concepts": (
        "**Why this page exists.** All three models on the other tabs are built from the same two "
        "ideas explained here: *attention* (each word decides which other words to look at) and "
        "*position* (telling the model where each word sits). Everything else is plumbing. If you "
        "only read one thing, read the shapes table: it shows that an attention layer takes a "
        "sentence in and gives a sentence of the same length back."
    ),
    "encoder_block": (
        "**When to use an encoder like this one.** Use it when you need to *understand or label* "
        "text you already have: is this a complaint, which product does it mention, does it match "
        "another document. This toy one was written by hand and only answers one question (is a "
        "list of words a palindrome?), but real encoders like BERT work the same way. Flip "
        "*positional encoding* off to see what happens when a model can no longer tell word order."
    ),
    "bert": (
        "**When to use an encoder (BERT).** Choose it when the answer is a *label, a score, a span "
        "or a search vector* — routing support tickets, tagging names, finding similar documents. "
        "It reads the whole text at once, in both directions, so it is fast and its cost does not "
        "grow with the length of the answer. It cannot write new text, and it is not a source of "
        "facts: ask it to fill in a capital city and it may pick a plausible-sounding wrong one."
    ),
    "gpt2": (
        "**When to use a decoder (GPT-2).** Choose it when you want *new text* that continues what "
        "you started: drafting, autocomplete, chat. It writes one word-piece at a time, so a longer "
        "answer takes proportionally longer and costs more. It writes fluently but invents details, "
        "so a person should check anything it produces before anyone relies on it."
    ),
    "t5": (
        "**When to use an encoder-decoder (T5).** Choose it when the output is a *rewritten version "
        "of an input*: a summary, a translation, a simpler wording. One half reads the source once, "
        "the other half writes while looking back at it, which keeps the result closer to the "
        "original than a decoder's. You pay for both halves, so it is the slowest per answer here."
    ),
    "compare": (
        "**How to read this comparison.** Same paragraph, three jobs. The encoder answers fastest "
        "because it does one pass and writes nothing new; the two writers spend most of their time "
        "producing words one at a time. On this free server only one model fits in memory, so "
        "running all three means loading each model in turn — that wait is real cost, and the "
        "clocks below show where it goes."
    ),
    "latency": (
        "**What this chart is for.** Every answer any model gives is timed on the server and "
        "logged. This page shows the typical (median) time per model, hour by hour, straight from "
        "that log. Use it to compare *types* of model on the same hardware, not to judge a model's "
        "best possible speed: this is a deliberately tiny free server."
    ),
}

# ---------------------------------------------------------------------------
# Concepts tab
# ---------------------------------------------------------------------------
SDPA = r"\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V"
SCALE_WHY = (
    r"If the entries of a query $q$ and key $k$ are independent with mean 0 and variance 1, then "
    r"$q\cdot k=\sum_{j=1}^{d_k} q_j k_j$ has variance $d_k$. Dividing by $\sqrt{d_k}$ brings the "
    r"scores back to unit variance; without it, large $d_k$ pushes the softmax into saturation "
    r"(one weight $\approx 1$) where its gradient vanishes. $M$ is $0$ for allowed keys and "
    r"$-\infty$ for masked ones (padding, or future tokens in a decoder), so they get exactly zero weight."
)
HEAD = r"\mathrm{head}_i=\operatorname{Attention}\!\left(XW_i^{Q},\;XW_i^{K},\;XW_i^{V}\right),\qquad W_i^{Q},W_i^{K},W_i^{V}\in\mathbb{R}^{d\times d_h},\; d_h=\tfrac{d}{h}"
MHA = r"\operatorname{MultiHead}(X)=\operatorname{Concat}\!\left(\mathrm{head}_1,\dots,\mathrm{head}_h\right)W^{O},\qquad W^{O}\in\mathbb{R}^{hd_h\times d}"
MHA_WHY = (
    "Each head runs its own scaled dot-product attention on a $d_h$-wide projection, in parallel, so "
    "different heads can attend to different relations (the next word, the matching word, the "
    "subject of the verb). Concatenating $h$ heads of width $d_h = d/h$ restores width $d$, and "
    "$W^O$ mixes them. The cost is about the same as one head of full width: the parameters "
    "$3d\\cdot d + d\\cdot d = 4d^2$ do not depend on $h$."
)
SHAPES = [
    ("Input $X$", "$[B, T, d]$", "batch $B$, sequence length $T$, model width $d$"),
    ("$Q, K, V = XW^Q, XW^K, XW^V$", "$[B, T, d]$ each", "one fused $[d, 3d]$ projection in `src/transformer.py`"),
    ("split into heads", "$[B, h, T, d_h]$", "reshape $d \\to h \\times d_h$, then move heads forward"),
    ("scores $QK^\\top/\\sqrt{d_h}$", "$[B, h, T, T]$", "every query against every key, per head"),
    ("softmax over keys", "$[B, h, T, T]$", "each row sums to 1 (the heatmap on My Encoder Block)"),
    ("weights $\\cdot V$", "$[B, h, T, d_h]$", "weighted average of value vectors"),
    ("concat heads", "$[B, T, h\\,d_h] = [B, T, d]$", "move heads back, merge"),
    ("$\\cdot\\, W^O$", "$[B, T, d]$", "same shape as the input: blocks stack"),
]
BLOCK = r"h=x+\operatorname{MultiHead}\big(\operatorname{LN}_1(x)\big),\qquad \text{out}=h+W_2\,\operatorname{ReLU}\!\big(W_1\operatorname{LN}_2(h)\big)"
EQUIVARIANCE = (
    "Attention alone is **permutation-equivariant**: shuffle the input rows and the output rows "
    "shuffle the same way, because every score is a dot product between two rows and nothing "
    "records where a row came from. A transformer is therefore blind to word order until "
    "position is added. The *My Encoder Block* tab lets you switch that off and watch it happen."
)
SINUSOIDAL = r"\mathrm{PE}_{p,2i}=\sin\!\big(p\,\omega_i\big),\qquad \mathrm{PE}_{p,2i+1}=\cos\!\big(p\,\omega_i\big),\qquad \omega_i=10000^{-2i/d}"
ROPE = r"\tilde q_m = R_{\Theta,m}\,q_m,\quad \tilde k_n = R_{\Theta,n}\,k_n,\qquad \tilde q_m^{\top}\tilde k_n = q_m^{\top}R_{\Theta,\,n-m}\,k_n"
ROPE_PAIR = r"R_{\Theta,m}\Big|_{(2i,\,2i+1)}=\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix},\qquad \theta_i=10000^{-2i/d_h}"
PE_TABLE = [
    # scheme, how it enters, what attention sees, params, longer-than-trained inputs, used by
    ("Sinusoidal (2017)", "added to the embeddings once, before block 1",
     "absolute position mixed into every content vector; relative offsets only indirectly",
     "none", "defined for any position, but quality drops past training length",
     "original Transformer; this project's hand-written encoder"),
    ("Learned absolute", "a trained vector per position, added to the embeddings",
     "absolute position, whatever the model learned",
     "$T_{\\max}\\times d$", "impossible: no vector exists for position $T_{\\max}+1$",
     "BERT, GPT-2"),
    ("Rotary (RoPE, 2021)", "rotates $q$ and $k$ inside every attention layer",
     "the score depends on content and on the **relative** offset $n-m$ only",
     "none", "rotations are defined everywhere; can be stretched (NTK / YaRN scaling) to longer contexts",
     "LLaMA, Mistral, Qwen, most open LLMs since 2023"),
]
ROPE_WHY = [
    "**Relative position for free.** Rotating $q_m$ by angle $m\\theta$ and $k_n$ by $n\\theta$ makes "
    "their dot product depend only on $n-m$: language cares far more about how far apart two words are "
    "than about their absolute index, and RoPE gives exactly that without extra parameters.",
    "**It leaves the content alone.** Sinusoidal and learned schemes *add* position into the same vector "
    "that carries meaning, all the way through the network. RoPE applies only where position matters "
    "(the $q\\cdot k$ score) and is reapplied in every layer, so no layer has to preserve it.",
    "**No length ceiling, and it can be stretched.** A learned table stops at its last row. Rotations "
    "are defined for any $m$, and rescaling $\\theta_i$ (position interpolation, NTK-aware scaling, "
    "YaRN) extends a trained model's context, which is how many models went from 4k to 100k+ tokens.",
    "**It fits fast inference.** Each key is rotated once by its own absolute position and cached; "
    "new queries are rotated by theirs, so a KV cache needs no recomputation as the text grows.",
]
