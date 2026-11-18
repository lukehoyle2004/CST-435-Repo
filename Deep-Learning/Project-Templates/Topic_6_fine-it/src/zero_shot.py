"""The zero-shot baseline: an instruction-tuned T5 asked to route the message.

Free-text generation from a 77M-parameter model often answers with something
that is not a queue at all, which would score the baseline on its formatting
rather than its understanding. So the baseline is *scored*, not sampled: for each
of the eight queues we compute the length-normalised log-likelihood that the
model would write that queue's label after the prompt, and a softmax over the
eight scores gives a probability per queue. The prediction is always a valid
queue, and the baseline gets the same kind of probability vector the fine-tuned
classifier returns -- an apples-to-apples comparison.

The prompt variant was chosen on the VALIDATION split (scripts/run_sweep.py
--pick-prompt); the test split never informed it.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from shared.checkpoints import MAX_LENGTH, ZERO_SHOT_MODEL, ZERO_SHOT_REVISION
from shared.queues import QUEUE_NAMES, QUEUES

# Plain-English names the model can actually say; index-aligned with QUEUE_NAMES.
LABEL_WORDS = {
    "card_services": "cards",
    "transfers": "transfers",
    "top_ups": "top-ups",
    "cash_atm": "cash machines",
    "fees_fx": "fees and exchange rates",
    "payments_disputes": "card payments and refunds",
    "security_fraud": "fraud and lost cards",
    "account_identity": "account and identity",
}
LETTERS = "ABCDEFGH"


def _prompt_described(text: str) -> tuple[str, list[str]]:
    lines = [f"- {LABEL_WORDS[q]}: {QUEUES[q]}" for q in QUEUE_NAMES]
    prompt = ("A customer sent this message to their bank's support desk:\n"
              f"\"{text}\"\n\nWhich team should handle it?\n" + "\n".join(lines) + "\n\nTeam:")
    return prompt, [LABEL_WORDS[q] for q in QUEUE_NAMES]


def _prompt_options(text: str) -> tuple[str, list[str]]:
    lines = [f"{LETTERS[i]}. {LABEL_WORDS[q]} ({QUEUES[q]})" for i, q in enumerate(QUEUE_NAMES)]
    prompt = (f"Message to a bank: \"{text}\"\n\nWhich team should handle this message?\n\nOPTIONS:\n"
              + "\n".join(lines) + "\n\nAnswer with the letter.")
    return prompt, list(LETTERS)


def _prompt_names(text: str) -> tuple[str, list[str]]:
    names = ", ".join(LABEL_WORDS[q] for q in QUEUE_NAMES)
    prompt = (f"Route this bank customer message to one team ({names}).\n"
              f"Message: {text}\nTeam:")
    return prompt, [LABEL_WORDS[q] for q in QUEUE_NAMES]


PROMPTS = {"described": _prompt_described, "options": _prompt_options, "names": _prompt_names}
DEFAULT_PROMPT = "options"
CONTENT_FREE = ["N/A", "", "[MASK]"]


class ZeroShotRouter:
    """Label-likelihood scoring with flan-t5-small. Thread-unsafe; one instance per process."""

    def __init__(self, prompt: str = DEFAULT_PROMPT, calibrate: bool = True, device: str = "cpu"):
        self.prompt_name = prompt
        self.calibrate = calibrate
        self.build = PROMPTS[prompt]
        self.device = device
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(ZERO_SHOT_MODEL, revision=ZERO_SHOT_REVISION)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(ZERO_SHOT_MODEL, revision=ZERO_SHOT_REVISION)
        self.model.eval().to(device)
        self.load_ms = (time.perf_counter() - t0) * 1000
        _, labels = self.build("x")
        enc = self.tok(labels, padding=True, return_tensors="pt")
        self.label_ids = enc.input_ids.to(device)             # (8, L) incl. </s>
        self.label_mask = enc.attention_mask.to(device)
        self.prior = self._raw(CONTENT_FREE).mean(0) if calibrate else None

    @property
    def checkpoint(self) -> str:
        return f"{ZERO_SHOT_MODEL}@{ZERO_SHOT_REVISION}"

    def scores(self, texts: list[str]) -> torch.Tensor:
        """(len(texts), 8) probabilities over queues (calibrated when enabled)."""
        p = self._raw(texts)
        if self.prior is None:
            return p
        return self.apply_prior(p)

    @torch.inference_mode()
    def _raw(self, texts: list[str]) -> torch.Tensor:
        prompts = [self.build(t)[0] for t in texts]
        enc = self.tok(prompts, padding=True, truncation=True, max_length=4 * MAX_LENGTH,
                       return_tensors="pt").to(self.device)
        enc_out = self.model.get_encoder()(**enc)
        n, k = len(texts), self.label_ids.shape[0]
        hidden = enc_out.last_hidden_state.repeat_interleave(k, dim=0)
        attn = enc.attention_mask.repeat_interleave(k, dim=0)
        labels = self.label_ids.repeat(n, 1)
        mask = self.label_mask.repeat(n, 1)
        out = self.model(encoder_outputs=(hidden,), attention_mask=attn,
                         labels=labels.masked_fill(mask == 0, -100))
        logp = out.logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ll = (logp * mask).sum(-1) / mask.sum(-1)              # mean log-prob per label token
        return ll.view(n, k).softmax(-1)

    def apply_prior(self, p: torch.Tensor) -> torch.Tensor:
        p = p / self.prior
        return p / p.sum(-1, keepdim=True)

    def predict_batch(self, texts: list[str], batch_size: int = 32, calibrated: bool = True) -> torch.Tensor:
        fn = self.scores if calibrated else self._raw
        return torch.cat([fn(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)])
