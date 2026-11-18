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

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from shared.checkpoints import MAX_LENGTH, ZERO_SHOT_MODEL, ZERO_SHOT_REVISION
from shared.queues import QUEUE_NAMES, QUEUES

# A float16 copy of the pinned flan-t5-small, written once by scripts/fetch_model.py.
# Converting the float32 checkpoint at load time leaves both copies' buffers behind;
# loading a file that is already float16 does not.
HALF_DIR = Path(os.environ.get("FTF_ZERO_SHOT_HALF_DIR",
                               Path(__file__).resolve().parents[1] / "models" / "zero_shot_fp16"))


def save_half_copy(dest: Path = HALF_DIR) -> Path:
    model = AutoModelForSeq2SeqLM.from_pretrained(ZERO_SHOT_MODEL, revision=ZERO_SHOT_REVISION)
    model.half().save_pretrained(dest)
    return dest

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


def _compute_in_float32(model: torch.nn.Module) -> None:
    """Keep Linear/Embedding weights in float16 but do every computation in float32.

    Halves the resident weights (~310 MB -> ~155 MB) so the model fits a 512 MB
    instance. Only the stored weights are rounded; activations, norms and the
    softmax stay float32 (a plain float16 model overflows inside T5). Each layer
    upcasts its own weight just for its matmul, so the extra memory is one layer.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.forward = (lambda mod: lambda x: F.linear(
                x, mod.weight.float(), None if mod.bias is None else mod.bias.float()))(m)
        elif isinstance(m, torch.nn.Embedding):
            m.forward = (lambda mod: lambda ids: F.embedding(ids, mod.weight).float())(m)
        else:
            for p in m.parameters(recurse=False):      # layer norms: tiny, keep float32
                p.data = p.data.float()


class ZeroShotRouter:
    """Label-likelihood scoring with flan-t5-small. Thread-unsafe; one instance per process."""

    def __init__(self, prompt: str = DEFAULT_PROMPT, calibrate: bool = True, device: str = "cpu",
                 half_storage: bool = False, on_stage=lambda stage: None):
        self.prompt_name = prompt
        self.calibrate = calibrate
        self.build = PROMPTS[prompt]
        self.device = device
        self.half_storage = half_storage
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(ZERO_SHOT_MODEL, revision=ZERO_SHOT_REVISION)
        on_stage("zero-shot tokenizer loaded")
        if half_storage and (HALF_DIR / "config.json").exists():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(HALF_DIR, dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                ZERO_SHOT_MODEL, revision=ZERO_SHOT_REVISION, low_cpu_mem_usage=True,
                dtype=torch.float16 if half_storage else torch.float32)
        if half_storage:
            on_stage("zero-shot float16 weights loaded")
            _compute_in_float32(self.model)
        self.model.eval().to(device)
        self.load_ms = (time.perf_counter() - t0) * 1000
        _, labels = self.build("x")
        enc = self.tok(labels, padding=True, return_tensors="pt")
        self.label_ids = enc.input_ids.to(device)             # (8, L) incl. </s>
        self.label_mask = enc.attention_mask.to(device)
        on_stage("zero-shot model ready, before calibration")
        self.prior = self._raw(CONTENT_FREE).mean(0) if calibrate else None
        on_stage("zero-shot calibrated")

    @property
    def checkpoint(self) -> str:
        tag = "[fp16-weights]" if self.half_storage else ""
        return f"{ZERO_SHOT_MODEL}@{ZERO_SHOT_REVISION}{tag}"

    def scores(self, texts: list[str]) -> torch.Tensor:
        """(len(texts), 8) probabilities over queues (calibrated when enabled)."""
        # Small chunks: the decoder logits are (texts x 8 labels x tokens x 32k vocab).
        p = torch.cat([self._raw(texts[i:i + 4]) for i in range(0, len(texts), 4)])
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
        # decoder_input_ids instead of labels: same logits, but no loss (and no
        # second vocabulary-sized tensor) is computed.
        dec_in = self.model._shift_right(labels.masked_fill(mask == 0, -100))
        logits = self.model(encoder_outputs=(hidden,), attention_mask=attn,
                            decoder_input_ids=dec_in).logits.float()
        logp = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
        ll = (logp * mask).sum(-1) / mask.sum(-1)              # mean log-prob per label token
        return ll.view(n, k).softmax(-1)

    def apply_prior(self, p: torch.Tensor) -> torch.Tensor:
        p = p / self.prior
        return p / p.sum(-1, keepdim=True)

    def predict_batch(self, texts: list[str], batch_size: int = 32, calibrated: bool = True) -> torch.Tensor:
        fn = self.scores if calibrated else self._raw
        return torch.cat([fn(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)])
