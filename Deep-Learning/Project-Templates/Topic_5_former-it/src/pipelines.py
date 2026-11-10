"""The three pretrained archetypes, loaded lazily inside a memory budget.

=============  ===============  ============================  =================
model          archetype        default checkpoint            task
=============  ===============  ============================  =================
``bert``       encoder          ``distilbert-base-uncased``   fill one [MASK]
``gpt2``       decoder          ``distilgpt2``                continue the text
``t5``         encoder-decoder  ``t5-small``                  summarise the text
=============  ===============  ============================  =================

Render's free instance has 512 MB of RAM, and torch alone takes a large part of
it, so the three checkpoints cannot all stay resident. ``ModelRegistry`` keeps at
most ``TRIO_MAX_RESIDENT`` (default 1) loaded, evicting the least recently used
one, and serialises model use behind a lock so two requests can never hold two
different models in memory at once. Every call reports how long loading took
(``load_ms``, 0 when the model was already resident) separately from the model
computation itself (``latency_ms``), because on this instance the cold-start
cost is a large part of the honest cost story.

Checkpoints are pinned to exact Hub commits (overridable through
``TRIO_BERT_CHECKPOINT`` / ``TRIO_BERT_REVISION`` and so on) and are reported by
``GET /version`` together with the commit hash actually loaded (provenance).
"""
from __future__ import annotations

import gc
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch

MASK_PLACEHOLDER = "[MASK]"


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one served archetype."""

    name: str
    archetype: str
    task: str
    checkpoint: str
    revision: str
    auto_class: str            # transformers Auto* class name


def _spec(name: str, archetype: str, task: str, default_ckpt: str, default_rev: str, auto_class: str) -> ModelSpec:
    key = name.upper()
    return ModelSpec(
        name=name,
        archetype=archetype,
        task=task,
        checkpoint=os.environ.get(f"TRIO_{key}_CHECKPOINT", default_ckpt),
        revision=os.environ.get(f"TRIO_{key}_REVISION", default_rev),
        auto_class=auto_class,
    )


# Revisions are pinned to the exact Hub commits the service was tested and
# benchmarked with, so a silent upstream change cannot alter what is served.
SPECS: Dict[str, ModelSpec] = {
    "bert": _spec("bert", "encoder", "fill-mask", "distilbert-base-uncased",
                  "12040accade4e8a0f71eabdb258fecc2e7e948be", "AutoModelForMaskedLM"),
    "gpt2": _spec("gpt2", "decoder", "continue", "distilgpt2",
                  "2290a62682d06624634c1f46a6ad5be0f47f38aa", "AutoModelForCausalLM"),
    "t5": _spec("t5", "encoder-decoder", "summarize", "t5-small",
                "df1b051c49625cf57a3d0d8d3863ed4d13564fe4", "AutoModelForSeq2SeqLM"),
}


@dataclass
class Bundle:
    """A loaded checkpoint: model, tokenizer and where it came from."""

    spec: ModelSpec
    model: Any
    tokenizer: Any
    commit: Optional[str]
    params: int
    loaded_at: float

    @property
    def checkpoint_id(self) -> str:
        """``checkpoint@commit`` (first 10 hex chars) for provenance in the log."""
        return f"{self.spec.checkpoint}@{(self.commit or self.spec.revision)[:10]}"


def _dtype() -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[os.environ.get("TRIO_DTYPE", "float32")]


def release_memory() -> None:
    """Collect garbage and, on Linux, hand freed heap pages back to the OS."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def load_bundle(spec: ModelSpec) -> Bundle:
    """Load ``spec`` from the Hugging Face cache (downloading on first use)."""
    import transformers

    auto = getattr(transformers, spec.auto_class)
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.checkpoint, revision=spec.revision)
    model = auto.from_pretrained(spec.checkpoint, revision=spec.revision, torch_dtype=_dtype())
    model.eval()
    commit = getattr(model.config, "_commit_hash", None)
    return Bundle(spec, model, tokenizer, commit, sum(p.numel() for p in model.parameters()), time.time())


class ModelRegistry:
    """LRU cache of loaded checkpoints with a hard cap on how many are resident.

    Args:
        max_resident: How many checkpoints may be in memory at once.
        loader: Function that turns a ``ModelSpec`` into a ``Bundle`` (tests
            inject a fake one so no weights are downloaded).
    """

    def __init__(self, max_resident: int = 1, loader: Callable[[ModelSpec], Bundle] = load_bundle) -> None:
        self.max_resident = max(1, max_resident)
        self._loader = loader
        self._resident: "OrderedDict[str, Bundle]" = OrderedDict()
        self._lock = threading.RLock()
        self.loads = 0
        self.evictions = 0
        self.last_error: Optional[str] = None

    @contextmanager
    def use(self, name: str) -> Iterator[Tuple[Bundle, float]]:
        """Hold model ``name`` for the duration of a request.

        Yields ``(bundle, load_ms)``. The lock is held while the caller runs
        inference, which keeps peak memory at ``max_resident`` checkpoints.
        """
        spec = SPECS[name]
        with self._lock:
            load_ms = 0.0
            bundle = self._resident.get(name)
            if bundle is None:
                while len(self._resident) >= self.max_resident:
                    self._resident.popitem(last=False)
                    self.evictions += 1
                    release_memory()
                t0 = time.perf_counter()
                try:
                    bundle = self._loader(spec)
                except Exception as exc:
                    self.last_error = f"{name}: {exc}"
                    raise
                load_ms = (time.perf_counter() - t0) * 1000
                self._resident[name] = bundle
                self.loads += 1
                self.last_error = None
            self._resident.move_to_end(name)
            yield bundle, load_ms

    def status(self) -> dict:
        """Snapshot for ``/healthz``: what is loaded and whether loading works."""
        return {
            "max_resident": self.max_resident,
            "resident": list(self._resident),
            "available": list(SPECS),
            "loads": self.loads,
            "evictions": self.evictions,
            "last_error": self.last_error,
        }

    def provenance(self) -> Dict[str, dict]:
        """Configured checkpoint/revision per model, plus the loaded commit if resident."""
        out = {}
        for name, spec in SPECS.items():
            b = self._resident.get(name)
            out[name] = {
                "archetype": spec.archetype,
                "task": spec.task,
                "checkpoint": spec.checkpoint,
                "revision": spec.revision,
                "loaded_commit": b.commit if b else None,
                "params": b.params if b else None,
            }
        return out


# ---------------------------------------------------------------------------
# Inference (each returns (result dict, input_tokens, output_tokens, latency_ms))
# ---------------------------------------------------------------------------
def _timed(fn: Callable[[], Tuple[dict, int, int]]) -> Tuple[dict, int, int, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        result, n_in, n_out = fn()
    return result, n_in, n_out, (time.perf_counter() - t0) * 1000


def prepare_masked_text(text: str, mask_word: Optional[str]) -> str:
    """Return the text with exactly one ``[MASK]``.

    Either the caller already wrote one ``[MASK]``, or they name a ``mask_word``
    whose first whole-word occurrence is replaced. Raises ``ValueError`` otherwise.
    """
    count = text.count(MASK_PLACEHOLDER)
    if count == 1 and mask_word is None:
        return text
    if count > 1:
        raise ValueError("the text must contain exactly one [MASK]")
    if count == 1 and mask_word is not None:
        raise ValueError("give either a [MASK] in the text or mask_word, not both")
    if not mask_word:
        raise ValueError("put [MASK] in the text, or name a mask_word to hide")
    import re

    pattern = re.compile(rf"(?<!\w){re.escape(mask_word)}(?!\w)")
    if not pattern.search(text):
        raise ValueError(f"mask_word {mask_word!r} does not occur in the text")
    return pattern.sub(MASK_PLACEHOLDER, text, count=1)


def bert_fill_mask(bundle: Bundle, masked_text: str, top_k: int = 5) -> Tuple[dict, int, int, float]:
    """Rank candidates for the single ``[MASK]`` in ``masked_text``."""
    tok, model = bundle.tokenizer, bundle.model

    def run():
        text = masked_text.replace(MASK_PLACEHOLDER, tok.mask_token)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        positions = (enc["input_ids"][0] == tok.mask_token_id).nonzero().flatten()
        if len(positions) != 1:
            raise ValueError("the [MASK] was cut off by the 512-token limit; shorten the text")
        logits = model(**enc).logits[0, positions[0]].float()          # [vocab]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, top_k)
        candidates = [
            {"token": tok.decode([int(i)]).strip(), "score": round(float(p), 5)}
            for p, i in zip(top.values, top.indices)
        ]
        filled = masked_text.replace(MASK_PLACEHOLDER, candidates[0]["token"], 1)
        return {"candidates": candidates, "filled_text": filled}, int(enc["input_ids"].shape[1]), 1

    return _timed(run)


def gpt2_continue(
    bundle: Bundle,
    text: str,
    max_new_tokens: int = 40,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
) -> Tuple[dict, int, int, float]:
    """Continue ``text`` left to right (greedy unless ``do_sample``)."""
    tok, model = bundle.tokenizer, bundle.model

    def run():
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"][:, -(1024 - max_new_tokens):]           # keep the most recent context
        attn = enc["attention_mask"][:, -ids.shape[1]:]
        kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=do_sample,
        )
        if do_sample:
            torch.manual_seed(seed)
            kwargs.update(temperature=temperature, top_p=top_p)
        out = model.generate(ids, attention_mask=attn, **kwargs)
        new = out[0, ids.shape[1]:]
        return {"continuation": tok.decode(new, skip_special_tokens=True)}, int(ids.shape[1]), int(new.numel())

    return _timed(run)


def t5_summarize(
    bundle: Bundle, text: str, max_new_tokens: int = 60, min_new_tokens: int = 10, num_beams: int = 2
) -> Tuple[dict, int, int, float]:
    """Summarise ``text`` with T5's ``summarize:`` task prefix."""
    tok, model = bundle.tokenizer, bundle.model

    def run():
        enc = tok("summarize: " + text, return_tensors="pt", truncation=True, max_length=512)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min(min_new_tokens, max_new_tokens),
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        summary = tok.decode(out[0], skip_special_tokens=True)
        n_out = int((out[0] != tok.pad_token_id).sum())
        return {"summary": summary}, int(enc["input_ids"].shape[1]), n_out

    return _timed(run)
