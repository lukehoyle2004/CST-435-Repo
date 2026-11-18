"""Load and run the two routers the service exposes.

* ``fine_tuned``: the deployed DistilBERT checkpoint (the sweep's best on validation).
* ``zero_shot``:  flan-t5-small scored by label likelihood (src/zero_shot.py).

Render's free instance has 512 MB. The two models together (~250 MB + ~310 MB of
float32 weights, plus torch itself) do not fit, so the registry keeps at most
``FTF_MAX_RESIDENT`` models in memory and evicts the least recently used one.
A request that has to load its model reports that time as ``load_ms`` and is
flagged ``cold_start`` -- the cost is visible rather than hidden in the latency.

The deployed checkpoint is found, in order, at ``FTF_MODEL_DIR`` (default
``models/deployed``) or downloaded from the Hugging Face repo ``FTF_MODEL_REPO``
at build time (scripts/fetch_model.py).
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.checkpoints import MAX_LENGTH
from shared.queues import PRIORITY_QUEUE, QUEUE_NAMES

ROOT = Path(__file__).resolve().parents[1]

# The routing policy, set by design before testing: below REVIEW_THRESHOLD the top
# queue is not trusted, and any message whose fraud probability reaches FRAUD_FLOOR
# without winning goes to a human anyway -- a fraud report routed to the fees team
# is the expensive mistake. docs/experiments/review_policy.json measures what it
# catches on the test set (1.3% of messages flagged; 22 of 100 errors caught).
REVIEW_THRESHOLD = float(os.environ.get("FTF_REVIEW_THRESHOLD", "0.60"))
FRAUD_FLOOR = float(os.environ.get("FTF_FRAUD_FLOOR", "0.10"))


def memory_mb() -> dict:
    """Resident and peak memory of this process (Linux /proc; empty elsewhere)."""
    try:
        status = Path("/proc/self/status").read_text()
    except OSError:
        return {}
    out = {}
    for line in status.splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            out["rss_mb" if key == "VmRSS" else "peak_mb"] = round(int(rest.split()[0]) / 1024, 1)
    return out


def release_memory() -> None:
    """Give freed model memory back to the OS before the next model loads.

    glibc keeps freed heap memory for reuse; on a 512 MB instance the evicted model's
    pages must actually be returned, or loading the next model runs out of memory.
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass


def model_dir() -> Path:
    return Path(os.environ.get("FTF_MODEL_DIR", ROOT / "models" / "deployed"))


def deployed_meta() -> dict:
    """``{run_name, artifact_sha256}`` written next to the deployed weights by the sweep."""
    f = model_dir() / "ftf_run.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"run_name": os.environ.get("FTF_DEPLOYED_RUN", "unknown"), "artifact_sha256": None}


def needs_review(scores: dict[str, float]) -> bool:
    top = max(scores, key=scores.get)
    return scores[top] < REVIEW_THRESHOLD or (top != PRIORITY_QUEUE and scores[PRIORITY_QUEUE] >= FRAUD_FLOOR)


@dataclass
class Router:
    kind: str
    run_name: str
    checkpoint: str
    proba: Callable[[list[str]], torch.Tensor]       # texts -> (n, 8) probabilities
    count_tokens: Callable[[str], int]


def load_fine_tuned() -> Router:
    d = model_dir()
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"no deployed checkpoint at {d} (run scripts/run_sweep.py or scripts/fetch_model.py)")
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if labels != QUEUE_NAMES:
        raise ValueError(f"checkpoint labels {labels} do not match the queue contract")
    meta = deployed_meta()

    @torch.inference_mode()
    def proba(texts: list[str]) -> torch.Tensor:
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return model(**enc).logits.softmax(-1)

    return Router("fine_tuned", meta["run_name"], f"{meta['run_name']}@{(meta.get('artifact_sha256') or 'local')[:12]}",
                  proba, lambda t: len(tok(t, truncation=True, max_length=MAX_LENGTH)["input_ids"]))


def load_zero_shot() -> Router:
    from src.zero_shot import ZeroShotRouter
    prompt_file = ROOT / "docs" / "experiments" / "zero_shot_prompt_selection.json"
    choice = json.loads(prompt_file.read_text()) if prompt_file.exists() else {}
    # FTF_ZERO_SHOT_HALF_WEIGHTS=1 stores the weights in float16 (computing in float32)
    # so the model fits Render's 512 MB instance; docs/experiments/zero_shot_served_variant.json
    # measures how closely it agrees with the float32 baseline on the leaderboard.
    half = os.environ.get("FTF_ZERO_SHOT_HALF_WEIGHTS", "0") == "1"
    z = ZeroShotRouter(choice.get("chosen", "options"), calibrate=choice.get("calibrate", True), half_storage=half)
    return Router("zero_shot", "zero-shot-flan-t5-small", z.checkpoint, z.scores,
                  lambda t: len(z.tok(z.build(t)[0])["input_ids"]))


LOADERS: dict[str, Callable[[], Router]] = {"fine_tuned": load_fine_tuned, "zero_shot": load_zero_shot}


class Registry:
    """At most ``max_resident`` routers in memory, least recently used evicted first."""

    def __init__(self, max_resident: int):
        self.max_resident = max(1, max_resident)
        self._models: OrderedDict[str, Router] = OrderedDict()
        self._lock = threading.Lock()
        self.last_error: dict[str, str] = {}
        self.memory_after_load: dict[str, dict] = {}

    def resident(self) -> list[str]:
        return list(self._models)

    def get(self, kind: str) -> tuple[Router, float]:
        """Return ``(router, load_ms)``; ``load_ms`` is 0 when it was already in memory."""
        with self._lock:
            if kind in self._models:
                self._models.move_to_end(kind)
                return self._models[kind], 0.0
            evicted = False
            while len(self._models) >= self.max_resident:
                self._models.popitem(last=False)
                evicted = True
            if evicted:
                release_memory()              # free the evicted weights before loading the next model
            t0 = time.perf_counter()
            try:
                router = LOADERS[kind]()
            except Exception as exc:
                self.last_error[kind] = f"{type(exc).__name__}: {exc}"
                raise
            self.last_error.pop(kind, None)
            self._models[kind] = router
            self.memory_after_load[kind] = memory_mb()
            # Never report a zero load for a real load (the DB check ties cold_start to load_ms > 0).
            return router, max((time.perf_counter() - t0) * 1000, 0.001)


def predict(router: Router, texts: list[str]) -> tuple[list[dict[str, float]], float]:
    """Return per-text ``{queue: prob}`` and the model compute time in ms."""
    t0 = time.perf_counter()
    probs = router.proba(texts)
    ms = (time.perf_counter() - t0) * 1000
    return [{q: float(p) for q, p in zip(QUEUE_NAMES, row.tolist())} for row in probs], ms
