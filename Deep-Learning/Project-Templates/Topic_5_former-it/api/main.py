"""TrioFormer -- FastAPI inference service (Render.com).

One service, four transformers, one honest latency log:

====================  ======================  ===============================
endpoint              model                   archetype
====================  ======================  ===============================
POST /encoder_attention  hand-written block    encoder (built in src/transformer.py)
POST /bert_fillmask      distilbert-base-uncased  encoder
POST /gpt2_continue      distilgpt2              decoder
POST /t5_summarize       t5-small                encoder-decoder
GET  /benchmarks         trio_benchmarks materialized view (Supabase)
GET  /healthz            model loader + database client state
GET  /version            code, library and checkpoint provenance
====================  ======================  ===============================

Every model endpoint measures its latency on the server and writes the input's
SHA-256 hash, the output and the latency to Supabase *before* it responds. If
the database is unreachable the answer is still returned, with ``logged: false``
and an ``X-Trio-Logged: false`` header, so an outage degrades the audit trail
loudly instead of taking the product down.

    uvicorn api.main:app --reload --port 8000      # then open /docs
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api import db
from shared.schemas import (
    BenchmarksResponse,
    BertFillMaskRequest,
    BertFillMaskResponse,
    EncoderAttentionRequest,
    EncoderAttentionResponse,
    Gpt2ContinueRequest,
    Gpt2ContinueResponse,
    Health,
    T5SummarizeRequest,
    T5SummarizeResponse,
    Version,
)
from src import pipelines, toy

log = logging.getLogger("trioformer")

# Render's free tier is a fraction of one CPU; more threads only contend.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

API_VERSION = "1.0.0"
GIT_SHA = os.environ.get("RENDER_GIT_COMMIT", "local")[:7]
REFRESH_EVERY_S = float(os.environ.get("TRIO_REFRESH_EVERY_S", "15"))
MAX_STORED_CHARS = 1000

REGISTRY = pipelines.ModelRegistry(int(os.environ.get("TRIO_MAX_RESIDENT", "1")))
STATE: Dict[str, Any] = {"last_refresh": 0.0}


def load_encoder_block() -> None:
    """Load the toy classifier built from the hand-written block (tiny; kept resident)."""
    if "toy" not in STATE:
        STATE["toy"], STATE["toy_meta"] = toy.load_classifier()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_encoder_block()
    yield


TAGS = [
    {"name": "models", "description": "The four transformers. Each call is timed on the server and "
                                      "logged (input hash, output, latency) before the response is sent."},
    {"name": "benchmarks", "description": "Per-model latency aggregated by the `trio_benchmarks` materialized view."},
    {"name": "ops", "description": "Health and provenance."},
]

app = FastAPI(
    title="TrioFormer API",
    summary="BERT vs GPT-2 vs T5 (plus a hand-written encoder block), side by side, with server-side latency.",
    description=(
        "Give it a paragraph: **BERT** fills a masked word, **GPT-2** continues the text and **T5** "
        "summarises it. `/encoder_attention` runs an encoder written from scratch and returns its "
        "attention heatmaps.\n\n"
        "**Privacy.** The submitted text is never stored: Supabase keeps its SHA-256 hash and size, "
        "the model's output and the latency. Only the aggregate `/benchmarks` numbers are public.\n\n"
        "**Latency fields.** `latency_ms` is model compute only; `load_ms` is time spent loading the "
        "checkpoint during this request (the instance keeps one pretrained model in memory at a time, "
        "so switching models costs a cold start); `total_ms` is the whole server-side request."
    ),
    version=API_VERSION,
    openapi_tags=TAGS,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trio-Logged", "X-Trio-Latency-Ms"],
)


# --------------------------------------------------------------------------- helpers
def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _cap(value: Any) -> Any:
    """Truncate long strings in an output before storing it (keeps rows small)."""
    if isinstance(value, str):
        return value[:MAX_STORED_CHARS]
    if isinstance(value, dict):
        return {k: _cap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cap(v) for v in value]
    return value


def record(
    response: Response,
    *,
    text: str,
    model: str,
    archetype: str,
    checkpoint: str,
    task: str,
    stored_output: dict,
    latency_ms: float,
    load_ms: float,
    input_tokens: int,
    output_tokens: int,
    source: str,
    started: float,
) -> dict:
    """Log one inference to Supabase (before responding) and build the shared response fields."""
    meta = dict(
        model=model, archetype=archetype, task=task, checkpoint=checkpoint,
        input_sha256=sha256_hex(text), input_tokens=input_tokens, output_tokens=output_tokens,
        latency_ms=round(latency_ms, 2), load_ms=round(load_ms, 2), cold_start=load_ms > 0,
    )
    inference_id: Optional[int] = None
    try:
        inference_id = db.log_inference(dict(
            input_sha256=meta["input_sha256"], n_chars=len(text.strip()), n_words=len(text.split()),
            model=model, archetype=archetype, checkpoint=checkpoint, task=task,
            output=_cap(stored_output), latency_ms=meta["latency_ms"], load_ms=meta["load_ms"],
            cold_start=meta["cold_start"], input_tokens=input_tokens, output_tokens=output_tokens,
            source=source, api_version=GIT_SHA,
        ))
    except Exception as exc:  # outage: the answer still goes back, flagged as unlogged
        log.warning("could not log %s inference: %s", model, type(exc).__name__)
    logged = inference_id is not None
    response.headers["X-Trio-Logged"] = str(logged).lower()
    response.headers["X-Trio-Latency-Ms"] = str(meta["latency_ms"])
    return {**meta, "total_ms": round((time.perf_counter() - started) * 1000, 2),
            "inference_id": inference_id, "logged": logged}


def run_pretrained(name: str, fn: Callable[[pipelines.Bundle], tuple]):
    """Run ``fn`` against model ``name``; map bad input to 422 and load failures to 503."""
    try:
        with REGISTRY.use(name) as (bundle, load_ms):
            result, n_in, n_out, latency_ms = fn(bundle)
            return bundle, load_ms, result, n_in, n_out, latency_ms
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except (OSError, RuntimeError, ImportError) as exc:
        log.error("model %s unavailable: %s", name, exc)
        raise HTTPException(503, f"Model '{name}' is unavailable right now ({type(exc).__name__}). Retry shortly.")


# --------------------------------------------------------------------------- models
@app.post("/encoder_attention", response_model=EncoderAttentionResponse, tags=["models"])
def encoder_attention(req: EncoderAttentionRequest, response: Response) -> dict:
    """Run the **hand-written** transformer encoder (`src/transformer.py`) on 2-12 words.

    The encoder was trained offline to tell whether a sequence of words is a palindrome
    ("dog bites man bites dog"). Words are relabelled by first appearance, so the model never
    needs the words themselves. Returns the prediction and every layer's per-head attention matrix
    (`attention[layer][head][query][key]`) plus per-head summary statistics.

    Set `positional_encoding: false` to run the same weights without positional encoding: the
    encoder then sees a bag of words, so "a b a" and "a a b" get the same answer."""
    started = time.perf_counter()
    words = re.findall(r"[a-z0-9']+", req.text.lower())
    if not toy.MIN_LEN <= len(words) <= toy.MAX_LEN:
        raise HTTPException(422, f"give between {toy.MIN_LEN} and {toy.MAX_LEN} words (got {len(words)})")
    load_encoder_block()
    model, meta = STATE["toy"], STATE["toy_meta"]
    ids = toy.canonical_ids(words)
    tokens, mask = toy.pad_batch([ids])
    n = len(ids)

    t0 = time.perf_counter()
    with torch.inference_mode():
        logits, attns = model(tokens, mask, req.positional_encoding)
        probs = torch.softmax(logits[0], dim=-1)
    latency_ms = (time.perf_counter() - t0) * 1000

    label = int(probs.argmax())
    prediction = toy.CLASS_NAMES[label]
    attention = [[[[round(float(v), 5) for v in row[:n]] for row in a[0, h, :n]] for h in range(a.shape[1])]
                 for a in attns]
    summary = [toy.attention_summary(ids, a[0]) for a in attns]
    meta_fields = record(
        response, text=req.text, model="encoder_block", archetype="encoder",
        checkpoint=f"hand-written@{meta['weights_sha256'][:10]}", task="attention",
        stored_output={"prediction": prediction, "confidence": round(float(probs[label]), 4),
                       "positional_encoding": req.positional_encoding},
        latency_ms=latency_ms, load_ms=0.0, input_tokens=n, output_tokens=1,
        source=req.source, started=started,
    )
    return {
        **meta_fields, "words": words, "ids": ids, "prediction": prediction,
        "probabilities": {c: round(float(p), 5) for c, p in zip(toy.CLASS_NAMES, probs)},
        "n_layers": len(attns), "n_heads": attns[0].shape[1], "attention": attention,
        "head_summary": summary, "model_accuracy": meta["test"]["accuracy"],
        "positional_encoding": req.positional_encoding,
    }


@app.post("/bert_fillmask", response_model=BertFillMaskResponse, tags=["models"])
def bert_fillmask(req: BertFillMaskRequest, response: Response) -> dict:
    """**Encoder.** BERT reads the whole paragraph in both directions at once and ranks the
    words that fit the one masked position. Send text containing exactly one `[MASK]`, or plain
    text plus `mask_word` (its first occurrence is hidden). The stored output is the ranked
    candidate list, not the filled-in paragraph."""
    started = time.perf_counter()
    try:
        masked = pipelines.prepare_masked_text(req.text, req.mask_word)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    bundle, load_ms, result, n_in, n_out, latency_ms = run_pretrained(
        "bert", lambda b: pipelines.bert_fill_mask(b, masked, req.top_k))
    meta = record(
        response, text=req.text, model="bert", archetype="encoder", checkpoint=bundle.checkpoint_id,
        task="fill-mask", stored_output={"candidates": result["candidates"]},
        latency_ms=latency_ms, load_ms=load_ms, input_tokens=n_in, output_tokens=n_out,
        source=req.source, started=started,
    )
    return {**meta, "masked_text": masked, **result}


@app.post("/gpt2_continue", response_model=Gpt2ContinueResponse, tags=["models"])
def gpt2_continue(req: Gpt2ContinueRequest, response: Response) -> dict:
    """**Decoder.** GPT-2 predicts one token at a time, left to right, each conditioned only on
    what came before, and appends `max_new_tokens` of continuation. Greedy by default so latency
    comparisons are repeatable."""
    started = time.perf_counter()
    bundle, load_ms, result, n_in, n_out, latency_ms = run_pretrained(
        "gpt2", lambda b: pipelines.gpt2_continue(
            b, req.text, req.max_new_tokens, req.do_sample, req.temperature, req.top_p, req.seed))
    meta = record(
        response, text=req.text, model="gpt2", archetype="decoder", checkpoint=bundle.checkpoint_id,
        task="continue", stored_output=result, latency_ms=latency_ms, load_ms=load_ms,
        input_tokens=n_in, output_tokens=n_out, source=req.source, started=started,
    )
    return {**meta, **result}


@app.post("/t5_summarize", response_model=T5SummarizeResponse, tags=["models"])
def t5_summarize(req: T5SummarizeRequest, response: Response) -> dict:
    """**Encoder-decoder.** T5's encoder reads the whole paragraph once; its decoder then writes
    a new, shorter sequence while cross-attending to that encoding (`summarize:` task prefix)."""
    started = time.perf_counter()
    bundle, load_ms, result, n_in, n_out, latency_ms = run_pretrained(
        "t5", lambda b: pipelines.t5_summarize(
            b, req.text, req.max_new_tokens, req.min_new_tokens, req.num_beams))
    meta = record(
        response, text=req.text, model="t5", archetype="encoder-decoder", checkpoint=bundle.checkpoint_id,
        task="summarize", stored_output=result, latency_ms=latency_ms, load_ms=load_ms,
        input_tokens=n_in, output_tokens=n_out, source=req.source, started=started,
    )
    return {**meta, **result}


# --------------------------------------------------------------------------- benchmarks
@app.get("/benchmarks", response_model=BenchmarksResponse, tags=["benchmarks"])
def benchmarks(refresh: bool = Query(True, description=f"Refresh the materialized view first "
                                                       f"(at most once every {REFRESH_EVERY_S:g} s).")) -> dict:
    """Per-model latency from the `trio_benchmarks` materialized view: call counts (warm vs cold
    start), mean / p50 / p95 / min / max model latency, mean checkpoint load time, tokens in and
    out, and milliseconds per output token."""
    refreshed = False
    try:
        if refresh and time.monotonic() - STATE["last_refresh"] >= REFRESH_EVERY_S:
            db.refresh_benchmarks()
            STATE["last_refresh"] = time.monotonic()
            refreshed = True
        rows = db.get_benchmarks()
    except Exception as exc:
        log.warning("benchmarks unavailable: %s", type(exc).__name__)
        raise HTTPException(503, "The benchmarks database is temporarily unavailable. Please retry shortly.")
    return {"refreshed": refreshed, "rows": rows}


# --------------------------------------------------------------------------- ops
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> dict:
    """Always 200 while the process is up. `status` is `degraded` when the encoder block is not
    loaded, the last checkpoint load failed, or Supabase did not answer."""
    loader = REGISTRY.status()
    loader["encoder_block_loaded"] = "toy" in STATE
    loader["ok"] = loader["encoder_block_loaded"] and loader["last_error"] is None
    database = db.ping()
    status = "ok" if loader["ok"] and database["ok"] else "degraded"
    return {"status": status, "model_loader": loader, "database": database}


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> dict:
    """Which code, libraries and checkpoints are serving (provenance for every logged row)."""
    try:
        import transformers

        tf_version = transformers.__version__
    except ImportError:  # pragma: no cover
        tf_version = None
    load_encoder_block()
    meta = STATE["toy_meta"]
    return {
        "api": API_VERSION, "git_sha": GIT_SHA, "python": platform.python_version(),
        "torch": torch.__version__, "transformers": tf_version,
        "dtype": os.environ.get("TRIO_DTYPE", "float32"),
        "models": REGISTRY.provenance(),
        "encoder_block": {"source": "src/transformer.py", "task": meta["task"],
                          "architecture": meta["architecture"], "params": meta["params"],
                          "test_accuracy": meta["test"]["accuracy"],
                          "weights_sha256": meta["weights_sha256"]},
    }


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
