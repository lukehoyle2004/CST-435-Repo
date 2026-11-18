"""FineTuneFleet -- the fine-tuned support-ticket router as a FastAPI service (Render.com).

======================  ================================================================
endpoint                what it does
======================  ================================================================
POST /predict             route a message with the deployed fine-tuned DistilBERT
POST /predict_zero_shot   route the same message with the zero-shot flan-t5-small baseline
POST /predict_batch       up to 32 messages through either model (the cost benchmark)
GET  /experiments         the leaderboard, read from the experiment store (Supabase)
GET  /metrics             latency and error counters: this process + all-time from the store
GET  /queues              the eight queues and what each one owns
GET  /healthz             database + model loader state
GET  /version             code, library and checkpoint provenance
======================  ================================================================

Every prediction is written to ``ftf_predictions`` *before* the response goes out,
joined to the ``ftf_experiments`` row of the model that served it. If Supabase is
unreachable the prediction is still returned, with ``logged: false`` and an
``X-FTF-Logged: false`` header, so an outage degrades the audit trail loudly
instead of taking routing down. The message itself is never stored.

    uvicorn api.main:app --reload --port 8000      # then open /docs
"""
from __future__ import annotations

import logging
import os
import platform
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import torch
import transformers
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api import db
from shared.checkpoints import ZERO_SHOT_MODEL, ZERO_SHOT_REVISION
from shared.queues import QUEUE_NAMES, QUEUES
from shared.schemas import (BatchRequest, BatchResponse, ExperimentsResponse, Health,
                            Metrics, Prediction, PredictRequest, Version)
from shared.text import clean, sha256
from src import serving

log = logging.getLogger("finetunefleet")

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

API_VERSION = "1.0.0"
GIT_SHA = os.environ.get("RENDER_GIT_COMMIT", "local")[:7]
SOURCE = os.environ.get("FTF_SOURCE", "api")          # tests set "test" so their rows never count
EXPERIMENT_TTL_S = 300.0

REGISTRY = serving.Registry(int(os.environ.get("FTF_MAX_RESIDENT", "2")))
STATE: Dict[str, Any] = {"experiment_ids": {}, "ids_at": 0.0, "started": time.time(), "counters": {}}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the fine-tuned model so the first /predict is not a cold start.
    try:
        REGISTRY.get("fine_tuned")
    except Exception as exc:                            # reported by /healthz, not fatal
        log.error("fine-tuned model failed to load at startup: %s", exc)
    yield


TAGS = [
    {"name": "routing", "description": "Route a customer message to one of eight support queues. "
                                       "Each call is timed on the server and logged (hash, answer, latency, "
                                       "experiment) before the response is sent."},
    {"name": "experiments", "description": "The fine-tuning sweep and the zero-shot baseline, from the experiment store."},
    {"name": "ops", "description": "Metrics, health and provenance."},
]

app = FastAPI(
    title="FineTuneFleet API",
    summary="A fine-tuned DistilBERT support-ticket router, served next to its zero-shot baseline.",
    description=(
        "A digital bank's support desk receives messages that must reach one of eight teams. "
        "`/predict` routes with DistilBERT fine-tuned on Banking77; `/predict_zero_shot` asks "
        "flan-t5-small the same question with no training, for comparison. `/experiments` is the "
        "leaderboard of every configuration tried, with 95% bootstrap confidence intervals.\n\n"
        "**Privacy.** A message is never stored: Supabase keeps its SHA-256 hash, its length, the "
        "queue chosen, the probabilities and the latency.\n\n"
        "**Human review.** `needs_review` is true when the model is not confident or when the fraud "
        "queue was a close call; the bank's process sends those messages to a person."
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
    expose_headers=["X-FTF-Logged", "X-FTF-Latency-Ms"],
)


# --------------------------------------------------------------------------- helpers
def _count(endpoint: str, ms: float = 0.0, error: bool = False, unlogged: int = 0, n: int = 1) -> None:
    c = STATE["counters"].setdefault(endpoint, {"calls": 0, "errors": 0, "unlogged": 0, "latency_ms_sum": 0.0})
    c["calls"] += n
    c["errors"] += int(error)
    c["unlogged"] += unlogged
    c["latency_ms_sum"] += ms


def experiment_id(run_name: str) -> Optional[int]:
    """The ftf_experiments id for a run, cached; None if the store cannot be reached."""
    ids = STATE["experiment_ids"]
    if run_name not in ids or time.time() - STATE["ids_at"] > EXPERIMENT_TTL_S:
        try:
            ids.update(db.experiment_ids(["zero-shot-flan-t5-small", serving.deployed_meta()["run_name"]]))
            STATE["ids_at"] = time.time()
        except Exception as exc:
            log.warning("experiment lookup failed: %s", type(exc).__name__)
    return ids.get(run_name)


def _route(kind: str, texts: list[str], endpoint: str, source: str) -> list[Prediction]:
    try:
        router, load_ms = REGISTRY.get(kind)
    except Exception:
        _count(endpoint, error=True, n=len(texts))
        raise HTTPException(503, f"The {kind.replace('_', '-')} model is not available right now. "
                                 "Retry in a minute; /healthz shows the loader state.")
    cleaned = [clean(t) for t in texts]
    if any(not t for t in cleaned):
        raise HTTPException(422, "Every message must contain non-whitespace text.")
    scores, ms = serving.predict(router, cleaned)
    per_ms = ms / len(cleaned)
    exp_id = experiment_id(router.run_name)
    out = []
    for i, (text, s) in enumerate(zip(cleaned, scores)):
        top = max(s, key=s.get)
        this_load = load_ms if i == 0 else 0.0
        pred = Prediction(queue=top, confidence=round(s[top], 4), needs_review=serving.needs_review(s),
                          scores={q: round(v, 4) for q, v in s.items()}, model=kind,
                          run_name=router.run_name, experiment_id=exp_id, checkpoint=router.checkpoint,
                          latency_ms=round(per_ms, 2), load_ms=max(round(this_load, 1), 0.1) if this_load > 0 else 0.0, cold_start=this_load > 0,
                          n_tokens=router.count_tokens(text), prediction_id=None, logged=False)
        if exp_id is not None:
            try:
                pred.prediction_id = db.log_prediction({
                    "experiment_id": exp_id, "endpoint": "predict" if kind == "fine_tuned" else "predict_zero_shot",
                    "input_sha256": sha256(text), "n_chars": len(text), "n_tokens": pred.n_tokens,
                    "predicted_queue": top, "confidence": pred.confidence, "needs_review": pred.needs_review,
                    "scores": pred.scores, "latency_ms": pred.latency_ms, "load_ms": pred.load_ms,
                    "cold_start": pred.cold_start, "source": source, "api_version": GIT_SHA,
                })
                pred.logged = True
            except Exception as exc:
                log.warning("prediction not logged: %s", type(exc).__name__)
        out.append(pred)
    _count(endpoint, ms + load_ms, unlogged=sum(not p.logged for p in out), n=len(out))
    return out


def _headers(response: Response, preds: list[Prediction]) -> None:
    response.headers["X-FTF-Logged"] = "true" if all(p.logged for p in preds) else "false"
    response.headers["X-FTF-Latency-Ms"] = f"{sum(p.latency_ms for p in preds):.2f}"


# --------------------------------------------------------------------------- routing
@app.post("/predict", response_model=Prediction, tags=["routing"])
def predict(req: PredictRequest, response: Response) -> Prediction:
    """Route one message with the deployed fine-tuned checkpoint."""
    [pred] = _route("fine_tuned", [req.text], "predict", SOURCE)
    _headers(response, [pred])
    return pred


@app.post("/predict_zero_shot", response_model=Prediction, tags=["routing"])
def predict_zero_shot(req: PredictRequest, response: Response) -> Prediction:
    """Route the same message with the zero-shot baseline (no task-specific training)."""
    [pred] = _route("zero_shot", [req.text], "predict_zero_shot", SOURCE)
    _headers(response, [pred])
    return pred


@app.post("/predict_batch", response_model=BatchResponse, tags=["routing"])
def predict_batch(req: BatchRequest, response: Response) -> BatchResponse:
    """Route up to 32 messages in one forward pass. Rows are logged with source='benchmark'."""
    for t in req.texts:
        if not 1 <= len(t) <= 2000:
            raise HTTPException(422, "Each message must be 1-2000 characters.")
    t0 = time.perf_counter()
    source = "test" if SOURCE == "test" else "benchmark"
    preds = _route(req.model, req.texts, "predict_batch", source)
    _headers(response, preds)
    return BatchResponse(model=req.model, predictions=preds,
                         total_latency_ms=round(sum(p.latency_ms for p in preds), 2),
                         wall_ms=round((time.perf_counter() - t0) * 1000, 2),
                         logged=sum(p.logged for p in preds))


# --------------------------------------------------------------------------- experiments
@app.get("/experiments", response_model=ExperimentsResponse, tags=["experiments"])
def experiments() -> ExperimentsResponse:
    """The leaderboard: every sweep configuration plus the zero-shot row, ranked by test macro-F1."""
    try:
        rows = db.get_leaderboard()
    except Exception as exc:
        log.warning("leaderboard unavailable: %s", type(exc).__name__)
        raise HTTPException(503, "The experiment store is temporarily unavailable; try again shortly.")
    deployed = next((r["run_name"] for r in rows if r["is_deployed"]), None)
    tuned = [r for r in rows if r["kind"] == "fine_tuned"]
    best = tuned[0]["run_name"] if tuned else None
    return ExperimentsResponse(experiments=rows, deployed_run=deployed, best_run=best,
                               deployed_is_best=deployed is not None and deployed == best)


@app.get("/queues", tags=["experiments"])
def queues() -> dict:
    """The routing contract: eight queues and what each one owns."""
    return {"queues": [{"name": q, "description": QUEUES[q]} for q in QUEUE_NAMES],
            "review_threshold": serving.REVIEW_THRESHOLD, "fraud_floor": serving.FRAUD_FLOOR}


# --------------------------------------------------------------------------- ops
@app.get("/metrics", response_model=Metrics, tags=["ops"])
def metrics() -> Metrics:
    """Counters since this process started, plus all-time latency aggregates from the store."""
    since = {k: {**v, "mean_latency_ms": round(v["latency_ms_sum"] / v["calls"], 2) if v["calls"] else None}
             for k, v in STATE["counters"].items()}
    store, err = None, None
    try:
        store = db.get_latency()
    except Exception as exc:
        err = f"store unavailable ({type(exc).__name__})"
    return Metrics(since_start=since, store=store, store_error=err,
                   resident_models=REGISTRY.resident(), uptime_s=round(time.time() - STATE["started"], 1))


@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    database = db.ping()
    models = {"resident": REGISTRY.resident(), "max_resident": REGISTRY.max_resident,
              "errors": dict(REGISTRY.last_error), "memory_mb": serving.memory_mb(),
              "memory_after_load_mb": dict(REGISTRY.memory_after_load)}
    ok = database["ok"] and not REGISTRY.last_error.get("fine_tuned")
    return Health(status="ok" if ok else "degraded", database=database, models=models,
                  deployed_run=serving.deployed_meta()["run_name"])


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    meta = serving.deployed_meta()
    return Version(api_version=API_VERSION, git_sha=GIT_SHA, python=platform.python_version(),
                   torch=torch.__version__, transformers=transformers.__version__,
                   fine_tuned={"run_name": meta["run_name"], "artifact_sha256": meta.get("artifact_sha256"),
                               "source": str(serving.model_dir().name)},
                   zero_shot={"checkpoint": f"{ZERO_SHOT_MODEL}@{ZERO_SHOT_REVISION}"},
                   queues=QUEUE_NAMES)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")
