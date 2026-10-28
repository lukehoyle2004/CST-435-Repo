"""FastAPI model service -- Cloud #2 (Render.com).

Hosts both trained IMDB sentiment models (plain LSTM and LSTM + attention), the
tokenizer + vocab they were trained with, and the attention read-out. Logs every
prediction to Supabase as a hash + token counts + answer + attention vector --
never the review text. No UI code lives here.

    uvicorn api.main:app --reload --port 8000      # then open /docs
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import db
from seqsight import MODELS_DIR
from seqsight.inference import classify_tokens, guardrails, load_vocab, sha256_hex, top_tokens
from seqsight.model import MODEL_KINDS, count_parameters, load_exported
from shared.imdb import CLASS_NAMES
from shared.schemas import PredictExampleRequest, PredictRequest, PredictResponse
from shared.text import tokenize

log = logging.getLogger("seqsight")

# Render's free tier is a fraction of one CPU; more threads only contend.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

MODELS: dict[str, tuple] = {}     # kind -> (model, config)
STATE: dict = {}                  # vocab, gallery, gallery_index


def load_models() -> None:
    if "vocab" not in STATE:
        STATE["vocab"] = load_vocab()
        gallery = json.loads((MODELS_DIR / "gallery.json").read_text())
        STATE["gallery"] = gallery
        STATE["gallery_index"] = {c["test_index"]: c for cards in gallery.values() for c in cards}
    for kind in MODEL_KINDS:
        if kind not in MODELS:
            model, config = load_exported(MODELS_DIR / kind)
            if config["vocab_sha256"] != STATE["vocab"].sha256():
                raise RuntimeError(f"{kind} was trained with a different vocab.json")
            MODELS[kind] = (model, config)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_models()   # load once at startup, not on the first request
    yield


TAGS = [
    {"name": "predict", "description": "Classify a review and return per-token attention."},
    {"name": "runs", "description": "The two matched training runs persisted in Supabase."},
    {"name": "predictions", "description": "The anonymized prediction log (hashes + attention, no text)."},
    {"name": "meta", "description": "Health, version, models and labels."},
]

app = FastAPI(
    title="Seq-Sight API",
    description="IMDB sentiment: plain LSTM vs LSTM + scaled-dot-product attention. "
                "Review text is never stored.",
    version="1.0.0",
    openapi_tags=TAGS,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model(kind: str):
    load_models()
    return MODELS[kind]


def db_unavailable(exc: Exception) -> HTTPException:
    log.warning("supabase error: %s", exc)
    return HTTPException(503, "Database temporarily unavailable. Please retry shortly.")


# --------------------------------------------------------------------------- predict
def serve(kind: str, tokens: list[str], raw_tokens: int | None, digest: str,
          test_index: int | None = None, true_class: str | None = None) -> PredictResponse:
    """Classify, log to seq_predictions (best effort), and build the response."""
    net, config = get_model(kind)
    start = time.perf_counter()
    try:
        res = classify_tokens(net, STATE["vocab"], tokens, raw_tokens)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    guard = guardrails(res)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    source = "user" if test_index is None else "imdb_test"

    prediction_id, logged = None, False
    try:
        prediction_id = db.log_prediction(dict(
            text_sha256=digest, source=source, test_index=test_index,
            n_tokens=res["n_tokens"], raw_tokens=res["raw_tokens"], unk_tokens=res["unk_tokens"],
            predicted_label=res["label"], predicted_class=res["class_name"],
            confidence=res["confidence"], probabilities=res["probabilities"],
            abstained=guard["abstained"], flags=guard["flags"], model=kind,
            attention=res["attention"], served_by_run_id=config["run_id"], latency_ms=latency_ms,
        ))
        logged = True
    except Exception as exc:  # the prediction still goes back to the user
        log.warning("could not log prediction: %s", exc)

    return PredictResponse(
        **res, **guard,
        top_tokens=top_tokens(res["tokens"], res["attention"]) if res["attention"] else [],
        text_sha256=digest, source=source, test_index=test_index, true_class=true_class,
        model=kind, served_by_run_id=config["run_id"], latency_ms=latency_ms,
        prediction_id=prediction_id, logged=logged,
    )


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(req: PredictRequest) -> PredictResponse:
    """Classify a review. Returns the label, both probabilities, the scored tokens, and
    (attention model) one weight per token. Logs the text's SHA-256 hash, token counts,
    the answer and the attention vector -- not the text -- to `seq_predictions`."""
    return serve(req.model, tokenize(req.text), None, sha256_hex(req.text))


@app.post("/predict_example", response_model=PredictResponse, tags=["predict"])
def predict_example(req: PredictExampleRequest) -> PredictResponse:
    """Classify one of the public IMDB test reviews in /gallery by its test index."""
    load_models()
    card = STATE["gallery_index"].get(req.test_index)
    if card is None:
        raise HTTPException(404, f"test review {req.test_index} is not in the gallery")
    return serve(req.model, card["tokens"], card["raw_length"], sha256_hex(" ".join(card["tokens"])),
                 test_index=req.test_index, true_class=card["true"])


@app.get("/gallery", tags=["predict"])
def gallery():
    """Precomputed IMDB test reviews: clean examples, the Failure Gallery (the attention
    model's most confident mistakes), and reviews only one of the two models got right."""
    load_models()
    return STATE["gallery"]


# --------------------------------------------------------------------------- runs
@app.get("/runs", tags=["runs"])
def runs(limit: int = Query(20, ge=1, le=100)):
    """Training runs from Supabase."""
    try:
        return db.list_runs(limit)
    except Exception as exc:
        raise db_unavailable(exc)


@app.get("/runs/{run_id}", tags=["runs"])
def run_detail(run_id: str):
    """One run with per-class metrics, length buckets, calibration and training curves."""
    try:
        row = db.get_run(run_id)
    except Exception as exc:
        raise db_unavailable(exc)
    if row is None:
        raise HTTPException(404, f"run {run_id} not found")
    return row


# --------------------------------------------------------------------------- predictions
@app.get("/predictions", tags=["predictions"])
def predictions(limit: int = Query(50, ge=1, le=500)):
    """Recent predictions joined to the run that served them (no attention vectors)."""
    try:
        return db.recent_predictions(limit)
    except Exception as exc:
        raise db_unavailable(exc)


@app.get("/predictions/{prediction_id}", tags=["predictions"])
def prediction_detail(prediction_id: int):
    """One prediction with its stored attention vector. Words are shown only for public
    IMDB test reviews (looked up in the gallery); user text was never stored."""
    try:
        row = db.get_prediction(prediction_id)
    except Exception as exc:
        raise db_unavailable(exc)
    if row is None:
        raise HTTPException(404, f"prediction {prediction_id} not found")
    load_models()
    card = STATE["gallery_index"].get(row.get("test_index")) if row["source"] == "imdb_test" else None
    tokens = card["tokens"] if card else None
    return {**row, "tokens": tokens,
            "top_tokens": top_tokens(tokens, row["attention"]) if tokens and row["attention"] else []}


# --------------------------------------------------------------------------- meta
@app.get("/classes", tags=["meta"])
def classes():
    return {"classes": CLASS_NAMES}


@app.get("/models", tags=["meta"])
def models():
    """The served models and the run each one came from."""
    out = []
    for kind in MODEL_KINDS:
        net, config = get_model(kind)
        out.append(dict(name=kind, run_id=config["run_id"], params=count_parameters(net),
                        epochs=config["epochs"], best_epoch=config["best_epoch"],
                        test_acc=config["test_acc"], returns_attention=kind == "attention"))
    return out


@app.get("/healthz", tags=["meta"])
def healthz():
    """Always 200 while the process is up; `db` says whether Supabase answered."""
    return {"status": "ok", "models_loaded": sorted(MODELS), "db": db.ping()}


@app.get("/version", tags=["meta"])
def version():
    return {"git_sha": os.environ.get("RENDER_GIT_COMMIT", "local")[:7],
            "torch": torch.__version__, "api": app.version}
