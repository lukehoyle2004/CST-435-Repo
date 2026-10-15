"""FastAPI model service -- Cloud #2 (Render.com).

Hosts both trained CIFAR-10 CNNs (baseline + improved), the preprocessing, and
the Grad-CAM hook. Logs every prediction to Supabase as a hash + dimensions +
answer -- never the image. No UI code lives here.

    uvicorn api.main:app --reload --port 8000      # then open /docs
"""
from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api import db
from seesense import CLASSES, MODELS_DIR
from seesense.inference import explain, open_image, png_b64, sha256_hex
from seesense.model import count_parameters, load_exported

log = logging.getLogger("seesense")

# Render's free tier is a fraction of one CPU; more threads only contend.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

MODEL_NAMES = ("improved", "baseline")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MODELS: dict[str, tuple] = {}     # name -> (model, config)


def load_models() -> None:
    for name in MODEL_NAMES:
        if name not in MODELS:
            MODELS[name] = load_exported(MODELS_DIR / name)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_models()   # load once at startup, not on the first request
    yield


TAGS = [
    {"name": "predict", "description": "Classify an uploaded image and explain it with Grad-CAM."},
    {"name": "runs", "description": "Training runs persisted in Supabase."},
    {"name": "predictions", "description": "The anonymized prediction log (hashes only)."},
    {"name": "meta", "description": "Health, version, and label metadata."},
]

app = FastAPI(
    title="See-Sense API",
    description="CIFAR-10 CNN classifier with Grad-CAM explanations. Raw images are never stored.",
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


def get_model(name: str):
    load_models()
    if name not in MODELS:
        raise HTTPException(422, f"model must be one of {list(MODEL_NAMES)}")
    return MODELS[name]


def db_unavailable(exc: Exception) -> HTTPException:
    log.warning("supabase error: %s", exc)
    return HTTPException(503, "Database temporarily unavailable. Please retry shortly.")


# --------------------------------------------------------------------------- predict
@app.post("/predict", tags=["predict"])
async def predict(
    file: UploadFile = File(..., description="PNG / JPEG / WebP / BMP, at most 5 MB"),
    model: Literal["improved", "baseline"] = Form("improved"),
    tta: bool = Form(True, description="Average with the mirror image (flip test-time augmentation)"),
):
    """Classify one image. Returns the label, all 10 probabilities, the top 5, and two
    base64 PNG Grad-CAM overlays (full size and the 32x32 model view). Logs the image's
    SHA-256 hash and dimensions -- not the image -- to `predictions` + `image_metadata`."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 5 MB.")
    try:
        img = open_image(data)
    except ValueError as exc:
        raise HTTPException(415, str(exc))
    fmt = Image.open(io.BytesIO(data)).format

    net, config = get_model(model)
    start = time.perf_counter()
    res = explain(net, img, tta=tta)
    latency_ms = (time.perf_counter() - start) * 1000
    digest = sha256_hex(data)

    prediction_id, logged = None, False
    try:
        prediction_id = db.log_prediction(
            dict(image_sha256=digest, predicted_label=res["label"], predicted_class=res["class_name"],
                 confidence=res["confidence"], top5=res["top5"], served_by_run_id=config["run_id"],
                 tta=tta, latency_ms=round(latency_ms, 2)),
            dict(image_sha256=digest, width=img.width, height=img.height, format=fmt,
                 size_bytes=len(data), predicted_class=res["class_name"]),
        )
        logged = True
    except Exception as exc:  # the prediction still goes back to the user
        log.warning("could not log prediction: %s", exc)

    return dict(
        label=res["label"],
        class_name=res["class_name"],
        confidence=res["confidence"],
        probabilities=res["probabilities"],
        top5=res["top5"],
        gradcam_png=png_b64(res["overlay"]),
        gradcam_size=list(res["overlay"].size),
        model_view_png=png_b64(res["model_view"]),
        image_sha256=digest,
        width=img.width,
        height=img.height,
        model=model,
        served_by_run_id=config["run_id"],
        tta=tta,
        latency_ms=round(latency_ms, 2),
        prediction_id=prediction_id,
        logged=logged,
    )


@app.get("/gallery", tags=["predict"])
def gallery():
    """Both models on the same held-out gallery of CIFAR-10 test images (Compare Models tab)."""
    g = np.load(MODELS_DIR / "gallery.npz")
    items = []
    for x, y, idx in zip(g["images"], g["labels"], g["test_index"]):
        img = Image.fromarray(x.transpose(1, 2, 0)).resize((96, 96), Image.Resampling.NEAREST)
        row = dict(test_index=int(idx), true_class=CLASSES[int(y)], image_png=png_b64(img), models={})
        for name in MODEL_NAMES:
            net, config = get_model(name)
            r = explain(net, Image.fromarray(x.transpose(1, 2, 0)))
            row["models"][name] = dict(class_name=r["class_name"], confidence=r["confidence"],
                                       correct=r["label"] == int(y), run_id=config["run_id"],
                                       gradcam_png=png_b64(r["model_view"]))
        items.append(row)
    return items


# --------------------------------------------------------------------------- runs
@app.get("/runs", tags=["runs"])
def runs(limit: int = Query(20, ge=1, le=100)):
    """Recent training runs from Supabase."""
    try:
        return db.list_runs(limit)
    except Exception as exc:
        raise db_unavailable(exc)


@app.get("/runs/{run_id}", tags=["runs"])
def run_detail(run_id: str):
    """One run with its per-class report, confusion matrix, and training curves."""
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
    """Recent predictions joined to the run that served them."""
    try:
        return db.recent_predictions(limit)
    except Exception as exc:
        raise db_unavailable(exc)


# --------------------------------------------------------------------------- meta
@app.get("/classes", tags=["meta"])
def classes():
    return {"classes": CLASSES}


@app.get("/models", tags=["meta"])
def models():
    """The served models and the run each one came from."""
    out = []
    for name in MODEL_NAMES:
        net, config = get_model(name)
        out.append(dict(name=name, run_id=config["run_id"], params=count_parameters(net),
                        epochs=config["epochs"], baseline=bool(config.get("baseline"))))
    return out


@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "ok", "models_loaded": sorted(MODELS), "db": db.ping()}


@app.get("/version", tags=["meta"])
def version():
    return {"git_sha": os.environ.get("RENDER_GIT_COMMIT", "local")[:7],
            "torch": torch.__version__, "api": app.version}
