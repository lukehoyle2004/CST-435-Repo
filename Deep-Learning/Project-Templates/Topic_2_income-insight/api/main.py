"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Responsibilities:
  * expose the MLP behind HTTP so the Streamlit UI can call it over HTTPS,
  * read the UCI Adult rows (adult_income) from Supabase before training,
  * write run rows, model artifacts, and prediction rows back to Supabase,
  * serve the SQL-computed bias audit and activation comparison.

There is NO UI code here and NO business logic in the UI -- separation of
concerns across the three clouds. This is "Income-Insight": tabular binary
classification with a PyTorch MLP + sklearn preprocessing pipeline.
"""
from __future__ import annotations

import logging
import os
import subprocess
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import sklearn
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import db
from api.training import (
    ACTIVATIONS,
    STATUS_DIVERGED,
    STATUS_OK,
    load_artifact,
    predict_records,
    train_income_classifier,
)
from shared.data import (
    CATEGORICAL_COLS,
    CATEGORIES,
    NUMERIC_COLS,
    PROTECTED_COLS,
    TARGET_CLASSES,
    TARGET_NAME,
)
from shared.schemas import (
    ActivationRow,
    Attribute,
    BatchItem,
    BiasAuditResponse,
    BiasGroup,
    ClassMetrics,
    Health,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    Run,
    RunDetail,
    SchemaResponse,
    TrainRequest,
    TrainResponse,
    Version,
)

log = logging.getLogger("income_insight")

# Render's free tier gives a fraction of one CPU; extra torch threads only fight
# over it. Override with TORCH_NUM_THREADS on bigger hardware.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

# Loaded (preprocessor, network, columns) per run_id, newest last.
_ARTIFACT_CACHE: "OrderedDict[int, Tuple]" = OrderedDict()
_CACHE_SIZE = 8


def _cache_artifact(run_id: int, model_b64: str) -> Tuple:
    loaded = load_artifact(model_b64)
    _ARTIFACT_CACHE[run_id] = loaded
    _ARTIFACT_CACHE.move_to_end(run_id)
    while len(_ARTIFACT_CACHE) > _CACHE_SIZE:
        _ARTIFACT_CACHE.popitem(last=False)
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the newest successful run's sklearn+torch pipeline at startup, so
    the first /predict does not pay for a download + unpickle. A missing model
    or an unreachable database is logged, not fatal: /healthz reports it."""
    app.state.default_run_id = None
    try:
        run_id = db.latest_ok_run_id()
        if run_id is not None:
            model_b64 = db.get_run_artifact(run_id)
            if model_b64:
                _cache_artifact(run_id, model_b64)
                app.state.default_run_id = run_id
                log.info("loaded pipeline for run %s at startup", run_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("startup model load skipped: %s", exc)
    yield


app = FastAPI(
    title="Income-Insight API",
    description="Deep MLP income classifier on UCI Adult, with calibration and a bias audit.",
    version="2.0.0",
    lifespan=lifespan,
)

# The UI lives on a different origin (Streamlit Cloud), so CORS must allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(db.DatabaseError)
async def database_error_handler(_request: Request, exc: db.DatabaseError) -> JSONResponse:
    """Supabase down or misconfigured -> 503 with a readable message, never a 500."""
    log.error("database error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": f"database unavailable: {exc}"})


def _git_sha() -> str:
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _project_ref() -> str | None:
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


def _income(label: int) -> str:
    return TARGET_CLASSES[label]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(req: TrainRequest) -> TrainResponse:
    """Train on the fixed adult_income splits, persist the run, its artifact,
    and one prediction per test row (the input to the SQL bias audit)."""
    data = db.load_adult_frame()
    result = train_income_classifier(
        data,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        n_layers=req.n_layers,
        activation=req.activation,
        seed=req.seed,
        max_train_rows=req.max_train_rows,
        include_protected=req.include_protected,
    )
    fields = {
        "data_source": "adult_income",
        "activation": req.activation,
        "n_layers": req.n_layers,
        "hidden_dim": req.hidden_dim,
        "lr": req.lr,
        "batch_size": req.batch_size,
        "epochs": req.epochs,
        "seed": req.seed,
        "max_train_rows": req.max_train_rows,
        "include_protected": req.include_protected,
        "status": result.status,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "n_test": result.n_test,
        "train_accuracy": result.train_accuracy,
        "val_accuracy": result.val_accuracy,
        **result.metrics,
        "brier": result.brier,
        "ece": result.ece,
        "confusion": result.confusion,
        "class_report": result.class_report,
        "calibration": result.calibration,
        "perm_importance": result.perm_importance,
    }
    run = db.insert_run(fields, result.loss_history, result.val_loss_history,
                        result.model_b64)

    logged = 0
    if result.status == STATUS_OK:
        logged = db.insert_predictions(
            db.eval_prediction_rows(run["id"], result.test_predictions)
        )
        _cache_artifact(run["id"], result.model_b64)
        app.state.default_run_id = run["id"]

    return TrainResponse(
        run_id=run["id"],
        status=result.status,
        final_loss=run.get("final_loss"),
        train_accuracy=result.train_accuracy,
        val_accuracy=result.val_accuracy,
        metrics=ClassMetrics(**result.metrics),
        n_test_predictions_logged=logged,
    )


@app.get("/runs/{run_id}", response_model=RunDetail, tags=["training"])
def get_run(run_id: int) -> RunDetail:
    """One run with its confusion matrix, per-class scores, calibration bins,
    permutation importance, and loss curves (the Model Performance tab)."""
    row = db.get_run_detail(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return RunDetail(**row)


@app.get("/runs", response_model=list[Run], tags=["training"])
def list_runs() -> list[Run]:
    """Return the latest 50 runs from Supabase."""
    return [Run(**r) for r in db.latest_runs(limit=50)]


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
def _resolve_run_id(run_id: Optional[int]) -> int:
    if run_id is not None:
        return run_id
    default = getattr(app.state, "default_run_id", None) or db.latest_ok_run_id()
    if default is None:
        raise HTTPException(status_code=404, detail="no trained model yet; POST /train first")
    return default


def _load_model_or_error(run_id: int) -> Tuple:
    if run_id in _ARTIFACT_CACHE:
        _ARTIFACT_CACHE.move_to_end(run_id)
        return _ARTIFACT_CACHE[run_id]
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    if run.get("status") == STATUS_DIVERGED:
        # Serving a diverged model would return garbage with a straight face.
        raise HTTPException(
            status_code=409,
            detail="run diverged during training; retrain with a smaller lr",
        )
    model_b64 = db.get_run_artifact(run_id)
    if model_b64 is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return _cache_artifact(run_id, model_b64)


def _score(model, records, threshold):
    try:
        return predict_records(model, records, threshold)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"could not score records: {exc}")


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """Score one row (the Score-a-Row tab); log it to Supabase."""
    run_id = _resolve_run_id(req.run_id)
    model = _load_model_or_error(run_id)
    label, proba = _score(model, [req.features], req.threshold)[0]
    db.insert_predictions([{
        "run_id": run_id, "features": req.features, "proba": proba, "label": label,
        "threshold": req.threshold, "source": "row",
    }])
    return PredictResponse(
        run_id=run_id, label=label, income=_income(label), proba=proba,
        threshold=req.threshold,
    )


@app.post("/predict_batch", response_model=PredictBatchResponse, tags=["prediction"])
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    """Score many rows at once (the Score-a-CSV tab); each row is logged."""
    run_id = _resolve_run_id(req.run_id)
    model = _load_model_or_error(run_id)
    results = _score(model, req.records, req.threshold)
    db.insert_predictions([
        {"run_id": run_id, "features": features, "proba": proba, "label": label,
         "threshold": req.threshold, "source": "csv"}
        for features, (label, proba) in zip(req.records, results)
    ])
    items = [BatchItem(label=l, income=_income(l), proba=p) for l, p in results]
    return PredictBatchResponse(
        run_id=run_id, threshold=req.threshold, n_rows=len(items), predictions=items
    )


# ---------------------------------------------------------------------------
# schema, bias audit, activation comparison
# ---------------------------------------------------------------------------
@app.get("/schema", response_model=SchemaResponse, tags=["meta"])
def schema() -> SchemaResponse:
    """Expose the feature contract so the UI can build its form dynamically."""
    return SchemaResponse(
        numeric_features=NUMERIC_COLS,
        categorical_features=CATEGORICAL_COLS,
        protected_features=PROTECTED_COLS,
        categories=CATEGORIES,
        target_name=TARGET_NAME,
        target_classes=TARGET_CLASSES,
        activations=sorted(ACTIVATIONS),
    )


@app.get("/bias_audit", response_model=BiasAuditResponse, tags=["fairness"])
def bias_audit(run_id: int, attribute: Attribute = "sex") -> BiasAuditResponse:
    """FPR / FNR per group of a protected attribute, computed by the SQL
    ``bias_audit`` view over the run's test-split predictions."""
    rows = db.bias_audit(run_id, attribute)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="no test-split predictions for this run (diverged or pre-audit run)",
        )
    groups = [BiasGroup(**r) for r in rows]
    fprs = [g.fpr for g in groups if g.fpr is not None]
    fnrs = [g.fnr for g in groups if g.fnr is not None]
    return BiasAuditResponse(
        run_id=run_id,
        attribute=attribute,
        groups=groups,
        fpr_gap=max(fprs) - min(fprs) if fprs else None,
        fnr_gap=max(fnrs) - min(fnrs) if fnrs else None,
    )


@app.get("/activation_comparison", response_model=list[ActivationRow], tags=["fairness"])
def activation_comparison() -> list[ActivationRow]:
    """The SQL ``activation_comparison`` view: ok runs grouped by matched controls."""
    return [ActivationRow(**r) for r in db.activation_comparison()]


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    """200 when the model loader (torch) and the Supabase client are reachable."""
    supabase_ok = db.ping()
    model_ok = torch.tensor([1.0]).sum().item() == 1.0
    status = "ok" if (supabase_ok and model_ok) else "degraded"
    return Health(
        status=status, model_loader=model_ok, supabase=supabase_ok,
        default_run_id=getattr(app.state, "default_run_id", None),
    )


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    return Version(
        git_sha=_git_sha(),
        torch_version=torch.__version__,
        sklearn_version=sklearn.__version__,
        supabase_project_ref=_project_ref(),
    )
