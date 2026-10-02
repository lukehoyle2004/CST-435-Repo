"""Pydantic request/response models shared by the API and (optionally) the UI.

Keeping every wire-format type in one module is the contract between the three
clouds. The Streamlit UI never imports model or SQL code -- it only mirrors these
schemas so that the payloads it sends match what FastAPI expects.

This is "Income-Insight": tabular binary classification (income >50K) on the
UCI Adult census data, with a deep MLP, calibration, and a bias audit.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Activation = Literal["relu", "leaky_relu", "tanh", "sigmoid", "gelu"]
Attribute = Literal["sex", "race"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for POST /train. Trains on the fixed adult_income splits."""

    n_layers: int = Field(2, ge=1, le=6, description="Number of hidden layers (depth).")
    hidden_dim: int = Field(32, ge=1, le=512, description="Width of each hidden layer.")
    activation: Activation = Field(
        "tanh",
        description="Hidden-layer non-linearity. tanh is the default: in the matched "
                    "comparison it had the best >50K F1 and better calibration than relu.",
    )
    lr: float = Field(0.005, gt=0.0, description="Adam learning rate.")
    batch_size: int = Field(256, ge=1)
    epochs: int = Field(15, ge=1, le=500)
    seed: int = Field(0, description="Fixes weight init, batch order, and subsample.")
    max_train_rows: Optional[int] = Field(
        15000, ge=500,
        description="Subsample of the training split (None = all ~34k rows). Keeps "
                    "Render's free tier under the request timeout.",
    )
    include_protected: bool = Field(
        False, description="Also feed sex and race to the model (for the audit).",
    )


class ClassMetrics(BaseModel):
    """Held-out (test-split) metrics. All None when the run diverged."""

    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    roc_auc: Optional[float]


class Run(BaseModel):
    """A training-run row as stored in Supabase (summary; no blobs)."""

    model_config = ConfigDict(protected_namespaces=())

    id: int
    data_source: str
    activation: str
    n_layers: int
    hidden_dim: int
    lr: float
    batch_size: int
    epochs: int
    seed: int
    max_train_rows: Optional[int]
    include_protected: bool
    status: str = Field(..., description="'ok' or 'diverged'.")
    final_loss: Optional[float]
    train_accuracy: Optional[float]
    val_accuracy: Optional[float]
    accuracy: Optional[float] = Field(..., description="Test-split accuracy.")
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    roc_auc: Optional[float]
    brier: Optional[float]
    ece: Optional[float]
    created_at: datetime


class CalibrationBin(BaseModel):
    bin_lower: float
    bin_upper: float
    mean_pred: float
    frac_positive: float
    count: int


class ClassScores(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class Importance(BaseModel):
    feature: str
    importance: float = Field(..., description="Mean drop in val ROC-AUC when shuffled.")
    std: float


class RunDetail(Run):
    """Everything the Model Performance tab shows for one run."""

    n_train: Optional[int]
    n_val: Optional[int]
    n_test: Optional[int]
    confusion: Optional[Dict[str, int]]
    class_report: Optional[Dict[str, ClassScores]]
    calibration: Optional[List[CalibrationBin]]
    perm_importance: Optional[List[Importance]]
    loss_history: List[Optional[float]]
    val_loss_history: List[Optional[float]]


class TrainResponse(BaseModel):
    run_id: int
    status: str = Field(..., description="'ok' or 'diverged'.")
    final_loss: Optional[float]
    train_accuracy: Optional[float]
    val_accuracy: Optional[float]
    metrics: ClassMetrics
    n_test_predictions_logged: int


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Request body for POST /predict -- one record keyed by the schema features."""

    run_id: Optional[int] = Field(None, description="Defaults to the newest ok run.")
    features: Dict[str, object] = Field(
        ..., description="One record, e.g. {'age': 39, 'education_num': 13, ...}."
    )
    threshold: float = Field(
        0.5, gt=0.0, lt=1.0, description="Predict >50K when P(>50K) >= threshold."
    )


class PredictResponse(BaseModel):
    run_id: int
    label: int = Field(..., description="0 = <=50K, 1 = >50K.")
    income: str = Field(..., description="Human-readable class label.")
    proba: float = Field(..., description="P(income > 50K).")
    threshold: float


class PredictBatchRequest(BaseModel):
    """Request body for POST /predict_batch (the Score-a-CSV tab)."""

    run_id: Optional[int] = None
    records: List[Dict[str, object]] = Field(..., min_length=1, max_length=5000)
    threshold: float = Field(0.5, gt=0.0, lt=1.0)


class BatchItem(BaseModel):
    label: int
    income: str
    proba: float


class PredictBatchResponse(BaseModel):
    run_id: int
    threshold: float
    n_rows: int
    predictions: List[BatchItem]


# ---------------------------------------------------------------------------
# Schema, bias audit, activation comparison
# ---------------------------------------------------------------------------
class SchemaResponse(BaseModel):
    """Feature contract, so the UI can build its input form dynamically."""

    numeric_features: List[str]
    categorical_features: List[str]
    protected_features: List[str]
    categories: Dict[str, List[str]]
    target_name: str
    target_classes: List[str]
    activations: List[str]


class BiasGroup(BaseModel):
    grp: str
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    positive_rate: float
    base_rate: float
    fpr: Optional[float] = Field(..., description="FP / (FP + TN).")
    fnr: Optional[float] = Field(..., description="FN / (FN + TP).")


class BiasAuditResponse(BaseModel):
    run_id: int
    attribute: str
    groups: List[BiasGroup]
    fpr_gap: Optional[float] = Field(..., description="max FPR - min FPR across groups.")
    fnr_gap: Optional[float] = Field(..., description="max FNR - min FNR across groups.")


class ActivationRow(BaseModel):
    activation: str
    n_layers: int
    hidden_dim: int
    lr: float
    batch_size: int
    epochs: int
    seed: int
    max_train_rows: Optional[int]
    include_protected: bool
    runs: int
    latest_run_id: int
    train_accuracy: Optional[float]
    val_accuracy: Optional[float]
    test_accuracy: Optional[float]
    test_f1: Optional[float]
    test_roc_auc: Optional[float]
    brier: Optional[float]
    ece: Optional[float]
    final_train_loss: Optional[float]


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loader: bool
    supabase: bool
    default_run_id: Optional[int] = Field(
        None, description="Run whose pipeline was loaded at startup."
    )


class Version(BaseModel):
    git_sha: str
    torch_version: str
    sklearn_version: str
    supabase_project_ref: Optional[str]
