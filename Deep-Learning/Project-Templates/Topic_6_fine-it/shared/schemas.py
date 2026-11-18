"""Pydantic request/response models: the contract between FineTuneFleet and its callers.

FleetDeck (Product 6B) and any other client depend only on these shapes.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

EXAMPLE_MESSAGE = "I don't recognise a payment on my statement from a shop I've never been to"

ModelKind = Literal["fine_tuned", "zero_shot"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=[EXAMPLE_MESSAGE],
                      description="One customer message. It is never stored -- only its SHA-256.")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32,
                             description="Up to 32 messages, each 1-2000 characters.")
    model: ModelKind = "fine_tuned"


class Prediction(BaseModel):
    queue: str = Field(..., description="The queue the message is routed to.")
    confidence: float = Field(..., ge=0, le=1, description="Probability of that queue.")
    needs_review: bool = Field(..., description="Confidence below the review threshold, or the "
                                                "runner-up is the fraud queue: send to a human.")
    scores: Dict[str, float] = Field(..., description="Probability per queue, summing to 1.")
    model: ModelKind
    run_name: str = Field(..., description="The ftf_experiments row that served this prediction.")
    experiment_id: Optional[int] = Field(None, description="Its id in the experiment store (null if the store was unreachable at lookup).")
    checkpoint: str
    latency_ms: float = Field(..., description="Model compute only, measured on the server.")
    load_ms: float = Field(..., description="Time spent loading the model during this request (0 when warm).")
    cold_start: bool
    n_tokens: int
    prediction_id: Optional[int] = Field(None, description="ftf_predictions.id, or null if it could not be logged.")
    logged: bool


class BatchResponse(BaseModel):
    model: ModelKind
    predictions: List[Prediction]
    total_latency_ms: float = Field(..., description="Sum of model compute across the batch.")
    wall_ms: float = Field(..., description="Server-side wall time for the whole batch, including logging.")
    logged: int = Field(..., description="How many of the predictions reached the store.")


class Experiment(BaseModel):
    rank: int
    id: int
    run_name: str
    kind: ModelKind
    base_checkpoint: str
    seed: int
    hyperparams: dict
    n_train: int
    val_macro_f1: float
    test_accuracy: float
    test_accuracy_lo: float
    test_accuracy_hi: float
    test_macro_f1: float
    test_macro_f1_lo: float
    test_macro_f1_hi: float
    fraud_recall: float
    fraud_recall_lo: float
    fraud_recall_hi: float
    delta_macro_f1: Optional[float] = None
    delta_macro_f1_lo: Optional[float] = None
    delta_macro_f1_hi: Optional[float] = None
    train_runtime_s: float
    eval_ms_per_message: float
    is_deployed: bool
    created_at: Optional[str] = None
    distinguishable_from_best: bool


class ExperimentsResponse(BaseModel):
    experiments: List[Experiment]
    deployed_run: Optional[str]
    best_run: Optional[str]
    deployed_is_best: bool = Field(..., description="The served checkpoint is the top fine-tuned row on the leaderboard.")
    source: str = "ftf_leaderboard (Supabase)"


class LatencyRow(BaseModel):
    endpoint: str
    run_name: Optional[str] = None
    n_calls: int
    n_cold: int
    mean_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    mean_load_ms: Optional[float] = None
    mean_tokens: Optional[float] = None
    review_rate: Optional[float] = None
    last_served_at: Optional[str] = None


class Metrics(BaseModel):
    since_start: Dict[str, dict] = Field(..., description="This process only: calls, errors, unlogged, mean latency per endpoint.")
    store: Optional[List[LatencyRow]] = Field(None, description="All-time aggregates from the ftf_latency view (null if unreachable).")
    store_error: Optional[str] = None
    resident_models: List[str]
    uptime_s: float


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    database: dict
    models: dict
    deployed_run: str


class Version(BaseModel):
    api_version: str
    git_sha: str
    python: str
    torch: str
    transformers: str
    fine_tuned: dict
    zero_shot: dict
    queues: List[str]
