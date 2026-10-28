"""Pydantic request/response models: the contract between the UI and the API.

Seq-Sight classifies a movie review as negative or positive with either the
plain LSTM or the LSTM + attention model, and returns one attention weight per
kept token so the UI can draw a heatmap over the words.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ModelName = Literal["attention", "lstm"]
MAX_TEXT_CHARS = 20_000   # the longest IMDB review is ~14k characters


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS, description="Review text.")
    model: ModelName = "attention"


class PredictExampleRequest(BaseModel):
    """Request body for POST /predict_example: score a public IMDB test review."""

    test_index: int = Field(..., ge=0, description="Index into the IMDB test split (see /gallery).")
    model: ModelName = "attention"


class TopToken(BaseModel):
    position: int
    token: str
    weight: float


class PredictResponse(BaseModel):
    label: int
    class_name: str
    confidence: float = Field(..., description="Softmax probability of the predicted class.")
    probabilities: Dict[str, float]
    tokens: List[str] = Field(..., description="The tokens the model scored (first 256).")
    unk: List[bool] = Field(..., description="True where a token is out of vocabulary.")
    attention: Optional[List[float]] = Field(
        None, description="One weight per token, summing to 1 (attention model only).")
    top_tokens: List[TopToken] = Field(default_factory=list)
    n_tokens: int
    raw_tokens: int
    truncated: bool
    unk_tokens: int
    text_sha256: str = Field(..., description="Hash of the input; the text itself is not stored.")
    source: Literal["user", "imdb_test"]
    test_index: Optional[int] = None
    true_class: Optional[str] = Field(None, description="Ground truth for IMDB test reviews.")
    model: ModelName
    served_by_run_id: str
    latency_ms: float
    prediction_id: Optional[int] = None
    logged: bool = Field(..., description="False if the database was unreachable.")
