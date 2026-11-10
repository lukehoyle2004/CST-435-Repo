"""Request and response models for the TrioFormer API (the public contract).

Product 5B interfaces are built against these shapes, so every field carries a
description and an example that show up in the OpenAPI docs at ``/docs``.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PARAGRAPH = (
    "The city council voted on Tuesday to expand the bike lane network after a year of public "
    "meetings. Supporters said the new lanes would make downtown streets safer for commuters, "
    "while several shop owners worried about losing parking in front of their stores."
)

Source = Literal["api", "benchmark"]


class TextRequest(BaseModel):
    """Fields shared by the three pretrained-model endpoints."""

    text: str = Field(..., min_length=1, max_length=2000,
                      description="The paragraph. Only its SHA-256 hash is stored, never the text.")
    source: Source = Field("api", description="'benchmark' marks calls made by scripts/benchmark_latency.py.")


class BertFillMaskRequest(TextRequest):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"text": EXAMPLE_PARAGRAPH, "mask_word": "safer", "top_k": 5},
        {"text": "The capital of France is [MASK].", "top_k": 5},
    ]})
    mask_word: Optional[str] = Field(
        None, min_length=1, max_length=50,
        description="Hide the first whole-word occurrence of this word. Omit it if the text already "
                    "contains exactly one [MASK].")
    top_k: int = Field(5, ge=1, le=10, description="How many candidates to return.")


class Gpt2ContinueRequest(TextRequest):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"text": EXAMPLE_PARAGRAPH, "max_new_tokens": 40}]})
    max_new_tokens: int = Field(40, ge=1, le=80, description="Tokens to generate.")
    do_sample: bool = Field(False, description="False = greedy (deterministic, comparable latency).")
    temperature: float = Field(0.8, gt=0.0, le=2.0, description="Used only when do_sample is true.")
    top_p: float = Field(0.95, gt=0.0, le=1.0, description="Used only when do_sample is true.")
    seed: int = Field(0, ge=0, description="Sampling seed (do_sample only).")


class T5SummarizeRequest(TextRequest):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"text": EXAMPLE_PARAGRAPH, "max_new_tokens": 60}]})
    max_new_tokens: int = Field(60, ge=8, le=120, description="Upper bound on summary length (tokens).")
    min_new_tokens: int = Field(10, ge=0, le=60, description="Lower bound on summary length (tokens).")
    num_beams: int = Field(2, ge=1, le=4, description="Beam width; more beams cost more latency.")


class EncoderAttentionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"text": "dog bites man bites dog"}, {"text": "never odd or even"}]})
    text: str = Field("dog bites man bites dog", min_length=1, max_length=200,
                      description="2-12 words. Punctuation and case are ignored.")
    source: Source = "api"


class InferenceMeta(BaseModel):
    """Fields every model endpoint returns: what ran, how long it took, whether it was logged."""

    model: Literal["encoder_block", "bert", "gpt2", "t5"]
    archetype: Literal["encoder", "decoder", "encoder-decoder"]
    task: str
    checkpoint: str = Field(..., description="checkpoint@commit that served the call (provenance).")
    input_sha256: str = Field(..., description="SHA-256 of the stripped input text; the only trace of it we keep.")
    input_tokens: int
    output_tokens: int
    latency_ms: float = Field(..., description="Server-side model compute time (tokenise -> forward/generate -> decode).")
    load_ms: float = Field(..., description="Time spent loading the checkpoint during this request (0 if already resident).")
    cold_start: bool
    total_ms: float = Field(..., description="Server-side wall time for the whole request before logging, including any wait for the model lock.")
    inference_id: Optional[int] = Field(None, description="trio_inferences.id, or null if the database was unreachable.")
    logged: bool = Field(..., description="False when the result could not be written to Supabase (outage).")


class Candidate(BaseModel):
    token: str
    score: float


class BertFillMaskResponse(InferenceMeta):
    masked_text: str
    candidates: List[Candidate]
    filled_text: str


class Gpt2ContinueResponse(InferenceMeta):
    continuation: str


class T5SummarizeResponse(InferenceMeta):
    summary: str


class HeadSummary(BaseModel):
    head: int
    mirror: float = Field(..., description="Mean weight on the palindrome partner (anti-diagonal).")
    self: float = Field(..., description="Mean weight on the query's own position (diagonal).")
    same_word: float = Field(..., description="Mean weight on other positions holding the same word.")
    uniform: float = Field(..., description="1/n: what a head that ignored its input would give.")


class EncoderAttentionResponse(InferenceMeta):
    model_config = ConfigDict(protected_namespaces=())
    words: List[str]
    ids: List[int] = Field(..., description="Words relabelled by first appearance (what the model sees).")
    prediction: str
    probabilities: Dict[str, float]
    n_layers: int
    n_heads: int
    attention: List[List[List[List[float]]]] = Field(
        ..., description="attention[layer][head][query][key], n x n over the real words; rows sum to 1.")
    head_summary: List[List[HeadSummary]] = Field(..., description="head_summary[layer][head].")
    model_accuracy: float = Field(..., description="Held-out accuracy of the toy classifier.")


class BenchmarkRow(BaseModel):
    model: str
    archetype: str
    checkpoint: str
    n_calls: int
    n_warm: int
    n_cold: int
    mean_latency_ms: Optional[float]
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    min_latency_ms: Optional[float]
    max_latency_ms: Optional[float]
    mean_load_ms: Optional[float]
    mean_total_ms: Optional[float]
    mean_input_tokens: Optional[float]
    mean_output_tokens: Optional[float]
    ms_per_output_token: Optional[float]
    distinct_inputs: int
    first_served_at: Optional[str]
    last_served_at: Optional[str]
    refreshed_at: Optional[str]


class BenchmarksResponse(BaseModel):
    refreshed: bool = Field(..., description="True if this request refreshed the materialized view.")
    rows: List[BenchmarkRow]


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: Literal["ok", "degraded"]
    model_loader: dict
    database: dict


class Version(BaseModel):
    api: str
    git_sha: str
    python: str
    torch: str
    transformers: Optional[str]
    dtype: str
    models: dict
    encoder_block: dict
