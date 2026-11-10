"""Supabase persistence for the API tier (service-role key, server side only).

Every database call the model service makes goes through this module, so the
tests can swap it for an in-memory fake or make it fail on purpose (the outage
tests). Nothing here ever receives the submitted text -- only its hash and size.

Environment (a local .env, and Render's dashboard):
    SUPABASE_URL           https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY   service-role key (secret!)
"""
from __future__ import annotations

import os
import time
from typing import Optional

from supabase import Client, create_client

_client: Optional[Client] = None

BENCHMARK_COLS = (
    "model,archetype,checkpoint,n_calls,n_warm,n_cold,mean_latency_ms,p50_latency_ms,"
    "p95_latency_ms,min_latency_ms,max_latency_ms,mean_load_ms,mean_total_ms,"
    "mean_input_tokens,mean_output_tokens,ms_per_output_token,distinct_inputs,"
    "first_served_at,last_served_at,refreshed_at"
)


def configured() -> bool:
    """True when both Supabase environment variables are set."""
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))


def get_client() -> Client:
    global _client
    if not configured():
        raise RuntimeError("Supabase is not configured (SUPABASE_URL / SUPABASE_SERVICE_KEY)")
    if _client is None:
        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    return _client


def ping() -> dict:
    """Round-trip to the benchmarks view. Never raises: returns ``{ok, latency_ms, error}``."""
    if not configured():
        return {"ok": False, "configured": False, "latency_ms": None, "error": "SUPABASE_URL / SUPABASE_SERVICE_KEY not set"}
    t0 = time.perf_counter()
    try:
        get_client().table("trio_benchmarks").select("model").limit(1).execute()
        return {"ok": True, "configured": True,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "error": None}
    except Exception as exc:
        return {"ok": False, "configured": True, "latency_ms": None, "error": type(exc).__name__}


def log_inference(row: dict) -> int:
    """Upsert the input hash and insert one inference in a single transaction.

    ``row`` keys: input_sha256, n_chars, n_words, model, archetype, checkpoint,
    task, output, latency_ms, load_ms, cold_start, input_tokens, output_tokens,
    source, api_version. Returns the new ``trio_inferences.id``.
    """
    params = {f"p_{k}": v for k, v in row.items()}
    return int(get_client().rpc("trio_log_inference", params).execute().data)


def refresh_benchmarks() -> str:
    """``REFRESH MATERIALIZED VIEW CONCURRENTLY trio_benchmarks``; returns the timestamp."""
    return get_client().rpc("trio_refresh_benchmarks", {}).execute().data


def get_benchmarks() -> list[dict]:
    """Every row of the benchmarks view, ordered by model."""
    return get_client().table("trio_benchmarks").select(BENCHMARK_COLS).order("model").execute().data
