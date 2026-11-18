"""Supabase persistence for the API tier and the offline scripts (service-role key).

Every database call goes through this module, so the tests can swap it for an
in-memory fake or make it fail on purpose (the outage tests). Nothing here ever
receives a message sent to the live service -- only its hash and its length.

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

LEADERBOARD_COLS = (
    "rank,id,run_name,kind,base_checkpoint,seed,hyperparams,n_train,val_macro_f1,"
    "test_accuracy,test_accuracy_lo,test_accuracy_hi,test_macro_f1,test_macro_f1_lo,test_macro_f1_hi,"
    "fraud_recall,fraud_recall_lo,fraud_recall_hi,delta_macro_f1,delta_macro_f1_lo,delta_macro_f1_hi,"
    "train_runtime_s,eval_ms_per_message,is_deployed,created_at,distinguishable_from_best"
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
    """Round-trip to the experiment store. Never raises: returns ``{ok, latency_ms, error}``."""
    if not configured():
        return {"ok": False, "configured": False, "latency_ms": None,
                "error": "SUPABASE_URL / SUPABASE_SERVICE_KEY not set"}
    t0 = time.perf_counter()
    try:
        get_client().table("ftf_experiments").select("id").limit(1).execute()
        return {"ok": True, "configured": True,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "error": None}
    except Exception as exc:
        return {"ok": False, "configured": True, "latency_ms": None, "error": type(exc).__name__}


# --- corpus -----------------------------------------------------------------
def upsert_documents(rows: list[dict]) -> None:
    """Insert corpus rows; a re-run updates them in place (keyed on text_sha256)."""
    get_client().table("ftf_documents").upsert(rows, on_conflict="text_sha256").execute()


# --- experiments ------------------------------------------------------------
def upsert_experiment(row: dict) -> int:
    """Insert or replace one experiment (keyed on run_name). Returns its id."""
    data = get_client().table("ftf_experiments").upsert(row, on_conflict="run_name").execute().data
    return int(data[0]["id"])


def set_deployed(run_name: str) -> None:
    """Mark exactly one fine-tuned run as the deployed checkpoint."""
    c = get_client()
    c.table("ftf_experiments").update({"is_deployed": False}).eq("is_deployed", True).execute()
    c.table("ftf_experiments").update({"is_deployed": True}).eq("run_name", run_name).execute()


def experiment_ids(run_names: list[str]) -> dict[str, int]:
    """``run_name -> id`` for the runs the service serves (looked up once at startup)."""
    rows = (get_client().table("ftf_experiments").select("id,run_name")
            .in_("run_name", run_names).execute().data)
    return {r["run_name"]: int(r["id"]) for r in rows}


def get_leaderboard() -> list[dict]:
    return get_client().table("ftf_leaderboard").select(LEADERBOARD_COLS).execute().data


# --- predictions ------------------------------------------------------------
def log_prediction(row: dict) -> int:
    """Insert one prediction row and return its id."""
    data = get_client().table("ftf_predictions").insert(row).execute().data
    return int(data[0]["id"])


def get_latency() -> list[dict]:
    return get_client().table("ftf_latency").select("*").order("endpoint").execute().data
