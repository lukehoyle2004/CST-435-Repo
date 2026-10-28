"""Supabase persistence for the API tier (service-role key, server side only).

Every Supabase call the model service makes goes through this module, so the
tests can swap it for an in-memory fake. The Streamlit UI never imports it: it
does its own read-only queries with the anon key.

A prediction is stored as the review's SHA-256 hash, token counts, the answer,
and the attention vector -- never the review text.

Environment (a local .env, and Render's dashboard):
    SUPABASE_URL           https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY   service-role key (secret!)
"""
from __future__ import annotations

import os
from typing import Optional

from supabase import Client, create_client

_client: Optional[Client] = None

RUN_SUMMARY_COLS = ("run_id,name,display_name,architecture,epochs,batch_size,lr,seed,params,"
                    "best_epoch,val_acc,test_acc,macro_f1,ece,confident_errors,created_at")
PREDICTION_LIST_COLS = ("id,text_sha256,source,test_index,n_tokens,raw_tokens,unk_tokens,"
                        "predicted_class,confidence,model,served_by_run_id,latency_ms,created_at,"
                        "seq_runs(name,display_name)")


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    return _client


def ping() -> bool:
    try:
        get_client().table("seq_runs").select("run_id").limit(1).execute()
        return True
    except Exception:
        return False


def list_runs(limit: int = 20) -> list[dict]:
    return (get_client().table("seq_runs").select(RUN_SUMMARY_COLS)
            .order("created_at", desc=True).limit(limit).execute().data)


def get_run(run_id: str) -> Optional[dict]:
    data = get_client().table("seq_runs").select("*").eq("run_id", run_id).limit(1).execute().data
    return data[0] if data else None


def log_prediction(prediction: dict) -> int:
    """Insert one seq_predictions row; return its id."""
    return get_client().table("seq_predictions").insert(prediction).execute().data[0]["id"]


def recent_predictions(limit: int = 50) -> list[dict]:
    """Recent predictions (without the attention vector) joined to the serving run."""
    return (get_client().table("seq_predictions").select(PREDICTION_LIST_COLS)
            .order("created_at", desc=True).limit(limit).execute().data)


def get_prediction(prediction_id: int) -> Optional[dict]:
    """One prediction including its attention vector (the drilldown)."""
    data = (get_client().table("seq_predictions").select("*,seq_runs(name,display_name)")
            .eq("id", prediction_id).limit(1).execute().data)
    return data[0] if data else None
