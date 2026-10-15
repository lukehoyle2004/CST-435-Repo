"""Supabase persistence for the API tier (service-role key, server side only).

Every Supabase call the model service makes goes through this module, so the
tests can swap it for an in-memory fake. The Streamlit UI never imports it: it
does its own read-only queries with the anon key.

Environment (a local .env, and Render's dashboard):
    SUPABASE_URL           https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY   service-role key (secret!)
"""
from __future__ import annotations

import os
from typing import Optional

from supabase import Client, create_client

_client: Optional[Client] = None

RUN_SUMMARY_COLS = ("run_id,name,display_name,architecture,techniques,epochs,batch_size,lr,"
                    "lr_schedule,weight_decay,label_smoothing,params,best_epoch,train_acc,val_acc,"
                    "test_acc,test_acc_tta,macro_f1,weights_path,created_at")


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    return _client


def ping() -> bool:
    try:
        get_client().table("runs").select("run_id").limit(1).execute()
        return True
    except Exception:
        return False


def list_runs(limit: int = 20) -> list[dict]:
    return (get_client().table("runs").select(RUN_SUMMARY_COLS)
            .order("created_at", desc=True).limit(limit).execute().data)


def get_run(run_id: str) -> Optional[dict]:
    data = get_client().table("runs").select("*").eq("run_id", run_id).limit(1).execute().data
    return data[0] if data else None


def log_prediction(prediction: dict, metadata: dict) -> int:
    """Insert one predictions row and its image_metadata row; return the prediction id."""
    client = get_client()
    row = client.table("predictions").insert(prediction).execute().data[0]
    client.table("image_metadata").insert({**metadata, "prediction_id": row["id"]}).execute()
    return row["id"]


def recent_predictions(limit: int = 50) -> list[dict]:
    """Recent predictions joined to the run that served them (PostgREST embed on the FK)."""
    return (get_client().table("predictions")
            .select("id,image_sha256,predicted_class,confidence,top5,tta,latency_ms,created_at,"
                    "served_by_run_id,runs(name,display_name,test_acc_tta)")
            .order("created_at", desc=True).limit(limit).execute().data)
