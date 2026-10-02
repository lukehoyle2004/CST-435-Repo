"""Supabase persistence helpers for the API tier.

All Supabase access for the model service is funneled through this module. The
FastAPI app calls these functions; it never talks to Supabase directly. The
Streamlit UI NEVER imports this module -- it does its one read-only query with
the anon key on its own side.

The fitted model artifact is kept in a SEPARATE ``run_artifacts`` table (no anon
RLS policy) so the large base64 blob is never exposed to the public anon key --
only run metrics are anon-readable. ``adult_income``, ``predictions`` and the
``bias_audit`` / ``activation_comparison`` views are service-role only.

Every public function raises :class:`DatabaseError` on any Supabase/network
failure, which the API turns into a 503 instead of a stack trace.

Environment variables (set locally in a .env, and in the Render dashboard):
    SUPABASE_URL              -> https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY      -> the service-role key (server-side only, secret!)
"""
from __future__ import annotations

import functools
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

# Load a local .env for development; on Render there is no .env and the real
# environment variables are used instead.
load_dotenv()

_client: Optional[Client] = None
_adult_cache: Optional[pd.DataFrame] = None

PAGE = 1000  # PostgREST returns at most 1000 rows per request by default

# Columns returned to callers as a "run" (summary; no curves or model blob).
_RUN_COLS = (
    "id,data_source,activation,n_layers,hidden_dim,lr,batch_size,epochs,seed,"
    "max_train_rows,include_protected,status,final_loss,train_accuracy,"
    "val_accuracy,accuracy,precision,recall,f1,roc_auc,brier,ece,created_at"
)
_RUN_DETAIL_COLS = (
    _RUN_COLS + ",n_train,n_val,n_test,confusion,class_report,calibration,"
    "perm_importance,loss_history,val_loss_history"
)

T = TypeVar("T")


class DatabaseError(RuntimeError):
    """Supabase was unreachable or rejected the request."""


def _wrap(fn: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> T:
        try:
            return fn(*args, **kwargs)
        except DatabaseError:
            raise
        except Exception as exc:  # noqa: BLE001 -- any client/network failure
            raise DatabaseError(f"{fn.__name__} failed: {exc}") from exc
    return inner


def get_client() -> Client:
    """Lazily create and cache a Supabase client."""
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _client = create_client(url, key)
    return _client


def ping() -> bool:
    """Return True if the Supabase client can reach the runs table."""
    try:
        get_client().table("runs").select("id").limit(1).execute()
        return True
    except Exception:
        return False


def _finite_or_none(values: List[float]) -> List[Optional[float]]:
    # JSON has no NaN/Inf, so non-finite values are stored as null.
    return [v if math.isfinite(v) else None for v in values]


# ---------------------------------------------------------------------------
# adult_income (read-only for the API; filled by db/load_adult.py)
# ---------------------------------------------------------------------------
@_wrap
def load_adult_frame(refresh: bool = False) -> pd.DataFrame:
    """All adult_income rows as a DataFrame, cached in process after first load.

    The table is static (only the loader writes it), so one paged read per API
    process is enough and later /train calls skip ~49 round trips.
    """
    global _adult_cache
    if _adult_cache is not None and not refresh:
        return _adult_cache
    client = get_client()
    rows: List[dict] = []
    start = 0
    while True:
        batch = (
            client.table("adult_income").select("*").order("id")
            .range(start, start + PAGE - 1).execute().data
        )
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        start += PAGE
    if not rows:
        raise DatabaseError("adult_income is empty -- run `python -m db.load_adult`")
    _adult_cache = pd.DataFrame(rows)
    return _adult_cache


# ---------------------------------------------------------------------------
# runs (metrics)  +  run_artifacts (model blob)
# ---------------------------------------------------------------------------
@_wrap
def insert_run(fields: Dict[str, Any], loss_history: List[float],
               val_loss_history: List[float], model_b64: str) -> dict:
    """Insert a run row + its artifact. Diverged runs carry NULL metrics."""
    client = get_client()
    losses = _finite_or_none(loss_history)
    row = {
        **fields,
        "final_loss": losses[-1] if losses else None,
        "loss_history": losses,
        "val_loss_history": _finite_or_none(val_loss_history),
    }
    run = client.table("runs").insert(row).execute().data[0]
    client.table("run_artifacts").insert(
        {"run_id": run["id"], "model_b64": model_b64}
    ).execute()
    return run


@_wrap
def get_run(run_id: int) -> Optional[dict]:
    resp = get_client().table("runs").select(_RUN_COLS).eq("id", run_id).limit(1).execute()
    return resp.data[0] if resp.data else None


@_wrap
def get_run_detail(run_id: int) -> Optional[dict]:
    resp = (
        get_client().table("runs").select(_RUN_DETAIL_COLS)
        .eq("id", run_id).limit(1).execute()
    )
    return resp.data[0] if resp.data else None


@_wrap
def get_run_artifact(run_id: int) -> Optional[str]:
    resp = (
        get_client().table("run_artifacts").select("model_b64")
        .eq("run_id", run_id).limit(1).execute()
    )
    return resp.data[0]["model_b64"] if resp.data else None


@_wrap
def latest_runs(limit: int = 50) -> List[dict]:
    return (
        get_client().table("runs").select(_RUN_COLS)
        .order("created_at", desc=True).limit(limit).execute().data
    )


@_wrap
def latest_ok_run_id() -> Optional[int]:
    """Newest successful run on the real data (the default model to serve)."""
    resp = (
        get_client().table("runs").select("id")
        .eq("status", "ok").eq("data_source", "adult_income")
        .order("created_at", desc=True).limit(1).execute()
    )
    return resp.data[0]["id"] if resp.data else None


# ---------------------------------------------------------------------------
# predictions (the audit log)
# ---------------------------------------------------------------------------
@_wrap
def insert_predictions(rows: List[Dict[str, Any]]) -> int:
    """Bulk-insert prediction rows (run_id, features, proba, label, threshold,
    source, adult_id). Returns the number inserted."""
    client = get_client()
    for start in range(0, len(rows), PAGE):
        client.table("predictions").insert(rows[start:start + PAGE]).execute()
    return len(rows)


def eval_prediction_rows(run_id: int, scored: List[Tuple[int, int, float]],
                         threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Prediction rows for a run's test split, linked to adult_income by id."""
    return [
        {"run_id": run_id, "adult_id": adult_id, "label": label, "proba": proba,
         "threshold": threshold, "source": "test_eval", "features": None}
        for adult_id, label, proba in scored
    ]


# ---------------------------------------------------------------------------
# SQL views
# ---------------------------------------------------------------------------
@_wrap
def bias_audit(run_id: int, attribute: str) -> List[dict]:
    """Per-group confusion counts, FPR and FNR from the bias_audit view."""
    return (
        get_client().table("bias_audit").select("*")
        .eq("run_id", run_id).eq("attribute", attribute).order("grp")
        .execute().data
    )


@_wrap
def activation_comparison() -> List[dict]:
    """Rows of the activation_comparison view (ok runs grouped by controls)."""
    return (
        get_client().table("activation_comparison").select("*")
        .order("latest_run_id", desc=True).execute().data
    )
