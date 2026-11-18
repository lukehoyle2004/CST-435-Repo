"""Fixtures: a fake FineTuneFleet API (as an HTTP transport) and a fake read-only Supabase.

The fake responses copy the shapes the deployed service returns (shared/schemas.py
in Product 6), so the interface is tested against the same contract it uses in
production. Modes switch the "service" between up, cold-starting (Render's own
HTML 503 page) and unreachable (connection refused).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))

API = "http://api.test"
QUEUES = ["card_services", "transfers", "top_ups", "cash_atm", "fees_fx", "payments_disputes",
          "security_fraud", "account_identity"]


def _prediction(kind: str, text: str, i: int = 0, load_ms: float = 0.0) -> dict:
    fraud = any(w in text.lower() for w in ("without", "stolen", "recognise", "wasn't me"))
    top = "security_fraud" if fraud else "card_services"
    conf = 0.98 if kind == "fine_tuned" else 0.41
    rest = (1 - conf) / 7
    scores = {q: (conf if q == top else rest) for q in QUEUES}
    return {"queue": top, "confidence": conf, "needs_review": conf < 0.6, "scores": scores, "model": kind,
            "run_name": "ft-lr5e-5-ep3" if kind == "fine_tuned" else "zero-shot-flan-t5-small",
            "experiment_id": 1 if kind == "fine_tuned" else 6,
            "checkpoint": "ft-lr5e-5-ep3@4748e601f4af" if kind == "fine_tuned" else "google/flan-t5-small@0fc9[fp16-weights]",
            "latency_ms": 45.0 if kind == "fine_tuned" else 1900.0, "load_ms": load_ms, "cold_start": load_ms > 0,
            "n_tokens": 12, "prediction_id": 100 + i, "logged": True}


def _predict(kind):
    def handler(body):
        if not body.get("text", "").strip():
            return 422, {"detail": [{"loc": ["body", "text"], "msg": "String should have at least 1 character"}]}
        return 200, _prediction(kind, body["text"], load_ms=2500.0 if kind == "zero_shot" else 0.0)
    return handler


def _batch(body):
    kind = body.get("model", "fine_tuned")
    preds = [_prediction(kind, t, i, load_ms=1800.0 if i == 0 else 0.0) for i, t in enumerate(body["texts"])]
    total = sum(p["latency_ms"] for p in preds)
    return 200, {"model": kind, "predictions": preds, "total_latency_ms": total, "wall_ms": total + 2100,
                 "logged": len(preds)}


ROUTES = {
    ("GET", "/healthz"): lambda b: (200, {"status": "ok", "database": {"ok": True}, "models": {},
                                          "deployed_run": "ft-lr5e-5-ep3"}),
    ("POST", "/predict"): _predict("fine_tuned"),
    ("POST", "/predict_zero_shot"): _predict("zero_shot"),
    ("POST", "/predict_batch"): _batch,
}


def make_response(status: int, payload, html: bool = False) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    if html:
        r._content = b"<html><body>Service Unavailable</body></html>"
        r.headers["content-type"] = "text/html"
    else:
        r._content = json.dumps(payload).encode()
        r.headers["content-type"] = "application/json"
    return r


class FakeAPI:
    """Callable with the signature of ``requests.request``; records every call."""

    def __init__(self):
        self.mode = "up"            # up | cold | down
        self.calls: list[tuple[str, str]] = []
        self.bodies: list[dict] = []

    def __call__(self, method, url, json=None, timeout=None, **_):
        path = url.replace(API, "")
        self.calls.append((method, path))
        self.bodies.append(json or {})
        if self.mode == "down":
            raise requests.ConnectionError("connection refused")
        if self.mode == "cold":
            return make_response(503, None, html=True)
        status, body = ROUTES[(method, path)](json or {})
        return make_response(status, body)


def _row(rank, run, kind, f1, lo, hi, val, deployed=False, dist=True, lr=5e-5, epochs=3, n=8952, frozen=False):
    zero = kind == "zero_shot"
    return {"rank": rank, "id": rank, "run_name": run, "kind": kind, "base_checkpoint": "x@y", "seed": 42,
            "hyperparams": {} if zero else {"learning_rate": lr, "epochs": epochs, "freeze_encoder": frozen},
            "n_train": 0 if zero else n, "val_macro_f1": val,
            "test_accuracy": f1 + 0.003, "test_accuracy_lo": lo, "test_accuracy_hi": hi,
            "test_macro_f1": f1, "test_macro_f1_lo": lo, "test_macro_f1_hi": hi,
            "fraud_recall": 0.97 if not zero else 0.60, "fraud_recall_lo": 0.95 if not zero else 0.54,
            "fraud_recall_hi": 0.99 if not zero else 0.66,
            "delta_macro_f1": None if zero else f1 - 0.669, "delta_macro_f1_lo": None if zero else f1 - 0.69,
            "delta_macro_f1_hi": None if zero else f1 - 0.65, "train_runtime_s": 0 if zero else 2300.0,
            "eval_ms_per_message": 605.7 if zero else 46.8, "is_deployed": deployed,
            "created_at": "2026-11-17T00:00:00+00:00", "distinguishable_from_best": dist}


LEADERBOARD = [
    _row(1, "ft-lr5e-5-ep3", "fine_tuned", 0.965, 0.958, 0.972, 0.970, deployed=True),
    _row(2, "ft-lr2e-5-ep3", "fine_tuned", 0.954, 0.946, 0.962, 0.958, dist=False, lr=2e-5),
    _row(3, "head-only-lr1e-3-ep3", "fine_tuned", 0.8075, 0.791, 0.822, 0.804, lr=1e-3, frozen=True),
    _row(4, "zero-shot-flan-t5-small", "zero_shot", 0.669, 0.649, 0.686, 0.667),
]
LATENCY = [
    {"endpoint": "predict", "run_name": "ft-lr5e-5-ep3", "n_calls": 20, "n_cold": 3, "mean_latency_ms": 60.1,
     "p50_latency_ms": 48.0, "p95_latency_ms": 120.0, "mean_load_ms": 2400.0, "mean_tokens": 13.2,
     "review_rate": 0.05, "last_served_at": "2026-11-18T20:00:00+00:00"},
    {"endpoint": "predict_zero_shot", "run_name": "zero-shot-flan-t5-small", "n_calls": 8, "n_cold": 4,
     "mean_latency_ms": 2100.0, "p50_latency_ms": 1900.0, "p95_latency_ms": 3500.0, "mean_load_ms": 6100.0,
     "mean_tokens": 290.0, "review_rate": 0.6, "last_served_at": "2026-11-18T20:00:00+00:00"},
]


class FakeQuery:
    def __init__(self, rows):
        self.rows = [dict(r) for r in rows]

    def select(self, *_a, **_k):
        return self

    def order(self, col, desc=False):
        self.rows.sort(key=lambda r: r[col], reverse=desc)
        return self

    def execute(self):
        return type("Resp", (), {"data": self.rows})()


class FakeSupabase:
    """Read-only, like the anon key: only the leaderboard and latency views exist for it."""

    def __init__(self):
        self.tables_read: list[str] = []
        self.fail = False

    def table(self, name):
        self.tables_read.append(name)
        if self.fail:
            raise ConnectionError("supabase down")
        views = {"ftf_leaderboard": LEADERBOARD, "ftf_latency": LATENCY}
        if name not in views:
            raise PermissionError(f"permission denied for table {name}")
        return FakeQuery(views[name])


@pytest.fixture
def fake_api(monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(requests, "request", api)
    return api


@pytest.fixture
def fake_db(monkeypatch):
    import supabase

    db = FakeSupabase()
    monkeypatch.setattr(supabase, "create_client", lambda url, key: db)
    return db
