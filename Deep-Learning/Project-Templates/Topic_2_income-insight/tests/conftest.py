"""Shared pytest fixtures.

The schema, health, numerical, and fairness tests run WITHOUT any cloud access
by stubbing the db module with an in-memory store that holds a small synthetic
``adult_income`` table in the real schema. The Supabase round-trip test is
skipped unless real credentials are present, so `pytest` stays green in CI.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict

import pytest
from dotenv import load_dotenv

from shared.data import generate_adult_like

# A local .env (never present in CI) opts in to the live Supabase test.
load_dotenv()

ADULT_ROWS = 1500


class FakeDB:
    """In-memory stand-in for api.db with the same function signatures."""

    def __init__(self):
        self.adult = generate_adult_like(ADULT_ROWS, seed=1)
        self.runs: dict[int, dict] = {}
        self.artifacts: dict[int, str] = {}
        self.predictions: list[dict] = []

    # -- adult_income -------------------------------------------------------
    def load_adult_frame(self, refresh=False):
        return self.adult

    # -- runs ---------------------------------------------------------------
    def insert_run(self, fields, loss_history, val_loss_history, model_b64):
        run_id = len(self.runs) + 1
        losses = [v if math.isfinite(v) else None for v in loss_history]
        row = {
            "id": run_id,
            **fields,
            "final_loss": losses[-1] if losses else None,
            "loss_history": losses,
            "val_loss_history": [v if math.isfinite(v) else None for v in val_loss_history],
            "created_at": f"2026-01-01T00:00:{run_id:02d}+00:00",
        }
        self.runs[run_id] = row
        self.artifacts[run_id] = model_b64
        return row

    def get_run(self, run_id):
        return self.runs.get(run_id)

    get_run_detail = get_run

    def get_run_artifact(self, run_id):
        return self.artifacts.get(run_id)

    def latest_runs(self, limit=50):
        return list(self.runs.values())[-limit:][::-1]

    def latest_ok_run_id(self):
        ok = [r["id"] for r in self.runs.values()
              if r["status"] == "ok" and r["data_source"] == "adult_income"]
        return max(ok) if ok else None

    # -- predictions --------------------------------------------------------
    def insert_predictions(self, rows):
        for r in rows:
            self.predictions.append({"id": len(self.predictions) + 1, **r})
        return len(rows)

    # -- views (Python mirrors of the SQL in 003_adult_income_audit.sql) -----
    def bias_audit(self, run_id, attribute):
        truth = self.adult.set_index("id")
        cells = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "pos": 0, "y": 0})
        for p in self.predictions:
            if p["run_id"] != run_id or p["source"] != "test_eval":
                continue
            row = truth.loc[p["adult_id"]]
            y, yhat = int(row["label"]), int(p["label"])
            c = cells[row[attribute]]
            c["tp" if y and yhat else "fn" if y else "fp" if yhat else "tn"] += 1
            c["pos"] += yhat
            c["y"] += y
        out = []
        for grp, c in sorted(cells.items()):
            n = c["tp"] + c["fp"] + c["tn"] + c["fn"]
            out.append({
                "run_id": run_id, "attribute": attribute, "grp": grp, "n": n,
                "tp": c["tp"], "fp": c["fp"], "tn": c["tn"], "fn": c["fn"],
                "positive_rate": c["pos"] / n, "base_rate": c["y"] / n,
                "fpr": c["fp"] / (c["fp"] + c["tn"]) if c["fp"] + c["tn"] else None,
                "fnr": c["fn"] / (c["fn"] + c["tp"]) if c["fn"] + c["tp"] else None,
            })
        return out

    def activation_comparison(self):
        keys = ("activation", "n_layers", "hidden_dim", "lr", "batch_size", "epochs",
                "seed", "max_train_rows", "include_protected")
        groups = defaultdict(list)
        for r in self.runs.values():
            if r["status"] == "ok" and r["data_source"] == "adult_income":
                groups[tuple(r[k] for k in keys)].append(r)

        def avg(rows, col):
            return round(sum(r[col] for r in rows) / len(rows), 4)

        return [
            {**dict(zip(keys, key)), "runs": len(rows),
             "latest_run_id": max(r["id"] for r in rows),
             "train_accuracy": avg(rows, "train_accuracy"),
             "val_accuracy": avg(rows, "val_accuracy"),
             "test_accuracy": avg(rows, "accuracy"), "test_f1": avg(rows, "f1"),
             "test_roc_auc": avg(rows, "roc_auc"), "brier": avg(rows, "brier"),
             "ece": avg(rows, "ece"), "final_train_loss": avg(rows, "final_loss")}
            for key, rows in groups.items()
        ]


@pytest.fixture
def fake_db(monkeypatch):
    from api import db
    from api import main as api_main

    fake = FakeDB()
    monkeypatch.setattr(db, "ping", lambda: True)
    for name in ("load_adult_frame", "insert_run", "get_run", "get_run_detail",
                 "get_run_artifact", "latest_runs", "latest_ok_run_id",
                 "insert_predictions", "bias_audit", "activation_comparison"):
        monkeypatch.setattr(db, name, getattr(fake, name))
    # Run ids restart at 1 for every fake store, so drop models cached by
    # earlier tests.
    api_main._ARTIFACT_CACHE.clear()
    return fake


@pytest.fixture
def client(fake_db):
    """A TestClient with the Supabase layer stubbed by an in-memory fake."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        c._store = fake_db  # expose for assertions
        yield c


# Small, fast settings for training through the API in tests.
FAST_TRAIN = {"hidden_dim": 16, "batch_size": 64, "epochs": 15, "max_train_rows": None}


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))
