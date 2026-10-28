"""Shared pytest fixtures.

API tests run WITHOUT cloud access: the db module is stubbed by an in-memory
fake with the same function signatures, which also enforces the migration's
CHECK constraints. The live Supabase round-trip test is skipped unless real
credentials are present (a local .env), so CI stays green.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DB_FUNCS = ("ping", "list_runs", "get_run", "log_prediction", "recent_predictions", "get_prediction")


class FakeDB:
    def __init__(self):
        self.runs = {r["run_id"]: r for r in json.loads((ROOT / "models" / "runs.json").read_text())}
        self.predictions: list[dict] = []

    def ping(self):
        return True

    def list_runs(self, limit=20):
        return list(self.runs.values())[:limit]

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def log_prediction(self, p):
        # The same invariants 001_init.sql enforces.
        assert p["served_by_run_id"] in self.runs, "FK: served_by_run_id must exist in seq_runs"
        assert re.fullmatch(r"[0-9a-f]{64}", p["text_sha256"])
        assert (p["source"] == "imdb_test") == (p["test_index"] is not None)
        assert (p["model"] == "attention") == (p["attention"] is not None)
        assert 1 <= p["n_tokens"] <= 256 and p["raw_tokens"] >= p["n_tokens"]
        if p["attention"] is not None:
            assert len(p["attention"]) == p["n_tokens"]
        pid = len(self.predictions) + 1
        self.predictions.append({"id": pid, **p})
        return pid

    def _with_run(self, p):
        return {**p, "seq_runs": {"name": self.runs[p["served_by_run_id"]]["name"]}}

    def recent_predictions(self, limit=50):
        return [{k: v for k, v in self._with_run(p).items() if k not in ("attention", "probabilities")}
                for p in self.predictions[::-1][:limit]]

    def get_prediction(self, prediction_id):
        for p in self.predictions:
            if p["id"] == prediction_id:
                return self._with_run(p)
        return None


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from api import db
    from api.main import app

    fake = FakeDB()
    for name in DB_FUNCS:
        monkeypatch.setattr(db, name, getattr(fake, name))
    with TestClient(app) as c:
        c.store = fake
        yield c


@pytest.fixture
def outage_client(monkeypatch):
    """Every Supabase call raises, as when the project is paused or the network is down."""
    from fastapi.testclient import TestClient

    from api import db
    from api.main import app

    def boom(*_args, **_kwargs):
        raise ConnectionError("supabase unreachable")

    for name in DB_FUNCS[1:]:
        monkeypatch.setattr(db, name, boom)
    monkeypatch.setattr(db, "ping", lambda: False)
    with TestClient(app) as c:
        yield c


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))
