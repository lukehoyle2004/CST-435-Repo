"""Shared pytest fixtures.

API tests run WITHOUT cloud access: the db module is stubbed by an in-memory
fake with the same function signatures. The live Supabase round-trip test is
skipped unless real credentials are present (a local .env), so CI stays green.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class FakeDB:
    def __init__(self):
        self.runs = {r["run_id"]: r for r in json.loads((ROOT / "models" / "runs.json").read_text())}
        self.predictions: list[dict] = []
        self.image_metadata: list[dict] = []

    def ping(self):
        return True

    def list_runs(self, limit=20):
        return list(self.runs.values())[:limit]

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def log_prediction(self, prediction, metadata):
        assert prediction["served_by_run_id"] in self.runs, "FK: served_by_run_id must exist in runs"
        pid = len(self.predictions) + 1
        self.predictions.append({"id": pid, **prediction})
        self.image_metadata.append({"id": pid, "prediction_id": pid, **metadata})
        return pid

    def recent_predictions(self, limit=50):
        return [{**p, "runs": {"name": self.runs[p["served_by_run_id"]]["name"]}}
                for p in self.predictions[::-1][:limit]]


def _stub(monkeypatch, fake):
    from api import db
    for name in ("ping", "list_runs", "get_run", "log_prediction", "recent_predictions"):
        monkeypatch.setattr(db, name, getattr(fake, name))


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    fake = FakeDB()
    _stub(monkeypatch, fake)
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

    for name in ("list_runs", "get_run", "log_prediction", "recent_predictions"):
        monkeypatch.setattr(db, name, boom)
    monkeypatch.setattr(db, "ping", lambda: False)
    with TestClient(app) as c:
        yield c


def png_bytes(width=64, height=48, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))
