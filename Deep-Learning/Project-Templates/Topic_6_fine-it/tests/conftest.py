"""Shared pytest fixtures.

API tests run WITHOUT Supabase: ``api.db`` is swapped for ``FakeDB``, an
in-memory store with the same function signatures that also enforces the CHECK
constraints of ``db/migrations/001_init.sql``. The ``outage`` fixture instead
makes every database call fail, to exercise graceful degradation.

Models: by default the API is served by the REAL deployed checkpoint when it is
present (models/deployed, or FTF_MODEL_DIR). On a machine without the weights,
a tiny randomly initialised DistilBERT with the same label contract stands in,
so the contract, logging and outage tests still run; the behavioural tests that
need the real model skip with a clear reason instead.
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
os.environ["FTF_SOURCE"] = "test"

DB_FUNCS = ("ping", "experiment_ids", "get_leaderboard", "log_prediction", "get_latency")
REAL_DIR = Path(os.environ.get("FTF_MODEL_DIR", ROOT / "models" / "deployed"))
HAVE_REAL = (REAL_DIR / "config.json").exists()


def results() -> list[dict]:
    return json.loads((ROOT / "docs" / "experiments" / "results.json").read_text())


class FakeDB:
    """In-memory stand-in for ftf_experiments / ftf_predictions and the two views."""

    def __init__(self):
        self.experiments = [{**r, "id": i + 1} for i, r in enumerate(results())]
        self.predictions: list[dict] = []

    def ping(self):
        return {"ok": True, "configured": True, "latency_ms": 1.0, "error": None}

    def experiment_ids(self, names):
        return {e["run_name"]: e["id"] for e in self.experiments if e["run_name"] in names}

    def get_leaderboard(self):
        rows = sorted(self.experiments, key=lambda e: -e["test_macro_f1"])
        best = rows[0]
        return [{**e, "rank": i + 1, "created_at": None,
                 "distinguishable_from_best": e is best or e["test_macro_f1_hi"] < best["test_macro_f1_lo"]}
                for i, e in enumerate(rows)]

    def log_prediction(self, row):
        # The invariants 001_init.sql enforces.
        assert row["experiment_id"] in {e["id"] for e in self.experiments}
        assert row["endpoint"] in ("predict", "predict_zero_shot")
        assert re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"])
        assert row["n_chars"] >= 1 and row["n_tokens"] >= 0
        assert 0 <= row["confidence"] <= 1
        assert row["latency_ms"] >= 0 and row["load_ms"] >= 0
        assert row["cold_start"] == (row["load_ms"] > 0)
        assert row["source"] in ("api", "benchmark", "test")
        assert "text" not in row
        self.predictions.append({**row, "id": len(self.predictions) + 1})
        return len(self.predictions)

    def get_latency(self):
        out = {}
        for p in self.predictions:
            if p["source"] == "test":
                continue
            o = out.setdefault(p["endpoint"], {"endpoint": p["endpoint"], "n_calls": 0, "n_cold": 0})
            o["n_calls"] += 1
            o["n_cold"] += p["cold_start"]
        return list(out.values())


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    """The real deployed checkpoint, or a tiny random stand-in with the same labels."""
    if HAVE_REAL:
        return REAL_DIR
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    from shared.queues import QUEUE_NAMES, QUEUE_TO_ID
    from src.finetune import tokenizer

    d = tmp_path_factory.mktemp("tiny_model")
    cfg = DistilBertConfig(dim=32, hidden_dim=64, n_layers=1, n_heads=2, num_labels=8,
                           id2label=dict(enumerate(QUEUE_NAMES)), label2id=QUEUE_TO_ID)
    DistilBertForSequenceClassification(cfg).save_pretrained(d)
    tokenizer().save_pretrained(d)
    (d / "ftf_run.json").write_text(json.dumps({"run_name": next(r["run_name"] for r in results() if r["is_deployed"]),
                                                "artifact_sha256": "0" * 64}))
    return d


@pytest.fixture
def stub_zero_shot(monkeypatch):
    """A fast stand-in for flan-t5-small (the real one is exercised in test_zero_shot.py)."""
    import torch

    from src import serving

    def load():
        def proba(texts):
            p = torch.full((len(texts), 8), 0.05)
            p[:, 4] = 0.65
            return p
        return serving.Router("zero_shot", "zero-shot-flan-t5-small", "stub", proba, lambda t: len(t.split()))

    monkeypatch.setitem(serving.LOADERS, "zero_shot", load)


def _client(monkeypatch, model_dir, patch_db):
    from fastapi.testclient import TestClient

    from api import main

    monkeypatch.setenv("FTF_MODEL_DIR", str(model_dir))
    monkeypatch.setattr(main, "REGISTRY", main.serving.Registry(2))
    monkeypatch.setitem(main.STATE, "experiment_ids", {})
    monkeypatch.setitem(main.STATE, "counters", {})
    patch_db()
    return TestClient(main.app)


@pytest.fixture
def client(monkeypatch, model_dir, stub_zero_shot):
    from api import db

    fake = FakeDB()

    def patch():
        for name in DB_FUNCS:
            monkeypatch.setattr(db, name, getattr(fake, name))

    with _client(monkeypatch, model_dir, patch) as c:
        c.store = fake
        yield c


@pytest.fixture
def outage(monkeypatch, model_dir, stub_zero_shot):
    """A client whose Supabase is down: every database call raises."""
    from api import db

    def down(*_a, **_k):
        raise ConnectionError("supabase unreachable")

    def patch():
        for name in ("experiment_ids", "get_leaderboard", "log_prediction", "get_latency"):
            monkeypatch.setattr(db, name, down)
        monkeypatch.setattr(db, "ping", lambda: {"ok": False, "configured": True, "latency_ms": None,
                                                "error": "ConnectionError"})

    with _client(monkeypatch, model_dir, patch) as c:
        yield c


requires_real_model = pytest.mark.skipif(
    not HAVE_REAL, reason=f"deployed checkpoint not found at {REAL_DIR} (run scripts/fetch_model.py)")
