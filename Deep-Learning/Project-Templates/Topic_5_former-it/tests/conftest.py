"""Shared pytest fixtures.

API tests run WITHOUT Supabase: ``api.db`` is swapped for ``FakeDB``, an
in-memory store with the same function signatures that also enforces the
CHECK constraints and the upsert behaviour of ``db/migrations/001_init.sql``.
The ``outage`` fixture instead makes every database call fail, to exercise the
graceful-degradation path. The pretrained checkpoints are real (pinned
revisions, cached by CI), so the model endpoints are tested end to end.
"""
from __future__ import annotations

import os
import re
import statistics
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DB_FUNCS = ("ping", "log_inference", "refresh_benchmarks", "get_benchmarks")
MODEL_TASK = {"encoder_block": ("encoder", "attention"), "bert": ("encoder", "fill-mask"),
              "gpt2": ("decoder", "continue"), "t5": ("encoder-decoder", "summarize")}


class FakeDB:
    """In-memory stand-in for the trio_ tables, RPCs and materialized view."""

    def __init__(self):
        self.inputs: dict[str, dict] = {}
        self.inferences: list[dict] = []
        self.view: list[dict] = []
        self.refreshes = 0

    def ping(self):
        return {"ok": True, "configured": True, "latency_ms": 1.0, "error": None}

    def log_inference(self, row):
        # The invariants 001_init.sql enforces.
        assert re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"])
        assert row["n_chars"] >= 1 and row["n_words"] >= 0
        assert MODEL_TASK[row["model"]] == (row["archetype"], row["task"])
        assert isinstance(row["output"], dict) and row["output"]
        assert row["latency_ms"] >= 0 and row["load_ms"] >= 0
        assert row["cold_start"] == (row["load_ms"] > 0)
        assert row["input_tokens"] >= 0 and row["output_tokens"] >= 0
        assert row["source"] in ("api", "benchmark", "test")
        inp = self.inputs.get(row["input_sha256"])
        if inp is None:
            inp = self.inputs[row["input_sha256"]] = {
                "id": len(self.inputs) + 1, "n_chars": row["n_chars"], "n_words": row["n_words"],
                "times_submitted": 1}
        else:
            inp["times_submitted"] += 1
        inference = {"id": len(self.inferences) + 1, "input_id": inp["id"], **row,
                     "total_ms": row["latency_ms"] + row["load_ms"]}
        self.inferences.append(inference)
        return inference["id"]

    def refresh_benchmarks(self):
        self.refreshes += 1
        rows = []
        for model in sorted({i["model"] for i in self.inferences if i["source"] != "test"}):
            group = [i for i in self.inferences if i["model"] == model and i["source"] != "test"]
            lat = [i["latency_ms"] for i in group]
            cold = [i for i in group if i["cold_start"]]
            out_tok = sum(i["output_tokens"] for i in group)
            rows.append({
                "model": model, "archetype": group[0]["archetype"], "checkpoint": group[0]["checkpoint"],
                "n_calls": len(group), "n_warm": len(group) - len(cold), "n_cold": len(cold),
                "mean_latency_ms": statistics.mean(lat), "p50_latency_ms": statistics.median(lat),
                "p95_latency_ms": max(lat), "min_latency_ms": min(lat), "max_latency_ms": max(lat),
                "mean_load_ms": statistics.mean(i["load_ms"] for i in cold) if cold else None,
                "mean_total_ms": statistics.mean(i["total_ms"] for i in group),
                "mean_input_tokens": statistics.mean(i["input_tokens"] for i in group),
                "mean_output_tokens": statistics.mean(i["output_tokens"] for i in group),
                "ms_per_output_token": sum(lat) / out_tok if out_tok else None,
                "distinct_inputs": len({i["input_id"] for i in group}),
                "first_served_at": None, "last_served_at": None, "refreshed_at": "2026-11-10T00:00:00+00:00",
            })
        self.view = rows
        return "2026-11-10T00:00:00+00:00"

    def get_benchmarks(self):
        return list(self.view)


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from api import db, main

    fake = FakeDB()
    for name in DB_FUNCS:
        monkeypatch.setattr(db, name, getattr(fake, name))
    monkeypatch.setitem(main.STATE, "last_refresh", 0.0)
    with TestClient(main.app) as c:
        c.store = fake
        yield c


@pytest.fixture
def outage(monkeypatch):
    """A client whose Supabase is down: every database call raises."""
    from fastapi.testclient import TestClient

    from api import db, main

    def down(*_a, **_k):
        raise ConnectionError("supabase unreachable")

    for name in ("log_inference", "refresh_benchmarks", "get_benchmarks"):
        monkeypatch.setattr(db, name, down)
    monkeypatch.setattr(db, "ping", lambda: {"ok": False, "configured": True, "latency_ms": None,
                                            "error": "ConnectionError"})
    monkeypatch.setitem(main.STATE, "last_refresh", 0.0)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="session")
def paragraph() -> str:
    from shared.schemas import EXAMPLE_PARAGRAPH

    return EXAMPLE_PARAGRAPH


def pytest_report_header(config):
    return f"HF_HOME={os.environ.get('HF_HOME', '(default)')}"
