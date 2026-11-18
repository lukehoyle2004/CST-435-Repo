"""External-dependency failures: Supabase down, and a checkpoint that will not load.

The contract under failure:
* routing still answers, flagged ``logged: false`` / ``X-FTF-Logged: false``;
* ``/experiments`` is a clear 503 -- there is no honest leaderboard to show;
* ``/metrics`` still reports this process's counters, including the unlogged calls;
* ``/healthz`` stays 200 but reports ``degraded``;
* a model that cannot load is a 503 for that model only, and the loader recovers.
"""
from __future__ import annotations

import pytest

from shared.schemas import EXAMPLE_MESSAGE


@pytest.mark.parametrize("path", ["/predict", "/predict_zero_shot"])
def test_routing_still_answers_when_supabase_is_down(outage, path):
    r = outage.post(path, json={"text": EXAMPLE_MESSAGE})
    assert r.status_code == 200
    body = r.json()
    assert body["logged"] is False and body["prediction_id"] is None and body["experiment_id"] is None
    assert r.headers["X-FTF-Logged"] == "false"
    assert body["latency_ms"] > 0


def test_experiments_is_503_without_leaking_internals(outage):
    r = outage.get("/experiments")
    assert r.status_code == 503
    assert "temporarily unavailable" in r.json()["detail"]
    assert "supabase unreachable" not in r.text


def test_metrics_report_unlogged_calls_during_outage(outage):
    outage.post("/predict", json={"text": "card not arrived"})
    m = outage.get("/metrics").json()
    assert m["since_start"]["predict"]["unlogged"] == 1
    assert m["store"] is None and "unavailable" in m["store_error"]


def test_healthz_is_degraded_but_up(outage):
    r = outage.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


def test_unconfigured_database_is_reported_not_crashed(monkeypatch):
    from api import db
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    assert db.ping()["configured"] is False
    with pytest.raises(RuntimeError, match="not configured"):
        db.log_prediction({})


def test_failed_model_load_is_a_503_and_the_loader_recovers(client, monkeypatch):
    from src import serving

    good = serving.LOADERS["zero_shot"]

    def broken():
        raise OSError("cannot reach the Hub")

    monkeypatch.setitem(serving.LOADERS, "zero_shot", broken)
    r = client.post("/predict_zero_shot", json={"text": "hi"})
    assert r.status_code == 503 and "not available" in r.json()["detail"]
    assert "zero_shot" in client.get("/healthz").json()["models"]["errors"]
    assert client.post("/predict", json={"text": "hi"}).status_code == 200   # the other model is unaffected

    monkeypatch.setitem(serving.LOADERS, "zero_shot", good)
    assert client.post("/predict_zero_shot", json={"text": "hi"}).status_code == 200
    assert client.get("/healthz").json()["models"]["errors"] == {}


def test_missing_checkpoint_is_reported(monkeypatch, tmp_path):
    from src import serving
    monkeypatch.setenv("FTF_MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no deployed checkpoint"):
        serving.load_fine_tuned()
