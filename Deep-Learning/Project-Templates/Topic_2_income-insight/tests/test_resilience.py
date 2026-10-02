"""Supabase-failure path and startup model loading.

When the database is down the API must answer 503 with a readable message (not
crash with a 500), and /healthz must report "degraded". At startup the API loads
the newest successful run's sklearn + torch pipeline so it is ready to serve.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api import db
from api import main as api_main
from shared.data import FEATURE_COLS
from tests.conftest import FAST_TRAIN


def _boom(*_args, **_kwargs):
    raise db.DatabaseError("connection refused")


def test_train_returns_503_when_supabase_write_fails(client, monkeypatch):
    monkeypatch.setattr(db, "insert_run", _boom)
    resp = client.post("/train", json=FAST_TRAIN)
    assert resp.status_code == 503
    assert "database unavailable" in resp.json()["detail"]


def test_train_returns_503_when_data_cannot_be_read(client, monkeypatch):
    monkeypatch.setattr(db, "load_adult_frame", _boom)
    assert client.post("/train", json=FAST_TRAIN).status_code == 503


def test_predict_returns_503_when_logging_fails(client, monkeypatch):
    run_id = client.post("/train", json=FAST_TRAIN).json()["run_id"]
    record = client._store.adult.iloc[0][FEATURE_COLS].to_dict()
    monkeypatch.setattr(db, "insert_predictions", _boom)
    resp = client.post("/predict", json={"run_id": run_id, "features": record})
    assert resp.status_code == 503


def test_healthz_degraded_when_supabase_unreachable(client, monkeypatch):
    monkeypatch.setattr(db, "ping", lambda: False)
    body = client.get("/healthz").json()
    assert body["status"] == "degraded" and body["supabase"] is False


def test_db_wrapper_converts_client_errors():
    @db._wrap
    def broken():
        raise ConnectionError("network down")

    try:
        broken()
    except db.DatabaseError as exc:
        assert "broken failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected DatabaseError")


def test_startup_loads_newest_pipeline(client):
    run_id = client.post("/train", json=FAST_TRAIN).json()["run_id"]
    api_main._ARTIFACT_CACHE.clear()

    # A fresh app start (new TestClient context) must load that run's pipeline.
    with TestClient(api_main.app) as fresh:
        assert run_id in api_main._ARTIFACT_CACHE
        assert fresh.get("/healthz").json()["default_run_id"] == run_id


def test_startup_survives_database_outage(fake_db, monkeypatch):
    monkeypatch.setattr(db, "latest_ok_run_id", _boom)
    with TestClient(api_main.app) as c:
        assert c.get("/healthz").json()["default_run_id"] is None
