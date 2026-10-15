"""Smoke tests for /healthz, /version, /classes, /models, and the OpenAPI docs."""
from __future__ import annotations

from seesense import CLASSES


def test_healthz_ok(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["db"] is True
    assert set(body["models_loaded"]) == {"baseline", "improved"}


def test_version_reports_torch(client):
    body = client.get("/version").json()
    assert body["torch"].startswith("2.")
    assert body["git_sha"]


def test_classes_lists_labels(client):
    assert client.get("/classes").json()["classes"] == CLASSES


def test_models_point_at_persisted_runs(client):
    served = {m["name"]: m["run_id"] for m in client.get("/models").json()}
    assert {client.store.runs[rid]["name"] for rid in served.values()} == {"baseline", "improved"}


def test_docs_are_tagged(client):
    spec = client.get("/openapi.json").json()
    assert {t["name"] for t in spec["tags"]} == {"predict", "runs", "predictions", "meta"}
    assert spec["paths"]["/predict"]["post"]["tags"] == ["predict"]
