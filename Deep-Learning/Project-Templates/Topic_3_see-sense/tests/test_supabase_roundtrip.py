"""Supabase round-trip: /predict must write a `predictions` row linked to a `runs` row.

Talks to the real database, so it is skipped unless SUPABASE_URL and
SUPABASE_SERVICE_KEY are set (a local .env). The rows it creates are deleted.

    pytest tests/test_supabase_roundtrip.py -v
"""
from __future__ import annotations

import hashlib

import pytest

from .conftest import has_supabase_creds, png_bytes

pytestmark = pytest.mark.skipif(not has_supabase_creds(), reason="No live Supabase credentials.")


def test_predict_logs_prediction_linked_to_run():
    from fastapi.testclient import TestClient

    from api import db
    from api.main import app

    data = png_bytes(seed=435)
    digest = hashlib.sha256(data).hexdigest()
    with TestClient(app) as c:
        r = c.post("/predict", files={"file": ("roundtrip.png", data, "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["logged"] is True
    client = db.get_client()
    try:
        rows = (client.table("predictions")
                .select("id,image_sha256,predicted_class,served_by_run_id,runs(run_id,name)")
                .eq("id", body["prediction_id"]).execute().data)
        assert len(rows) == 1
        row = rows[0]
        assert row["image_sha256"] == digest
        assert row["predicted_class"] == body["class_name"]
        assert row["runs"]["run_id"] == row["served_by_run_id"] == body["served_by_run_id"]
        meta = client.table("image_metadata").select("*").eq("prediction_id", row["id"]).execute().data
        assert meta and meta[0]["image_sha256"] == digest and meta[0]["width"] == 64
    finally:
        client.table("predictions").delete().eq("id", body["prediction_id"]).execute()
