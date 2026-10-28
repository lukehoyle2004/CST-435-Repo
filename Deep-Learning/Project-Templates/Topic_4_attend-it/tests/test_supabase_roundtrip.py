"""Supabase round-trip: /predict must write a `seq_predictions` row, with its attention
vector, linked to a `seq_runs` row.

Talks to the real database, so it is skipped unless SUPABASE_URL and
SUPABASE_SERVICE_KEY are set (a local .env). The row it creates is deleted.

    pytest tests/test_supabase_roundtrip.py -v
"""
from __future__ import annotations

import hashlib

import pytest

from tests.conftest import has_supabase_creds

pytestmark = pytest.mark.skipif(not has_supabase_creds(), reason="No live Supabase credentials.")


def test_predict_logs_attention_linked_to_run():
    from fastapi.testclient import TestClient

    from api import db
    from api.main import app

    text = "Round-trip test: a dull, lifeless film with one wonderful scene."
    with TestClient(app) as c:
        r = c.post("/predict", json={"text": text})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["logged"] is True
    client = db.get_client()
    try:
        rows = (client.table("seq_predictions")
                .select("*,seq_runs(run_id,name)").eq("id", body["prediction_id"]).execute().data)
        assert len(rows) == 1
        row = rows[0]
        assert row["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert row["predicted_class"] == body["class_name"]
        assert row["seq_runs"]["name"] == "attention"
        assert row["attention"] == pytest.approx(body["attention"])
        assert text not in str(row)
    finally:
        client.table("seq_predictions").delete().eq("id", body["prediction_id"]).execute()
