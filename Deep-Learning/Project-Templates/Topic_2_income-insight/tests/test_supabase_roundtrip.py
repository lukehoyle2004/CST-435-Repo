"""Live Supabase round trip: write a run, read it back, and query both SQL views.

Skipped automatically unless real Supabase credentials are present (a local .env
or exported env vars), so CI and offline runs stay green. Requires migrations
001-003 and the loaded adult_income table. The run it creates is deleted at the
end (cascading to its artifact and predictions) so it never pollutes Run History.
"""
from __future__ import annotations

import pytest

from tests.conftest import has_supabase_creds

pytestmark = pytest.mark.skipif(
    not has_supabase_creds(), reason="No live Supabase credentials in environment."
)


def test_train_writes_run_and_sql_views_read_it():
    from fastapi.testclient import TestClient

    from api import db
    from api.main import app

    run_id = None
    try:
        with TestClient(app) as c:
            resp = c.post(
                "/train",
                json={"hidden_dim": 8, "epochs": 2, "max_train_rows": 1000,
                      "activation": "tanh", "seed": 4242},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            run_id = body["run_id"]
            assert body["n_test_predictions_logged"] > 0

            detail = c.get(f"/runs/{run_id}").json()
            assert detail["data_source"] == "adult_income"
            assert detail["confusion"] is not None

            audit = c.get("/bias_audit", params={"run_id": run_id, "attribute": "sex"})
            assert audit.status_code == 200, audit.text
            groups = {g["grp"] for g in audit.json()["groups"]}
            assert groups == {"Female", "Male"}

            rows = c.get("/activation_comparison").json()
            assert any(r["seed"] == 4242 and r["activation"] == "tanh" for r in rows)
    finally:
        if run_id is not None:
            db.get_client().table("runs").delete().eq("id", run_id).execute()
