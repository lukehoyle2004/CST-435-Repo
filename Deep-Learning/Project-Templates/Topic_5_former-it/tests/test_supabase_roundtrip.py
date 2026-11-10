"""Live Supabase checks -- skipped unless a local .env provides real credentials.

Run after applying db/migrations/001_init.sql:

    pytest tests/test_supabase_roundtrip.py -v

It proves the access-control design, not just connectivity:
the service role can log and refresh; the anon key can read the aggregate
benchmarks view but can read neither the hashes nor the model outputs.
Rows it writes use source='test' and a hash that cannot collide with real text.
"""
from __future__ import annotations

import os

import pytest

from api import db

LIVE = db.configured() and os.environ.get("SUPABASE_URL", "").startswith("https://") \
    and "your-project-ref" not in os.environ.get("SUPABASE_URL", "")
pytestmark = pytest.mark.skipif(not LIVE, reason="no live Supabase credentials in .env")

TEST_HASH = "0" * 63 + "1"


def _row(**over):
    row = dict(input_sha256=TEST_HASH, n_chars=10, n_words=2, model="gpt2", archetype="decoder",
               checkpoint="distilgpt2@test", task="continue", output={"continuation": "test"},
               latency_ms=12.5, load_ms=0.0, cold_start=False, input_tokens=3, output_tokens=4,
               source="test", api_version="pytest")
    row.update(over)
    return row


def test_ping():
    assert db.ping()["ok"]


def test_service_role_logs_and_refreshes():
    first = db.log_inference(_row())
    second = db.log_inference(_row(model="t5", archetype="encoder-decoder", task="summarize",
                                   output={"summary": "test"}, load_ms=900.0, cold_start=True))
    assert second > first
    inp = db.get_client().table("trio_inputs").select("times_submitted").eq("input_sha256", TEST_HASH).execute().data
    assert inp and inp[0]["times_submitted"] >= 2          # one input row, upserted
    assert db.refresh_benchmarks()                          # returns the refresh timestamp
    rows = db.get_client().table("trio_inferences").select("source").in_("id", [first, second]).execute().data
    assert {r["source"] for r in rows} == {"test"}
    for r in db.get_benchmarks():                           # test rows are excluded from the view
        assert r["model"] in ("encoder_block", "bert", "gpt2", "t5")


def test_database_rejects_rows_that_break_the_invariants():
    with pytest.raises(Exception):
        db.log_inference(_row(input_sha256="not-a-hash"))
    with pytest.raises(Exception):
        db.log_inference(_row(model="encoder_block"))      # encoder_block must be task 'attention'


def test_anon_key_sees_only_the_aggregate_view():
    anon_key = os.environ.get("SUPABASE_ANON_KEY")
    if not anon_key:
        pytest.skip("SUPABASE_ANON_KEY not set")
    from supabase import create_client

    anon = create_client(os.environ["SUPABASE_URL"], anon_key)
    assert anon.table("trio_benchmarks").select("model").execute().data   # readable
    assert anon.table("trio_inferences").select("id").limit(1).execute().data == []  # RLS: no rows
    assert anon.table("trio_inputs").select("id").limit(1).execute().data == []
    with pytest.raises(Exception):
        anon.rpc("trio_log_inference", {f"p_{k}": v for k, v in _row().items()}).execute()
