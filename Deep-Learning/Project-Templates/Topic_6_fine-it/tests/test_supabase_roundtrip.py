"""Live Supabase checks -- skipped unless a local .env provides real credentials.

Run after applying db/migrations/001_init.sql and loading the data:

    pytest tests/test_supabase_roundtrip.py -v

It proves the access-control design, not just connectivity: the service role
writes a prediction joined to its experiment and reads it back; the database
itself rejects rows that break the schema's invariants; and the anon key (the
only one FleetDeck holds) can read the leaderboard and latency aggregates but
neither the corpus text nor a single prediction row. Rows it writes use
source='test' (excluded from ftf_latency) and are deleted afterwards.
"""
from __future__ import annotations

import os

import pytest

from api import db

URL = os.environ.get("SUPABASE_URL", "")
LIVE = db.configured() and URL.startswith("https://") and "your-project-ref" not in URL
pytestmark = pytest.mark.skipif(not LIVE, reason="no live Supabase credentials in .env")

TEST_HASH = "0" * 63 + "1"


@pytest.fixture
def written():
    ids: list[int] = []
    yield ids
    if ids:
        db.get_client().table("ftf_predictions").delete().in_("id", ids).execute()


def _row(experiment_id: int, **over) -> dict:
    row = dict(experiment_id=experiment_id, endpoint="predict", input_sha256=TEST_HASH, n_chars=12, n_tokens=5,
               predicted_queue="security_fraud", confidence=0.97, needs_review=False,
               scores={"security_fraud": 0.97}, latency_ms=8.5, load_ms=0.0, cold_start=False,
               source="test", api_version="pytest")
    row.update(over)
    return row


def _deployed_id() -> int:
    rows = db.get_client().table("ftf_experiments").select("id,run_name").eq("is_deployed", True).execute().data
    assert len(rows) == 1
    return rows[0]["id"]


def test_ping():
    assert db.ping()["ok"]


def test_corpus_and_experiments_are_loaded():
    c = db.get_client()
    for split, n in (("train", 8952), ("validation", 995), ("test", 3076)):
        got = c.table("ftf_documents").select("id", count="exact").eq("split", split).limit(1).execute().count
        assert got == n, split
    board = db.get_leaderboard()
    assert len(board) >= 4 and board[0]["is_deployed"]


def test_service_role_logs_a_prediction_joined_to_its_experiment(written):
    exp = _deployed_id()
    pid = db.log_prediction(_row(exp))
    written.append(pid)
    got = (db.get_client().table("ftf_predictions")
           .select("id,experiment_id,input_sha256,source,ftf_experiments(run_name,is_deployed)")
           .eq("id", pid).execute().data[0])
    assert got["experiment_id"] == exp and got["source"] == "test"
    assert got["ftf_experiments"]["is_deployed"] is True


def test_database_rejects_rows_that_break_the_invariants(written):
    exp = _deployed_id()
    for bad in (dict(input_sha256="not-a-hash"), dict(load_ms=0.0, cold_start=True),
                dict(endpoint="classify"), dict(experiment_id=-1)):
        with pytest.raises(Exception):
            written.append(db.log_prediction(_row(exp, **bad)))


def test_anon_key_reads_aggregates_but_not_text_or_predictions():
    anon_key = os.environ.get("SUPABASE_ANON_KEY")
    if not anon_key:
        pytest.skip("SUPABASE_ANON_KEY not set")
    from supabase import create_client

    anon = create_client(URL, anon_key)
    assert anon.table("ftf_leaderboard").select("run_name").execute().data
    assert anon.table("ftf_experiments").select("run_name").limit(1).execute().data
    assert isinstance(anon.table("ftf_latency").select("endpoint").execute().data, list)
    for table in ("ftf_documents", "ftf_predictions"):
        with pytest.raises(Exception, match="permission denied"):
            anon.table(table).select("id").limit(1).execute()
    with pytest.raises(Exception):
        anon.table("ftf_experiments").update({"is_deployed": False}).eq("is_deployed", True).execute()
