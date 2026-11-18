"""Static checks on db/migrations/001_init.sql (offline, in CI).

They pin the parts of the schema the API, FleetDeck and the privacy statement
rely on, so a careless edit to the SQL fails a test instead of a deploy.
"""
from __future__ import annotations

import re
from pathlib import Path

from shared.queues import QUEUE_NAMES

SQL = (Path(__file__).resolve().parents[1] / "db" / "migrations" / "001_init.sql").read_text(encoding="utf-8")
CODE = re.sub(r"--[^\n]*", "", SQL)
LOWER = CODE.lower()


def _table_body(name: str) -> str:
    m = re.search(rf"create table {name} \((.*?)\n\);", CODE, re.S)
    assert m, f"{name} missing"
    return m.group(1)


def _columns(name: str) -> set[str]:
    return {line.split()[0] for line in _table_body(name).strip().splitlines()
            if line.strip() and not line.strip().startswith("check")}


def test_every_object_is_prefixed_so_it_cannot_touch_other_products():
    names = re.findall(r"create (?:table|view|unique index|index|policy)\s+(\w+)", LOWER)
    assert names and all(n.startswith(("ftf_", "idx_ftf_")) for n in names), names
    for dropped in re.findall(r"drop \w+\s+if exists (\w+)", LOWER):
        assert dropped.startswith("ftf_")


def test_predictions_store_a_hash_never_the_message():
    cols = _columns("ftf_predictions")
    assert "input_sha256" in cols
    assert not cols & {"text", "message", "input", "input_text", "raw_text", "prompt"}
    assert "'^[0-9a-f]{64}$'" in _table_body("ftf_predictions")


def test_predictions_join_to_the_experiment_that_served_them():
    body = _table_body("ftf_predictions")
    assert re.search(r"experiment_id\s+bigint\s+not null references ftf_experiments", body)
    assert "check (cold_start = (load_ms > 0))" in body


def test_documents_carry_split_labels_and_the_queue_contract():
    body = _table_body("ftf_documents")
    assert "('train', 'validation', 'test')" in body
    for q in QUEUE_NAMES:
        assert f"'{q}'" in body


def test_experiments_store_intervals_and_one_deployed_checkpoint():
    cols = _columns("ftf_experiments")
    for metric in ("test_macro_f1", "test_accuracy", "fraud_recall", "delta_macro_f1"):
        assert {metric, f"{metric}_lo", f"{metric}_hi"} <= cols, metric
    assert "where is_deployed" in LOWER                       # partial unique index
    assert "check ((kind = 'zero_shot') = (delta_macro_f1 is null))" in LOWER


def test_leaderboard_says_when_runs_are_not_meaningfully_different():
    assert "distinguishable_from_best" in LOWER
    assert "e.test_macro_f1_hi < b.best_lo" in LOWER


def test_anon_reads_experiments_and_aggregates_only():
    for t in ("ftf_documents", "ftf_experiments", "ftf_predictions"):
        assert re.search(rf"alter table {t}\s+enable row level security", LOWER), t
    assert "revoke all on ftf_documents, ftf_predictions from anon, authenticated" in LOWER
    assert "grant select on ftf_experiments to anon" in LOWER
    assert "grant select on ftf_leaderboard to anon" in LOWER
    assert "grant select on ftf_latency to anon" in LOWER
    policies = re.findall(r"create policy \w+ on (\w+)", LOWER)
    assert policies == ["ftf_experiments"]                    # no anon policy on raw text or predictions
    assert "where p.source <> 'test'" in LOWER                # test traffic never reaches the dashboard
