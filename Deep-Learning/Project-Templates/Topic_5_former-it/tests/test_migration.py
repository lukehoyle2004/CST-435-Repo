"""Static checks on db/migrations/001_init.sql (runs offline, in CI).

They pin the parts of the schema the API and the privacy statement rely on, so
a careless edit to the SQL fails a test instead of a deploy.
"""
from __future__ import annotations

import re
from pathlib import Path

SQL = (Path(__file__).resolve().parents[1] / "db" / "migrations" / "001_init.sql").read_text(encoding="utf-8")
CODE = re.sub(r"--[^\n]*", "", SQL)        # strip comments
LOWER = CODE.lower()


def _table_body(name: str) -> str:
    m = re.search(rf"create table {name} \((.*?)\n\);", CODE, re.S)
    assert m, f"{name} missing"
    return m.group(1)


def test_required_objects_exist():
    for obj in ("create table trio_inputs", "create table trio_inferences",
                "create materialized view trio_benchmarks",
                "create function trio_log_inference", "create function trio_refresh_benchmarks"):
        assert obj in LOWER, obj


def test_inputs_table_stores_a_hash_not_the_text():
    body = _table_body("trio_inputs")
    assert "input_sha256" in body and "'^[0-9a-f]{64}$'" in body
    columns = {line.split()[0] for line in body.strip().splitlines() if line.strip()}
    assert not columns & {"text", "prompt", "input", "input_text", "raw_text"}


def test_inferences_table_has_the_required_columns():
    body = _table_body("trio_inferences")
    for col in ("model", "output", "latency_ms", "served_at", "input_id", "load_ms", "cold_start"):
        assert re.search(rf"^\s+{col}\s", body, re.M), col
    assert "('encoder_block', 'bert', 'gpt2', 't5')" in body


def test_benchmarks_view_aggregates_by_model_and_can_refresh_concurrently():
    assert "group by i.model" in LOWER
    assert "where i.source <> 'test'" in LOWER
    assert "percentile_cont(0.95)" in LOWER
    assert "create unique index idx_trio_benchmarks_model on trio_benchmarks (model)" in LOWER
    assert "refresh materialized view concurrently trio_benchmarks" in LOWER


def test_anon_can_read_only_the_aggregate():
    assert "alter table trio_inputs     enable row level security" in LOWER
    assert "alter table trio_inferences enable row level security" in LOWER
    assert "grant select on trio_benchmarks to anon" in LOWER
    assert not re.search(r"create policy[^;]*to anon", LOWER), "no anon policies on the raw tables"
    assert re.search(r"revoke execute on function trio_log_inference\([^)]*\) from public, anon", LOWER)


def test_rpc_parameters_match_the_api_row():
    from api.db import log_inference  # noqa: F401  (module imports cleanly)

    sig = re.search(r"create function trio_log_inference\((.*?)\) returns bigint", CODE, re.S).group(1)
    params = re.findall(r"p_(\w+)", sig)
    assert params == ["input_sha256", "n_chars", "n_words", "model", "archetype", "checkpoint", "task",
                      "output", "latency_ms", "load_ms", "cold_start", "input_tokens", "output_tokens",
                      "source", "api_version"]
