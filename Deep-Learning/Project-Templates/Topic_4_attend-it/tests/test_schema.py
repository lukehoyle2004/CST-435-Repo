"""The database contract: what the code writes must match db/migrations/001_init.sql."""
from __future__ import annotations

import re

from db.seed_runs import load_rows
from tests.conftest import ROOT

SQL = (ROOT / "db" / "migrations" / "001_init.sql").read_text()


def table_columns(table: str) -> set[str]:
    body = re.search(rf"create table {table} \((.*?)\n\);", SQL, re.S).group(1)
    cols = set()
    for line in body.splitlines():
        m = re.match(r"\s+([a-z_0-9]+)\s+(text|integer|bigint|double|jsonb|timestamptz|boolean)", line)
        if m:
            cols.add(m.group(1))
    return cols


def test_runs_json_matches_seq_runs_columns():
    rows = load_rows()
    assert {r["name"] for r in rows} == {"lstm", "attention"}
    assert len({frozenset(r) for r in rows}) == 1          # same keys on every row
    assert set(rows[0]) == table_columns("seq_runs") - {"created_at"}


def test_logged_prediction_matches_seq_predictions_columns(client):
    client.post("/predict", json={"text": "a fine film"})
    logged = set(client.store.predictions[-1]) - {"id"}
    assert logged == table_columns("seq_predictions") - {"id", "created_at"}


def test_every_table_has_rls_and_only_select_for_anon():
    tables = re.findall(r"create table (\w+)", SQL)
    for t in tables:
        assert re.search(rf"alter table {t}\s+enable row level security", SQL), t
    policies = re.findall(r"create policy .* on (\w+)\s+for (\w+) to anon", SQL)
    assert sorted(policies) == sorted((t, "select") for t in tables)


def test_migration_only_touches_seq_objects():
    # It may share a Supabase project with See-Sense, whose tables are `runs` / `predictions`.
    for name in re.findall(r"(?:create|drop) (?:table|view)(?: if exists)? (\w+)", SQL):
        assert name.startswith("seq_"), name
