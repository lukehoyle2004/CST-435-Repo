"""Persist the two offline training runs (models/runs.json) to the Supabase `seq_runs` table.

    python -m db.seed_runs          # needs SUPABASE_URL + SUPABASE_SERVICE_KEY (.env)

Apply db/migrations/001_init.sql in the SQL Editor first. Upserts on run_id, so
re-running after re-exporting the models is safe.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent


def load_rows(path: Path = ROOT / "models" / "runs.json") -> list[dict]:
    """runs.json rows with the same keys each (PostgREST bulk upserts need that)."""
    runs = json.loads(path.read_text())
    keys = sorted({k for r in runs for k in r})
    return [{k: r.get(k) for k in keys} for r in runs]


def main():
    load_dotenv(ROOT / ".env")
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")):
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY first (see .env.example).")
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    client.table("seq_runs").upsert(load_rows(), on_conflict="run_id").execute()
    for row in client.table("seq_model_comparison").select("*").execute().data:
        print(f"{row['name']:9s} {row['run_id']}  test acc {row['test_acc_pct']}%  "
              f"macro F1 {row['macro_f1']}  ECE {row['ece']}  gain {row['gain_over_lstm_pts']:+} pts")


if __name__ == "__main__":
    main()
