"""Persist the two offline training runs (models/runs.json) to the Supabase `runs` table.

    python -m db.seed_runs          # needs SUPABASE_URL + SUPABASE_SERVICE_KEY (.env)

Upserts on run_id, so re-running after re-exporting the models is safe.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent


def main():
    load_dotenv(ROOT / ".env")
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    runs = json.loads((ROOT / "models" / "runs.json").read_text())
    client.table("runs").upsert(runs, on_conflict="run_id").execute()
    for row in client.table("model_comparison").select("*").execute().data:
        print(f"{row['name']:9s} {row['run_id']}  test acc {row['test_acc_tta']}%  "
              f"macro F1 {row['macro_f1']}  gain {row['gain_over_baseline']:+} pts")


if __name__ == "__main__":
    main()
