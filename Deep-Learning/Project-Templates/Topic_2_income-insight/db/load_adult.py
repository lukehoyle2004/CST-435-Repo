"""Download the UCI Adult dataset, clean it, split it, and load it into Supabase.

Usage (from the project folder, with a .env holding the SERVICE-ROLE key, after
applying migrations 001-003 in the Supabase SQL Editor):

    python -m db.load_adult            # loads only if adult_income is empty
    python -m db.load_adult --force    # deletes and reloads every row

The split is stratified and seeded (70/15/15, seed 42), so reloading always
reproduces the same train/val/test membership.
"""
from __future__ import annotations

import argparse
import io
import urllib.request

import pandas as pd
from dotenv import load_dotenv

from shared.data import (
    RAW_COLUMNS,
    STORED_COLS,
    UCI_BASE_URL,
    UCI_FILES,
    assign_splits,
    clean_adult,
)

BATCH = 1000


def download_adult() -> pd.DataFrame:
    """Fetch adult.data + adult.test from the UCI archive and combine them."""
    frames = []
    for name in UCI_FILES:
        with urllib.request.urlopen(f"{UCI_BASE_URL}/{name}", timeout=60) as resp:
            text = resp.read().decode("utf-8")
        # adult.test starts with a "|1x3 Cross validator" comment line.
        skip = 1 if name == "adult.test" else 0
        frames.append(
            pd.read_csv(io.StringIO(text), names=RAW_COLUMNS, skipinitialspace=True,
                        skiprows=skip)
        )
    return pd.concat(frames, ignore_index=True).dropna(how="all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="delete and reload all rows")
    args = parser.parse_args()

    load_dotenv()
    from api import db  # after load_dotenv so the client sees the env vars

    client = db.get_client()
    existing = client.table("adult_income").select("id", count="exact").limit(1).execute()
    if existing.count and not args.force:
        print(f"adult_income already has {existing.count} rows; use --force to reload.")
        return
    if existing.count:
        client.table("adult_income").delete().gte("id", 0).execute()
        print(f"Deleted {existing.count} existing rows.")

    df = assign_splits(clean_adult(download_adult()))
    rows = df[["id"] + STORED_COLS].to_dict(orient="records")
    for start in range(0, len(rows), BATCH):
        client.table("adult_income").insert(rows[start:start + BATCH]).execute()
        print(f"  inserted {min(start + BATCH, len(rows))}/{len(rows)}", end="\r")

    counts = df["split"].value_counts().to_dict()
    print(f"\nLoaded {len(df)} rows into adult_income: {counts}, "
          f">50K rate {df['label'].mean():.3f}")


if __name__ == "__main__":
    main()
