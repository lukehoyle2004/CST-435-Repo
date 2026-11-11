"""The tier boundary, checked against the code: no model, no SQL writes, no secrets.

These are the README's claims about the Streamlit tier, turned into tests.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from conftest import ROOT

UI = sorted((ROOT / "ui").glob("*.py"))
SOURCE = {p.name: p.read_text(encoding="utf-8") for p in UI}
CODE = {name: re.sub(r'(?s)""".*?"""|#[^\n]*', "", text) for name, text in SOURCE.items()}


def test_no_model_code_in_this_tier():
    for name, code in CODE.items():
        assert not re.search(r"^\s*(import|from)\s+(torch|transformers|tensorflow|sklearn|onnx)", code, re.M), name
    reqs = (ROOT / "ui" / "requirements.txt").read_text(encoding="utf-8")
    reqs = "\n".join(line for line in reqs.splitlines() if not line.startswith("#"))
    assert not re.search(r"torch|transformers|sentencepiece", reqs)


def test_no_sql_write_path():
    for name, code in CODE.items():
        # Supabase writes are .table(...).insert/update/upsert/delete(...) or .rpc(...).
        # (st.status(...).update(label=...) is a UI call and is allowed.)
        assert not re.search(r"\.(insert|upsert|delete|rpc)\s*\(", code), name
        assert not re.search(r"\.table\([^)]*\)(\s*\.\w+\([^)]*\))*\s*\.update\s*\(", code), name
        assert not re.search(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE)\s", code), name


def test_database_reads_touch_only_the_two_aggregate_views():
    tables = set(re.findall(r"\.table\(\s*\"([^\"]+)\"", CODE["app.py"]))
    assert tables == {"trio_benchmarks", "trio_benchmarks_hourly"}


def test_only_the_anon_key_is_referenced_and_none_is_committed():
    for name, text in SOURCE.items():
        assert "SUPABASE_SERVICE_KEY" not in text, name
        assert "service_role" not in CODE[name].lower(), name          # comments may mention it; code may not
        assert not re.search(r"eyJ[A-Za-z0-9_-]{20,}\.", text), f"a JWT is hard-coded in {name}"
    tracked = subprocess.run(["git", "ls-files", str(ROOT)], capture_output=True, text=True, cwd=ROOT).stdout
    assert "secrets.toml\n" not in tracked.replace("secrets.toml.example", "")


def test_every_inference_goes_through_the_http_client():
    assert "TrioClient(" in CODE["app.py"]
    assert "requests" in CODE["client.py"]
