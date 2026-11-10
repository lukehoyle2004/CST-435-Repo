"""Regression fixture: the service still gives the answers it gave when the fixture was made.

tests/fixtures/reference_outputs.json is produced by scripts/make_reference.py.
Comparisons are tolerant only where CPU floating-point differences between
machines could legitimately move a number (probabilities to 2e-3); the ranked
candidates, the predicted class and the checkpoints must match exactly, and
generated text must stay at least 90% similar.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest

CASES = json.loads((Path(__file__).parent / "fixtures" / "reference_outputs.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[f"{c['path']}:{i}" for i, c in enumerate(CASES)])
def test_matches_reference(client, case):
    r = client.post(case["path"], json=case["request"])
    assert r.status_code == 200
    got, exp = r.json(), case["expected"]
    assert got["checkpoint"] == exp["checkpoint"]

    if "prediction" in exp:
        assert got["prediction"] == exp["prediction"]
        for k, v in exp["probabilities"].items():
            assert got["probabilities"][k] == pytest.approx(v, abs=2e-3)
    if "candidates" in exp:
        assert [c["token"] for c in got["candidates"]] == [c["token"] for c in exp["candidates"]]
        for g, e in zip(got["candidates"], exp["candidates"]):
            assert g["score"] == pytest.approx(e["score"], abs=2e-3)
    for key in ("continuation", "summary"):
        if key in exp:
            ratio = difflib.SequenceMatcher(None, got[key], exp[key]).ratio()
            assert ratio >= 0.9, f"{key} drifted (similarity {ratio:.2f}):\n{got[key]!r}\nvs\n{exp[key]!r}"
