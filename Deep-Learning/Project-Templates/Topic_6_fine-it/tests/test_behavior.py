"""Behavioural tests: does the deployed router actually do its job?

* It must beat the zero-shot baseline on held-out test data by more than the
  noise (the paired-bootstrap interval of the difference excludes zero).
* It must keep answering a fixed set of hand-written messages the same way
  (tests/fixtures/regression.json), so a retrained or re-uploaded checkpoint
  that silently changes behaviour fails CI.
* Its answer must not depend on surface noise: casing, extra spaces, a
  trailing "please" or punctuation.
* The zero-shot variant the service runs (float16-stored weights, so it fits
  512 MB) must agree with the float32 baseline the leaderboard scored.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_real_model, results

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "regression.json").read_text(encoding="utf-8"))


def _rows():
    rows = results()
    deployed = next(r for r in rows if r["is_deployed"])
    baseline = next(r for r in rows if r["kind"] == "zero_shot")
    return rows, deployed, baseline


def test_deployed_beats_zero_shot_by_more_than_the_noise():
    _, deployed, baseline = _rows()
    assert deployed["delta_macro_f1_lo"] > 0.10            # paired 95% CI of the gain sits well above 0
    assert deployed["test_macro_f1_lo"] > baseline["test_macro_f1_hi"]
    assert deployed["test_accuracy"] > baseline["test_accuracy"]


def test_deployed_catches_fraud_reports():
    _, deployed, baseline = _rows()
    assert deployed["fraud_recall"] >= 0.90
    assert deployed["fraud_recall_lo"] > baseline["fraud_recall_hi"]


def test_deployed_run_was_chosen_on_validation_not_test():
    rows, deployed, _ = _rows()
    fine_tuned = [r for r in rows if r["kind"] == "fine_tuned"]
    assert deployed["val_macro_f1"] == max(r["val_macro_f1"] for r in fine_tuned)


def test_every_fine_tuned_run_beats_zero_shot():
    rows, _, _ = _rows()
    for r in rows:
        if r["kind"] == "fine_tuned":
            assert r["delta_macro_f1_lo"] > 0, r["run_name"]


@requires_real_model
def test_regression_fixture(client):
    assert FIXTURE["run_name"] == next(r["run_name"] for r in results() if r["is_deployed"])
    for case in FIXTURE["cases"]:
        body = client.post("/predict", json={"text": case["text"]}).json()
        assert body["queue"] == case["queue"] == case["intended"], case["text"]
        assert body["confidence"] == pytest.approx(case["confidence"], abs=0.02), case["text"]


@requires_real_model
@pytest.mark.parametrize("variant", [str.lower, str.upper, lambda t: "  " + t.replace(" ", "  ") + "  ",
                                     lambda t: t.rstrip(".?!") + " please", lambda t: t.rstrip(".?!")])
def test_surface_noise_does_not_change_the_queue(client, variant):
    for case in FIXTURE["cases"]:
        body = client.post("/predict", json={"text": variant(case["text"])}).json()
        assert body["queue"] == case["queue"], variant(case["text"])


def test_served_zero_shot_variant_matches_the_leaderboard_baseline():
    f = ROOT / "docs" / "experiments" / "zero_shot_served_variant.json"
    v = json.loads(f.read_text())
    _, _, baseline = _rows()
    assert v["leaderboard_run"] == baseline["run_name"]
    assert v["n_test"] == 3076
    assert v["agreement_with_float32"] >= 0.99
    assert v["float32_test_macro_f1"] == pytest.approx(baseline["test_macro_f1"], abs=1e-4)
    lo, hi = baseline["test_macro_f1_lo"], baseline["test_macro_f1_hi"]
    assert lo <= v["served_test_macro_f1"] <= hi             # inside the baseline's own 95% CI
