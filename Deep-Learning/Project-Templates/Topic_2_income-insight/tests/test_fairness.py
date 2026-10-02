"""Bias-audit and activation-comparison tests.

The FakeDB mirrors the SQL views, so these check the API plumbing and the FPR /
FNR definitions against an independent sklearn computation. The SQL itself is
exercised against real Supabase by test_supabase_roundtrip.py.
"""
from __future__ import annotations

import math

from sklearn.metrics import confusion_matrix

from tests.conftest import FAST_TRAIN


def test_bias_audit_fpr_fnr_match_sklearn(client):
    run_id = client.post("/train", json=FAST_TRAIN).json()["run_id"]
    body = client.get("/bias_audit", params={"run_id": run_id, "attribute": "sex"}).json()

    adult = client._store.adult.set_index("id")
    preds = [p for p in client._store.predictions
             if p["run_id"] == run_id and p["source"] == "test_eval"]
    assert {g["grp"] for g in body["groups"]} == {"Female", "Male"}
    assert sum(g["n"] for g in body["groups"]) == len(preds)

    for g in body["groups"]:
        rows = [p for p in preds if adult.loc[p["adult_id"], "sex"] == g["grp"]]
        y = [int(adult.loc[p["adult_id"], "label"]) for p in rows]
        yhat = [p["label"] for p in rows]
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        assert (g["tp"], g["fp"], g["tn"], g["fn"]) == (tp, fp, tn, fn)
        assert math.isclose(g["fpr"], fp / (fp + tn))
        assert math.isclose(g["fnr"], fn / (fn + tp))

    fprs = [g["fpr"] for g in body["groups"]]
    assert math.isclose(body["fpr_gap"], max(fprs) - min(fprs))


def test_bias_audit_by_race(client):
    run_id = client.post("/train", json=FAST_TRAIN).json()["run_id"]
    body = client.get("/bias_audit", params={"run_id": run_id, "attribute": "race"}).json()
    assert body["attribute"] == "race"
    assert {g["grp"] for g in body["groups"]} <= set(client._store.adult["race"])


def test_bias_audit_404_without_test_predictions(client):
    resp = client.get("/bias_audit", params={"run_id": 999, "attribute": "sex"})
    assert resp.status_code == 404


def test_activation_comparison_groups_matched_runs(client):
    for activation in ("relu", "tanh", "sigmoid"):
        client.post("/train", json={**FAST_TRAIN, "activation": activation, "seed": 0})
    rows = client.get("/activation_comparison").json()
    assert {r["activation"] for r in rows} == {"relu", "tanh", "sigmoid"}
    # Same controls everywhere -> each row differs only in activation.
    controls = {(r["n_layers"], r["hidden_dim"], r["lr"], r["epochs"], r["seed"]) for r in rows}
    assert len(controls) == 1
    assert all(r["runs"] == 1 and 0.0 <= r["test_accuracy"] <= 1.0 for r in rows)
