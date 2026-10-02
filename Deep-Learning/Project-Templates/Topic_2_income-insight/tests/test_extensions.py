"""Tests for network depth, activation choice, diverged-run handling (NULL
metrics + status instead of a sentinel), and the decision threshold."""
from __future__ import annotations

import base64
import math
import pickle

import numpy as np
import pytest
from torch import nn

from api import main as api_main
from api.training import (
    ACTIVATIONS,
    MLP,
    STATUS_DIVERGED,
    STATUS_OK,
    TrainResult,
    predict_records,
    run_status,
    train_income_classifier,
)
from shared.data import FEATURE_COLS, generate_adult_like
from tests.conftest import FAST_TRAIN

DATA = generate_adult_like(1200, seed=3)
RECORD = DATA.iloc[0][FEATURE_COLS].to_dict()


# ---------------------------------------------------------------------------
# depth and activation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_mlp_depth_matches_n_layers(n_layers):
    model = MLP(in_dim=10, hidden_dim=8, n_layers=n_layers)
    # n hidden layers + 1 output layer.
    assert sum(isinstance(m, nn.Linear) for m in model.modules()) == n_layers + 1


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_every_activation_builds_and_is_used(activation):
    model = MLP(in_dim=10, hidden_dim=8, n_layers=2, activation=activation)
    acts = [m for m in model.modules() if isinstance(m, ACTIVATIONS[activation])]
    assert len(acts) == 2


def test_artifact_records_depth_activation_and_columns():
    r = train_income_classifier(DATA, hidden_dim=8, epochs=3, n_layers=3, activation="tanh")
    obj = pickle.loads(base64.b64decode(r.model_b64))
    assert (obj["n_layers"], obj["activation"]) == (3, "tanh")
    assert obj["numeric"] + obj["categorical"] == FEATURE_COLS


def test_include_protected_adds_sex_and_race_as_inputs():
    r = train_income_classifier(DATA, hidden_dim=8, epochs=3, include_protected=True)
    obj = pickle.loads(base64.b64decode(r.model_b64))
    assert {"sex", "race"} <= set(obj["categorical"])
    with pytest.raises(ValueError, match="missing required columns"):
        predict_records(r.model_b64, [RECORD])  # record lacks sex/race


# ---------------------------------------------------------------------------
# divergence
# ---------------------------------------------------------------------------
def test_run_status_flags_nonfinite_and_rising_loss():
    ok_prob = np.array([0.2, 0.8])
    assert run_status([0.7, 0.5, 0.3], ok_prob) == STATUS_OK
    assert run_status([0.7, float("nan")], ok_prob) == STATUS_DIVERGED
    assert run_status([0.7, float("inf")], ok_prob) == STATUS_DIVERGED
    assert run_status([0.7, 0.5], np.array([0.2, np.nan])) == STATUS_DIVERGED
    assert run_status([0.5, 0.9], ok_prob) == STATUS_DIVERGED  # loss went up
    assert run_status([0.7], ok_prob) == STATUS_OK  # single epoch: nothing to compare


def test_diverged_run_stores_null_metrics_and_blocks_predict(client, monkeypatch):
    def fake_train(*_args, **_kwargs):
        return TrainResult(
            status=STATUS_DIVERGED,
            metrics={k: None for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            train_accuracy=None, val_accuracy=None,
            loss_history=[0.7, float("nan")], val_loss_history=[0.7, float("nan")],
            model_b64="unused", n_train=1, n_val=1, n_test=1,
        )

    monkeypatch.setattr(api_main, "train_income_classifier", fake_train)
    body = client.post("/train", json={"lr": 50.0}).json()

    assert body["status"] == "diverged"
    assert all(v is None for v in body["metrics"].values())
    assert body["final_loss"] is None  # NaN is stored as null, never a number
    assert body["n_test_predictions_logged"] == 0  # nothing to audit

    run = client.get(f"/runs/{body['run_id']}").json()
    assert run["status"] == "diverged" and run["accuracy"] is None

    resp = client.post("/predict", json={"run_id": body["run_id"], "features": RECORD})
    assert resp.status_code == 409  # a diverged model is never served


# ---------------------------------------------------------------------------
# threshold
# ---------------------------------------------------------------------------
def test_threshold_controls_the_label(client):
    run_id = client.post("/train", json=FAST_TRAIN).json()["run_id"]
    p = client.post("/predict", json={"run_id": run_id, "features": RECORD}).json()["proba"]
    assert 0.0 < p < 1.0

    below = client.post(
        "/predict", json={"run_id": run_id, "features": RECORD, "threshold": p / 2}
    ).json()
    above = client.post(
        "/predict", json={"run_id": run_id, "features": RECORD, "threshold": (1 + p) / 2}
    ).json()
    assert below["label"] == 1  # p >= threshold
    assert above["label"] == 0  # p <  threshold
    assert math.isclose(below["proba"], above["proba"])  # same model output

    logged = [r["threshold"] for r in client._store.predictions if r["source"] == "row"]
    assert logged[-2:] == [below["threshold"], above["threshold"]]


def test_legacy_artifact_without_new_keys_still_loads():
    # Artifacts saved before depth/activation/columns were stored.
    r = train_income_classifier(DATA, hidden_dim=8, epochs=2, n_layers=1)
    obj = pickle.loads(base64.b64decode(r.model_b64))
    for key in ("n_layers", "activation", "numeric", "categorical"):
        del obj[key]
    legacy = base64.b64encode(pickle.dumps(obj)).decode("ascii")
    [(label, proba)] = predict_records(legacy, [RECORD])
    assert label in (0, 1) and 0.0 <= proba <= 1.0
