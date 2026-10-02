"""Regression tests for the training pipeline.

The synthetic rows follow a known logistic rule, so a working MLP must clear an
accuracy/AUC floor; if a change breaks learning, these fail. Also checks that the
evaluation artifacts (splits, confusion matrix, calibration, permutation
importance) are internally consistent and that runs are reproducible.
"""
from __future__ import annotations

import math

from api.training import STATUS_OK, train_income_classifier
from shared.data import FEATURE_COLS, generate_adult_like
from tests.conftest import FAST_TRAIN

DATA = generate_adult_like(2000, seed=1)
FAST = {"hidden_dim": 16, "batch_size": 64, "epochs": 20}


def test_classifier_recovers_signal():
    r = train_income_classifier(DATA, **FAST)
    assert r.status == STATUS_OK
    assert r.metrics["accuracy"] > 0.75
    assert r.metrics["roc_auc"] > 0.8
    assert r.loss_history[-1] < r.loss_history[0]  # training loss decreased
    assert isinstance(r.model_b64, str) and r.model_b64


def test_uses_fixed_train_val_test_splits():
    r = train_income_classifier(DATA, **FAST)
    counts = DATA["split"].value_counts()
    assert (r.n_train, r.n_val, r.n_test) == (counts["train"], counts["val"], counts["test"])
    # One logged prediction per test row, keyed by adult_income id.
    test_ids = set(DATA.loc[DATA["split"] == "test", "id"])
    assert {i for i, _l, _p in r.test_predictions} == test_ids


def test_evaluation_artifacts_are_consistent():
    r = train_income_classifier(DATA, **FAST)
    assert sum(r.confusion.values()) == r.n_test
    assert sum(b["count"] for b in r.calibration) == r.n_test
    assert all(0.0 <= b["mean_pred"] <= 1.0 for b in r.calibration)
    assert 0.0 <= r.brier <= 1.0 and 0.0 <= r.ece <= 1.0
    # Accuracy from the confusion matrix matches the reported accuracy.
    c = r.confusion
    assert math.isclose((c["tp"] + c["tn"]) / r.n_test, r.metrics["accuracy"])
    # Per-class supports are the true class counts in the test split.
    assert r.class_report[">50K"]["support"] == c["tp"] + c["fn"]
    assert r.class_report["<=50K"]["support"] == c["tn"] + c["fp"]
    # Permutation importance covers every input column; protected ones excluded.
    assert {d["feature"] for d in r.perm_importance} == set(FEATURE_COLS)


def test_same_seed_is_reproducible_and_seed_matters():
    a = train_income_classifier(DATA, seed=7, **FAST)
    b = train_income_classifier(DATA, seed=7, **FAST)
    c = train_income_classifier(DATA, seed=8, **FAST)
    assert a.loss_history == b.loss_history
    assert a.metrics == b.metrics
    assert a.loss_history != c.loss_history


def test_max_train_rows_subsamples_only_the_training_split():
    r = train_income_classifier(DATA, max_train_rows=500, **FAST)
    assert r.n_train == 500
    assert r.n_test == (DATA["split"] == "test").sum()


def test_train_and_score_through_the_api(client):
    resp = client.post("/train", json={**FAST_TRAIN, "activation": "relu"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["metrics"]["accuracy"] > 0.75
    assert body["n_test_predictions_logged"] == (client._store.adult["split"] == "test").sum()

    detail = client.get(f"/runs/{body['run_id']}").json()
    assert detail["data_source"] == "adult_income"
    assert set(detail["class_report"]) == {"<=50K", ">50K"}
    assert len(detail["loss_history"]) == FAST_TRAIN["epochs"]

    record = client._store.adult.iloc[0][FEATURE_COLS].to_dict()
    pred = client.post("/predict", json={"features": record}).json()  # default run
    assert pred["run_id"] == body["run_id"]
    assert pred["label"] in (0, 1) and 0.0 <= pred["proba"] <= 1.0
