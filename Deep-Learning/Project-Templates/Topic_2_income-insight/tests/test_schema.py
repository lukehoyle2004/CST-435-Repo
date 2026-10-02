"""Request-schema validation tests: bad input is a 422, never a 500."""
from __future__ import annotations

import pytest


def test_predict_rejects_missing_fields(client):
    # Missing 'features' -> 422 Unprocessable Entity from Pydantic validation.
    assert client.post("/predict", json={"run_id": 1}).status_code == 422


def test_predict_rejects_wrong_type(client):
    resp = client.post("/predict", json={"run_id": "not-an-int", "features": {}})
    assert resp.status_code == 422


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5])
def test_threshold_must_be_strictly_between_0_and_1(client, bad):
    resp = client.post("/predict", json={"features": {}, "threshold": bad})
    assert resp.status_code == 422


def test_train_rejects_unknown_activation(client):
    assert client.post("/train", json={"activation": "softplus"}).status_code == 422


@pytest.mark.parametrize("body", [{"n_layers": 0}, {"epochs": 0}, {"lr": 0}, {"max_train_rows": 10}])
def test_train_rejects_out_of_range_hyperparameters(client, body):
    assert client.post("/train", json=body).status_code == 422


def test_bias_audit_rejects_unprotected_attribute(client):
    resp = client.get("/bias_audit", params={"run_id": 1, "attribute": "occupation"})
    assert resp.status_code == 422


def test_schema_lists_features_and_protected_attributes(client):
    body = client.get("/schema").json()
    assert "age" in body["numeric_features"]
    assert "occupation" in body["categorical_features"]
    assert body["protected_features"] == ["sex", "race"]
    assert "sex" not in body["categorical_features"]  # not a model input by default
    assert body["target_classes"] == ["<=50K", ">50K"]
    assert {"relu", "tanh", "sigmoid"} <= set(body["activations"])
