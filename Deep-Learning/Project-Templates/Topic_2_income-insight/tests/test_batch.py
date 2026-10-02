"""Score-a-CSV tests: /predict_batch returns (and logs) exactly one prediction
per input row, in order."""
from __future__ import annotations

import pytest

from shared.data import FEATURE_COLS
from tests.conftest import FAST_TRAIN


def _train(client) -> int:
    return client.post("/train", json=FAST_TRAIN).json()["run_id"]


def test_batch_row_count_matches_input(client):
    run_id = _train(client)
    records = client._store.adult.head(37)[FEATURE_COLS].to_dict(orient="records")
    before = len(client._store.predictions)

    body = client.post("/predict_batch", json={"run_id": run_id, "records": records}).json()

    assert body["n_rows"] == len(body["predictions"]) == 37
    logged = client._store.predictions[before:]
    assert len(logged) == 37
    assert all(p["source"] == "csv" for p in logged)
    # Order is preserved: row i of the response is row i of the request.
    assert [p["features"] for p in logged] == records
    assert [p["proba"] for p in logged] == [p["proba"] for p in body["predictions"]]


def test_batch_matches_single_row_scoring(client):
    run_id = _train(client)
    records = client._store.adult.head(5)[FEATURE_COLS].to_dict(orient="records")
    batch = client.post("/predict_batch", json={"run_id": run_id, "records": records}).json()
    singles = [
        client.post("/predict", json={"run_id": run_id, "features": r}).json()["proba"]
        for r in records
    ]
    # float32 matrix products round slightly differently for a batch of 5 vs 1.
    assert [p["proba"] for p in batch["predictions"]] == pytest.approx(singles, abs=1e-6)


def test_batch_rejects_empty_and_missing_columns(client):
    run_id = _train(client)
    assert client.post("/predict_batch", json={"run_id": run_id, "records": []}).status_code == 422
    resp = client.post("/predict_batch", json={"run_id": run_id, "records": [{"age": 40}]})
    assert resp.status_code == 422
    assert "missing required columns" in resp.json()["detail"]
