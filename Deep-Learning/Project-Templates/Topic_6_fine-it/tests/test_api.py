"""Every endpoint against the in-memory store: schema, logging, joins, privacy."""
from __future__ import annotations

import pytest

from shared.queues import QUEUE_NAMES
from shared.schemas import EXAMPLE_MESSAGE
from shared.text import sha256


def _check_prediction(body, kind):
    assert body["queue"] in QUEUE_NAMES
    assert set(body["scores"]) == set(QUEUE_NAMES)
    assert abs(sum(body["scores"].values()) - 1) < 1e-3
    assert body["confidence"] == max(body["scores"].values())
    assert body["model"] == kind
    assert body["latency_ms"] > 0
    assert body["cold_start"] == (body["load_ms"] > 0)


def test_predict_logs_a_hash_joined_to_the_deployed_experiment(client):
    r = client.post("/predict", json={"text": EXAMPLE_MESSAGE})
    assert r.status_code == 200
    body = r.json()
    _check_prediction(body, "fine_tuned")
    assert body["logged"] and r.headers["X-FTF-Logged"] == "true"

    [row] = client.store.predictions
    deployed = next(e for e in client.store.experiments if e["is_deployed"])
    assert row["experiment_id"] == deployed["id"] == body["experiment_id"]
    assert body["run_name"] == deployed["run_name"]
    assert row["endpoint"] == "predict" and row["source"] == "test"
    assert row["input_sha256"] == sha256(EXAMPLE_MESSAGE)
    assert EXAMPLE_MESSAGE not in str(row)                  # the text itself is never persisted
    assert row["id"] == body["prediction_id"]


def test_zero_shot_is_logged_against_the_baseline_row(client):
    r = client.post("/predict_zero_shot", json={"text": EXAMPLE_MESSAGE})
    assert r.status_code == 200
    body = r.json()
    _check_prediction(body, "zero_shot")
    [row] = client.store.predictions
    baseline = next(e for e in client.store.experiments if e["kind"] == "zero_shot")
    assert row["experiment_id"] == baseline["id"] and row["endpoint"] == "predict_zero_shot"


def test_first_zero_shot_call_is_a_cold_start_then_warm(client):
    first = client.post("/predict_zero_shot", json={"text": "hi"}).json()
    second = client.post("/predict_zero_shot", json={"text": "hi"}).json()
    assert first["cold_start"] and first["load_ms"] > 0
    assert not second["cold_start"] and second["load_ms"] == 0


def test_batch_routes_and_logs_every_message(client):
    texts = ["my card hasn't arrived", "why was I charged a fee?", "someone stole my phone"]
    r = client.post("/predict_batch", json={"texts": texts, "model": "fine_tuned"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3 and body["logged"] == 3
    assert body["wall_ms"] >= body["total_latency_ms"] > 0
    assert [p["input_sha256"] for p in client.store.predictions] == [sha256(t) for t in texts]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "x" * 2001}, {}, {"text": "   "}])
def test_bad_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422
    assert client.store.predictions == []


def test_batch_limits(client):
    assert client.post("/predict_batch", json={"texts": []}).status_code == 422
    assert client.post("/predict_batch", json={"texts": ["a"] * 33}).status_code == 422
    assert client.post("/predict_batch", json={"texts": ["ok", ""]}).status_code == 422


def test_experiments_leaderboard_comes_from_the_store(client):
    body = client.get("/experiments").json()
    rows = body["experiments"]
    assert len(rows) == len(client.store.experiments) >= 4          # >= 3 fine-tuned + baseline
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    f1 = [r["test_macro_f1"] for r in rows]
    assert f1 == sorted(f1, reverse=True)
    for r in rows:
        assert r["test_macro_f1_lo"] <= r["test_macro_f1"] <= r["test_macro_f1_hi"]
    assert body["deployed_is_best"], "the served checkpoint must be the best fine-tuned row"
    assert sum(r["is_deployed"] for r in rows) == 1


def test_metrics_count_this_process(client):
    client.post("/predict", json={"text": "hello"})
    client.post("/predict", json={"text": "hello again"})
    m = client.get("/metrics").json()
    assert m["since_start"]["predict"]["calls"] == 2
    assert m["since_start"]["predict"]["unlogged"] == 0
    assert "fine_tuned" in m["resident_models"]


def test_healthz_version_queues(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["database"]["ok"]
    v = client.get("/version").json()
    assert v["queues"] == QUEUE_NAMES and v["fine_tuned"]["run_name"] == h["deployed_run"]
    q = client.get("/queues").json()
    assert [x["name"] for x in q["queues"]] == QUEUE_NAMES
    assert client.get("/", follow_redirects=False).status_code in (302, 307)


def test_needs_review_policy():
    from src.serving import FRAUD_FLOOR, REVIEW_THRESHOLD, needs_review
    base = {q: 0.0 for q in QUEUE_NAMES}
    assert not needs_review({**base, "transfers": 0.95, "cash_atm": 0.05})
    assert needs_review({**base, "transfers": REVIEW_THRESHOLD - 0.01, "cash_atm": 1 - REVIEW_THRESHOLD + 0.01})
    assert needs_review({**base, "fees_fx": 0.85, "security_fraud": FRAUD_FLOOR + 0.01, "transfers": 0.04})
    assert not needs_review({**base, "security_fraud": 0.9, "fees_fx": 0.1})
