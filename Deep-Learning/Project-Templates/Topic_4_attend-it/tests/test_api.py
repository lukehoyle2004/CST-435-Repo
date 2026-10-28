"""/predict + attention tests, the other endpoints, and the Supabase-outage path."""
import hashlib

import pytest

from shared.imdb import CLASS_NAMES
from shared.text import MAX_LEN

REVIEW = "An absolutely wonderful film. The acting was superb and I loved every minute!"


def predict(client, text=REVIEW, **body):
    return client.post("/predict", json={"text": text, **body})


def test_predict_smoke(client):
    r = predict(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["class_name"] == CLASS_NAMES[body["label"]] == "positive"
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))
    assert body["model"] == "attention" and body["source"] == "user"
    assert body["text_sha256"] == hashlib.sha256(REVIEW.encode()).hexdigest()
    assert body["logged"] is True and body["prediction_id"] == 1


def test_attention_is_one_weight_per_token_and_sums_to_one(client):
    body = predict(client).json()
    assert len(body["attention"]) == len(body["tokens"]) == body["n_tokens"] == len(body["unk"])
    assert sum(body["attention"]) == pytest.approx(1.0, abs=1e-4)
    assert all(w >= 0 for w in body["attention"])
    top = body["top_tokens"]
    assert top[0]["weight"] == pytest.approx(max(body["attention"]))
    assert body["tokens"][top[0]["position"]] == top[0]["token"]


def test_lstm_model_has_no_attention(client):
    body = predict(client, model="lstm").json()
    assert body["attention"] is None and body["top_tokens"] == []
    assert client.store.runs[body["served_by_run_id"]]["name"] == "lstm"


def test_predict_logs_hash_and_attention_never_text(client):
    predict(client)
    row = client.store.predictions[-1]
    assert row["text_sha256"] == hashlib.sha256(REVIEW.encode()).hexdigest()
    assert len(row["attention"]) == row["n_tokens"]
    words = {w.strip(".!").lower() for w in REVIEW.split()}
    for v in row.values():
        assert not (isinstance(v, str) and v.lower() in words)
        assert not (isinstance(v, str) and len(v) > 64)
        assert not isinstance(v, list) or all(isinstance(x, float) for x in v)


def test_long_reviews_are_truncated_and_flagged(client):
    body = predict(client, text="great " * (MAX_LEN + 40)).json()
    assert body["n_tokens"] == MAX_LEN and body["raw_tokens"] == MAX_LEN + 40
    assert body["truncated"] is True and len(body["attention"]) == MAX_LEN


def test_unknown_words_are_counted(client):
    body = predict(client, text="zzqxv wonderful").json()
    assert body["unk"] == [True, False] and body["unk_tokens"] == 1


@pytest.mark.parametrize("payload,status", [
    ({"text": ""}, 422),                       # empty
    ({"text": "... --- ..."}, 422),            # nothing tokenizable
    ({"text": "x" * 20_001}, 422),             # too long
    ({"text": "fine", "model": "gpt"}, 422),   # unknown model
    ({}, 422),
])
def test_predict_rejects_bad_input(client, payload, status):
    assert client.post("/predict", json=payload).status_code == status
    assert client.store.predictions == []


def test_gallery_and_predict_example(client):
    g = client.get("/gallery").json()
    assert {"failures", "examples", "attention_wins", "lstm_wins"} <= set(g)
    assert len(g["examples"]) >= 3 and len(g["failures"]) >= 3
    card = g["failures"][0]
    body = client.post("/predict_example", json={"test_index": card["test_index"]}).json()
    assert body["source"] == "imdb_test" and body["true_class"] == card["true"]
    assert body["class_name"] == card["attention_pred"] != card["true"]    # still wrong, same answer
    assert body["confidence"] == pytest.approx(card["attention_confidence"], abs=1e-4)
    assert body["attention"] == pytest.approx(card["attention"], abs=1e-4)
    assert body["raw_tokens"] == card["raw_length"]
    assert client.post("/predict_example", json={"test_index": 10**7}).status_code == 404


def test_recent_predictions_and_drilldown(client):
    user = predict(client).json()
    card = client.get("/gallery").json()["examples"][0]
    example = client.post("/predict_example", json={"test_index": card["test_index"]}).json()
    recent = client.get("/predictions").json()
    assert [p["id"] for p in recent] == [example["prediction_id"], user["prediction_id"]]
    assert recent[0]["seq_runs"]["name"] == "attention"

    d = client.get(f"/predictions/{example['prediction_id']}").json()
    assert d["tokens"] == card["tokens"] and len(d["attention"]) == len(d["tokens"])
    assert d["top_tokens"][0]["weight"] == pytest.approx(max(d["attention"]))
    d = client.get(f"/predictions/{user['prediction_id']}").json()
    assert d["tokens"] is None and len(d["attention"]) == user["n_tokens"]   # text never stored
    assert client.get("/predictions/999").status_code == 404


def test_runs_endpoints(client):
    runs = client.get("/runs").json()
    assert {r["name"] for r in runs} == {"lstm", "attention"}
    assert client.get(f"/runs/{runs[0]['run_id']}").status_code == 200
    assert client.get("/runs/nope").status_code == 404


def test_meta_endpoints(client):
    assert client.get("/classes").json() == {"classes": CLASS_NAMES}
    health = client.get("/healthz").json()
    assert health["status"] == "ok" and health["models_loaded"] == ["attention", "lstm"]
    assert "torch" in client.get("/version").json()
    models = {m["name"]: m for m in client.get("/models").json()}
    assert models["attention"]["returns_attention"] and not models["lstm"]["returns_attention"]
    assert models["attention"]["params"] > models["lstm"]["params"]


# ------------------------------------------------------------------ Supabase outage
def test_predict_still_works_when_supabase_is_down(outage_client):
    r = predict(outage_client)
    assert r.status_code == 200
    assert r.json()["logged"] is False and r.json()["prediction_id"] is None


@pytest.mark.parametrize("path", ["/runs", "/runs/abc", "/predictions", "/predictions/1"])
def test_db_endpoints_return_503_when_supabase_is_down(outage_client, path):
    r = outage_client.get(path)
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]


def test_healthz_reports_db_down_but_stays_up(outage_client):
    body = outage_client.get("/healthz").json()
    assert body["status"] == "ok" and body["db"] is False
