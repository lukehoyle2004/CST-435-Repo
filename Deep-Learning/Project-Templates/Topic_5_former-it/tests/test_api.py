"""End-to-end tests of the seven endpoints against the real (pinned) checkpoints and FakeDB."""
from __future__ import annotations

import hashlib
import json

import pytest

MODEL_ENDPOINTS = ("/encoder_attention", "/bert_fillmask", "/gpt2_continue", "/t5_summarize")
SHARED_FIELDS = ("model", "archetype", "task", "checkpoint", "input_sha256", "input_tokens",
                 "output_tokens", "latency_ms", "load_ms", "cold_start", "total_ms",
                 "inference_id", "logged")


def _sha(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


# ------------------------------------------------------------------ the seven endpoints
def test_openapi_lists_exactly_the_seven_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/encoder_attention", "/bert_fillmask", "/gpt2_continue", "/t5_summarize",
                          "/benchmarks", "/healthz", "/version"}
    for path, ops in paths.items():
        for op in ops.values():
            assert op.get("description"), f"{path} has no human-readable description"


def test_encoder_attention(client):
    r = client.post("/encoder_attention", json={"text": "Dog bites man, bites DOG!"})
    assert r.status_code == 200
    body = r.json()
    assert body["words"] == ["dog", "bites", "man", "bites", "dog"]
    assert body["ids"] == [1, 2, 3, 2, 1]
    assert body["prediction"] == "palindrome"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert len(body["attention"]) == body["n_layers"] == 2
    for layer in body["attention"]:
        assert len(layer) == body["n_heads"]
        for head in layer:
            assert len(head) == 5 and all(len(row) == 5 for row in head)
            assert all(abs(sum(row) - 1) < 1e-3 for row in head)        # rows sum to 1
    assert body["head_summary"][0][0]["uniform"] == 0.2
    assert body["logged"] and body["model"] == "encoder_block" and not body["cold_start"]


def test_encoder_attention_rejects_bad_lengths(client):
    assert client.post("/encoder_attention", json={"text": "one"}).status_code == 422
    assert client.post("/encoder_attention", json={"text": " ".join(["w"] * 13)}).status_code == 422


def test_bert_fillmask_with_mask_word(client, paragraph):
    r = client.post("/bert_fillmask", json={"text": paragraph, "mask_word": "safer", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "[MASK]" in body["masked_text"] and "safer" not in body["masked_text"]
    assert len(body["candidates"]) == 3
    assert body["candidates"][0]["token"] == "safer"
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True) and 0 < sum(scores) <= 1.0001
    assert body["archetype"] == "encoder" and body["checkpoint"].startswith("distilbert-base-uncased@")


def test_bert_fillmask_with_explicit_mask(client):
    r = client.post("/bert_fillmask", json={"text": "I drank a cup of [MASK] this morning."})
    assert r.status_code == 200
    assert r.json()["candidates"][0]["token"] in {"coffee", "tea", "water", "milk"}


@pytest.mark.parametrize("payload, message", [
    ({"text": "no mask here"}, "put [MASK]"),
    ({"text": "[MASK] and [MASK]"}, "exactly one"),
    ({"text": "a [MASK] b", "mask_word": "b"}, "not both"),
    ({"text": "the cat sat", "mask_word": "dog"}, "does not occur"),
])
def test_bert_fillmask_validation(client, payload, message):
    r = client.post("/bert_fillmask", json=payload)
    assert r.status_code == 422 and message in r.json()["detail"]
    assert client.store.inferences == []                              # nothing logged


def test_gpt2_continue(client, paragraph):
    r = client.post("/gpt2_continue", json={"text": paragraph, "max_new_tokens": 12})
    assert r.status_code == 200
    body = r.json()
    assert body["continuation"].strip()
    assert body["output_tokens"] == 12 and body["archetype"] == "decoder"


def test_gpt2_greedy_is_deterministic_and_sampling_is_seeded(client):
    p = {"text": "Once upon a time", "max_new_tokens": 8}
    a, b = (client.post("/gpt2_continue", json=p).json()["continuation"] for _ in range(2))
    assert a == b
    s = {**p, "do_sample": True, "seed": 7}
    c, d = (client.post("/gpt2_continue", json=s).json()["continuation"] for _ in range(2))
    assert c == d


def test_t5_summarize(client, paragraph):
    r = client.post("/t5_summarize", json={"text": paragraph, "max_new_tokens": 30})
    assert r.status_code == 200
    body = r.json()
    assert 0 < len(body["summary"]) < len(paragraph)
    assert body["output_tokens"] <= 31 and body["archetype"] == "encoder-decoder"


@pytest.mark.parametrize("path,payload", [
    ("/gpt2_continue", {"text": ""}),
    ("/gpt2_continue", {"text": "x" * 2001}),
    ("/gpt2_continue", {"text": "hi", "max_new_tokens": 500}),
    ("/t5_summarize", {"text": "hi", "num_beams": 9}),
    ("/bert_fillmask", {"text": "a [MASK]", "top_k": 0}),
    ("/gpt2_continue", {"text": "hi", "source": "test"}),
])
def test_schema_validation(client, path, payload):
    assert client.post(path, json=payload).status_code == 422


# ------------------------------------------------------------------ logging and privacy
def test_every_model_endpoint_logs_hash_output_and_latency_before_responding(client, paragraph):
    calls = [("/encoder_attention", {"text": "never odd or even"}),
             ("/bert_fillmask", {"text": paragraph, "mask_word": "safer"}),
             ("/gpt2_continue", {"text": paragraph, "max_new_tokens": 5}),
             ("/t5_summarize", {"text": paragraph, "max_new_tokens": 20})]
    for path, payload in calls:
        r = client.post(path, json=payload)
        body = r.json()
        assert r.status_code == 200 and body["logged"] and r.headers["X-Trio-Logged"] == "true"
        assert set(SHARED_FIELDS) <= set(body)
        row = client.store.inferences[-1]
        assert row["id"] == body["inference_id"]
        assert row["input_sha256"] == body["input_sha256"] == _sha(payload["text"])
        assert row["latency_ms"] == body["latency_ms"] > 0
        assert row["model"] == body["model"]
        assert row["output"]


def test_same_paragraph_across_models_is_one_input_row(client, paragraph):
    client.post("/bert_fillmask", json={"text": paragraph, "mask_word": "safer"})
    client.post("/gpt2_continue", json={"text": paragraph, "max_new_tokens": 3})
    client.post("/t5_summarize", json={"text": "  " + paragraph + "\n", "max_new_tokens": 10})
    assert len(client.store.inputs) == 1
    assert next(iter(client.store.inputs.values()))["times_submitted"] == 3
    assert {i["model"] for i in client.store.inferences} == {"bert", "gpt2", "t5"}


def test_raw_text_is_never_stored(client, paragraph):
    secret = "Zanzibar-7731 is my private passphrase and nobody should keep it"
    client.post("/bert_fillmask", json={"text": secret + " [MASK]."})
    client.post("/encoder_attention", json={"text": "zanzibar 7731 zanzibar"})
    dump = json.dumps({"inputs": client.store.inputs, "inferences": client.store.inferences})
    assert "Zanzibar-7731" not in dump and "zanzibar" not in dump.lower()
    assert "private passphrase" not in dump
    bert_row = next(i for i in client.store.inferences if i["model"] == "bert")
    assert set(bert_row["output"]) == {"candidates"}              # not the filled-in paragraph


def test_cold_start_is_measured_separately(client, paragraph):
    from api.main import REGISTRY

    client.post("/gpt2_continue", json={"text": "warm up", "max_new_tokens": 1})
    assert REGISTRY.status()["resident"] == ["gpt2"]
    warm = client.post("/gpt2_continue", json={"text": paragraph, "max_new_tokens": 2}).json()
    assert warm["load_ms"] == 0 and not warm["cold_start"]
    cold = client.post("/t5_summarize", json={"text": paragraph, "max_new_tokens": 10}).json()
    assert cold["cold_start"] and cold["load_ms"] > 0
    assert cold["total_ms"] >= cold["latency_ms"] + cold["load_ms"] - 1
    assert REGISTRY.status()["resident"] == ["t5"]                 # only one pretrained model in memory


# ------------------------------------------------------------------ benchmarks / ops
def test_benchmarks_aggregate_by_model(client, paragraph):
    for _ in range(2):
        client.post("/gpt2_continue", json={"text": paragraph, "max_new_tokens": 4, "source": "benchmark"})
    client.post("/encoder_attention", json={"text": "a b a"})
    r = client.get("/benchmarks")
    assert r.status_code == 200 and r.json()["refreshed"]
    rows = {row["model"]: row for row in r.json()["rows"]}
    assert set(rows) == {"gpt2", "encoder_block"}
    assert rows["gpt2"]["n_calls"] == 2 and rows["gpt2"]["mean_output_tokens"] == 4
    assert rows["gpt2"]["distinct_inputs"] == 1


def test_benchmarks_refresh_is_throttled(client):
    client.get("/benchmarks")
    second = client.get("/benchmarks").json()
    assert not second["refreshed"] and client.store.refreshes == 1
    assert client.get("/benchmarks", params={"refresh": False}).json()["refreshed"] is False


def test_healthz_reports_loader_and_database(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_loader"]["encoder_block_loaded"] and body["model_loader"]["ok"]
    assert body["model_loader"]["available"] == ["bert", "gpt2", "t5"]
    assert body["database"]["ok"]


def test_version_reports_pinned_provenance(client):
    body = client.get("/version").json()
    for name in ("bert", "gpt2", "t5"):
        assert len(body["models"][name]["revision"]) == 40          # a full commit hash
    assert body["encoder_block"]["source"] == "src/transformer.py"
    assert len(body["encoder_block"]["weights_sha256"]) == 64


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_positional_encoding_toggle_makes_order_visible_or_invisible(client):
    """With PE the encoder separates "a b a" (palindrome) from "a a b"; without it, it cannot."""
    def run(text, pe):
        r = client.post("/encoder_attention", json={"text": text, "positional_encoding": pe})
        assert r.status_code == 200 and r.json()["positional_encoding"] is pe
        return r.json()

    with_pe = [run(t, True) for t in ("a b a", "a a b")]
    assert [b["prediction"] for b in with_pe] == ["palindrome", "not a palindrome"]

    no_pe = [run(t, False) for t in ("a b a", "a a b")]
    assert no_pe[0]["probabilities"] == pytest.approx(no_pe[1]["probabilities"], abs=1e-4)
    assert client.store.inferences[-1]["output"]["positional_encoding"] is False
