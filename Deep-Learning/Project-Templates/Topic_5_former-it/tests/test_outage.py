"""External-dependency failures: Supabase down, and a checkpoint that will not load.

The product's contract under failure:
* model endpoints still answer, flagged ``logged: false`` / ``X-Trio-Logged: false``;
* ``/benchmarks`` returns a clear 503 (there is nothing honest to show);
* ``/healthz`` stays 200 but reports ``degraded`` and says which side failed;
* a model that cannot load gives a 503 for that model only, and the loader recovers.
"""
from __future__ import annotations

import pytest

from src import pipelines


def test_model_endpoints_still_answer_when_supabase_is_down(outage, paragraph):
    for path, payload in [("/encoder_attention", {"text": "step on no pets on step"}),
                          ("/bert_fillmask", {"text": paragraph, "mask_word": "safer"}),
                          ("/gpt2_continue", {"text": paragraph, "max_new_tokens": 3}),
                          ("/t5_summarize", {"text": paragraph, "max_new_tokens": 10})]:
        r = outage.post(path, json=payload)
        assert r.status_code == 200, path
        body = r.json()
        assert body["logged"] is False and body["inference_id"] is None
        assert r.headers["X-Trio-Logged"] == "false"
        assert body["latency_ms"] > 0


def test_benchmarks_return_503_when_supabase_is_down(outage):
    r = outage.get("/benchmarks")
    assert r.status_code == 503
    assert "temporarily unavailable" in r.json()["detail"]
    assert "supabase unreachable" not in r.text                     # no internals leaked


def test_healthz_is_degraded_but_up_when_supabase_is_down(outage):
    r = outage.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["database"] == {"ok": False, "configured": True, "latency_ms": None, "error": "ConnectionError"}
    assert body["model_loader"]["ok"]


def test_unconfigured_database_is_reported_not_crashed(monkeypatch):
    from api import db

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    assert db.ping()["configured"] is False
    with pytest.raises(RuntimeError, match="not configured"):
        db.log_inference({})


def test_failed_checkpoint_load_is_a_503_and_the_loader_recovers(client, monkeypatch, paragraph):
    from api import main

    def broken(spec):
        raise OSError(f"cannot reach the Hub for {spec.checkpoint}")

    fresh = pipelines.ModelRegistry(1, loader=broken)
    monkeypatch.setattr(main, "REGISTRY", fresh)
    r = client.post("/t5_summarize", json={"text": paragraph})
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]
    health = client.get("/healthz").json()
    assert health["status"] == "degraded" and "t5" in health["model_loader"]["last_error"]
    assert client.store.inferences == []                             # nothing logged for a failed call

    fresh._loader = pipelines.load_bundle                            # the Hub is back
    assert client.post("/t5_summarize", json={"text": paragraph, "max_new_tokens": 10}).status_code == 200
    assert client.get("/healthz").json()["model_loader"]["last_error"] is None


# ------------------------------------------------------------------ the registry alone
class _Fake:
    def __init__(self):
        self.loaded = []

    def __call__(self, spec):
        self.loaded.append(spec.name)
        return pipelines.Bundle(spec, model=object(), tokenizer=object(), commit="c" * 40, params=1, loaded_at=0)


def test_registry_keeps_at_most_max_resident_and_evicts_lru():
    loader = _Fake()
    reg = pipelines.ModelRegistry(2, loader=loader)
    for name in ("bert", "gpt2", "bert", "t5"):
        with reg.use(name) as (_, load_ms):
            pass
    assert reg.status()["resident"] == ["bert", "t5"]                # gpt2 was least recently used
    assert loader.loaded == ["bert", "gpt2", "t5"] and reg.evictions == 1


def test_registry_reports_zero_load_ms_when_warm():
    reg = pipelines.ModelRegistry(1, loader=_Fake())
    with reg.use("gpt2") as (_, first):
        pass
    with reg.use("gpt2") as (_, second):
        pass
    assert first > 0 and second == 0


def test_prepare_masked_text_whole_word_only():
    assert pipelines.prepare_masked_text("cart art art", "art") == "cart [MASK] art"
    with pytest.raises(ValueError):
        pipelines.prepare_masked_text("cartography", "art")
