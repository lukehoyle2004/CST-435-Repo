"""Fixtures: a fake TrioFormer API (as an HTTP transport) and a fake read-only Supabase.

The fake responses copy the shapes the deployed service returns, so the
interface is tested against the same contract it uses in production. Modes let a
test switch the "service" between up, cold-starting (Render's own 502 page),
and unreachable (connection refused).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))

API = "http://api.test"


def _meta(model, archetype, task, latency, load=0.0, n_in=10, n_out=1):
    return {"model": model, "archetype": archetype, "task": task, "checkpoint": f"{model}@abc1234567",
            "input_sha256": "0" * 64, "input_tokens": n_in, "output_tokens": n_out,
            "latency_ms": latency, "load_ms": load, "cold_start": load > 0,
            "total_ms": latency + load + 3, "inference_id": 1, "logged": True}


def _encoder(body):
    words = body["text"].lower().split()
    ids, seen = [], {}
    for w in words:
        ids.append(seen.setdefault(w, len(seen) + 1))
    n = len(words)
    pal = ids == ids[::-1] if body.get("positional_encoding", True) else False
    row = [round(1 / n, 4)] * n
    return {**_meta("encoder_block", "encoder", "attention", 1.2), "words": words, "ids": ids,
            "prediction": "palindrome" if pal else "not a palindrome",
            "probabilities": {"not a palindrome": 0.1 if pal else 0.9, "palindrome": 0.9 if pal else 0.1},
            "n_layers": 2, "n_heads": 4, "attention": [[[row] * n] * 4] * 2,
            "head_summary": [[{"head": h, "mirror": 0.2, "self": 0.2, "same_word": 0.1, "uniform": 1 / n}
                              for h in range(4)]] * 2,
            "model_accuracy": 0.957, "positional_encoding": body.get("positional_encoding", True)}


def _bert(body):
    text = body["text"]
    if "[MASK]" not in text and not body.get("mask_word"):
        return 422, {"detail": "put [MASK] in the text, or name a mask_word to hide"}
    masked = text if "[MASK]" in text else text.replace(body["mask_word"], "[MASK]", 1)
    return 200, {**_meta("bert", "encoder", "fill-mask", 1370.0, load=2900.0), "masked_text": masked,
                 "candidates": [{"token": "coffee", "score": 0.54}, {"token": "tea", "score": 0.14},
                                {"token": "water", "score": 0.08}, {"token": "wine", "score": 0.04}],
                 "filled_text": masked.replace("[MASK]", "coffee")}


ROUTES = {
    ("GET", "/healthz"): lambda b: (200, {"status": "ok", "model_loader": {"ok": True},
                                          "database": {"ok": True}}),
    ("POST", "/encoder_attention"): lambda b: (200, _encoder(b)),
    ("POST", "/bert_fillmask"): _bert,
    ("POST", "/gpt2_continue"): lambda b: (200, {**_meta("gpt2", "decoder", "continue", 10460.0, n_out=40),
                                                 "continuation": " The plan was approved in November."}),
    ("POST", "/t5_summarize"): lambda b: (200, {**_meta("t5", "encoder-decoder", "summarize", 12570.0, n_out=42),
                                                "summary": "the council voted to expand bike lanes."}),
    ("GET", "/benchmarks"): lambda b: (200, {"refreshed": True, "rows": []}),
}


def make_response(status: int, payload, html: bool = False) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    if html:
        r._content = b"<html><body>Service Unavailable</body></html>"
        r.headers["content-type"] = "text/html"
    else:
        r._content = json.dumps(payload).encode()
        r.headers["content-type"] = "application/json"
    return r


class FakeAPI:
    """Callable with the signature of ``requests.request``; records every call."""

    def __init__(self):
        self.mode = "up"            # up | cold | down
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, url, json=None, timeout=None, **_):
        path = url.replace(API, "")
        self.calls.append((method, path))
        if self.mode == "down":
            raise requests.ConnectionError("connection refused")
        if self.mode == "cold":
            return make_response(503, None, html=True)
        status, body = ROUTES[(method, path)](json or {})
        return make_response(status, body)


HOURLY = [
    {"model": m, "hour": f"2026-11-10T{h:02d}:00:00+00:00", "n_calls": 5, "n_cold": 1,
     "p50_latency_ms": base * (1 + 0.1 * h), "p95_latency_ms": base * 1.3, "p50_total_ms": base * 1.4,
     "mean_load_ms": 2000.0, "refreshed_at": "2026-11-10T23:34:16+00:00"}
    for m, base in (("bert", 1370.0), ("gpt2", 10460.0), ("t5", 12570.0), ("encoder_block", 1.2))
    for h in (22, 23)
]
OVERALL = [{"model": m, "n_calls": 10, "n_warm": 8, "n_cold": 2, "p50_latency_ms": 1.0, "p95_latency_ms": 2.0,
            "mean_load_ms": 1.0, "ms_per_output_token": 1.0} for m in ("bert", "gpt2", "t5", "encoder_block")]


class FakeQuery:
    def __init__(self, rows):
        self.rows = list(rows)

    def select(self, *_a, **_k):
        return self

    def order(self, col, desc=False):
        self.rows.sort(key=lambda r: r[col], reverse=desc)
        return self

    def execute(self):
        return type("Resp", (), {"data": self.rows})()


class FakeSupabase:
    """Read-only, like the anon key: only the two aggregate views exist for it."""

    def __init__(self):
        self.tables_read: list[str] = []
        self.fail = False

    def table(self, name):
        self.tables_read.append(name)
        if self.fail:
            raise ConnectionError("supabase down")
        views = {"trio_benchmarks_hourly": HOURLY, "trio_benchmarks": OVERALL}
        if name not in views:
            raise PermissionError(f"anon cannot read {name}")
        return FakeQuery(views[name])


@pytest.fixture
def fake_api(monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(requests, "request", api)
    return api


@pytest.fixture
def fake_db(monkeypatch):
    import supabase

    db = FakeSupabase()
    monkeypatch.setattr(supabase, "create_client", lambda url, key: db)
    return db
