"""TrioClient: two clocks kept apart, and failures classified so the UI can act on them."""
from __future__ import annotations

import requests

from client import TrioClient, classify
from conftest import API, FakeAPI, make_response


def test_success_keeps_server_and_end_to_end_time_separate():
    c = TrioClient(API, transport=FakeAPI())
    r = c.post("/gpt2_continue", {"text": "hi"})
    assert r.ok and r.kind is None
    assert r.server_ms == 10460.0           # what Render measured
    assert r.e2e_ms > 0 and r.e2e_ms != r.server_ms   # what this tier waited


def test_render_proxy_page_is_a_cold_start_not_a_result():
    api = FakeAPI()
    api.mode = "cold"
    r = TrioClient(API, transport=api).post("/t5_summarize", {"text": "hi"})
    assert not r.ok and r.kind == "cold_start" and r.data == {}


def test_connection_refused_is_an_outage():
    api = FakeAPI()
    api.mode = "down"
    r = TrioClient(API, transport=api).get("/healthz")
    assert not r.ok and r.kind == "outage"


def test_timeout_is_its_own_kind_not_called_a_cold_start():
    kind, msg = classify(None, requests.Timeout())
    assert kind == "timeout" and "waking up" in msg and "busy" in msg


def test_422_is_bad_input_with_the_api_reason():
    r = TrioClient(API, transport=FakeAPI()).post("/bert_fillmask", {"text": "no mask"})
    assert r.kind == "bad_input" and "put [MASK]" in r.message


def test_json_503_from_the_api_is_a_service_error_not_a_cold_start():
    kind, msg = classify(make_response(503, {"detail": "Model 't5' is unavailable"}), None)
    assert kind == "service" and "unavailable" in msg


def test_wait_until_awake_gives_up_as_outage_after_budget():
    api = FakeAPI()
    api.mode = "cold"
    slept = []
    r = TrioClient(API, transport=api).wait_until_awake(budget_s=0.05, every_s=0.01, sleep=slept.append)
    assert not r.ok and r.kind == "outage" and "did not wake up" in r.message


def test_wait_until_awake_returns_as_soon_as_the_service_answers():
    api = FakeAPI()
    api.mode = "cold"

    def sleep(_):
        api.mode = "up"                       # the instance finishes booting

    r = TrioClient(API, transport=api).wait_until_awake(budget_s=10, every_s=0.01, sleep=sleep)
    assert r.ok and [c for c in api.calls if c[1] == "/healthz"] == [("GET", "/healthz")] * 2
