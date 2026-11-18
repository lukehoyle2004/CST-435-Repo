"""FleetClient: two clocks kept apart, and failures classified so the UI can act on them."""
from __future__ import annotations

import requests

from client import FleetClient, classify
from conftest import API, FakeAPI, make_response


def test_success_keeps_server_and_end_to_end_time_separate():
    r = FleetClient(API, transport=FakeAPI()).post("/predict", {"text": "my card was stolen"})
    assert r.ok and r.kind is None
    assert r.data["latency_ms"] == 45.0                 # what Render measured
    assert r.e2e_ms > 0 and r.e2e_ms != r.data["latency_ms"]   # what this tier waited


def test_render_proxy_page_is_a_cold_start_not_a_result():
    api = FakeAPI()
    api.mode = "cold"
    r = FleetClient(API, transport=api).post("/predict", {"text": "hi"})
    assert not r.ok and r.kind == "cold_start" and r.data == {}


def test_connection_refused_is_an_outage():
    api = FakeAPI()
    api.mode = "down"
    r = FleetClient(API, transport=api).get("/healthz")
    assert not r.ok and r.kind == "outage"


def test_timeout_is_its_own_kind_not_called_a_cold_start():
    kind, msg = classify(None, requests.Timeout())
    assert kind == "timeout" and "waking up" in msg and "busy" in msg


def test_422_is_bad_input_with_the_api_reason():
    r = FleetClient(API, transport=FakeAPI()).post("/predict", {"text": "  "})
    assert r.kind == "bad_input" and "text: String should have at least 1 character" in r.message


def test_json_503_from_the_api_is_a_service_error_not_a_cold_start():
    kind, msg = classify(make_response(503, {"detail": "experiment store unavailable"}), None)
    assert kind == "service" and "unavailable" in msg


def test_wait_until_awake_gives_up_as_outage_after_budget():
    api = FakeAPI()
    api.mode = "cold"
    r = FleetClient(API, transport=api).wait_until_awake(budget_s=0.05, every_s=0.01, sleep=lambda s: None)
    assert not r.ok and r.kind == "outage" and "did not wake up" in r.message


def test_wait_until_awake_returns_as_soon_as_the_service_answers():
    api = FakeAPI()
    api.mode = "cold"

    def sleep(_):
        api.mode = "up"                       # the instance finishes booting

    r = FleetClient(API, transport=api).wait_until_awake(budget_s=10, every_s=0.01, sleep=sleep)
    assert r.ok and api.calls == [("GET", "/healthz")] * 2
