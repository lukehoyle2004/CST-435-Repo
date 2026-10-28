"""Run the real Streamlit script (ui/app.py) headlessly with streamlit.testing.

The UI's HTTP calls are routed into the FastAPI app (with the in-memory FakeDB
from conftest), and its Supabase anon client is a fake that reads the same
store. That exercises every tab end to end, plus the cold-start / outage path.
"""
from __future__ import annotations

import sys

import pytest
import requests
import streamlit as st
import supabase
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT

API = "http://api.test"
sys.path.insert(0, str(ROOT / "ui"))


class FakeQuery:
    """Just enough of the supabase-py query builder for ui/app.py."""

    def __init__(self, rows):
        self.rows = list(rows)

    def select(self, *_):
        return self

    def order(self, col, desc=False):
        self.rows.sort(key=lambda r: r[col], reverse=desc)
        return self

    def limit(self, n):
        self.rows = self.rows[:n]
        return self

    def eq(self, col, value):
        self.rows = [r for r in self.rows if r.get(col) == value]
        return self

    def execute(self):
        return type("Resp", (), {"data": self.rows})()


class FakeSupabase:
    def __init__(self, store):
        self.store = store

    def table(self, name):
        runs = self.store.runs
        if name == "seq_runs":
            return FakeQuery({**r, "created_at": r["name"]} for r in runs.values())
        if name == "seq_predictions":
            return FakeQuery({**p, "created_at": f"2026-10-27T12:00:{p['id']:02d}+00:00",
                              "seq_runs": {k: runs[p["served_by_run_id"]][k]
                                           for k in ("name", "display_name", "test_acc")}}
                             for p in self.store.predictions)
        raise KeyError(name)


def to_requests(r) -> requests.Response:
    resp = requests.Response()
    resp.status_code, resp._content, resp.url = r.status_code, r.content, str(r.request.url)
    resp.headers.update(r.headers)
    return resp


@pytest.fixture
def app(client, monkeypatch):
    """AppTest wired to the FastAPI test client and a fake Supabase over the same store."""
    monkeypatch.setattr(requests, "get", lambda url, **kw: to_requests(
        client.get(url.removeprefix(API), params=kw.get("params"))))
    monkeypatch.setattr(requests, "post", lambda url, **kw: to_requests(
        client.post(url.removeprefix(API), json=kw.get("json"))))
    monkeypatch.setattr(supabase, "create_client", lambda *_: FakeSupabase(client.store))
    yield new_app(), client.store


def new_app() -> AppTest:
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60)
    at.secrets["API_URL"] = API
    at.secrets["SUPABASE_URL"] = "https://fake.supabase.co"
    at.secrets["SUPABASE_ANON_KEY"] = "anon.fake.key"
    return at


def all_text(at: AppTest) -> str:
    return "\n".join(e.value for kind in ("markdown", "caption", "info", "warning", "error")
                     for e in getattr(at, kind))


def test_every_tab_renders(app):
    at, _ = app
    at.run()
    assert not at.exception
    text, heads = all_text(at), " | ".join(h.value for h in at.subheader)
    assert "Worked example" in heads and "13.54" in text                 # Concepts
    assert "Matched runs" in heads and "Accuracy by review length" in heads   # Compare, from seq_runs
    assert "No predictions yet" in text                                  # Recent tab, empty
    assert "Failure Gallery: confidently wrong" in heads
    assert len(at.expander) >= 12 + 8                                    # failures + one-model wins


def test_try_it_logs_and_shows_attention_then_recent_drilldown(app):
    at, store = app
    at.run()
    at.radio[0].set_value("Both, side by side")
    at.button[0].click().run()
    assert not at.exception
    assert [p["model"] for p in store.predictions] == ["attention", "lstm"]
    text = all_text(at)
    assert "Most attended:" in text and "has no attention" in text
    assert "logged to Supabase" in text

    at.run()   # the Recent Predictions tab now lists both rows; drill into the newest
    assert not at.exception
    assert at.dataframe[-1].value.shape[0] == 2
    assert at.selectbox[0].value == 2       # lstm row is newest
    assert "no attention vector" in all_text(at)
    at.selectbox[0].set_value(1).run()
    text = all_text(at)
    assert "text was never stored" in text and "weights summing to 1.0" in text


def test_examples_run_live_and_drilldown_shows_words(app):
    at, store = app
    at.run()
    run_live = next(b for b in at.button if b.label == "Run live")
    run_live.click().run()
    assert not at.exception
    assert {p["source"] for p in store.predictions} == {"imdb_test"}
    assert "true label" in all_text(at)
    at.run()
    at.selectbox[0].set_value(1).run()      # the attention-model prediction
    assert "Public IMDB test review" in all_text(at)


def test_cold_start_and_outage_degrade_gracefully(monkeypatch):
    def down(*_a, **_k):
        raise requests.ConnectionError("render is asleep")

    class DownSupabase:
        def table(self, _):
            raise ConnectionError("supabase paused")

    monkeypatch.setattr(requests, "get", down)
    monkeypatch.setattr(requests, "post", down)
    monkeypatch.setattr(supabase, "create_client", lambda *_: DownSupabase())
    monkeypatch.setattr("time.sleep", lambda _s: None)
    at = new_app()
    at.run()
    assert not at.exception
    text = all_text(at)
    assert "Supabase is unreachable" in text
    assert "Could not read runs from Supabase" in text
    assert "model service is unavailable" in text
    assert "still waking up" in text                                     # Examples tab
