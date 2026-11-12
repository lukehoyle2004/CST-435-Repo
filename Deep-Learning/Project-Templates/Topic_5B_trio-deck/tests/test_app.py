"""Run the real Streamlit script headlessly (streamlit.testing.v1.AppTest).

HTTP goes to the FakeAPI transport and Supabase reads go to a read-only fake, so
every tab is exercised end to end, plus the three failure situations the rubric
names: cold start, outage, and recovery without a reload.
"""
from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import API, ROOT

APP = str(ROOT / "ui" / "app.py")


@pytest.fixture(autouse=True)
def fast_wake(monkeypatch):
    monkeypatch.setenv("TRIODECK_WAKE_BUDGET_S", "0.05")
    monkeypatch.setenv("TRIODECK_WAKE_EVERY_S", "0.01")
    st.cache_resource.clear()


def new_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["API_URL"] = API
    at.secrets["SUPABASE_URL"] = "https://example.supabase.co"
    at.secrets["SUPABASE_ANON_KEY"] = "anon-test-key"
    return at.run()


def text_of(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "header", "subheader", "success", "info", "warning", "error", "latex"):
        parts += [str(e.value) for e in getattr(at, kind)]
    parts += [str(m.label) + " " + str(m.value) for m in at.metric]
    return "\n".join(parts)


def press(at: AppTest, key: str) -> AppTest:
    return at.button(key=key).click().run()


def test_script_compiles():
    compile(open(APP, encoding="utf-8").read(), APP, "exec")


def test_missing_secrets_stop_with_a_clear_message():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.error and "missing secret" in at.error[0].value


def test_all_seven_tabs_render_with_their_sidebars(fake_api, fake_db):
    at = new_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Concepts", "My Encoder Block", "BERT — Fill Mask",
                                         "GPT-2 — Continue Text", "T5 — Summarize", "Compare", "Latency History"]
    page = text_of(at)
    assert page.count("When to use this archetype") == 4 and page.count("Read me first") == 3
    assert fake_api.calls == []                      # nothing is called until the user asks


def test_concepts_tab_typesets_the_derivation(fake_api, fake_db):
    at = new_app()
    latex = " ".join(str(e.value) for e in at.latex)
    assert r"\operatorname{Concat}" in latex and r"\sqrt{d_k}" in latex and "R_{\\Theta" in latex
    assert "Why rotary encodings dominate modern LLMs" in text_of(at)
    md = "\n".join(str(m.value) for m in at.markdown)
    # Both tables are Markdown (so their $...$ math is typeset), not st.table (which shows raw $).
    assert len(at.table) == 0
    assert "| Step | Shape | What happens |" in md and "| Scheme | How it enters |" in md
    assert "$[B, h, T, T]$" in md and "Rotary (RoPE, 2021)" in md


def test_encoder_block_with_and_without_positional_encoding(fake_api, fake_db):
    at = new_app()
    at.text_input(key="enc_text").set_value("a b a")
    at = press(at, "enc_go")
    assert "Prediction: palindrome" in text_of(at) and "positional encoding ON" in text_of(at)
    at.toggle(key="enc_pe").set_value(False)
    at = press(at, "enc_go")
    assert "positional encoding OFF" in text_of(at)
    sent = [c for c in fake_api.calls if c[1] == "/encoder_attention"]
    assert len(sent) == 2


@pytest.mark.parametrize("key,expect", [
    ("bert_go", "Best guess"), ("gpt_go", "invents details"), ("t5_go", "bike lanes")])
def test_each_model_tab_shows_a_live_result_and_both_latencies(fake_api, fake_db, key, expect):
    at = press(new_app(), key)
    page = text_of(at)
    assert expect in page and "Live result received" in page
    assert "Server-side model time (latency_ms)" in page and "End-to-end wait (this app ↔ Render)" in page


def test_bert_bad_input_is_explained(fake_api, fake_db):
    at = new_app()
    at.text_area(key="bert_text").set_value("no mask here")
    at = press(at, "bert_go")
    assert any("put [MASK]" in i.value for i in at.info)
    assert "Best guess" not in text_of(at)


def test_compare_renders_three_archetypes_with_per_model_latency(fake_api, fake_db):
    at = press(new_app(), "cmp_go")
    page = text_of(at)
    for name in ("BERT (encoder)", "GPT-2 (decoder)", "T5 (encoder-decoder)"):
        assert name in page
    labels = [m.label for m in at.metric]
    assert labels.count("Server model time") == 3 and labels.count("End-to-end wait") == 3
    assert page.count("measured by this app, Streamlit Cloud ↔ Render") == 3
    assert "run one after another" in page
    models = {c[1] for c in fake_api.calls}
    assert {"/bert_fillmask", "/gpt2_continue", "/t5_summarize"} <= models


def test_latency_history_reads_only_the_views_with_the_anon_client(fake_api, fake_db):
    at = press(new_app(), "hist_go")
    assert not at.exception
    assert set(fake_db.tables_read) == {"trio_benchmarks_hourly", "trio_benchmarks"}
    assert ("GET", "/benchmarks") in fake_api.calls          # asks the API to refresh first
    assert "View refreshed at" in text_of(at)


def test_cold_start_is_reported_and_nothing_stale_is_shown(fake_api, fake_db):
    at = press(new_app(), "t5_go")
    assert "bike lanes" in text_of(at)                       # a live result exists
    fake_api.mode = "cold"
    at.session_state["awake_until"] = 0
    at = press(at, "t5_go")
    page = text_of(at)
    assert "bike lanes" not in page                          # the old answer is gone, not passed off as new
    assert "model service is unavailable" in page.lower() and "did not wake up" in page


def test_outage_then_recovery_without_reload(fake_api, fake_db):
    fake_api.mode = "down"
    at = press(new_app(), "gpt_go")
    assert not at.exception
    assert "Could not reach the service" in text_of(at)
    fake_api.mode = "up"                                     # service comes back
    at = press(at, "gpt_go")                                 # same session, no reload
    assert "invents details" in text_of(at) and "Live result received" in text_of(at)


def test_latency_history_when_supabase_is_down_shows_nothing_old(fake_api, fake_db):
    fake_db.fail = True
    at = press(new_app(), "hist_go")
    assert any("Could not read the benchmarks view" in e.value for e in at.error)
    assert "View refreshed at" not in text_of(at)
