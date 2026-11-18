"""Run the real Streamlit script headlessly (streamlit.testing.v1.AppTest).

HTTP goes to the FakeAPI transport and Supabase reads go to a read-only fake, so
every tab is exercised end to end, plus the failure situations the rubric names:
cold start vs outage, never showing a stale prediction as live, and recovery
without a reload.
"""
from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import API, ROOT

APP = str(ROOT / "ui" / "app.py")


@pytest.fixture(autouse=True)
def fast_wake(monkeypatch):
    monkeypatch.setenv("FLEETDECK_WAKE_BUDGET_S", "0.05")
    monkeypatch.setenv("FLEETDECK_WAKE_EVERY_S", "0.01")
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


def frames(at: AppTest):
    return [d.value for d in at.dataframe]


def test_missing_secrets_stop_with_a_clear_message():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.error and "missing secret" in at.error[0].value


def test_five_tabs_render_and_nothing_is_called_until_asked(fake_api, fake_db):
    at = new_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Concepts", "Try It", "Leaderboard", "Cost & Latency", "Model Card"]
    assert text_of(at).count("Read me first") == 5
    assert fake_api.calls == [] and fake_db.tables_read == []


def test_concepts_typesets_the_four_objectives_and_the_matrix(fake_api, fake_db):
    at = new_app()
    latex = " ".join(str(e.value) for e in at.latex)
    for marker in (r"\mathcal{L}_{\text{MLM}}", r"\mathcal{L}_{\text{CLM}}", r"\mathcal{L}_{\text{span}}",
                   r"\mathcal{L}_{\text{InfoNCE}}", r"\log N"):
        assert marker in latex
    md = "\n".join(str(m.value) for m in at.markdown)
    assert "| Downstream task | MLM (encoder) |" in md and "Single-label classification (this product)" in md
    assert "When zero-shot prompting beats fine-tuning" in md and "0.965" in md and "0.669" in md


def test_try_it_shows_both_models_side_by_side_with_both_clocks(fake_api, fake_db):
    at = press(new_app(), "try_go")
    page = text_of(at)
    assert "Fine-tuned DistilBERT (deployed)" in page and "Zero-shot flan-t5-small" in page
    assert "Live result received" in page
    labels = [m.label for m in at.metric]
    assert labels.count("Server model time (latency_ms)") == 2
    assert labels.count("End-to-end wait (this app ↔ Render)") == 2
    assert "cold start: **yes**" in page                          # the zero-shot call had to load its model
    assert "logged server-side: yes (prediction #100)" in page    # logging happens in the API, not here
    assert ("POST", "/predict") in fake_api.calls and ("POST", "/predict_zero_shot") in fake_api.calls
    assert "Needs review" in page                                 # the zero-shot answer was only 41% sure
    assert "slower" in page


def test_try_it_empty_message_is_explained_and_not_sent(fake_api, fake_db):
    at = new_app()
    at.text_area(key=at.text_area[0].key).set_value("   ")
    at = press(at, "try_go")
    assert any("Type a message first" in i.value for i in at.info)
    assert not [c for c in fake_api.calls if c[1].startswith("/predict")]


def test_leaderboard_reads_the_view_with_cis_and_flags_ties(fake_api, fake_db):
    at = press(new_app(), "lb_go")
    assert not at.exception
    assert fake_db.tables_read == ["ftf_leaderboard"]
    df = frames(at)[0]
    assert "0.965 [0.958, 0.972]" in df["Test macro-F1 [95% CI]"].tolist()
    assert "+0.296 [+0.275, +0.315]" in df["Δ vs zero-shot [95% CI]"].tolist()
    assert df["vs best"].tolist().count("not meaningfully different") == 1
    page = text_of(at)
    assert "Not meaningfully different from the best run (ft-lr5e-5-ep3):** ft-lr2e-5-ep3" in page
    assert "The deployed checkpoint (ft-lr5e-5-ep3) is the top row." in page
    assert fake_api.calls == []                                   # read straight from Supabase, anon key


def test_leaderboard_can_be_resorted(fake_api, fake_db):
    at = press(new_app(), "lb_go")
    at.selectbox(key="lb_sort").set_value("ms / message").run()
    runs = frames(at)[0]["Run"].tolist()
    assert runs[0] == "ft-lr5e-5-ep3 ★ deployed" or runs[-1] == "zero-shot-flan-t5-small"
    assert runs[-1] == "zero-shot-flan-t5-small"                 # slowest last


def test_leaderboard_when_supabase_is_down_shows_nothing_old(fake_api, fake_db):
    at = press(new_app(), "lb_go")
    assert frames(at)
    fake_db.fail = True
    at = press(at, "lb_go")
    assert any("Could not read the leaderboard" in e.value for e in at.error)
    assert not frames(at)


def test_cost_tab_posts_the_fixed_batch_and_states_its_assumptions(fake_api, fake_db):
    at = press(new_app(), "cost_go")
    assert not at.exception
    batch = [b for (m, p), b in zip(fake_api.calls, fake_api.bodies) if p == "/predict_batch"]
    assert len(batch) == 1 and len(batch[0]["texts"]) == 16 and batch[0]["model"] == "fine_tuned"
    measured, priced = frames(at)[:2]
    assert measured["Server ms per message"].iloc[0] == 45.0
    assert measured["Logged to Supabase"].iloc[0] == "16/16"
    options = priced.index.tolist()
    assert any("Claude Haiku 5.5" in o for o in options) and any("Render Free" in o for o in options)
    page = text_of(at)
    assert "Prices as of" in page and "not measured" in "\n".join(priced["Quality (test macro-F1)"])
    assert "Frontier-model quality was not measured" in page


def test_cost_tab_can_time_the_zero_shot_baseline_too(fake_api, fake_db):
    at = new_app()
    at.checkbox(key="cost_zero").check()
    at = press(at, "cost_go")
    models = [b["model"] for (m, p), b in zip(fake_api.calls, fake_api.bodies) if p == "/predict_batch"]
    assert models == ["fine_tuned", "zero_shot"]
    assert len(frames(at)[0]) == 2


def test_latency_history_reads_only_the_aggregate_view(fake_api, fake_db):
    at = press(new_app(), "lat_go")
    assert fake_db.tables_read == ["ftf_latency"]
    assert set(frames(at)[0].index) == {"predict", "predict_zero_shot"}


def test_cold_start_is_reported_and_the_old_prediction_is_removed(fake_api, fake_db):
    at = press(new_app(), "try_go")
    assert "Live result received" in text_of(at)
    fake_api.mode = "cold"
    at.session_state["awake_until"] = 0
    at = press(at, "try_go")
    page = text_of(at)
    assert "Live result received" not in page and "Routed to" not in page   # no stale answer passed off as new
    assert "routing service is unavailable" in page.lower() and "did not wake up" in page


def test_outage_is_named_differently_from_a_cold_start(fake_api, fake_db):
    fake_api.mode = "down"
    at = press(new_app(), "try_go")
    page = text_of(at)
    assert "Could not reach the service at all" in page and "Cold start" not in page


def test_outage_then_recovery_without_reload(fake_api, fake_db):
    fake_api.mode = "down"
    at = press(new_app(), "try_go")
    assert "Live result received" not in text_of(at)
    fake_api.mode = "up"
    at = press(at, "try_go")                                   # same session, no reload
    assert "Live result received" in text_of(at) and not at.exception


def test_model_card_tab(fake_api, fake_db):
    md = "\n".join(str(m.value) for m in new_app().markdown)
    for heading in ("### Intended use", "### Limitations", "### Ethical considerations", "**Out of scope:**"):
        assert heading in md
    assert "**0.971** [0.948, 0.991]" in md and "**0.965** [0.958, 0.972]" in md
