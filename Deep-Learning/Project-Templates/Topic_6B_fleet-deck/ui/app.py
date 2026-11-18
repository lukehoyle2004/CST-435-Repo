"""FleetDeck -- the dashboard tier of FineTuneFleet (Streamlit Community Cloud).

A THIN client, on purpose:
  * every prediction is an HTTPS call to the FineTuneFleet FastAPI service on
    Render (``client.FleetClient``) -- no torch, no transformers, no weights here;
    the API logs each call to Supabase server-side;
  * the only database access is a READ of the experiment leaderboard and the
    latency aggregates with the public anon key -- no SQL is written from this
    tier, and the service-role key never reaches it.

Secrets (Streamlit secrets manager; locally ui/.streamlit/secrets.toml):
    API_URL, SUPABASE_URL, SUPABASE_ANON_KEY
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import content as C
from client import FleetClient, Result

st.set_page_config(page_title="FleetDeck", page_icon="🧭", layout="wide")

REQUIRED_SECRETS = ("API_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY")
QUEUE_LABELS = {"card_services": "Card services", "transfers": "Transfers", "top_ups": "Top-ups",
                "cash_atm": "Cash & ATM", "fees_fx": "Fees & FX", "payments_disputes": "Payments & disputes",
                "security_fraud": "Security & fraud", "account_identity": "Account & identity"}
MODEL_LABELS = {"fine_tuned": "Fine-tuned DistilBERT (deployed)", "zero_shot": "Zero-shot flan-t5-small"}
AWAKE_FOR_S = 120
WAKE_BUDGET_S = float(os.environ.get("FLEETDECK_WAKE_BUDGET_S", "120"))
WAKE_EVERY_S = float(os.environ.get("FLEETDECK_WAKE_EVERY_S", "5"))
SECONDS_PER_MONTH = 730 * 3600


# --------------------------------------------------------------------------- setup
def secret(name: str) -> str | None:
    if not st.secrets.load_if_toml_exists():
        return None
    value = st.secrets.get(name)
    return str(value) if value else None


missing = [n for n in REQUIRED_SECRETS if not secret(n)]
if missing:
    st.error(f"FleetDeck is not configured: missing secret(s) {', '.join(missing)}. Add them in the "
             "Streamlit Cloud secrets manager (or ui/.streamlit/secrets.toml locally).")
    st.stop()

client = FleetClient(secret("API_URL"))


@st.cache_resource
def supabase_anon():
    """Read-only client: the anon key can SELECT the leaderboard and latency views, nothing else."""
    from supabase import create_client

    return create_client(secret("SUPABASE_URL"), secret("SUPABASE_ANON_KEY"))


# --------------------------------------------------------------------------- service state
def ensure_awake() -> bool:
    """Make sure the Render service answers before a model call.

    Render's free instance sleeps after 15 idle minutes and takes about a minute to wake.
    This polls /healthz and says which situation the user is in. Nothing is cached across
    failures: the next click checks again, so the page recovers without a reload.
    """
    if time.time() < st.session_state.get("awake_until", 0):
        return True
    with st.status("Checking the routing service on Render…", expanded=False) as status:
        def on_try(i: int, r: Result) -> None:
            if not r.ok and r.kind in ("cold_start", "timeout"):
                status.update(label=f"The routing service is waking up (free-tier cold start, about a "
                                    f"minute). Attempt {i}…", state="running")

        r = client.wait_until_awake(budget_s=WAKE_BUDGET_S, every_s=WAKE_EVERY_S, on_try=on_try)
        if r.ok:
            st.session_state.awake_until = time.time() + AWAKE_FOR_S
            db_ok = r.data.get("database", {}).get("ok")
            status.update(label="Routing service is awake." + ("" if db_ok else
                          " Its database is unreachable, so predictions will not be logged."),
                          state="complete")
            return True
        st.session_state.awake_until = 0
        status.update(label=r.message, state="error")
    st.error(f"**The routing service is unavailable.** {r.message} Nothing below is a live result. "
             "Press the button again in a minute; the page picks up as soon as the service answers.")
    return False


def show_failure(r: Result) -> None:
    if r.kind in ("cold_start", "timeout", "outage"):
        st.session_state.awake_until = 0
    if r.kind == "cold_start":
        st.warning(f"⏳ **Cold start.** {r.message} Press the button again in about 30 seconds.")
    elif r.kind == "timeout":
        st.warning(f"⏳ **No answer in time.** {r.message} Press the button again; the next call checks "
                   "whether the service is awake first.")
    elif r.kind == "outage":
        st.error(f"**Service unreachable.** {r.message} Try again in a few minutes.")
    elif r.kind == "bad_input":
        st.info(f"**Check the input.** {r.message}")
    else:
        st.error(f"**The service reported an error.** {r.message} Try again; if it repeats, the model "
                 "may be failing to load.")


def two_columns(key: str):
    main, side = st.columns([3, 1], gap="large")
    with side:
        with st.container(border=True):
            st.markdown("#### Read me first")
            st.markdown(C.SIDEBARS[key])
    return main


def md_table(header: list[str], rows: list[tuple]) -> str:
    """A Markdown table: unlike st.table, st.markdown typesets math in its cells."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def with_ci(v, lo, hi, signed: bool = False) -> str:
    if v is None:
        return "–"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def received(key: str) -> dict | None:
    item = st.session_state.get(key)
    if item:
        st.caption(f"Live result received at {item['at']} from the Render service.")
    return item


def prediction_panel(kind: str, r: Result) -> None:
    st.subheader(MODEL_LABELS[kind])
    if not r.ok:
        show_failure(r)
        return
    d = r.data
    st.metric("Routed to", QUEUE_LABELS.get(d["queue"], d["queue"]))
    st.markdown(f"**{d['confidence']:.1%} sure**")
    if d["needs_review"]:
        st.warning("**Needs review** — under 60% sure, or the fraud queue is a live possibility.")
    else:
        st.success("Confident enough to route automatically.")
    scores = pd.DataFrame([{"queue": QUEUE_LABELS[q], "probability": p} for q, p in d["scores"].items()])
    st.altair_chart(alt.Chart(scores).mark_bar(color="#2a78d6" if kind == "fine_tuned" else "#eb6834").encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="probability"),
        y=alt.Y("queue:N", sort="-x", title=None),
        tooltip=["queue", alt.Tooltip("probability:Q", format=".3f")]).properties(height=300),
        use_container_width=True)
    network = max(r.e2e_ms - d["latency_ms"] - d["load_ms"], 0.0)
    # Stacked, not in columns: these panels already sit in the Try It tab's two columns,
    # and Streamlit allows only one level of nested columns.
    st.metric("Server model time (latency_ms)", f"{d['latency_ms']:,.0f} ms",
              help="Measured by the API on Render: tokenize → model → probabilities. Logged to Supabase.")
    st.metric("End-to-end wait (this app ↔ Render)", f"{r.e2e_ms:,.0f} ms",
              help="Measured by FleetDeck: request sent → response received.")
    st.caption(f"Model load during this request (load_ms): {d['load_ms']:,.0f} ms · cold start: "
               f"{'**yes**' if d['cold_start'] else 'no'} · network + proxy + logging: {network:,.0f} ms · "
               f"logged server-side: {'yes (prediction #' + str(d['prediction_id']) + ')' if d['logged'] else '**NO** (database unreachable)'} · "
               f"run `{d['run_name']}` · checkpoint `{d['checkpoint']}`")


def read_view(name: str, order: str) -> list[dict]:
    return supabase_anon().table(name).select("*").order(order).execute().data


# --------------------------------------------------------------------------- page
st.title("FleetDeck")
st.caption("A bank's support desk routes every message to one of eight queues. This dashboard compares the "
           "fine-tuned router we deployed with a zero-shot model that was never trained on the task. Every "
           "answer comes from the FineTuneFleet service on Render; this page runs no model and writes "
           "nothing to the database.")

tabs = st.tabs(["Concepts", "Try It", "Leaderboard", "Cost & Latency", "Model Card"])

# ---- Concepts ---------------------------------------------------------------
with tabs[0]:
    main = two_columns("concepts")
    with main:
        st.header("Pre-training objectives, compared")
        st.markdown("A model's pre-training objective decides what it is good at before anyone fine-tunes "
                    "it. Notation: a text is $x=(x_1,\\dots,x_T)$; the model with parameters $\\theta$ "
                    "gives $p_\\theta(\\cdot\\mid\\text{context})$.")
        st.markdown("**1. Masked language modeling** (BERT, DistilBERT, the model we fine-tune):")
        st.latex(C.MLM)
        st.markdown(C.MLM_WHY)
        st.markdown("**2. Causal language modeling** (GPT-2, Llama, the pre-training of chat models):")
        st.latex(C.CLM)
        st.markdown(C.CLM_WHY)
        st.markdown("**3. Span corruption** (T5, flan-T5, our zero-shot baseline):")
        st.latex(C.SPAN)
        st.markdown(C.SPAN_WHY)
        st.markdown("**4. Contrastive learning** (SimCSE, Sentence-BERT, CLIP):")
        st.latex(C.INFONCE)
        st.markdown(C.INFONCE_WHY)
        st.subheader("Which objective suits which task")
        st.markdown("★★★ natural fit · ★★ works with adaptation · ★ fighting the objective")
        st.markdown(md_table(C.MATRIX_HEADER, C.MATRIX))
        st.subheader("Fine-tuned vs zero-shot, honestly")
        st.markdown("On this task the fine-tuned encoder wins clearly: macro-F1 **0.965** against **0.669** "
                    "for the zero-shot baseline, and the gap's 95% interval is +0.279 to +0.315. The baseline "
                    "is not a straw man: its answers are scored by likelihood (so it always names a real "
                    "queue), calibrated for its bias toward some label words, and its prompt was picked on "
                    "validation data. Without calibration the same model sent all 995 validation messages "
                    "to one queue.")
        st.markdown("**When zero-shot prompting beats fine-tuning:**")
        for point in C.ZERO_SHOT_WINS:
            st.markdown(f"- {point}")

# ---- Try It -----------------------------------------------------------------
with tabs[1]:
    main = two_columns("try")
    with main:
        st.header("Try It: fine-tuned vs zero-shot, side by side")
        sample = st.selectbox("Start from an example (or type your own below)", list(C.SAMPLES), key="try_sample")
        text = st.text_area("Customer message", C.SAMPLES[sample], key=f"try_text_{sample}", height=90,
                            max_chars=2000)
        st.caption("Your message is sent to the service, which stores only its SHA-256 hash, its length and "
                   "the model's answer, never the text.")
        if st.button("Route with both models", key="try_go", type="primary"):
            st.session_state.pop("try", None)       # never leave a previous answer on screen as if current
            if not text.strip():
                st.info("**Check the input.** Type a message first.")
            elif ensure_awake():
                results = {}
                with st.status("Routing the message with both models, one after the other…") as status:
                    for kind, path in (("fine_tuned", "/predict"), ("zero_shot", "/predict_zero_shot")):
                        status.update(label=f"Asking the {MODEL_LABELS[kind]}…")
                        results[kind] = client.post(path, {"text": text})
                    status.update(label="Both answered." if all(r.ok for r in results.values())
                                  else "Finished, with errors (see below).", state="complete")
                st.session_state["try"] = {"results": results, "text": text,
                                           "at": datetime.now().strftime("%H:%M:%S")}
        item = received("try")
        if item:
            cols = st.columns(2, gap="large")
            for col, kind in zip(cols, ("fine_tuned", "zero_shot")):
                with col:
                    prediction_panel(kind, item["results"][kind])
            ok = [r for r in item["results"].values() if r.ok]
            if len(ok) == 2:
                a, b = (item["results"][k].data for k in ("fine_tuned", "zero_shot"))
                verdict = "agree" if a["queue"] == b["queue"] else "**disagree**"
                st.markdown(f"The two models {verdict}. The fine-tuned model's compute took "
                            f"{a['latency_ms']:,.0f} ms against {b['latency_ms']:,.0f} ms for the zero-shot "
                            f"model ({b['latency_ms'] / max(a['latency_ms'], 0.01):,.1f}× slower).")
            st.caption("The free server keeps one model in memory, so each model usually has to be loaded "
                       "when it is called after the other one; that is the load_ms shown above, and it is "
                       "not part of the model time.")

# ---- Leaderboard ------------------------------------------------------------
with tabs[2]:
    main = two_columns("leaderboard")
    with main:
        st.header("Leaderboard")
        st.markdown("Every experiment from the sweep, read with the read-only key from the `ftf_leaderboard` "
                    "view in Supabase. Metrics are on the held-out test set with 95% bootstrap intervals; "
                    "the deployed run was chosen on the validation set.")
        if st.button("Load the leaderboard", key="lb_go"):
            st.session_state.pop("lb", None)
            try:
                st.session_state.lb = {"rows": read_view("ftf_leaderboard", "rank"),
                                       "at": datetime.now().strftime("%H:%M:%S")}
            except Exception as exc:  # show nothing rather than old numbers
                st.error(f"**Could not read the leaderboard from Supabase** ({type(exc).__name__}). Nothing "
                         "is shown rather than old numbers. Try again shortly.")
        item = st.session_state.get("lb")
        if item:
            rows = item["rows"]
            if not rows:
                st.info("The experiment store is empty.")
            else:
                best = rows[0]
                table = pd.DataFrame([{
                    "Rank": r["rank"],
                    "Run": r["run_name"] + (" ★ deployed" if r["is_deployed"] else ""),
                    "Kind": "fine-tuned" if r["kind"] == "fine_tuned" else "zero-shot",
                    "Learning rate": (f"{r['hyperparams']['learning_rate']:g}"
                                      if r["hyperparams"].get("learning_rate") else "–"),
                    "Epochs": r["hyperparams"].get("epochs"),
                    "Train messages": r["n_train"],
                    "Encoder": "frozen" if r["hyperparams"].get("freeze_encoder") else
                               ("–" if r["kind"] == "zero_shot" else "trained"),
                    "Val macro-F1": round(r["val_macro_f1"], 3),
                    "Test macro-F1 [95% CI]": with_ci(r["test_macro_f1"], r["test_macro_f1_lo"], r["test_macro_f1_hi"]),
                    "Δ vs zero-shot [95% CI]": with_ci(r["delta_macro_f1"], r["delta_macro_f1_lo"],
                                                       r["delta_macro_f1_hi"], signed=True),
                    "Fraud recall [95% CI]": with_ci(r["fraud_recall"], r["fraud_recall_lo"], r["fraud_recall_hi"]),
                    "Test accuracy [95% CI]": with_ci(r["test_accuracy"], r["test_accuracy_lo"], r["test_accuracy_hi"]),
                    "vs best": ("best" if r["run_name"] == best["run_name"] else
                                "meaningfully worse" if r["distinguishable_from_best"] else
                                "not meaningfully different"),
                    "Train time (min)": round(r["train_runtime_s"] / 60, 1),
                    "ms / message (laptop)": round(r["eval_ms_per_message"], 1),
                    "_f1": r["test_macro_f1"], "_fraud": r["fraud_recall"], "_val": r["val_macro_f1"],
                } for r in rows])
                sort = st.selectbox("Sort by", ["Test macro-F1", "Fraud recall", "Val macro-F1",
                                                "Train time", "ms / message"], key="lb_sort")
                key = {"Test macro-F1": ("_f1", False), "Fraud recall": ("_fraud", False),
                       "Val macro-F1": ("_val", False), "Train time": ("Train time (min)", True),
                       "ms / message": ("ms / message (laptop)", True)}[sort]
                table = table.sort_values(key[0], ascending=key[1])
                st.dataframe(table.drop(columns=["_f1", "_fraud", "_val"]).set_index("Rank"),
                             use_container_width=True)
                st.caption(f"Read at {item['at']}. Click a column header to sort by it too.")
                same = [r["run_name"] for r in rows[1:] if not r["distinguishable_from_best"]]
                if same:
                    st.info(f"**Not meaningfully different from the best run ({best['run_name']}):** "
                            f"{', '.join(same)}. Their 95% intervals overlap the best run's, so the data "
                            "cannot tell them apart; the extra accuracy of the top row may be luck.")
                deployed = next((r for r in rows if r["is_deployed"]), None)
                if deployed and deployed["run_name"] == best["run_name"]:
                    st.success(f"The deployed checkpoint ({deployed['run_name']}) is the top row.")
                elif deployed:
                    st.warning(f"The deployed checkpoint ({deployed['run_name']}) is not the top row on test; "
                               f"it was chosen on validation.")
                chart_df = pd.DataFrame([{"run": r["run_name"], "f1": r["test_macro_f1"], "lo": r["test_macro_f1_lo"],
                                          "hi": r["test_macro_f1_hi"], "deployed": r["is_deployed"]} for r in rows])
                base = alt.Chart(chart_df).encode(y=alt.Y("run:N", sort="-x", title=None))
                st.altair_chart(
                    (base.mark_rule(strokeWidth=3).encode(x=alt.X("lo:Q", title="test macro-F1 (95% interval)",
                                                                   scale=alt.Scale(zero=False)), x2="hi:Q")
                     + base.mark_point(filled=True, size=80).encode(
                         x="f1:Q", color=alt.condition("datum.deployed", alt.value("#2a78d6"), alt.value("#7a7a7a")),
                         tooltip=["run", alt.Tooltip("f1:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f"),
                                  alt.Tooltip("hi:Q", format=".3f")])).properties(height=240),
                    use_container_width=True)

# ---- Cost & Latency -----------------------------------------------------------
with tabs[3]:
    main = two_columns("cost")
    with main:
        st.header("Cost & Latency: cheaper, faster, good enough?")
        st.markdown(f"Sends a fixed batch of **{len(C.FIXED_BATCH)} messages** to `POST /predict_batch` and "
                    "times the deployed model on the live server, then prices the same traffic on a frontier "
                    "LLM API.")
        also_zero = st.checkbox("Also time the zero-shot baseline (much slower on the free server)",
                                key="cost_zero")
        if st.button("Run the fixed batch", key="cost_go", type="primary"):
            st.session_state.pop("cost", None)
            if ensure_awake():
                runs = {}
                kinds = ["fine_tuned"] + (["zero_shot"] if also_zero else [])
                with st.status("Sending the batch…") as status:
                    for kind in kinds:
                        status.update(label=f"Batch through the {MODEL_LABELS[kind]}…")
                        runs[kind] = client.post("/predict_batch", {"texts": C.FIXED_BATCH, "model": kind})
                    status.update(label="Done." if all(r.ok for r in runs.values())
                                  else "Finished, with errors (see below).", state="complete")
                st.session_state.cost = {"runs": runs, "at": datetime.now().strftime("%H:%M:%S")}
        item = received("cost")
        if item:
            measured = {}
            for kind, r in item["runs"].items():
                if not r.ok:
                    st.markdown(f"**{MODEL_LABELS[kind]}:**")
                    show_failure(r)
                    continue
                preds = r.data["predictions"]
                load = sum(p["load_ms"] for p in preds)
                measured[kind] = {"ms_per_msg": r.data["total_latency_ms"] / len(preds), "load_ms": load,
                                  "wall_ms": r.data["wall_ms"], "e2e_ms": r.e2e_ms, "logged": r.data["logged"],
                                  "tokens": sum(p["n_tokens"] for p in preds) / len(preds), "n": len(preds)}
            if measured:
                st.subheader("Measured on the live server")
                st.dataframe(pd.DataFrame([{
                    "Model": MODEL_LABELS[k], "Server ms per message": round(m["ms_per_msg"], 1),
                    "Model load during the batch (ms)": round(m["load_ms"]),
                    "Server wall time for the batch (ms)": round(m["wall_ms"]),
                    "End-to-end wait for the batch (ms)": round(m["e2e_ms"]),
                    "Logged to Supabase": f"{m['logged']}/{m['n']}"} for k, m in measured.items()]).set_index("Model"),
                    use_container_width=True)
            if "fine_tuned" in measured:
                m = measured["fine_tuned"]
                volume = st.select_slider("Messages per month", options=[10_000, 100_000, 1_000_000, 10_000_000],
                                          value=1_000_000, format_func=lambda v: f"{v:,}", key="cost_volume")
                fr, sh = C.COST["frontier"], C.COST["self_hosted"]
                in_tok = fr["prompt_tokens"] + m["tokens"]
                rows = []
                base_cpu = sh["plans"][sh["measured_on"]]["cpu"]
                for plan, p in sh["plans"].items():
                    ms = m["ms_per_msg"] * base_cpu / p["cpu"]
                    capacity = SECONDS_PER_MONTH * 1000 / ms
                    instances = max(1, math.ceil(volume / capacity))
                    monthly = instances * p["usd_per_month"]
                    rows.append({"Option": f"Self-hosted DistilBERT, Render {plan}",
                                 "Quality (test macro-F1)": "0.965 [0.958, 0.972] measured",
                                 "Model time per message": f"{ms:,.0f} ms" + (" (measured)" if plan == sh["measured_on"]
                                                                              else " (scaled by CPU)"),
                                 "Monthly cost": f"${monthly:,.2f}" + (f" ({instances} instances)" if instances > 1 else ""),
                                 "Cost per 1,000 messages": f"${monthly / volume * 1000:,.4f}",
                                 "Notes": p["note"]})
                for name, p in fr["models"].items():
                    per_msg = (in_tok * p["input"] + (fr["output_tokens"] + p["reasoning_tokens"]) * p["output"]) / 1e6
                    rows.append({"Option": f"{name} API (prompted)", "Quality (test macro-F1)": "not measured",
                                 "Model time per message": "not measured",
                                 "Monthly cost": f"${per_msg * volume:,.2f}",
                                 "Cost per 1,000 messages": f"${per_msg * 1000:,.4f}", "Notes": p["note"]})
                st.subheader(f"At {volume:,} messages a month")
                st.dataframe(pd.DataFrame(rows).set_index("Option"), use_container_width=True)
                st.caption("The zero-shot flan-t5-small baseline scores 0.669 [0.649, 0.686] on the same test set. "
                           "Frontier-model quality was not measured: sending the test set to a paid API was out "
                           "of scope, so the table compares cost and speed only for those rows.")
                with st.expander("Assumptions behind these numbers", expanded=True):
                    st.markdown(f"**Prices as of {C.COST['as_of']}.** {fr['source']}. {sh['source']}.")
                    st.markdown(f"**Frontier calls.** {fr['explain']} Measured mean message length in this batch: "
                                f"{m['tokens']:.1f} tokens, so each call is about {in_tok:.0f} input tokens and "
                                f"{fr['output_tokens']} output tokens, plus each model's reasoning tokens:")
                    st.markdown("\n".join(f"- {n}: \\${p['input']:.2f} in / \\${p['output']:.2f} out per million tokens; "
                                          f"{p['reasoning_tokens']} reasoning tokens ({p['note']})"
                                          for n, p in fr["models"].items()))
                    st.markdown(f"**Self-hosting.** {sh['explain']}")
                    st.markdown("\n".join(f"- {n}: {p['cpu']} CPU, \\${p['usd_per_month']}/month ({p['note']})"
                                          for n, p in sh["plans"].items()))
        st.subheader("All logged traffic so far")
        if st.button("Load latency history", key="lat_go"):
            st.session_state.pop("lat", None)
            try:
                st.session_state.lat = {"rows": read_view("ftf_latency", "endpoint"),
                                        "at": datetime.now().strftime("%H:%M:%S")}
            except Exception as exc:
                st.error(f"**Could not read the latency view from Supabase** ({type(exc).__name__}). Nothing is "
                         "shown rather than old numbers.")
        lat = st.session_state.get("lat")
        if lat:
            if not lat["rows"]:
                st.info("No predictions have been logged yet. Use Try It, then load again.")
            else:
                st.dataframe(pd.DataFrame(lat["rows"]).set_index("endpoint"), use_container_width=True)
                st.caption(f"Server-side times from the `ftf_latency` view (test traffic excluded), read at "
                           f"{lat['at']}. n_cold counts calls that had to load their model first.")

# ---- Model Card ---------------------------------------------------------------
with tabs[4]:
    main = two_columns("card")
    with main:
        st.header("Model card: FineTuneFleet router")
        st.markdown(C.MODEL_CARD)
