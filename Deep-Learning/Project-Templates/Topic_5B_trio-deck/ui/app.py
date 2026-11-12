"""TrioDeck -- the customer-facing tier of TrioFormer (Streamlit Community Cloud).

A THIN client, on purpose:
  * every inference is an HTTPS call to the TrioFormer FastAPI service on Render
    (``client.TrioClient``) -- no torch, no transformers, no model code here;
  * the only database access is a READ of two aggregate views
    (``trio_benchmarks``, ``trio_benchmarks_hourly``) with the public anon key --
    no SQL is written from this tier, and the service-role key never reaches it.

Secrets (Streamlit secrets manager; locally .streamlit/secrets.toml):
    API_URL, SUPABASE_URL, SUPABASE_ANON_KEY
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import content as C
from client import Result, TrioClient

st.set_page_config(page_title="TrioDeck", page_icon="🔺", layout="wide")

REQUIRED_SECRETS = ("API_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY")
SAMPLE = ("The city council voted on Tuesday to expand the bike lane network after a year of public "
          "meetings. Supporters said the new lanes would make downtown streets safer for commuters, "
          "while several shop owners worried about losing parking in front of their stores.")
MODEL_NAMES = {"encoder_block": "Hand-written encoder", "bert": "BERT (encoder)",
               "gpt2": "GPT-2 (decoder)", "t5": "T5 (encoder-decoder)"}
MODEL_COLORS = alt.Scale(domain=list(MODEL_NAMES), range=["#7a7a7a", "#2a78d6", "#eb6834", "#2e9e5b"])
AWAKE_FOR_S = 120
# Short generations by default: on the free 0.1-CPU instance every generated token costs
# 0.2-0.3 s when healthy (and far more when Render throttles it). Sliders allow longer.
GPT2_TOKENS = 20
T5_TOKENS, T5_BEAMS = 40, 1
# How long to keep polling a sleeping Render instance before calling it an outage.
WAKE_BUDGET_S = float(os.environ.get("TRIODECK_WAKE_BUDGET_S", "90"))
WAKE_EVERY_S = float(os.environ.get("TRIODECK_WAKE_EVERY_S", "5"))


# --------------------------------------------------------------------------- setup
def secret(name: str) -> str | None:
    # load_if_toml_exists() checks quietly; indexing st.secrets with no secrets file would
    # make Streamlit render its own error once per lookup.
    if not st.secrets.load_if_toml_exists():
        return None
    value = st.secrets.get(name)
    return str(value) if value else None


missing = [n for n in REQUIRED_SECRETS if not secret(n)]
if missing:
    st.error(f"TrioDeck is not configured: missing secret(s) {', '.join(missing)}. Add them in the "
             "Streamlit Cloud secrets manager (or .streamlit/secrets.toml locally).")
    st.stop()

client = TrioClient(secret("API_URL"))


@st.cache_resource
def supabase_anon():
    """Read-only client: the anon key can SELECT the two aggregate views and nothing else."""
    from supabase import create_client

    return create_client(secret("SUPABASE_URL"), secret("SUPABASE_ANON_KEY"))


# --------------------------------------------------------------------------- service state
def ensure_awake() -> bool:
    """Make sure the Render service answers before a model call.

    Render's free instance sleeps after 15 idle minutes and takes about a minute to wake.
    This polls /healthz and tells the user which situation they are in. Nothing is cached
    across failures: the next click checks again, so the page recovers without a reload.
    """
    if time.time() < st.session_state.get("awake_until", 0):
        return True
    with st.status("Checking the model service on Render…", expanded=False) as status:
        def on_try(i: int, r: Result) -> None:
            if not r.ok and r.kind in ("cold_start", "timeout"):
                status.update(label=f"The model service is waking up (free-tier cold start, up to "
                                    f"about a minute). Attempt {i}…", state="running")

        r = client.wait_until_awake(budget_s=WAKE_BUDGET_S, every_s=WAKE_EVERY_S, on_try=on_try)
        if r.ok:
            st.session_state.awake_until = time.time() + AWAKE_FOR_S
            db_ok = r.data.get("database", {}).get("ok")
            status.update(label="Model service is awake." + ("" if db_ok else
                          " Its database is unreachable, so answers will not be logged."),
                          state="complete")
            return True
        st.session_state.awake_until = 0
        status.update(label=r.message, state="error")
    st.error(f"**The model service is unavailable.** {r.message} Nothing below is a live result. "
             "Press the button again in a minute; the page will pick up as soon as the service answers.")
    return False


def show_failure(r: Result) -> None:
    st.session_state.awake_until = 0 if r.kind in ("cold_start", "timeout", "outage") else st.session_state.get("awake_until", 0)
    if r.kind == "cold_start":
        st.warning(f"⏳ **Cold start.** {r.message} Press the button again in about 30 seconds.")
    elif r.kind == "timeout":
        st.warning(f"⏳ **No answer in time.** {r.message} Press the button again in about 30 seconds; "
                   "the next call checks whether the service is awake first.")
    elif r.kind == "outage":
        st.error(f"**Service unreachable.** {r.message} Try again in a few minutes.")
    elif r.kind == "bad_input":
        st.info(f"**Check the input.** {r.message}")
    else:
        st.error(f"**The service reported an error.** {r.message} Try again; if it repeats, "
                 "the model may be failing to load.")


def run_and_store(key: str, path: str, payload: dict) -> None:
    """Call the API. Keep the result only if it is live; otherwise drop the old one too."""
    st.session_state.pop(key, None)            # never leave a previous answer on screen as if current
    if not ensure_awake():
        return
    with st.spinner("Waiting for the model on Render…"):
        r = client.post(path, payload)
    if r.ok:
        st.session_state[key] = {"result": r, "at": datetime.now().strftime("%H:%M:%S")}
    else:
        show_failure(r)


def latency_panel(r: Result, compact: bool = False) -> None:
    """Server-side and end-to-end time, side by side and labelled separately."""
    d = r.data
    gap = max(r.e2e_ms - d["total_ms"], 0.0)
    # Compact panels sit inside the Compare tab's per-model columns; Streamlit allows only one
    # level of nested columns, so they stack vertically instead.
    cols = [st.container(), st.container()] if compact else st.columns(4)
    server_label = "Server model time" if compact else "Server-side model time (latency_ms)"
    e2e_label = "End-to-end wait" if compact else "End-to-end wait (this app ↔ Render)"
    cols[0].metric(server_label, f"{d['latency_ms']:,.0f} ms",
                   help="Measured by the API on Render: tokenise → model → decode. Logged to Supabase.")
    cols[1].metric(e2e_label, f"{r.e2e_ms:,.0f} ms",
                   help="Measured by TrioDeck: request sent → response received. Includes everything below.")
    if not compact:
        cols[2].metric("Checkpoint load on the server (load_ms)", f"{d['load_ms']:,.0f} ms",
                       help="Time spent loading this model into memory during the request (0 if it was "
                            "already loaded). The free server holds one model at a time.")
        cols[3].metric("Network + proxy (end-to-end − server total)", f"{gap:,.0f} ms",
                       help="Transit between Streamlit Cloud and Render, Render's proxy and TLS.")
    if compact:
        st.caption("Server model time = `latency_ms` measured on Render; end-to-end wait = measured by "
                   "this app, Streamlit Cloud ↔ Render.")
    st.caption(f"Server total (total_ms, includes load and any wait for the model lock): "
               f"{d['total_ms']:,.0f} ms · cold start: {'yes' if d['cold_start'] else 'no'} · "
               f"logged to Supabase: {'yes' if d['logged'] else 'NO (database unreachable)'} · "
               f"checkpoint `{d['checkpoint']}`")


def two_columns(sidebar_key: str):
    main, side = st.columns([3, 1], gap="large")
    with side:
        with st.container(border=True):
            st.markdown("#### When to use this archetype" if sidebar_key not in ("concepts", "latency", "compare")
                        else "#### Read me first")
            st.markdown(C.SIDEBARS[sidebar_key])
    return main


def one_line(text: str) -> str:
    """Collapse whitespace so Markdown emphasis survives GPT-2's leading blank lines.

    GPT-2 continuations often start with three newlines, which would split a **bold**
    span across paragraphs and leave stray asterisks on screen.
    """
    return " ".join(text.split()).replace("*", "\\*")


def md_table(header: list[str], rows: list[tuple]) -> str:
    """A Markdown table: unlike st.table, st.markdown typesets the $...$ math in its cells."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def received(key: str) -> dict | None:
    item = st.session_state.get(key)
    if item:
        st.caption(f"Live result received at {item['at']} from the Render service.")
    return item


# --------------------------------------------------------------------------- tabs
st.title("TrioDeck")
st.caption("Three kinds of transformer on one paragraph, with honest timings. Every answer comes from "
           "the TrioFormer service on Render; this page runs no model and writes nothing to the database.")

tabs = st.tabs(["Concepts", "My Encoder Block", "BERT — Fill Mask", "GPT-2 — Continue Text",
                "T5 — Summarize", "Compare", "Latency History"])

# ---- Concepts ---------------------------------------------------------------
with tabs[0]:
    main = two_columns("concepts")
    with main:
        st.header("Multi-head attention, derived")
        st.markdown("**1. Scaled dot-product attention.** Queries $Q\\in\\mathbb{R}^{T_q\\times d_k}$ are "
                    "compared with keys $K\\in\\mathbb{R}^{T_k\\times d_k}$; the softmax turns each row of "
                    "scores into weights over the values $V\\in\\mathbb{R}^{T_k\\times d_v}$:")
        st.latex(C.SDPA)
        st.markdown(C.SCALE_WHY)
        st.markdown("**2. One head** projects the input $X\\in\\mathbb{R}^{T\\times d}$ into its own "
                    "query, key and value spaces of width $d_h$:")
        st.latex(C.HEAD)
        st.markdown("**3. Multi-head attention** is $h$ of those heads run in parallel and concatenated:")
        st.latex(C.MHA)
        st.markdown(C.MHA_WHY)
        st.markdown("**4. Shapes, step by step** (as implemented in `src/transformer.py`):")
        st.markdown(md_table(["Step", "Shape", "What happens"], C.SHAPES))
        st.markdown("**5. The encoder block** wraps attention and a two-layer MLP in residual "
                    "connections, normalising first (pre-norm):")
        st.latex(C.BLOCK)
        st.markdown(C.EQUIVARIANCE)

        st.header("Positional encodings compared")
        st.markdown("**Sinusoidal** (fixed, added to the embeddings):")
        st.latex(C.SINUSOIDAL)
        st.markdown("**Rotary (RoPE)** rotates each $(2i, 2i{+}1)$ pair of a query or key by an angle "
                    "proportional to its position, inside every attention layer:")
        st.latex(C.ROPE_PAIR)
        st.latex(C.ROPE)
        st.markdown("Because rotations compose, $R_{\\Theta,m}^{\\top}R_{\\Theta,n}=R_{\\Theta,n-m}$: the "
                    "score between positions $m$ and $n$ depends on their **distance**, not on where "
                    "they are.")
        st.markdown(md_table(["Scheme", "How it enters", "What attention sees", "Extra parameters",
                              "Longer inputs than trained", "Used by"], C.PE_TABLE))
        st.subheader("Why rotary encodings dominate modern LLMs")
        for point in C.ROPE_WHY:
            st.markdown(f"- {point}")

# ---- My Encoder Block --------------------------------------------------------
with tabs[1]:
    main = two_columns("encoder_block")
    with main:
        st.header("My Encoder Block")
        st.markdown("A transformer encoder written from scratch (`src/transformer.py`, 2 layers × 4 heads), "
                    "trained to decide whether a list of **2–12 words** reads the same backwards. "
                    "Words are replaced by numbers in order of first appearance, so only the pattern matters.")
        c1, c2 = st.columns([3, 1])
        words = c1.text_input("Words", "dog bites man bites dog", key="enc_text")
        pe = c2.toggle("Positional encoding", value=True, key="enc_pe",
                       help="Off: the same weights run without knowing word order.")
        st.caption("Try **a b a** and **a a b** with the toggle on, then off. Without positional encoding "
                   "both are the same bag of words, so the model must give them the same answer.")
        if st.button("Run my encoder", key="enc_go"):
            run_and_store("enc", "/encoder_attention", {"text": words, "positional_encoding": pe})
        item = received("enc")
        if item:
            d = item["result"].data
            st.subheader(f"Prediction: {d['prediction']} "
                         f"({max(d['probabilities'].values()):.1%}) — positional encoding "
                         f"{'ON' if d['positional_encoding'] else 'OFF'}")
            st.caption(f"Words seen as ids {d['ids']}. Held-out accuracy of this toy model (with positional "
                       f"encoding): {d['model_accuracy']:.1%}.")
            labels = [f"{i + 1}:{w}" for i, w in enumerate(d["words"])]
            for layer, heads in enumerate(d["attention"]):
                st.markdown(f"**Layer {layer + 1}** — each row is a word, each column the word it attends to; "
                            "rows sum to 1.")
                cols = st.columns(len(heads))
                for h, matrix in enumerate(heads):
                    df = pd.DataFrame([(labels[q], labels[k], w) for q, row in enumerate(matrix)
                                       for k, w in enumerate(row)], columns=["query", "key", "weight"])
                    chart = alt.Chart(df).mark_rect().encode(
                        x=alt.X("key:N", sort=labels, title=None),
                        y=alt.Y("query:N", sort=labels, title=None),
                        color=alt.Color("weight:Q", scale=alt.Scale(domain=[0, 1], scheme="blues"), legend=None),
                        tooltip=["query", "key", alt.Tooltip("weight:Q", format=".3f")],
                    ).properties(title=f"Head {h + 1}", height=220)
                    cols[h].altair_chart(chart, use_container_width=True)
            summary = pd.DataFrame([{"layer": li + 1, **s} for li, layer in enumerate(d["head_summary"])
                                    for s in layer])
            summary["head"] += 1
            st.markdown("**What the heads look at, on average** — compare each column with *uniform* "
                        "(a head that ignored the input):")
            st.dataframe(summary.set_index(["layer", "head"]), use_container_width=True)
            latency_panel(item["result"], compact=True)

# ---- BERT --------------------------------------------------------------------
with tabs[2]:
    main = two_columns("bert")
    with main:
        st.header("BERT — Fill Mask")
        text = st.text_area("Text with exactly one [MASK], or plain text plus a word to hide",
                            "I drank a cup of [MASK] this morning.", key="bert_text")
        c1, c2 = st.columns(2)
        mask_word = c1.text_input("Word to hide (optional)", "", key="bert_word")
        top_k = c2.slider("Candidates", 1, 10, 5, key="bert_k")
        if st.button("Fill the mask", key="bert_go"):
            payload = {"text": text, "top_k": top_k}
            if mask_word.strip():
                payload["mask_word"] = mask_word.strip()
            run_and_store("bert", "/bert_fillmask", payload)
        item = received("bert")
        if item:
            d = item["result"].data
            st.markdown(f"**Best guess:** {d['filled_text']}")
            cand = pd.DataFrame(d["candidates"])
            st.altair_chart(alt.Chart(cand).mark_bar(color="#2a78d6").encode(
                x=alt.X("score:Q", title="probability"), y=alt.Y("token:N", sort="-x", title=None)),
                use_container_width=True)
            latency_panel(item["result"])

# ---- GPT-2 -------------------------------------------------------------------
with tabs[3]:
    main = two_columns("gpt2")
    with main:
        st.header("GPT-2 — Continue Text")
        text = st.text_area("Beginning of the text", SAMPLE, key="gpt_text", height=120)
        c1, c2 = st.columns(2)
        n = c1.slider("New tokens", 5, 80, GPT2_TOKENS, key="gpt_n",
                      help="Each extra token is another full pass of the decoder, so time grows with length.")
        sample = c2.toggle("Sample (creative) instead of greedy", value=False, key="gpt_sample")
        if st.button("Continue the text", key="gpt_go"):
            run_and_store("gpt2", "/gpt2_continue", {"text": text, "max_new_tokens": n, "do_sample": sample})
        item = received("gpt2")
        if item:
            d = item["result"].data
            st.markdown(f"> {one_line(text)} **{one_line(d['continuation']) or '(no new words)'}**")
            st.caption("Bold = generated. GPT-2 writes fluently and invents details; nothing here is checked.")
            latency_panel(item["result"])

# ---- T5 ----------------------------------------------------------------------
with tabs[4]:
    main = two_columns("t5")
    with main:
        st.header("T5 — Summarize")
        text = st.text_area("Paragraph to summarise", SAMPLE, key="t5_text", height=120)
        n = st.slider("Maximum summary length (tokens)", 8, 120, T5_TOKENS, key="t5_n")
        if st.button("Summarise", key="t5_go"):
            run_and_store("t5", "/t5_summarize", {"text": text, "max_new_tokens": n, "num_beams": T5_BEAMS})
        item = received("t5")
        if item:
            st.success(item["result"].data["summary"])
            latency_panel(item["result"])

# ---- Compare -----------------------------------------------------------------
with tabs[5]:
    main = two_columns("compare")
    with main:
        st.header("Compare: one paragraph, three archetypes")
        text = st.text_area("Paragraph", SAMPLE, key="cmp_text", height=120)
        mask_word = st.text_input("Word BERT should fill back in", "safer", key="cmp_word")
        st.caption(f"GPT-2 writes {st.session_state.get('gpt_n', GPT2_TOKENS)} new tokens here "
                   f"(set on the GPT-2 tab); T5 writes up to {T5_TOKENS}.")
        if st.button("Run all three", key="cmp_go"):
            st.session_state.pop("cmp", None)
            if ensure_awake():
                calls = {
                    "bert": ("/bert_fillmask", {"text": text, "mask_word": mask_word.strip()}),
                    # Follows the GPT-2 tab's slider, so a slow server can be met with fewer tokens.
                    "gpt2": ("/gpt2_continue", {"text": text,
                                                "max_new_tokens": st.session_state.get("gpt_n", GPT2_TOKENS)}),
                    "t5": ("/t5_summarize", {"text": text, "max_new_tokens": T5_TOKENS,
                                             "num_beams": T5_BEAMS}),
                }
                # One after another, on purpose: the free server has 0.1 CPU and one model slot,
                # so concurrent requests only queue and compete (measured: one passed the 120 s limit).
                t0 = time.perf_counter()
                results = {}
                with st.status("Running the three models one after another…") as status:
                    for m, (path, payload) in calls.items():
                        status.update(label=f"Running {MODEL_NAMES[m]}…")
                        results[m] = client.post(path, payload)
                    status.update(label="All three answered." if all(r.ok for r in results.values())
                                  else "Finished, with errors (see below).", state="complete")
                wall = (time.perf_counter() - t0) * 1000
                st.session_state.cmp = {"results": results, "wall": wall,
                                        "at": datetime.now().strftime("%H:%M:%S")}
        item = received("cmp")
        if item:
            results = item["results"]
            cols = st.columns(3)
            for col, model in zip(cols, ("bert", "gpt2", "t5")):
                r = results[model]
                with col:
                    st.subheader(MODEL_NAMES[model])
                    if not r.ok:
                        show_failure(r)
                        continue
                    d = r.data
                    if model == "bert":
                        st.markdown(f"Filled **[MASK]** with: **{d['candidates'][0]['token']}** "
                                    f"({d['candidates'][0]['score']:.0%}); then "
                                    + ", ".join(c["token"] for c in d["candidates"][1:4]))
                    elif model == "gpt2":
                        st.markdown(f"…**{one_line(d['continuation']) or '(no new words)'}**")
                    else:
                        st.markdown(d["summary"])
                    latency_panel(r, compact=True)
            ok = {m: r for m, r in results.items() if r.ok}
            if ok:
                rows = []
                for m, r in ok.items():
                    rows += [{"model": m, "clock": "1 · server model time (latency_ms)", "ms": r.data["latency_ms"]},
                             {"model": m, "clock": "2 · server total incl. model load (total_ms)", "ms": r.data["total_ms"]},
                             {"model": m, "clock": "3 · end-to-end wait (this app ↔ Render)", "ms": r.e2e_ms}]
                st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                    x=alt.X("ms:Q", title="milliseconds"), y=alt.Y("clock:N", title=None),
                    row=alt.Row("model:N", title=None, sort=["bert", "gpt2", "t5"]),
                    color=alt.Color("model:N", scale=MODEL_COLORS, legend=None),
                    tooltip=["model", "clock", alt.Tooltip("ms:Q", format=",.0f")],
                ).properties(height=70), use_container_width=False)
            st.caption(f"Whole comparison took {item['wall']:,.0f} ms of wall time. The models run one after "
                       "another because the free server holds one model at a time; switching models shows up "
                       "as load_ms inside total_ms, and the network as the gap to the end-to-end bar.")

# ---- Latency History ---------------------------------------------------------
with tabs[6]:
    main = two_columns("latency")
    with main:
        st.header("Latency History")
        st.markdown("Median server-side model time per model, per hour, read with the read-only key from "
                    "the `trio_benchmarks_hourly` materialized view in Supabase.")
        if st.button("Load the latest numbers", key="hist_go"):
            st.session_state.pop("hist", None)
            refresh = client.get("/benchmarks", timeout=30)   # asks the API to refresh the views
            try:
                db = supabase_anon()
                hourly = db.table("trio_benchmarks_hourly").select("*").order("hour").execute().data
                overall = db.table("trio_benchmarks").select("*").order("model").execute().data
                st.session_state.hist = {"hourly": hourly, "overall": overall, "refreshed": refresh.ok,
                                         "refresh_msg": refresh.message,
                                         "at": datetime.now().strftime("%H:%M:%S")}
            except Exception as exc:  # the read itself failed: show nothing rather than old numbers
                st.error(f"**Could not read the benchmarks view from Supabase** ({type(exc).__name__}). "
                         "Nothing is shown rather than old numbers. Try again shortly.")
        item = st.session_state.get("hist")
        if item:
            if not item["refreshed"]:
                st.warning("The API could not refresh the views just now "
                           f"({item['refresh_msg'] or 'no answer'}). The chart shows the view as of its "
                           "last refresh — see 'refreshed at' below.")
            if not item["hourly"]:
                st.info("No calls have been logged yet. Run a model on another tab, then load again.")
            else:
                h = pd.DataFrame(item["hourly"])
                h["hour"] = pd.to_datetime(h["hour"])
                st.altair_chart(alt.Chart(h).mark_line(point=True).encode(
                    x=alt.X("hour:T", title="hour (your local time)"),
                    y=alt.Y("p50_latency_ms:Q", title="median model time (ms, log scale)",
                            scale=alt.Scale(type="log")),
                    color=alt.Color("model:N", scale=MODEL_COLORS, title=None),
                    tooltip=["model", "hour:T", alt.Tooltip("p50_latency_ms:Q", format=",.0f"),
                             alt.Tooltip("p95_latency_ms:Q", format=",.0f"), "n_calls", "n_cold"],
                ), use_container_width=True)
                st.caption(f"View refreshed at {h['refreshed_at'].iloc[0]} (UTC); read at {item['at']}.")
            if item["overall"]:
                o = pd.DataFrame(item["overall"])[["model", "n_calls", "n_warm", "n_cold", "p50_latency_ms",
                                                   "p95_latency_ms", "mean_load_ms", "ms_per_output_token"]]
                st.markdown("**All time, per model** (`trio_benchmarks`):")
                st.dataframe(o.set_index("model"), use_container_width=True)
