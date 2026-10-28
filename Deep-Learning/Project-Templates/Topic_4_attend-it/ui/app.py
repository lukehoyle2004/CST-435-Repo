"""Streamlit UI -- Cloud #1 (Streamlit Community Cloud).

A THIN client: no torch, no tokenizer, no model code. It
  * posts review text to the FastAPI service on Render and renders the JSON it
    gets back (attention arrives as one plain number per token),
  * reads `seq_runs` and `seq_predictions` straight from Supabase with the
    public ANON key (RLS allows SELECT only).

Secrets (st.secrets; see .streamlit/secrets.toml.example):
    API_URL, SUPABASE_URL, SUPABASE_ANON_KEY
"""
from __future__ import annotations

import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from supabase import create_client

from viz import attention_html, peak_ratio, position_rows

st.set_page_config(page_title="Seq-Sight", page_icon="🔎", layout="wide")

API_URL = st.secrets["API_URL"].rstrip("/")
ROOT = Path(__file__).resolve().parent
MODEL_LABELS = {"attention": "LSTM + attention", "lstm": "Plain LSTM"}
MODEL_COLORS = alt.Scale(domain=["lstm", "attention"], range=["#eb6834", "#2a78d6"])
EMOJI = {"positive": "👍", "negative": "👎"}
SAMPLE = ("I expected a dull remake, but the performances are wonderful and the last twenty "
          "minutes had me on the edge of my seat. Not perfect, the middle drags, but well worth it.")


# --------------------------------------------------------------------------- helpers
@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def error_text(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            return f"{exc.response.status_code}: {exc.response.json().get('detail')}"
        except ValueError:
            return f"{exc.response.status_code}: {exc.response.text[:200]}"
    return str(exc)


def wake_api() -> bool:
    """Render's free tier sleeps after 15 idle minutes; the first call can take ~1 min."""
    if st.session_state.get("api_awake"):
        return True
    with st.status("Waking the model service on Render (free-tier cold start, up to a minute). "
                   "The Baseline vs. Attention and Recent Predictions tabs already work: they read Supabase.",
                   expanded=False) as status:
        for _ in range(12):
            try:
                if requests.get(f"{API_URL}/healthz", timeout=10).ok:
                    st.session_state.api_awake = True
                    status.update(label="Model service is awake.", state="complete")
                    return True
            except requests.RequestException:
                pass
            time.sleep(5)
        status.update(label="The model service did not answer. Refresh the page to retry; "
                            "the database-backed tabs still work.", state="error")
    return False


def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"Prediction failed: {error_text(exc)}")
        return None


@st.cache_data(ttl=3600, show_spinner="Loading the IMDB examples…")
def load_gallery() -> dict:
    r = requests.get(f"{API_URL}/gallery", timeout=120)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def load_runs() -> list[dict]:
    return supabase_anon().table("seq_runs").select("*").order("created_at").execute().data


def runs_by_name() -> dict[str, dict]:
    try:
        return {r["name"]: r for r in load_runs()}
    except Exception as exc:
        st.warning(f"Could not read runs from Supabase: {exc}")
        return {}


def heatmap(tokens, weights, unk=None):
    st.markdown(attention_html(tokens, weights, unk), unsafe_allow_html=True)


def position_chart(weights, tokens=None, height=160):
    """Attention by position, for predictions whose words were never stored."""
    df = pd.DataFrame(position_rows(weights, tokens))
    st.altair_chart(alt.Chart(df).mark_rect().encode(
        x=alt.X("position:O", title="token position", axis=alt.Axis(labels=len(df) <= 60)),
        color=alt.Color("relative:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), legend=None),
        tooltip=["position", "token", alt.Tooltip("weight:Q", format=".4f")],
    ).properties(height=height), use_container_width=True)


def prob_bar(probs: dict):
    df = pd.DataFrame({"class": list(probs), "probability": list(probs.values())})
    st.altair_chart(alt.Chart(df).mark_bar().encode(
        x=alt.X("probability:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("class:N", title=None, sort=["negative", "positive"]),
        color=alt.Color("class:N", scale=alt.Scale(domain=["negative", "positive"],
                                                   range=["#e45756", "#54a24b"]), legend=None),
        tooltip=["class", alt.Tooltip("probability:Q", format=".2%")]).properties(height=alt.Step(26)),
        use_container_width=True)


def show_prediction(res: dict):
    """One /predict response: verdict, probabilities, attention heatmap, provenance."""
    if res["abstained"]:
        head = (f"🤔 **uncertain** · the model leans {res['class_name']} at {res['confidence']:.1%}, "
                "too unsure to call")
    else:
        head = f"{EMOJI[res['class_name']]} **{res['class_name']}** · {res['confidence']:.1%} confident"
    if res.get("true_class"):
        mark = "✅" if res["true_class"] == res["class_name"] else "❌"
        head += f" · {mark} true label: {res['true_class']}"
    st.markdown(f"#### {MODEL_LABELS[res['model']]}\n{head}")
    if res["abstained"]:
        st.warning("Guardrail: " + "; ".join(res["flag_messages"]) + ". Treat this review as "
                   "unclassified (the IMDB base rate is 50/50) rather than acting on the lean.")
    prob_bar(res["probabilities"])
    if res["attention"]:
        heatmap(res["tokens"], res["attention"], res["unk"])
        top = ", ".join(f"`{t['token']}` {t['weight']:.3f}" for t in res["top_tokens"])
        st.caption(f"Most attended: {top} · peak ratio {peak_ratio(res['attention']):.1f}× uniform")
    else:
        heatmap(res["tokens"], None, res["unk"])
        st.caption("The plain LSTM has no attention: it classifies from its final hidden state "
                   "only, so there is nothing per-word to show.")
    notes = []
    if res["truncated"]:
        notes.append(f"⚠️ only the first {res['n_tokens']} of {res['raw_tokens']} tokens were scored")
    if res["unk_tokens"]:
        notes.append(f"{res['unk_tokens']} word(s) are outside the 20k vocabulary (struck through)")
    notes.append(f"run `{res['served_by_run_id']}` · {res['latency_ms']:.0f} ms · hash "
                 f"`{res['text_sha256'][:12]}…`")
    notes.append("logged to Supabase (hash + attention, no text)" if res["logged"]
                 else "⚠️ database unavailable, prediction not logged")
    st.caption(" · ".join(notes))


def show_card(card: dict):
    """A precomputed gallery review (public IMDB test set): both models' answers + attention."""
    a_ok = card["attention_pred"] == card["true"]
    l_ok = card["lstm_pred"] == card["true"]
    st.markdown(
        f"**True label: {card['true']}** · "
        f"{'✅' if a_ok else '❌'} attention: {card['attention_pred']} ({card['attention_confidence']:.1%}) · "
        f"{'✅' if l_ok else '❌'} plain LSTM: {card['lstm_pred']} ({card['lstm_confidence']:.1%})"
        + (f" · {card['raw_length']} tokens, first 256 shown" if card["raw_length"] > 256 else ""))
    heatmap(card["tokens"], card["attention"])


def snippet(card: dict, n: int = 12) -> str:
    return " ".join(card["tokens"][:n]) + ("…" if len(card["tokens"]) > n else "")


# --------------------------------------------------------------------------- header
st.title("🔎 Seq-Sight")
st.caption("Type a movie review; two matched LSTMs trained on 25,000 IMDB reviews call it positive or "
           "negative, and the attention model shows which words it weighted. Only a SHA-256 hash of "
           "your text and the attention numbers are logged, never the text.")
banner = st.container()

tab_concepts, tab_try, tab_compare, tab_recent, tab_failures, tab_card = st.tabs(
    ["📐 Concepts", "✍️ Try It", "⚖️ Baseline vs. Attention", "🕒 Recent Predictions",
     "❌ Failure Gallery", "🪪 Model Card"])

# The database-backed tabs render first, so they work while Render is still waking up.

# --------------------------------------------------------------------------- Concepts
with tab_concepts:
    st.header("Scaled dot-product attention, by hand")
    st.markdown("Each query $q$ is compared with every key $k_j$; the scores become weights with a "
                "softmax, and the output is the weighted average of the values $v_j$. Padding "
                "positions get a mask of $-\\infty$, so their weight is exactly 0:")
    st.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}+M\Big)V,"
             r"\qquad M_{ij}=\begin{cases}0 & \text{token } j \text{ is real}\\ -\infty & "
             r"\text{token } j \text{ is padding}\end{cases}")
    st.markdown("In Seq-Sight the LSTM states $h_1..h_T$ are the values, the keys are $k_t = W_k h_t + b_k$ "
                "($d_k = 64$), and there is **one learned query** $q$ that asks \"which words decide the "
                "sentiment?\". The pooled review vector is $c = \\sum_t \\alpha_t h_t$.")

    st.subheader("Worked example (d_k = 2, three tokens)")
    st.markdown("$q=[1,1]$, keys $k_1=[1,1],\\;k_2=[1,-1],\\;k_3=[-1,0]$, values $v=[10,20,30]$. "
                "This exact case is a unit test (`tests/test_attention.py`).")
    st.table(pd.DataFrame({
        "token": ["1", "2", "3"],
        "q·k": [2, 0, -1],
        "q·k / √2": [1.4142, 0.0, -0.7071],
        "exp": [4.1133, 1.0, 0.4931],
        "weight α (exp / 5.6064)": [0.7337, 0.1784, 0.0879],
        "α·v": [7.337, 3.567, 2.638],
    }).set_index("token"))
    st.markdown("Output $= 7.337 + 3.567 + 2.638 \\approx \\mathbf{13.54}$: mostly token 1's value, "
                "because its key points the same way as the query.")

    st.subheader("Why divide by √d_k?")
    st.markdown("If the entries of $q$ and $k$ are independent with mean 0 and variance 1, then "
                "$q\\cdot k=\\sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$. With $d_k=256$ the scores "
                "have standard deviation 16, so typical score gaps are tens of units and the softmax "
                "saturates. The softmax gradient $\\partial \\alpha_i/\\partial s_i = "
                "\\alpha_i(1-\\alpha_i)$ then vanishes and attention stops learning:")
    st.table(pd.DataFrame({
        "scores": ["[16, 0, −16]  (unscaled, d_k = 256)", "[1, 0, −1]  (÷ √256 = 16)"],
        "softmax": ["[1.000000, 0.000000, 0.000000]", "[0.665, 0.245, 0.090]"],
        "α(1−α)  (gradient size)": ["[0.000000, 0.000000, 0.000000]", "[0.223, 0.185, 0.082]"],
    }).set_index("scores"))
    st.markdown("Dividing by $\\sqrt{d_k}$ brings the variance back to 1 at any width "
                "(`test_sqrt_dk_scaling_keeps_scores_unit_variance`, `test_unscaled_scores_saturate_the_softmax`).")

    st.subheader("Attention vs. the recurrent step")
    st.markdown("The LSTM step is $h_t = f(h_{t-1}, x_t)$: everything the classifier knows about the "
                "review has to survive in one 128-number vector, updated 256 times.")
    st.table(pd.DataFrame([
        ("Path from word t to the output", "T − t recurrent steps", "1 step (direct weight α_t)"),
        ("Memory of the past", "fixed-size h_T (128 numbers)", "all T hidden states"),
        ("Sequential operations", "O(T): must run in order", "O(1): all scores at once"),
        ("Cost per layer", "O(T · d²)", "O(T² · d) (here O(T · d), one query)"),
        ("Gradient to early words", "multiplied through T gates; can vanish", "one softmax away"),
        ("Inspectable", "no per-word signal", "α_t per word (what this app shows)"),
        ("Order information", "built in", "none on its own; here it comes from the LSTM states"),
    ], columns=["", "Recurrent step (plain LSTM)", "Attention pooling"]).set_index(""))
    st.info("**Reading the heatmaps honestly:** attention weights show where the model *looked* "
            "when pooling, not a proof of *why* it decided. The LSTM states being weighted already "
            "mix in context from earlier words, and on IMDB our model's attention is fairly spread "
            "out (mean normalized entropy 0.975, where 1 means uniform).")

# --------------------------------------------------------------------------- Compare Models
with tab_compare:
    runs = runs_by_name()
    if {"lstm", "attention"} <= runs.keys():
        l, a = runs["lstm"], runs["attention"]
        st.subheader("Matched runs (from the Supabase `seq_runs` table)")
        st.caption(f"Same seed ({a['seed']}), vocabulary, data split ({a['train_reviews']:,} train / "
                   f"{a['val_reviews']:,} val), {a['epochs']} epochs, batch {a['batch_size']}, lr "
                   f"{a['lr']}, max {a['max_len']} tokens and hardware. The only difference is the "
                   f"attention block ({a['params'] - l['params']:,} extra parameters). "
                   "SQL: `select * from seq_model_comparison;`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Attention test accuracy", f"{100 * a['test_acc']:.2f}%",
                  f"{100 * (a['test_acc'] - l['test_acc']):+.2f} pts vs plain LSTM")
        m2.metric("Plain LSTM test accuracy", f"{100 * l['test_acc']:.2f}%")
        m3.metric("Calibration error (ECE)", f"{a['ece']:.3f}", f"{a['ece'] - l['ece']:+.3f}",
                  delta_color="inverse")
        m4.metric("Wrong at ≥90% confidence", f"{a['confident_errors']:,}",
                  f"{a['confident_errors'] - l['confident_errors']:+,}", delta_color="inverse")
        st.dataframe(pd.DataFrame([{
            "model": r["display_name"], "run_id": r["run_id"], "params": r["params"],
            "best epoch": f"{r['best_epoch']} / {r['epochs']}", "val acc": r["val_acc"],
            "test acc": r["test_acc"], "macro F1": r["macro_f1"], "ECE": r["ece"],
            "confident errors": r["confident_errors"],
            "train time (min)": round(r["train_seconds"] / 60, 1) if r.get("train_seconds") else None,
        } for r in (l, a)]), hide_index=True, use_container_width=True)
        if a.get("test_acc_ci95") and "mcnemar_p" in a["agreement"]:
            st.caption(
                f"95% intervals on 25,000 reviews: attention {100 * a['test_acc_ci95'][0]:.2f}–"
                f"{100 * a['test_acc_ci95'][1]:.2f}%, plain LSTM {100 * l['test_acc_ci95'][0]:.2f}–"
                f"{100 * l['test_acc_ci95'][1]:.2f}%. On the same reviews the models disagree "
                f"{a['agreement']['only_attention_right'] + a['agreement']['only_lstm_right']:,} times; "
                f"McNemar's test on those gives p = {a['agreement']['mcnemar_p']:.1e}, so the gap is not "
                "chance on this test set (one seed per model, so not a claim about every seed).")

        st.subheader("Accuracy by review length")
        st.caption("Reviews longer than 256 tokens are cut to their first 256. The plain LSTM must "
                   "carry every cue through its final hidden state; attention can reach back to any "
                   "position, and its advantage is largest on the longest reviews.")
        lb = pd.DataFrame([dict(model=r["name"], **b) for r in (l, a) for b in r["length_buckets"]])
        st.altair_chart(alt.Chart(lb).mark_bar().encode(
            x=alt.X("bucket:N", title="raw review length (tokens)",
                    sort=[b["bucket"] for b in a["length_buckets"]]),
            xOffset="model:N",
            y=alt.Y("acc:Q", title="test accuracy", scale=alt.Scale(domain=[0.75, 0.92])),
            color=alt.Color("model:N", scale=MODEL_COLORS),
            tooltip=["model", "bucket", "n", alt.Tooltip("acc:Q", format=".2%")],
        ).properties(height=300), use_container_width=True)

        st.subheader("Training curves")
        hist = pd.concat([pd.DataFrame(r["history"]).assign(model=r["name"]) for r in (l, a)])
        long = pd.concat([hist.assign(split="train", loss=hist.train_loss),
                          hist.assign(split="validation", loss=hist.val_loss)])
        c1, c2 = st.columns(2)
        c1.altair_chart(alt.Chart(long).mark_line(point=True).encode(
            x="epoch:O", y=alt.Y("loss:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=MODEL_COLORS), strokeDash="split:N",
            tooltip=["model", "split", "epoch", alt.Tooltip("loss:Q", format=".3f")],
        ).properties(height=260, title="loss"), use_container_width=True)
        c2.altair_chart(alt.Chart(hist).mark_line(point=True).encode(
            x="epoch:O", y=alt.Y("val_acc:Q", title="validation accuracy", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=MODEL_COLORS),
            tooltip=["model", "epoch", alt.Tooltip("val_acc:Q", format=".2%")],
        ).properties(height=260, title="validation accuracy"), use_container_width=True)
        st.caption(f"The attention model learns much faster (epoch 1 validation accuracy "
                   f"{a['history'][0]['val_acc']:.1%} vs {l['history'][0]['val_acc']:.1%}) and starts "
                   f"overfitting after epoch {a['best_epoch']}; each run keeps its best-validation epoch.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Where they disagree")
            ag = a["agreement"]
            st.table(pd.DataFrame(
                [[ag["both_right"], ag["only_lstm_right"]], [ag["only_attention_right"], ag["both_wrong"]]],
                index=["attention right", "attention wrong"],
                columns=["plain LSTM right", "plain LSTM wrong"]).T)
        with c2:
            st.subheader("Per-class F1")
            st.table(pd.DataFrame({r["display_name"]: {c: round(v["f1"], 4) for c, v in r["per_class"].items()}
                                   for r in (l, a)}))
        if a.get("attention_stats"):
            s = a["attention_stats"]
            st.caption(f"Attention shape over the test set: mean normalized entropy "
                       f"{s['mean_normalized_entropy']:.3f} (1 = uniform), median peak ratio "
                       f"{s['median_peak_ratio']:.2f}× uniform weight on the top token.")

        if a.get("selective"):
            st.subheader("Guardrail: answer only when confident")
            st.caption("If the service abstains below a confidence threshold (and says \"uncertain\" "
                       "instead), the answers it does give are far more reliable. The API abstains "
                       "below 75%.")
            sel = pd.DataFrame([dict(model=r["name"], **s) for r in (l, a) for s in r["selective"]])
            st.dataframe(sel.assign(
                threshold=sel.threshold.map("≥ {:.0%}".format),
                coverage=sel.coverage.map("{:.1%}".format),
                acc=sel.acc.map("{:.2%}".format),
                abstained_acc=sel.abstained_acc.map(lambda v: f"{v:.1%}" if v is not None else "—"),
            ).rename(columns={"threshold": "answers when confidence", "coverage": "reviews answered",
                              "acc": "accuracy on answered", "abstained_acc": "accuracy it would have had on the rest"}),
                hide_index=True, use_container_width=True)
    else:
        st.info("Run rows not found in Supabase yet (apply db/migrations/001_init.sql and run "
                "`python -m db.seed_runs`).")

# --------------------------------------------------------------------------- Recent Predictions
with tab_recent:
    st.caption("Read directly from Supabase with the public anon key: `seq_predictions` joined to "
               "`seq_runs`. Each row holds a hash, token counts, the answer and the attention vector; "
               "never the review text.")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
    try:
        rows = (supabase_anon().table("seq_predictions")
                .select("id,created_at,text_sha256,source,test_index,n_tokens,raw_tokens,unk_tokens,"
                        "predicted_class,confidence,abstained,model,served_by_run_id,latency_ms,"
                        "seq_runs(name,display_name)")
                .order("created_at", desc=True).limit(500).execute().data)
    except Exception as exc:
        rows = None
        st.warning(f"Supabase is unreachable right now ({exc}). Try again in a moment.")
    if rows == []:
        st.info("No predictions yet. Try one in the Try It tab.")
    elif rows:
        df = pd.DataFrame([{
            "id": r["id"], "time (UTC)": pd.to_datetime(r["created_at"]).strftime("%Y-%m-%d %H:%M:%S"),
            "predicted": r["predicted_class"], "confidence": r["confidence"],
            "guardrail": "uncertain" if r.get("abstained") else "", "model": r["model"],
            "source": r["source"] if r["source"] == "user" else f"IMDB test #{r['test_index']}",
            "tokens": r["n_tokens"], "truncated": r["raw_tokens"] > r["n_tokens"],
            "unknown words": r["unk_tokens"], "run_id": r["served_by_run_id"],
            "text hash": r["text_sha256"][:12] + "…", "latency ms": r["latency_ms"],
        } for r in rows])
        models_seen = sorted(df["model"].unique())
        pick = st.multiselect("Model", models_seen, default=models_seen)
        view = df[df["model"].isin(pick)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictions", len(view))
        c2.metric("Positive share", f"{(view['predicted'] == 'positive').mean():.0%}" if len(view) else "—")
        c3.metric("Mean confidence", f"{view['confidence'].mean():.1%}" if len(view) else "—")
        c4.metric("Distinct texts", view["text hash"].nunique())
        st.dataframe(view, hide_index=True, use_container_width=True,
                     column_config={"confidence": st.column_config.ProgressColumn(
                         "confidence", format="%.2f", min_value=0, max_value=1)})

        st.subheader("Drill into a prediction")
        pid = st.selectbox("Prediction id", view["id"].tolist(),
                           format_func=lambda i: f"#{i} · " + " · ".join(
                               str(v) for v in view.loc[view["id"] == i, ["predicted", "model", "source"]].iloc[0]))
        if pid is not None:
            try:
                row = (supabase_anon().table("seq_predictions").select("*,seq_runs(display_name,test_acc)")
                       .eq("id", pid).limit(1).execute().data[0])
            except Exception as exc:
                row = None
                st.warning(f"Could not load prediction {pid}: {exc}")
            if row:
                st.markdown(f"{EMOJI[row['predicted_class']]} **{row['predicted_class']}** · "
                            f"{row['confidence']:.1%} confident · served by "
                            f"{row['seq_runs']['display_name']} (run `{row['served_by_run_id']}`, test "
                            f"accuracy {row['seq_runs']['test_acc']:.2%})")
                prob_bar(row["probabilities"])
                if not row["attention"]:
                    st.info("This prediction came from the plain LSTM, which has no attention vector.")
                else:
                    tokens = None
                    if row["source"] == "imdb_test" and st.session_state.get("api_awake"):
                        try:
                            cards = [c for cs in load_gallery().values() for c in cs]
                            tokens = next((c["tokens"] for c in cards if c["test_index"] == row["test_index"]), None)
                        except requests.RequestException:
                            pass
                    if tokens:
                        st.caption(f"Public IMDB test review #{row['test_index']}: the words come from "
                                   "the dataset; the weights are the ones stored with this prediction.")
                        heatmap(tokens, row["attention"])
                    else:
                        st.caption("The review text was never stored, so the stored attention vector is "
                                   "drawn by token position (darker = more attention). Hover for weights."
                                   if row["source"] == "user" else
                                   "Words load once the model service is awake; showing positions.")
                    position_chart(row["attention"], tokens)
                    st.caption(f"{row['n_tokens']} weights summing to {sum(row['attention']):.4f} · peak at "
                               f"position {max(range(len(row['attention'])), key=row['attention'].__getitem__)} · "
                               f"peak ratio {peak_ratio(row['attention']):.1f}× uniform")

# --------------------------------------------------------------------------- Model Card
with tab_card:
    card_path = ROOT.parent / "MODEL_CARD.md"
    st.markdown(card_path.read_text(encoding="utf-8") if card_path.exists() else "MODEL_CARD.md not found.")

# --------------------------------------------------------------------------- the API-backed tabs
with banner:
    api_ok = wake_api()

# --------------------------------------------------------------------------- Try It
with tab_try:
    left, right = st.columns([2, 3])
    with left:
        with st.form("predict"):
            text = st.text_area("Movie review", SAMPLE, height=200, max_chars=20_000)
            choice = st.radio("Model", ["LSTM + attention", "Plain LSTM", "Both, side by side"],
                              horizontal=True)
            submitted = st.form_submit_button("Classify", type="primary", disabled=not api_ok)
        st.caption("Reviews are lowercased and split into words, `!` and `?`; only the first 256 "
                   "tokens are scored. Darker blue = more attention (relative to the top word); "
                   "struck-through words are outside the vocabulary.")
    if submitted and text.strip():
        kinds = {"LSTM + attention": ["attention"], "Plain LSTM": ["lstm"]}.get(choice, ["attention", "lstm"])
        with st.spinner("Classifying…"):
            st.session_state.try_results = [r for k in kinds
                                            if (r := api_post("/predict", {"text": text, "model": k}))]
    with right:
        results = st.session_state.get("try_results", [])
        if not results:
            st.info("Write or paste a review and press Classify." if api_ok
                    else "The model service is unavailable; refresh to retry.")
        cols = st.columns(len(results)) if len(results) > 1 else [st.container()]
        for res, col in zip(results, cols):
            with col:
                show_prediction(res)

# --------------------------------------------------------------------------- gallery-backed sections
gallery = None
if api_ok:
    try:
        gallery = load_gallery()
    except requests.RequestException as exc:
        st.error(f"Could not load the IMDB examples: {error_text(exc)}")
WAKING = "These examples come from the model service, which is still waking up. Refresh in a minute."

with tab_try:
    st.divider()
    st.subheader("Or try a held-out IMDB review")
    if gallery:
        st.caption("Short reviews from the IMDB **test** set (never seen in training) that the attention "
                   "model gets right with ≥90% confidence. Running one sends it through the live API and "
                   "logs the prediction and its attention vector to Supabase.")
        ex = gallery["examples"]
        idx = st.radio("Example", range(len(ex)),
                       format_func=lambda i: f"[{ex[i]['true']}] {snippet(ex[i])}")
        c1, c2 = st.columns([1, 4])
        live_both = c2.checkbox("Also run the plain LSTM", value=True)
        if c1.button("Run live", type="primary"):
            kinds = ["attention", "lstm"] if live_both else ["attention"]
            st.session_state.example_results = [
                r for k in kinds
                if (r := api_post("/predict_example", {"test_index": ex[idx]["test_index"], "model": k}))]
        results = [r for r in st.session_state.get("example_results", [])
                   if r["test_index"] == ex[idx]["test_index"]]
        if results:
            cols = st.columns(len(results))
            for res, col in zip(results, cols):
                with col:
                    show_prediction(res)
        else:
            show_card(ex[idx])
    else:
        st.info(WAKING)

with tab_compare:
    st.subheader("Reviews only one model got right")
    if gallery:
        st.caption("The shortest test reviews where the two matched models disagree and exactly one is "
                   "right, with the attention model's weights.")
        c1, c2 = st.columns(2)
        for col, key, title in ((c1, "attention_wins", "Attention right, plain LSTM wrong"),
                                (c2, "lstm_wins", "Plain LSTM right, attention wrong")):
            with col:
                st.markdown(f"**{title}**")
                for card in gallery[key]:
                    with st.expander(snippet(card, 8)):
                        show_card(card)
    else:
        st.info(WAKING)

# --------------------------------------------------------------------------- Failure Gallery
with tab_failures:
    st.header("Where the attention model is confidently wrong")
    st.markdown(
        "These are the attention model's most confident mistakes among all 25,000 test reviews, each "
        "with the weights it used. Three patterns recur:\n"
        "- **Sarcasm and \"so bad it's good\" praise.** A positive review built from words like "
        "*badly*, *poorly*, *awful*, or a negative one dressed up as *masterpiece … 10/10*. The "
        "attention query scores each word by its **content**; whether the word is meant ironically "
        "can only come from the LSTM state at that position, and a 128-number state rarely carries it.\n"
        "- **Labels that look wrong.** Some reviews read as glowing but are labeled negative (or the "
        "reverse). IMDB labels come from the star rating, and reviewers do not always write what they rate.\n"
        "- **Unusual word use.** e.g. *primary* used to mean *primitive*: the model only knows how a "
        "word was used in its 22,500 training reviews.\n\n"
        "**Why so confident?** The pooled vector is a weighted average of states, so a few strong "
        "sentiment words can dominate it and push the logit far from zero. Softmax confidence measures "
        "how far, not whether the reading is right; that is why the API abstains below 75% and why "
        "even 100% answers here are wrong.")
    if gallery:
        for card in gallery["failures"]:
            with st.expander(f"❌ said {card['attention_pred']} at {card['attention_confidence']:.1%} · "
                             f"true {card['true']} · {snippet(card)}"):
                show_card(card)
    else:
        st.info(WAKING)
