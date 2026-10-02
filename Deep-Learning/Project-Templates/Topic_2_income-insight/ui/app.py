"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job, prediction, bias audit, and activation comparison is an
    HTTPS call to the FastAPI service (no torch/sklearn import here),
  * the ONLY database access is a read-only anon-key query against the `runs`
    table for the 'Run History' tab (no SQL writes here).

Configuration comes from st.secrets (see .streamlit/secrets.toml.example):
    API_URL              -> your Render.com base URL
    SUPABASE_URL         -> https://<ref>.supabase.co
    SUPABASE_ANON_KEY    -> the public anon key (safe to ship to the browser)
"""
from __future__ import annotations

import io
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from supabase import create_client

API_URL = st.secrets["API_URL"].rstrip("/")
PROJECT_DIR = Path(__file__).resolve().parent.parent  # Streamlit Cloud runs from repo root
BATCH_LIMIT = 5000  # rows per /predict_batch call (API limit)

st.set_page_config(page_title="Income-Insight", page_icon="💼", layout="wide")
st.title("💼 Income-Insight — Income Classification with a Deep MLP")
st.caption(
    "Streamlit (this UI) → FastAPI on Render (MLP + sklearn) → Supabase (UCI Adult data, "
    "runs, predictions). Three clouds."
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def api_get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict, timeout: int = 300):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def error_text(exc: Exception) -> str:
    """The API's 'detail' message when there is one, else the exception."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            return f"{exc.response.status_code}: {exc.response.json().get('detail')}"
        except ValueError:
            pass
    return str(exc)


@st.cache_data(ttl=600)
def feature_schema():
    return api_get("/schema")


def recent_runs() -> list[dict]:
    try:
        return api_get("/runs")
    except requests.RequestException as exc:
        st.error(f"Could not load runs: {error_text(exc)}")
        return []


def run_picker(label: str, key: str, only_ok: bool = True) -> int | None:
    """Select box of recent runs on the real data, newest first."""
    runs = [r for r in recent_runs() if r["data_source"] == "adult_income"]
    if only_ok:
        runs = [r for r in runs if r["status"] == "ok"]
    if not runs:
        st.info("No trained runs on the UCI Adult data yet. Train one on the Train tab.")
        return None
    labels = {
        r["id"]: (f"run {r['id']} · {r['activation']} · {r['n_layers']}×{r['hidden_dim']} · "
                  f"lr {r['lr']} · test acc {r['accuracy']:.3f}" if r["accuracy"] is not None
                  else f"run {r['id']} · {r['status']}")
        for r in runs
    }
    default = st.session_state.get("last_run_id")
    ids = list(labels)
    index = ids.index(default) if default in ids else 0
    return st.selectbox(label, ids, index=index, format_func=labels.get, key=key)


def fmt(v, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def feature_form(schema: dict, key: str) -> dict:
    """Input widgets generated from the API's /schema contract."""
    defaults = {"age": 38, "education_num": 10, "capital_gain": 0, "capital_loss": 0,
                "hours_per_week": 40}
    record: dict = {}
    cols = st.columns(len(schema["numeric_features"]))
    for col, feat in zip(cols, schema["numeric_features"]):
        record[feat] = col.number_input(feat, value=defaults.get(feat, 0), step=1,
                                        min_value=0, key=f"{key}-{feat}")
    cols = st.columns(len(schema["categorical_features"]))
    for col, feat in zip(cols, schema["categorical_features"]):
        record[feat] = col.selectbox(feat, schema["categories"][feat], key=f"{key}-{feat}")
    return record


try:
    SCHEMA = feature_schema()
except requests.RequestException as exc:
    SCHEMA = None
    st.error(f"The API is unreachable ({error_text(exc)}). Render's free tier may be waking "
             "up — wait ~60 s and refresh.")


(concepts_tab, train_tab, row_tab, csv_tab, perf_tab, bias_tab, history_tab,
 card_tab) = st.tabs(["Concepts", "Train", "Score a Row", "Score a CSV",
                      "Model Performance", "Bias Audit", "Run History", "Model Card"])

# ---------------------------------------------------------------------------
# Concepts (rendered from docs/CONCEPTS.md so it can be edited as markdown)
# ---------------------------------------------------------------------------
with concepts_tab:
    concepts_md = PROJECT_DIR / "docs" / "CONCEPTS.md"
    if concepts_md.exists():
        st.markdown(concepts_md.read_text(encoding="utf-8"))
    else:
        st.warning("docs/CONCEPTS.md not found.")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
with train_tab:
    st.header("Train a model on UCI Adult")
    st.caption(
        "Every run uses the same fixed, stratified 70/15/15 train/val/test split stored in "
        "Supabase, so runs are directly comparable. Defaults are sized for Render's free tier."
    )
    activations = SCHEMA["activations"] if SCHEMA else ["relu", "tanh", "sigmoid"]
    c = st.columns(4)
    n_layers = c[0].number_input("Hidden layers", value=2, min_value=1, max_value=6)
    hidden_dim = c[1].number_input("Units per layer", value=32, min_value=1, max_value=512, step=8)
    activation = c[2].selectbox("Activation", activations, index=activations.index("tanh"))
    lr = c[3].number_input("Learning rate", value=0.005, format="%.4f", min_value=0.0001)
    c = st.columns(4)
    batch_size = c[0].number_input("Batch size", value=256, min_value=1)
    epochs = c[1].number_input("Epochs", value=15, min_value=1, max_value=500)
    seed = c[2].number_input("Seed", value=0, step=1)
    max_rows = c[3].number_input("Max training rows (0 = all ~34k)", value=15000,
                                 min_value=0, step=1000)
    include_protected = st.checkbox(
        "Also use sex and race as model inputs (for the bias-audit comparison)", value=False
    )
    train_body = {
        "n_layers": int(n_layers), "hidden_dim": int(hidden_dim), "activation": activation,
        "lr": float(lr), "batch_size": int(batch_size), "epochs": int(epochs),
        "seed": int(seed), "max_train_rows": int(max_rows) or None,
        "include_protected": include_protected,
    }

    resp = None
    if st.button("Run training", type="primary"):
        with st.spinner("Training on the FastAPI service (≈20–60 s on Render's free tier)..."):
            try:
                resp = api_post("/train", train_body)
            except requests.RequestException as exc:
                # A timeout here must not crash the other tabs. The run may still
                # finish on the API and appear in Run History.
                st.error(f"Training request failed: {error_text(exc)}. If it timed out, "
                         "check Run History, or use fewer epochs / training rows.")
    if resp and resp["status"] == "diverged":
        st.error(
            f"Run {resp['run_id']} diverged (loss went NaN/Inf or ended higher than it "
            "started). Its metrics are stored as NULL, not as a number. Try a smaller "
            "learning rate."
        )
    elif resp:
        st.session_state["last_run_id"] = resp["run_id"]
        m = resp["metrics"]
        st.success(f"Run {resp['run_id']} complete — final training loss "
                   f"{fmt(resp['final_loss'], 4)}, {resp['n_test_predictions_logged']} test "
                   "predictions logged for the bias audit. See Model Performance.")
        mc = st.columns(6)
        mc[0].metric("Train acc", fmt(resp["train_accuracy"]))
        mc[1].metric("Val acc", fmt(resp["val_accuracy"]))
        mc[2].metric("Test acc", fmt(m["accuracy"]))
        mc[3].metric("Test F1 (>50K)", fmt(m["f1"]))
        mc[4].metric("Test ROC-AUC", fmt(m["roc_auc"]))
        mc[5].metric("Test recall (>50K)", fmt(m["recall"]))

# ---------------------------------------------------------------------------
# Score a Row (form auto-generated from the API's /schema contract)
# ---------------------------------------------------------------------------
with row_tab:
    st.header("Score a single person")
    if SCHEMA:
        run_id = run_picker("Model", key="row-run")
        record = feature_form(SCHEMA, key="row")
        threshold = st.slider(
            "Decision threshold: predict >50K when P(>50K) ≥ threshold",
            0.05, 0.95, 0.50, 0.05, key="row-threshold",
            help="Higher = fewer >50K predictions (more precision, less recall).",
        )
        if run_id and st.button("Score", type="primary"):
            try:
                out = api_post("/predict", {"run_id": run_id, "features": record,
                                            "threshold": threshold}, timeout=60)
            except requests.RequestException as exc:
                st.error(f"Prediction failed: {error_text(exc)}")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Predicted income", out["income"])
                c2.metric("P(>50K)", f"{out['proba']:.3f}")
                st.caption(f"Threshold {out['threshold']:.2f}; logged to Supabase by the API.")

# ---------------------------------------------------------------------------
# Score a CSV
# ---------------------------------------------------------------------------
with csv_tab:
    st.header("Score a CSV file")
    if SCHEMA:
        required = SCHEMA["numeric_features"] + SCHEMA["categorical_features"]
        st.markdown("Required columns: " + ", ".join(f"`{c}`" for c in required))
        template = pd.DataFrame([
            {"age": 45, "education_num": 14, "capital_gain": 5000, "capital_loss": 0,
             "hours_per_week": 50, "workclass": "Private", "marital_status": "Married-civ-spouse",
             "occupation": "Exec-managerial", "relationship": "Husband",
             "native_country": "United-States"},
            {"age": 23, "education_num": 9, "capital_gain": 0, "capital_loss": 0,
             "hours_per_week": 30, "workclass": "Private", "marital_status": "Never-married",
             "occupation": "Other-service", "relationship": "Own-child",
             "native_country": "United-States"},
        ])[required]
        st.download_button("Download a template CSV", template.to_csv(index=False),
                           "income_insight_template.csv", "text/csv")

        run_id = run_picker("Model", key="csv-run")
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05, key="csv-threshold")
        upload = st.file_uploader("CSV file", type="csv")
        if upload is not None and run_id:
            df = pd.read_csv(upload)
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            elif st.button(f"Score {len(df)} rows", type="primary"):
                scored = []
                try:
                    for start in range(0, len(df), BATCH_LIMIT):
                        chunk = df.iloc[start:start + BATCH_LIMIT][required]
                        out = api_post("/predict_batch", {
                            "run_id": run_id, "threshold": threshold,
                            "records": chunk.to_dict(orient="records"),
                        })
                        scored.extend(out["predictions"])
                except requests.RequestException as exc:
                    st.error(f"Scoring failed: {error_text(exc)}")
                else:
                    result = df.copy()
                    result["p_gt_50k"] = [p["proba"] for p in scored]
                    result["predicted_income"] = [p["income"] for p in scored]
                    st.success(f"Scored {len(result)} rows — "
                               f"{(result['predicted_income'] == '>50K').mean():.1%} predicted >50K.")
                    st.dataframe(result, use_container_width=True)
                    buf = io.StringIO()
                    result.to_csv(buf, index=False)
                    st.download_button("Download scored CSV", buf.getvalue(),
                                       "scored.csv", "text/csv")

# ---------------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------------
with perf_tab:
    st.header("Model performance")
    run_id = run_picker("Run", key="perf-run")
    detail = None
    if run_id:
        try:
            detail = api_get(f"/runs/{run_id}")
        except requests.RequestException as exc:
            st.error(f"Could not load run: {error_text(exc)}")

    if detail and detail.get("confusion"):
        st.caption(f"{detail['n_train']} train / {detail['n_val']} val / {detail['n_test']} "
                   f"test rows · {detail['activation']} · {detail['n_layers']} hidden layers "
                   f"× {detail['hidden_dim']} · lr {detail['lr']} · {detail['epochs']} epochs")
        mc = st.columns(6)
        mc[0].metric("Train acc", fmt(detail["train_accuracy"]))
        mc[1].metric("Val acc", fmt(detail["val_accuracy"]))
        mc[2].metric("Test acc", fmt(detail["accuracy"]))
        mc[3].metric("Test ROC-AUC", fmt(detail["roc_auc"]))
        mc[4].metric("Brier score", fmt(detail["brier"]))
        mc[5].metric("ECE", fmt(detail["ece"]))

        left, right = st.columns(2)
        with left:
            st.subheader("Per-class precision / recall / F1 (test)")
            report = pd.DataFrame(detail["class_report"]).T
            report["support"] = report["support"].astype(int)
            st.dataframe(report.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                              "f1": "{:.3f}"}), use_container_width=True)
            harder = min(detail["class_report"], key=lambda k: detail["class_report"][k]["f1"])
            st.caption(f"Harder class (lower F1): **{harder}**.")

            st.subheader("Confusion matrix (test, threshold 0.5)")
            cm = detail["confusion"]
            cm_df = pd.DataFrame([
                {"actual": "<=50K", "predicted": "<=50K", "count": cm["tn"]},
                {"actual": "<=50K", "predicted": ">50K", "count": cm["fp"]},
                {"actual": ">50K", "predicted": "<=50K", "count": cm["fn"]},
                {"actual": ">50K", "predicted": ">50K", "count": cm["tp"]},
            ])
            base = alt.Chart(cm_df).encode(
                x=alt.X("predicted:N", title="Predicted"),
                y=alt.Y("actual:N", title="Actual"),
            )
            st.altair_chart(
                base.mark_rect().encode(color=alt.Color("count:Q", legend=None))
                + base.mark_text(fontSize=18).encode(text="count:Q"),
                use_container_width=True,
            )

        with right:
            st.subheader("Calibration (reliability diagram)")
            cal = pd.DataFrame(detail["calibration"])
            diag = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
            st.altair_chart(
                alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray")
                .encode(x="x:Q", y="y:Q")
                + alt.Chart(cal).mark_line(point=True).encode(
                    x=alt.X("mean_pred:Q", title="Mean predicted P(>50K)",
                            scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("frac_positive:Q", title="Observed fraction >50K",
                            scale=alt.Scale(domain=[0, 1])),
                    tooltip=["mean_pred", "frac_positive", "count"],
                ),
                use_container_width=True,
            )
            st.caption("Points on the dashed diagonal = perfectly calibrated probabilities.")

            st.subheader("Loss per epoch")
            curves = pd.DataFrame({"train": detail["loss_history"],
                                   "validation": detail["val_loss_history"]})
            curves.index = curves.index + 1
            curves.index.name = "epoch"
            st.line_chart(curves, x_label="epoch", y_label="BCE loss")

        st.subheader("Permutation importance (drop in validation ROC-AUC when shuffled)")
        imp = pd.DataFrame(detail["perm_importance"])
        st.altair_chart(
            alt.Chart(imp).mark_bar().encode(
                x=alt.X("importance:Q", title="ROC-AUC drop"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature", "importance", "std"],
            ),
            use_container_width=True,
        )
    elif detail:
        st.info("This run predates the evaluation upgrade; train a new run.")

    st.divider()
    st.subheader("Activation-function comparison (SQL view over the runs table)")
    st.caption(
        "Rows that share every control (layers, width, lr, batch, epochs, seed, training "
        "rows) differ only in activation — a controlled experiment."
    )
    if st.button("Run relu / tanh / sigmoid with the Train tab's settings"):
        progress = st.progress(0.0)
        for i, act in enumerate(["relu", "tanh", "sigmoid"]):
            try:
                api_post("/train", {**train_body, "activation": act})
            except requests.RequestException as exc:
                st.error(f"{act} failed: {error_text(exc)}")
            progress.progress((i + 1) / 3, text=f"{act} done")
    try:
        comparison = pd.DataFrame(api_get("/activation_comparison"))
    except requests.RequestException as exc:
        comparison = pd.DataFrame()
        st.error(f"Could not load comparison: {error_text(exc)}")
    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.info("No successful runs on the UCI Adult data yet.")

# ---------------------------------------------------------------------------
# Bias Audit (FPR / FNR by protected attribute, computed in SQL)
# ---------------------------------------------------------------------------
with bias_tab:
    st.header("Bias audit: error rates by protected attribute")
    st.markdown(
        "- **FPR** (false-positive rate) = share of people truly earning ≤50K whom the model "
        "predicts >50K.\n"
        "- **FNR** (false-negative rate) = share of people truly earning >50K whom the model "
        "misses.\n\n"
        "Computed by the `bias_audit` SQL view, joining each run's test-split predictions to "
        "the true labels and protected attributes in `adult_income`."
    )
    run_id = run_picker("Run", key="bias-run")
    attribute = st.radio("Protected attribute", ["sex", "race"], horizontal=True)
    if run_id:
        try:
            audit = api_get("/bias_audit", run_id=run_id, attribute=attribute)
        except requests.RequestException as exc:
            st.error(f"Could not load the audit: {error_text(exc)}")
        else:
            groups = pd.DataFrame(audit["groups"])
            c1, c2 = st.columns(2)
            c1.metric("FPR gap (max − min)", fmt(audit["fpr_gap"]))
            c2.metric("FNR gap (max − min)", fmt(audit["fnr_gap"]))
            st.dataframe(
                groups.rename(columns={"grp": attribute}).style.format(
                    {"positive_rate": "{:.3f}", "base_rate": "{:.3f}", "fpr": "{:.3f}",
                     "fnr": "{:.3f}"}),
                use_container_width=True, hide_index=True,
            )
            rates = groups.melt(id_vars="grp", value_vars=["fpr", "fnr"],
                                var_name="rate", value_name="value")
            st.altair_chart(
                alt.Chart(rates).mark_bar().encode(
                    x=alt.X("grp:N", title=attribute),
                    xOffset="rate:N",
                    y=alt.Y("value:Q", title="rate"),
                    color=alt.Color("rate:N", title=None),
                    tooltip=["grp", "rate", alt.Tooltip("value:Q", format=".3f")],
                ),
                use_container_width=True,
            )
            if attribute == "race":
                st.caption("Small groups (e.g. 'Other', 'Amer-Indian-Eskimo') have few test "
                           "rows, so their rates are noisy — check `n` before concluding.")
            st.caption(
                "Mitigations to compare: (1) train without vs. with protected inputs "
                "(Train tab checkbox) — proxies such as `relationship` can still leak sex; "
                "(2) per-group decision thresholds to equalize FPR or FNR; (3) reweighting "
                "or resampling the training data. See the README for the discussion."
            )

# ---------------------------------------------------------------------------
# Run History  (read-only anon-key query against Supabase)
# ---------------------------------------------------------------------------
with history_tab:
    st.header("Run History")
    st.caption("Read directly from Supabase with the anon key — no API call.")
    try:
        rows = (
            supabase_anon()
            .table("runs")
            .select(
                "id,data_source,activation,n_layers,hidden_dim,lr,batch_size,epochs,seed,"
                "max_train_rows,include_protected,status,final_loss,train_accuracy,"
                "val_accuracy,accuracy,precision,recall,f1,roc_auc,brier,ece,created_at"
            )
            .order("created_at", desc=True)
            .limit(50)
            .execute()
            .data
        )
        if rows:
            # Diverged runs have NULL metrics; they show as empty cells, not numbers.
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No runs yet. Train a model on the Train tab.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read runs: {exc}")

# ---------------------------------------------------------------------------
# Model Card  (rendered from a markdown file in the repo)
# ---------------------------------------------------------------------------
with card_tab:
    st.header("Model Card")
    try:
        # Resolve relative to this file: Streamlit Cloud runs from the repo root.
        with open(PROJECT_DIR / "MODEL_CARD.md", "r", encoding="utf-8") as fh:
            st.markdown(fh.read())
    except FileNotFoundError:
        st.warning("MODEL_CARD.md not found.")
