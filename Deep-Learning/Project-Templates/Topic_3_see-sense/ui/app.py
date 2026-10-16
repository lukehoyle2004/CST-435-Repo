"""Streamlit UI -- Cloud #1 (Streamlit Community Cloud).

A THIN client: no torch, no PIL, no model code. It
  * posts uploads to the FastAPI service on Render as multipart/form-data and
    renders the JSON it gets back (Grad-CAM arrives as base64 PNG),
  * reads `runs` and `predictions` straight from Supabase with the public ANON
    key (RLS allows SELECT only).

Secrets (st.secrets; see .streamlit/secrets.toml.example):
    API_URL, SUPABASE_URL, SUPABASE_ANON_KEY
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="See-Sense", page_icon="👁️", layout="wide")

API_URL = st.secrets["API_URL"].rstrip("/")
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
EMOJI = dict(airplane="✈️", automobile="🚗", bird="🐦", cat="🐈", deer="🦌",
             dog="🐕", frog="🐸", horse="🐎", ship="🚢", truck="🚚")
MODEL_COLORS = alt.Scale(domain=["baseline", "improved"], range=["#eb6834", "#2a78d6"])


# --------------------------------------------------------------------------- helpers
@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def b64_image(s: str) -> bytes:
    return base64.b64decode(s)


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
    with st.status("Waking the model service on Render (free tier cold start, up to a minute)…",
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
        status.update(label="The model service did not answer. Refresh the page to retry.", state="error")
    return False


@st.cache_data(ttl=300, show_spinner=False)
def load_runs() -> list[dict]:
    return supabase_anon().table("runs").select("*").order("created_at").execute().data


@st.cache_data(ttl=3600, show_spinner="Running both models on the gallery…")
def load_gallery() -> list[dict]:
    r = requests.get(f"{API_URL}/gallery", timeout=180)
    r.raise_for_status()
    return r.json()


def runs_by_name() -> dict[str, dict]:
    try:
        return {r["name"]: r for r in load_runs()}
    except Exception as exc:
        st.warning(f"Could not read runs from Supabase: {exc}")
        return {}


# --------------------------------------------------------------------------- header
st.title("👁️ See-Sense")
st.caption("Upload a photo; a CNN trained on 32×32 CIFAR-10 images names it and shows, with Grad-CAM, "
           "which pixels drove the decision. Only a SHA-256 hash of your image is logged, never the image.")

tab_concepts, tab_try, tab_compare, tab_recent, tab_card = st.tabs(
    ["📐 Concepts", "🔍 Try It", "⚖️ Compare Models", "🕒 Recent Classifications", "🪪 Model Card"])

# --------------------------------------------------------------------------- Concepts
with tab_concepts:
    st.header("How a CNN sees (and why it is not how you see)")
    st.markdown("**1. Convolution.** A kernel $K$ of size $k\\times k$ slides over a $C_{in}$-channel "
                "input $X$; each output value is a dot product plus a bias:")
    st.latex(r"Y_{c_{out}}[i,j] = b_{c_{out}} + \sum_{c=1}^{C_{in}} \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} "
             r"K_{c_{out},c}[u,v]\; X_c[\,s\,i+u-p,\; s\,j+v-p\,]")
    st.markdown("The same weights are reused at every position (**weight sharing**), so a 3×3 conv "
                "from 64 to 128 channels has $128\\,(64\\cdot 9+1)=73{,}856$ parameters whether the "
                "image is 32×32 or 1024×1024.")
    st.markdown("**2. Output size** with input $I$, kernel $k$, padding $p$, stride $s$:")
    st.latex(r"O = \left\lfloor \frac{I - k + 2p}{s} \right\rfloor + 1")
    st.markdown("Our convs use $k=3,\\,p=1,\\,s=1$, so $O = I$ (\"same\" padding); each 2×2 max-pool "
                "with $s=2$ halves it: $32 \\to 16 \\to 8 \\to 4$.")
    st.markdown("**3. Receptive field.** Layer $l$ with kernel $k_l$ and the product of earlier strides "
                "$j_{l-1}=\\prod_{m<l} s_m$ grows the field by $(k_l-1)\\,j_{l-1}$:")
    st.latex(r"r_l = r_{l-1} + (k_l - 1)\, j_{l-1}, \qquad j_l = j_{l-1}\, s_l, \qquad r_0 = 1,\; j_0 = 1")
    st.markdown("Three stacked 3×3, stride-1 convs: $r = 1 + 2 + 2 + 2 = 7$, the same view as one 7×7 "
                "kernel but with $3\\cdot 9C^2 = 27C^2$ weights instead of $49C^2$ and three ReLUs instead "
                "of one. In our network the pools double $j$, so the last conv (the Grad-CAM layer) has "
                "$r = 1+2+2+(1)+4+4+(2)+8+8 = 32$ — every 8×8 Grad-CAM cell can in principle see the whole "
                "32×32 image, but it is centered on a 4×4-pixel patch.")
    st.markdown("**4. Grad-CAM.** Take the last conv feature maps $A^k$ (128 maps of 8×8) and the class "
                "score $y^c$ before softmax. Weight each map by its average gradient, sum, and keep the "
                "positive part:")
    st.latex(r"\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j} \frac{\partial y^c}{\partial A^k_{ij}}, \qquad "
             r"L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\!\Big(\sum_k \alpha_k^c A^k\Big)")
    st.markdown("Because our head is global-average-pool → linear, $y^c = \\sum_k w^c_k \\frac{1}{Z}"
                "\\sum_{ij} A^k_{ij}$ (ignoring the last pool), so $\\alpha^c_k \\propto w^c_k$ and "
                "Grad-CAM reduces to the classic CAM: the heatmap is literally the evidence the final "
                "layer adds up. The ReLU keeps only evidence *for* class $c$. The 8×8 map is upsampled "
                "to the image, which is why the overlays look blobby.")
    st.info("**What a visitor should take away:** the model only ever sees a 32×32 thumbnail of the "
            "center square of your photo, it must pick one of 10 labels even for a person or a laptop, "
            "and Grad-CAM often shows it keying on background (water → ship, sky → airplane, grass → "
            "deer). That is texture and context statistics, not understanding.")

# --------------------------------------------------------------------------- Try It
with tab_try:
    left, right = st.columns([1, 2])
    with left:
        upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        model = st.radio("Model", ["improved", "baseline"], horizontal=True)
        tta = st.toggle("Flip test-time augmentation", value=True)
        st.caption("Tip: photos where one of the 10 CIFAR-10 objects fills the frame work best.")
    if upload is not None and wake_api():
        with st.spinner("Classifying…"):
            try:
                r = requests.post(f"{API_URL}/predict",
                                  files={"file": (upload.name, upload.getvalue(), upload.type)},
                                  data={"model": model, "tta": str(tta).lower()}, timeout=120)
                r.raise_for_status()
                res = r.json()
            except requests.RequestException as exc:
                st.error(f"Prediction failed: {error_text(exc)}")
                res = None
        if res:
            with right:
                st.subheader(f"{EMOJI[res['class_name']]} {res['class_name']} — {res['confidence']:.1%}")
                c1, c2, c3 = st.columns(3)
                c1.image(upload.getvalue(), caption="Your image", use_container_width=True)
                c2.image(b64_image(res["gradcam_png"]), use_container_width=True,
                         caption="Grad-CAM (dimmed margins were cropped away)")
                c3.image(b64_image(res["model_view_png"]), use_container_width=True,
                         caption="What the model sees (32×32) + Grad-CAM")
                probs = pd.DataFrame({"class": list(res["probabilities"]),
                                      "probability": list(res["probabilities"].values())})
                probs["top"] = probs["class"] == res["class_name"]
                st.altair_chart(alt.Chart(probs).mark_bar().encode(
                    x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    y=alt.Y("class:N", sort="-x", title=None),
                    color=alt.condition("datum.top", alt.value("#2a78d6"), alt.value("#b7c4d6")),
                    tooltip=["class", alt.Tooltip("probability:Q", format=".2%")]).properties(height=280),
                    use_container_width=True)
                st.table(pd.DataFrame(res["top5"]).assign(
                    probability=lambda d: d.probability.map("{:.2%}".format)).rename(
                    columns={"class_name": "top-5 class"}))
                st.caption(f"Served by run `{res['served_by_run_id']}` ({res['model']}) in "
                           f"{res['latency_ms']:.0f} ms · image hash `{res['image_sha256'][:16]}…` · "
                           + ("logged to Supabase (hash only)" if res["logged"]
                              else "⚠️ database unavailable, prediction not logged"))

# --------------------------------------------------------------------------- Compare Models
with tab_compare:
    runs = runs_by_name()
    if {"baseline", "improved"} <= runs.keys():
        b, i = runs["baseline"], runs["improved"]
        st.subheader("Headline numbers (from the Supabase `runs` table)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Improved test accuracy", f"{i['test_acc_tta']:.2f}%",
                  f"{i['test_acc_tta'] - b['test_acc_tta']:+.2f} pts vs baseline")
        m2.metric("Baseline test accuracy", f"{b['test_acc_tta']:.2f}%")
        m3.metric("Macro F1 (improved / baseline)", f"{i['macro_f1']:.3f} / {b['macro_f1']:.3f}")
        m4.metric("Same budget", f"{i['epochs']} epochs · batch {i['batch_size']}")
        table = pd.DataFrame([{
            "model": r["display_name"], "run_id": r["run_id"], "techniques": ", ".join(r["techniques"]) or "—",
            "epochs": r["epochs"], "params": r["params"], "lr": f"{r['lr']} ({r['lr_schedule']})",
            "train acc": r["train_acc"], "val acc": r["val_acc"], "test acc": r["test_acc"],
            "test acc (TTA)": r["test_acc_tta"], "macro F1": r["macro_f1"]} for r in (b, i)])
        st.dataframe(table, hide_index=True, use_container_width=True)

        st.subheader("Training curves")
        hist = pd.concat([pd.DataFrame(r["history"]).assign(model=r["name"]) for r in (b, i)])
        long = pd.concat([
            hist.assign(split="train", loss=hist.train_loss, accuracy=hist.train_acc),
            hist.assign(split="validation", loss=hist.val_loss, accuracy=hist.val_acc)])
        c1, c2 = st.columns(2)
        for metric, col in (("loss", c1), ("accuracy", c2)):
            col.altair_chart(alt.Chart(long).mark_line(strokeWidth=2).encode(
                x="epoch:Q", y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("model:N", scale=MODEL_COLORS),
                strokeDash=alt.StrokeDash("split:N"),
                tooltip=["model", "split", "epoch", alt.Tooltip(f"{metric}:Q", format=".3f")]
            ).properties(height=280, title=metric), use_container_width=True)

        st.subheader("Per-class precision and recall (test set)")
        pc = pd.DataFrame([dict(model=r["name"], cls=c, **v) for r in (b, i) for c, v in r["per_class"].items()])
        c1, c2 = st.columns(2)
        for metric, col in (("precision", c1), ("recall", c2)):
            col.altair_chart(alt.Chart(pc).mark_bar().encode(
                x=alt.X("cls:N", title=None, sort=CLASSES), xOffset="model:N",
                y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("model:N", scale=MODEL_COLORS),
                tooltip=["model", "cls", alt.Tooltip(f"{metric}:Q", format=".3f")]
            ).properties(height=260, title=metric), use_container_width=True)

        st.subheader("Confusion matrices (test set, row = true class)")
        cols = st.columns(2)
        for r, col in zip((b, i), cols):
            cm = pd.DataFrame([dict(true=CLASSES[t], pred=CLASSES[p], n=r["confusion_matrix"][t][p])
                               for t in range(10) for p in range(10)])
            base = alt.Chart(cm).encode(x=alt.X("pred:N", sort=CLASSES, title="predicted"),
                                        y=alt.Y("true:N", sort=CLASSES, title="true"))
            col.altair_chart((base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"),
                                                                       legend=None), tooltip=["true", "pred", "n"])
                              + base.mark_text(fontSize=10).encode(text="n:Q"))
                             .properties(height=380, title=r["display_name"]), use_container_width=True)

        st.subheader("Ablations: what each technique contributes")
        abl = [r for n, r in runs.items() if n.startswith("no_")]
        if abl:
            ab = pd.DataFrame([dict(
                technique=r["name"][3:].replace("_", " "), run_id=r["run_id"], test_acc=r["test_acc_tta"],
                contribution=round(i["test_acc_tta"] - r["test_acc_tta"], 2),
                gap=round(r["train_acc"] - r["val_acc"], 2), macro_f1=r["macro_f1"]) for r in abl]
            ).sort_values("contribution", ascending=False)
            st.caption(f"Leave-one-out: each run is the improved recipe with ONE technique removed, same seed "
                       f"({i['seed']}), {i['epochs']} epochs, batch {i['batch_size']} and data split. "
                       "Contribution = improved test accuracy − ablated test accuracy. "
                       "SQL: `select * from ablation_comparison;`")
            st.altair_chart(alt.Chart(ab).mark_bar().encode(
                x=alt.X("contribution:Q", title="test accuracy lost when removed (pts)"),
                y=alt.Y("technique:N", sort="-x", title=None),
                color=alt.condition(alt.datum.contribution > 0, alt.value("#4c78a8"), alt.value("#e45756")),
                tooltip=["technique", "run_id", alt.Tooltip("test_acc:Q", format=".2f"),
                         alt.Tooltip("contribution:Q", format="+.2f"), alt.Tooltip("gap:Q", format=".2f")]
            ).properties(height=230), use_container_width=True)
            st.dataframe(ab.rename(columns={
                "technique": "technique removed", "test_acc": "test acc (TTA)",
                "contribution": "contribution (pts)", "gap": "train − val gap", "macro_f1": "macro F1"}),
                hide_index=True, use_container_width=True)
        else:
            st.info("Ablation runs not seeded yet.")
    else:
        st.info("Run rows not found in Supabase yet (apply the migration and run `python -m db.seed_runs`).")

    st.subheader("Held-out gallery: both models, same test images")
    st.caption("Six test images the improved model gets right and six it gets wrong. Each cell shows "
               "the 32×32 input with that model's Grad-CAM.")
    if wake_api():
        try:
            gallery = load_gallery()
        except requests.RequestException as exc:
            st.error(f"Could not load the gallery: {error_text(exc)}")
            gallery = []
        for start in range(0, len(gallery), 6):
            cols = st.columns(6)
            for item, col in zip(gallery[start:start + 6], cols):
                with col:
                    st.image(b64_image(item["image_png"]), caption=f"true: {item['true_class']}",
                             use_container_width=True)
                    for name in ("baseline", "improved"):
                        m = item["models"][name]
                        st.image(b64_image(m["gradcam_png"]), use_container_width=True,
                                 caption=f"{'✅' if m['correct'] else '❌'} {name}: {m['class_name']} "
                                         f"({m['confidence']:.0%})")

# --------------------------------------------------------------------------- Recent Classifications
with tab_recent:
    st.caption("Read directly from Supabase with the public anon key: `predictions` joined to `runs`. "
               "Only hashes, labels and probabilities exist there, never images.")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
    try:
        rows = (supabase_anon().table("predictions")
                .select("id,created_at,image_sha256,predicted_class,confidence,top5,tta,latency_ms,"
                        "served_by_run_id,runs(name,display_name,test_acc_tta)")
                .order("created_at", desc=True).limit(500).execute().data)
    except Exception as exc:
        rows = None
        st.warning(f"Supabase is unreachable right now ({exc}). Try again in a moment.")
    if rows == []:
        st.info("No classifications yet. Try one in the Try It tab.")
    elif rows:
        df = pd.DataFrame([{
            "id": r["id"], "time (UTC)": pd.to_datetime(r["created_at"]).strftime("%Y-%m-%d %H:%M:%S"),
            "predicted": r["predicted_class"], "confidence": r["confidence"],
            "runner-up": r["top5"][1]["class_name"] if len(r["top5"]) > 1 else "",
            "model": r["runs"]["name"] if r.get("runs") else "", "run_id": r["served_by_run_id"],
            "image hash": r["image_sha256"][:12] + "…", "TTA": r["tta"], "latency ms": r["latency_ms"],
        } for r in rows])
        models_seen = sorted(df["model"].unique())
        pick = st.multiselect("Model", models_seen, default=models_seen)
        view = df[df["model"].isin(pick)]
        c1, c2, c3 = st.columns(3)
        c1.metric("Classifications", len(view))
        c2.metric("Mean confidence", f"{view['confidence'].mean():.1%}" if len(view) else "—")
        c3.metric("Distinct images", view["image hash"].nunique())
        st.dataframe(view, hide_index=True, use_container_width=True,
                     column_config={"confidence": st.column_config.ProgressColumn(
                         "confidence", format="%.2f", min_value=0, max_value=1)})
        st.subheader("Class distribution")
        st.altair_chart(alt.Chart(view).mark_bar().encode(
            x=alt.X("predicted:N", sort=CLASSES, title=None), y=alt.Y("count():Q", title="classifications"),
            color=alt.Color("model:N", scale=MODEL_COLORS), tooltip=["predicted", "model", "count()"]
        ).properties(height=280), use_container_width=True)

        st.subheader("Drill into the run behind a classification")
        pid = st.selectbox("Prediction id", view["id"].tolist())
        if pid is not None:
            run_id = view.loc[view["id"] == pid, "run_id"].iloc[0]
            run = next((r for r in load_runs() if r["run_id"] == run_id), None)
            if run:
                st.json({k: run[k] for k in ("run_id", "display_name", "architecture", "techniques", "epochs",
                                             "batch_size", "lr", "lr_schedule", "weight_decay", "params",
                                             "best_epoch", "val_acc", "test_acc_tta", "macro_f1",
                                             "weights_path")})

# --------------------------------------------------------------------------- Model Card
with tab_card:
    card = ROOT.parent / "MODEL_CARD.md"
    st.markdown(card.read_text(encoding="utf-8") if card.exists() else "MODEL_CARD.md not found.")
