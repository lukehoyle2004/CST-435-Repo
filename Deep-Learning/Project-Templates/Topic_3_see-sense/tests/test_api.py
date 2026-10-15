"""/predict smoke + Grad-CAM shape tests, the other endpoints, and the Supabase-outage path."""
import base64
import hashlib
import io

import pytest
from PIL import Image

from seesense import CLASSES

from .conftest import png_bytes


def post_image(client, data: bytes, **form):
    return client.post("/predict", files={"file": ("test.png", data, "image/png")}, data=form)


def decode_png(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s)))


def test_predict_multipart_smoke(client):
    data = png_bytes()
    r = post_image(client, data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["class_name"] in CLASSES
    assert body["class_name"] == CLASSES[body["label"]]
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert len(body["top5"]) == 5
    assert body["top5"][0]["class_name"] == body["class_name"]
    assert body["image_sha256"] == hashlib.sha256(data).hexdigest()
    assert body["logged"] is True


def test_predict_logs_hash_never_pixels(client):
    data = png_bytes(seed=1)
    post_image(client, data)
    pred, meta = client.store.predictions[-1], client.store.image_metadata[-1]
    assert pred["image_sha256"] == meta["image_sha256"] == hashlib.sha256(data).hexdigest()
    assert pred["served_by_run_id"] in client.store.runs
    assert (meta["width"], meta["height"]) == (64, 48)
    for row in (pred, meta):
        assert not any(isinstance(v, (bytes, bytearray)) for v in row.values())
        assert not any(isinstance(v, str) and len(v) > 200 for v in row.values())


@pytest.mark.parametrize("size", [(64, 48), (40, 90), (32, 32)])
def test_gradcam_overlay_matches_input_dimensions(client, size):
    r = post_image(client, png_bytes(*size))
    body = r.json()
    overlay = decode_png(body["gradcam_png"])
    assert overlay.size == size == (body["width"], body["height"])
    assert tuple(body["gradcam_size"]) == size
    assert decode_png(body["model_view_png"]).size == (224, 224)


def test_gradcam_overlay_capped_for_huge_images(client):
    buf = io.BytesIO()
    Image.new("RGB", (2048, 1024), (40, 120, 200)).save(buf, format="PNG")  # compresses to a few KB
    body = post_image(client, buf.getvalue()).json()
    assert (body["width"], body["height"]) == (2048, 1024)
    assert decode_png(body["gradcam_png"]).size == (1024, 512)   # same aspect ratio, capped


def test_predict_rejects_uploads_over_5mb(client):
    assert post_image(client, png_bytes(2048, 1024)).status_code == 413   # noise ~6 MB


def test_baseline_model_selectable(client):
    body = post_image(client, png_bytes(), model="baseline").json()
    runs = client.store.runs
    assert runs[body["served_by_run_id"]]["name"] == "baseline"


@pytest.mark.parametrize("data,status", [(b"", 400), (b"not an image", 415)])
def test_predict_rejects_bad_uploads(client, data, status):
    assert post_image(client, data).status_code == status


def test_predict_rejects_unknown_model(client):
    assert post_image(client, png_bytes(), model="resnet152").status_code == 422


def test_meta_endpoints(client):
    assert client.get("/classes").json() == {"classes": CLASSES}
    health = client.get("/healthz").json()
    assert health["status"] == "ok" and set(health["models_loaded"]) == {"baseline", "improved"}
    assert "torch" in client.get("/version").json()
    assert {m["name"] for m in client.get("/models").json()} == {"baseline", "improved"}
    assert "predict" in {t["name"] for t in client.get("/openapi.json").json()["tags"]}


def test_runs_and_predictions(client):
    runs = client.get("/runs").json()
    assert {r["name"] for r in runs} == {"baseline", "improved"}
    assert client.get(f"/runs/{runs[0]['run_id']}").status_code == 200
    assert client.get("/runs/nope").status_code == 404
    post_image(client, png_bytes())
    preds = client.get("/predictions").json()
    assert preds and preds[0]["runs"]["name"] == "improved"


def test_gallery_has_right_and_wrong_examples(client):
    items = client.get("/gallery").json()
    right = sum(i["models"]["improved"]["correct"] for i in items)
    assert right >= 5 and len(items) - right >= 5


# ------------------------------------------------------------------ Supabase outage
def test_predict_still_works_when_supabase_is_down(outage_client):
    r = post_image(outage_client, png_bytes())
    assert r.status_code == 200
    assert r.json()["logged"] is False and r.json()["prediction_id"] is None


@pytest.mark.parametrize("path", ["/runs", "/runs/abc", "/predictions"])
def test_db_endpoints_return_503_when_supabase_is_down(outage_client, path):
    r = outage_client.get(path)
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]


def test_healthz_reports_db_down_but_stays_up(outage_client):
    body = outage_client.get("/healthz").json()
    assert body["status"] == "ok" and body["db"] is False
