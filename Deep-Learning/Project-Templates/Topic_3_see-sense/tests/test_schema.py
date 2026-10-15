"""Request-schema tests for /predict and the list endpoints."""
from __future__ import annotations

from .conftest import png_bytes


def test_predict_requires_a_file(client):
    assert client.post("/predict", data={"model": "improved"}).status_code == 422


def test_predict_rejects_non_boolean_tta(client):
    r = client.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")}, data={"tta": "maybe"})
    assert r.status_code == 422


def test_list_limits_are_validated(client):
    assert client.get("/runs", params={"limit": 0}).status_code == 422
    assert client.get("/predictions", params={"limit": 10_000}).status_code == 422
