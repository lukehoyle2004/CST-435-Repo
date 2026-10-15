"""Regression test: a fixed reference image must keep getting the same answer.

tests/fixtures/reference.png is CIFAR-10 test image #<see reference.json>, and
reference.json holds the probabilities each shipped model gave it when exported.
If a retrain, a preprocessing change, or a torch upgrade changes them, this fails.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from seesense import MODELS_DIR
from seesense.inference import explain, grad_cam, preprocess
from seesense.model import load_exported

FIXTURES = Path(__file__).parent / "fixtures"
REF = json.loads((FIXTURES / "reference.json").read_text())


@pytest.mark.parametrize("name", ["improved", "baseline"])
def test_reference_image_prediction_is_stable(name):
    model, config = load_exported(MODELS_DIR / name)
    expected = REF["models"][name]
    assert config["run_id"] == expected["run_id"]
    res = explain(model, Image.open(FIXTURES / "reference.png").convert("RGB"))
    assert res["class_name"] == expected["class_name"]
    got = np.array([res["probabilities"][c] for c in expected["probabilities"]])
    want = np.array(list(expected["probabilities"].values()))
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_gradcam_map_shape_and_range():
    model, _ = load_exported(MODELS_DIR / "improved")
    x = preprocess(Image.open(FIXTURES / "reference.png").convert("RGB"))
    assert x.shape == (3, 32, 32) and x.dtype == torch.uint8
    cam = grad_cam(model, x, 0)
    assert cam.shape == (8, 8)
    assert cam.min() >= 0 and cam.max() <= 1
