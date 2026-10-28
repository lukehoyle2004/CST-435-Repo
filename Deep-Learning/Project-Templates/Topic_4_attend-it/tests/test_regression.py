"""Regression: the exported models must reproduce tests/fixtures/reference.json.

Covers the whole serving path (tokenizer -> vocab -> packing/masking -> weights ->
softmax -> attention). Regenerate with `python -m scripts.make_reference` only
after deliberately retraining.
"""
from __future__ import annotations

import json

import pytest

from seqsight import MODELS_DIR
from seqsight.inference import classify_text, load_vocab
from seqsight.model import MODEL_KINDS, load_exported
from tests.conftest import ROOT

REF = json.loads((ROOT / "tests" / "fixtures" / "reference.json").read_text())
VOCAB = load_vocab()
MODELS = {k: load_exported(MODELS_DIR / k) for k in MODEL_KINDS}


def test_vocab_is_the_one_the_reference_was_made_with():
    assert VOCAB.sha256() == REF["vocab_sha256"]
    assert all(cfg["vocab_sha256"] == REF["vocab_sha256"] for _, cfg in MODELS.values())


@pytest.mark.parametrize("case", REF["cases"], ids=lambda c: c["text"][:30])
@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_models_reproduce_reference_outputs(case, kind):
    model, config = MODELS[kind]
    want = case[kind]
    got = classify_text(model, VOCAB, case["text"])
    assert config["run_id"] == want["run_id"]
    assert got["tokens"] == want["tokens"]
    assert got["class_name"] == want["class_name"]
    for c, p in want["probabilities"].items():
        assert got["probabilities"][c] == pytest.approx(p, abs=1e-5)
    if want["attention"] is None:
        assert got["attention"] is None
    else:
        assert got["attention"] == pytest.approx(want["attention"], abs=1e-5)


def test_reference_reviews_get_the_obvious_answers():
    # Sanity on top of exact reproduction: clear-cut reviews go the right way for both models.
    by_text = {c["text"][:20]: c for c in REF["cases"]}
    for kind in MODEL_KINDS:
        assert by_text["An absolutely wonder"][kind]["class_name"] == "positive"
        assert by_text["This is the worst mo"][kind]["class_name"] == "negative"


def test_exported_metrics_match_runs_json():
    runs = {r["name"]: r for r in json.loads((MODELS_DIR / "runs.json").read_text())}
    for kind, (_, cfg) in MODELS.items():
        assert runs[kind]["run_id"] == cfg["run_id"]
        assert runs[kind]["test_acc"] == pytest.approx(cfg["test_acc"], abs=1e-9)
    assert runs["attention"]["test_acc"] > runs["lstm"]["test_acc"]
