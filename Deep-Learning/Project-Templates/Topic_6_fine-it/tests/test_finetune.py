"""The training pipeline is reproducible and keeps its label contract.

Trains the real DistilBERT base on a small stratified slice (64 messages, one
epoch) twice with the same seed and requires identical predictions -- the
property that makes each row of the experiment store re-runnable. Takes about
a minute on a laptop CPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("datasets")

from shared.queues import QUEUE_NAMES  # noqa: E402
from src.finetune import Config, load_splits, predict_proba, subsample, train_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus.csv"
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="data/corpus.csv missing (run scripts/prepare_data.py)")


@pytest.fixture(scope="module")
def tiny_splits():
    s = load_splits(CORPUS)
    return {"train": subsample(s["train"], 64, seed=0),
            "validation": subsample(s["validation"], 64, seed=0),
            "test": s["test"].head(32)}


def _train(splits, out: Path):
    cfg = Config(run_name="repro", epochs=1, batch_size=16, n_train=None, seed=7)
    res = train_config(cfg, splits, out)
    return res, predict_proba(res["model"], res["tokenizer"], splits["test"]["text"].tolist())


def test_same_seed_same_model_and_label_contract(tiny_splits, tmp_path):
    from transformers import AutoConfig

    res_a, p_a = _train(tiny_splits, tmp_path / "a")
    res_b, p_b = _train(tiny_splits, tmp_path / "b")
    assert res_a["val_macro_f1"] == res_b["val_macro_f1"]
    np.testing.assert_allclose(p_a, p_b, atol=1e-5)
    cfg = AutoConfig.from_pretrained(tmp_path / "a")
    assert [cfg.id2label[i] for i in range(cfg.num_labels)] == QUEUE_NAMES
