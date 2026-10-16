"""Model-definition tests: the exported runs match the controlled-comparison design.

Training itself runs offline (80+ min on a CPU), so these check the shipped
artifacts instead: same budget for both runs, and the baseline really is plain.
"""
from __future__ import annotations

import json

import torch.nn as nn

from seesense import MODELS_DIR
from seesense.model import count_parameters, load_exported

RUNS = {r["name"]: r for r in json.loads((MODELS_DIR / "runs.json").read_text())}


def test_matched_budget():
    b, i = RUNS["baseline"], RUNS["improved"]
    for key in ("epochs", "batch_size", "seed", "train_images", "val_images", "test_images"):
        assert b[key] == i[key], key


def test_baseline_is_plain_cnn():
    model, config = load_exported(MODELS_DIR / "baseline")
    assert config["baseline"] and not config["augment"]["flip"] and config["weight_decay"] == 0
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert all(m.p == 0 for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d)))
    assert sum(isinstance(m, nn.Conv2d) for m in model.modules()) >= 2


def test_improved_uses_at_least_three_techniques():
    model, _ = load_exported(MODELS_DIR / "improved")
    assert len(RUNS["improved"]["techniques"]) >= 3
    assert any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def test_param_counts_match_runs():
    for name in ("baseline", "improved"):  # only the served models ship weights
        assert count_parameters(load_exported(MODELS_DIR / name)[0]) == RUNS[name]["params"]


def test_ablations_are_controlled_leave_one_out():
    """Each ablation removes exactly one technique and keeps the improved run's seed and budget."""
    imp = RUNS["improved"]
    ablations = {n: r for n, r in RUNS.items() if n.startswith("no_")}
    assert len(ablations) == 6
    for name, run in ablations.items():
        removed = set(imp["techniques"]) - set(run["techniques"])
        assert removed == {name[3:]}, name
        for key in ("seed", "epochs", "batch_size", "train_images", "val_images", "test_images"):
            assert run[key] == imp[key], (name, key)


def test_improved_beats_baseline():
    assert RUNS["improved"]["test_acc_tta"] > RUNS["baseline"]["test_acc_tta"]
