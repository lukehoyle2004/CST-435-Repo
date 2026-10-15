"""Export the two trained CNNs from the training project into this deployable service.

Training runs offline in `AIT-204-CNN/cifar10_project` (a laptop CPU, ~80 min for
the improved model). This script reads its artifacts and writes, into models/:

    improved.pt / improved.json     state_dict + run config (loaded by the API)
    baseline.pt / baseline.json
    runs.json                       one summary row per run -> db/seed_runs.py -> Supabase `runs`
    gallery.npz                     12 CIFAR-10 test images (>= 5 right, >= 5 wrong for the
                                    improved model) for the Compare Models tab

and copies each run's figures into ui/assets/<run name>/.

    python scripts/export_models.py --src ../../../../AIT-204-CNN/cifar10_project

Needs the training project's environment (torch + torchvision for the test set).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seesense import CLASSES  # noqa: E402
from seesense.model import count_parameters, load_exported  # noqa: E402
from seesense.inference import predict_proba  # noqa: E402

RUNS = {
    # name: (artifact sub-folder, human label, techniques on top of the baseline)
    "baseline": ("baseline", "Baseline CNN", []),
    "improved": ("", "Improved CNN", ["augmentation", "batchnorm", "dropout", "weight_decay",
                                      "label_smoothing", "onecycle_lr"]),
}
FIGURES = ["training_curves.png", "confusion_matrix.png", "confusion_matrix_norm.png",
           "per_class_metrics.png", "misclassified.png", "augmentation_examples.png"]


def load_test_set(src: Path):
    sys.path.insert(0, str(src))
    from cifar.data import load_cifar10
    _, _, test_x, test_y = load_cifar10()
    return test_x, test_y.numpy()


def export_run(src: Path, name: str) -> dict:
    sub, label, techniques = RUNS[name]
    art = src / "artifacts" / sub
    ckpt = torch.load(art / "model.pth", map_location="cpu", weights_only=False)
    config = dict(ckpt["config"], baseline=name == "baseline")
    config.pop("out", None)
    torch.save(ckpt["state_dict"], ROOT / "models" / f"{name}.pt")
    (ROOT / "models" / f"{name}.json").write_text(json.dumps(config, indent=2))

    metrics = json.loads((art / "metrics.json").read_text())
    history = json.loads((art / "history.json").read_text())
    best = next(h for h in history if h["epoch"] == ckpt["epoch"])
    rep = metrics["report"]

    assets = ROOT / "ui" / "assets" / name
    assets.mkdir(parents=True, exist_ok=True)
    for fig in FIGURES:
        if (art / fig).exists():
            shutil.copy2(art / fig, assets / fig)

    return dict(
        run_id=config["run_id"],
        name=name,
        display_name=label,
        architecture=("6 conv (3x3) in 3 blocks, no BatchNorm/dropout, GAP + Linear" if name == "baseline"
                      else "6 conv (3x3) + BatchNorm in 3 blocks, Dropout2d, GAP + Dropout + Linear"),
        techniques=techniques,
        dataset="CIFAR-10",
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        lr_schedule="constant" if name == "baseline" else "one-cycle",
        weight_decay=config["weight_decay"],
        label_smoothing=config["label_smoothing"],
        augment=config["augment"],
        seed=config["seed"],
        params=config["params"],
        train_images=config["train_images"],
        val_images=config["val_images"],
        test_images=config["test_images"],
        hardware=config["hardware"],
        train_seconds=round(config.get("train_seconds", sum(h["epoch_seconds"] for h in history)), 1),
        best_epoch=ckpt["epoch"],
        train_loss=best["train_loss"],
        train_acc=best["train_acc"],
        val_loss=best["val_loss"],
        val_acc=best["val_acc"],
        test_acc=metrics["test_acc"],
        test_acc_tta=metrics["test_acc_tta"],
        macro_f1=metrics["macro_f1"],
        per_class={c: dict(precision=rep[c]["precision"], recall=rep[c]["recall"],
                           f1=rep[c]["f1-score"]) for c in CLASSES},
        confusion_matrix=metrics["confusion_matrix"],
        history=[{k: h[k] for k in ("epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc")}
                 for h in history],
        weights_path=f"models/{name}.pt",
    )


def build_gallery(test_x, test_y, n_right=6, n_wrong=6, seed=3):
    """Pick test images the improved model gets right / wrong, one class each where possible."""
    models = {n: load_exported(ROOT / "models" / n)[0] for n in RUNS}
    probs = {n: predict_proba(m, test_x, tta=True) for n, m in models.items()}
    pred = {n: p.argmax(1) for n, p in probs.items()}
    rng = np.random.default_rng(seed)
    picks = []
    for want_right, n in ((True, n_right), (False, n_wrong)):
        pool = np.where((pred["improved"] == test_y) == want_right)[0]
        if want_right:  # favor images where the improvement shows (baseline wrong)
            pool = np.concatenate([rng.permutation(pool[pred["baseline"][pool] != test_y[pool]]),
                                   rng.permutation(pool[pred["baseline"][pool] == test_y[pool]])])
        else:
            pool = rng.permutation(pool)
        used = set()
        for i in pool:
            if test_y[i] not in used:
                picks.append(int(i))
                used.add(test_y[i])
            if len(used) == n:
                break
    np.savez_compressed(ROOT / "models" / "gallery.npz", images=test_x[picks].numpy(),
                        labels=test_y[picks], test_index=np.array(picks))
    for n in RUNS:
        right = int((pred[n][picks] == test_y[picks]).sum())
        print(f"gallery: {n} gets {right}/{len(picks)} right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT.parents[3] / "AIT-204-CNN" / "cifar10_project")
    args = ap.parse_args()
    (ROOT / "models").mkdir(exist_ok=True)

    runs = [export_run(args.src, n) for n in RUNS]
    (ROOT / "models" / "runs.json").write_text(json.dumps(runs, indent=2))
    for r in runs:
        m, _ = load_exported(ROOT / "models" / r["name"])
        print(f"{r['name']:9s} run {r['run_id']}  test acc {r['test_acc_tta']:.2f}%  "
              f"params {count_parameters(m):,}")
    build_gallery(*load_test_set(args.src))


if __name__ == "__main__":
    main()
