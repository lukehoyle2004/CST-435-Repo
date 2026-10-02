"""PyTorch MLP + sklearn preprocessing for tabular binary classification.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. Given the ``adult_income`` rows
(with their fixed train/val/test split) and hyperparameters, it fits a deep
multi-layer perceptron on standardized + one-hot-encoded features and returns:

* train / val / test accuracy and test precision / recall / F1 / ROC-AUC,
* a confusion matrix, per-class precision / recall / F1, and calibration data
  (reliability bins, Brier score, expected calibration error),
* permutation importance of every input column (on the validation split),
* the per-epoch train and validation loss curves,
* a serialized (preprocessor + network) artifact, base64-encoded so it can be
  persisted to Supabase and reloaded at /predict time.

Diverged runs (non-finite or rising loss) return ``status="diverged"`` and None
metrics -- never a sentinel number that could pass for a real score.
"""
from __future__ import annotations

import base64
import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from shared.data import TARGET_CLASSES, feature_columns

STATUS_OK = "ok"
STATUS_DIVERGED = "diverged"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")

# Hidden-layer non-linearities available for the activation comparison.
ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}

N_CALIBRATION_BINS = 10
N_PERMUTATION_REPEATS = 3


class MLP(nn.Module):
    """Deep MLP: ``n_layers`` hidden (Linear -> activation) blocks, then one logit.

    In matrix form, for a mini-batch X of shape (B, d) and hidden width H:
        A1 = f(X  W1 + b1)      W1: (d, H)   A1: (B, H)
        A2 = f(A1 W2 + b2)      W2: (H, H)   A2: (B, H)
        z  = A2 W3 + b3         W3: (H, 1)   z:  (B, 1)
        p  = sigmoid(z)
    (nn.Linear stores W transposed, as (out, in).) The sigmoid is applied outside
    the module -- inside BCEWithLogitsLoss for training -- for numerical stability.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 2,
                 activation: str = "relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        layers: List[nn.Module] = []
        width = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden_dim), ACTIVATIONS[activation]()]
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainResult:
    status: str
    metrics: Dict[str, Optional[float]]
    train_accuracy: Optional[float]
    val_accuracy: Optional[float]
    loss_history: List[float]
    val_loss_history: List[float]
    model_b64: str
    n_train: int
    n_val: int
    n_test: int
    confusion: Optional[Dict[str, int]] = None
    class_report: Optional[Dict[str, Dict[str, float]]] = None
    calibration: Optional[List[Dict[str, float]]] = None
    brier: Optional[float] = None
    ece: Optional[float] = None
    perm_importance: Optional[List[Dict[str, float]]] = None
    # (adult_income id, predicted label, P(>50K)) for every test row, so the
    # bias audit can join predictions back to the protected attributes in SQL.
    test_predictions: List[Tuple[int, int, float]] = field(default_factory=list)


def _build_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    """Standardize numeric columns, one-hot the categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
        ]
    )


def run_status(loss_history: List[float], y_prob: np.ndarray) -> str:
    """Classify a finished run as ``ok`` or ``diverged``.

    A run is diverged when the loss or the output probabilities went non-finite
    (NaN/Inf), or when training ended with a higher loss than it started with.
    Diverged runs are stored with NULL metrics instead of a sentinel number, so
    they can never be mistaken for (or sorted above) a real result.
    """
    if not all(math.isfinite(v) for v in loss_history):
        return STATUS_DIVERGED
    if not np.all(np.isfinite(y_prob)):
        return STATUS_DIVERGED
    if len(loss_history) > 1 and loss_history[-1] > loss_history[0]:
        return STATUS_DIVERGED
    return STATUS_OK


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
             ) -> Dict[str, Optional[float]]:
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc: Optional[float] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # Only one class present -> AUC is undefined, so store NULL, not 0.0.
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def class_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per-class precision / recall / F1 / support, keyed by class name."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        TARGET_CLASSES[k]: {
            "precision": float(p[k]), "recall": float(r[k]),
            "f1": float(f[k]), "support": int(s[k]),
        }
        for k in (0, 1)
    }


def calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_CALIBRATION_BINS
                ) -> Tuple[List[Dict[str, float]], float, float]:
    """Reliability-diagram bins plus Brier score and expected calibration error.

    For a well-calibrated model, among rows predicted with p ~= 0.7, about 70%
    should truly be >50K, i.e. every bin's ``frac_positive`` ~= ``mean_pred``.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = which == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += count / len(y_prob) * abs(frac_pos - mean_pred)
        bins.append({
            "bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]),
            "mean_pred": mean_pred, "frac_positive": frac_pos, "count": count,
        })
    brier = float(np.mean((y_prob - y_true) ** 2))
    return bins, brier, float(ece)


def _predict_proba(pre: ColumnTransformer, model: MLP, df: pd.DataFrame) -> np.ndarray:
    x = torch.from_numpy(pre.transform(df).astype(np.float32))
    with torch.no_grad():
        return torch.sigmoid(model(x)).squeeze(1).numpy()


def permutation_importance(pre: ColumnTransformer, model: MLP, df: pd.DataFrame,
                           y: np.ndarray, columns: List[str], seed: int = 0
                           ) -> List[Dict[str, float]]:
    """Drop in ROC-AUC when one input column is shuffled (model-agnostic).

    Shuffling a column breaks its link to the target while keeping its
    distribution; a big AUC drop means the model leans on that column.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, _predict_proba(pre, model, df))
    out = []
    for col in columns:
        drops = []
        for _ in range(N_PERMUTATION_REPEATS):
            shuffled = df.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(base - roc_auc_score(y, _predict_proba(pre, model, shuffled)))
        out.append({"feature": col, "importance": float(np.mean(drops)),
                    "std": float(np.std(drops))})
    return sorted(out, key=lambda d: d["importance"], reverse=True)


def train_income_classifier(
    data: pd.DataFrame,
    hidden_dim: int = 32,
    lr: float = 0.01,
    batch_size: int = 128,
    epochs: int = 20,
    n_layers: int = 2,
    activation: str = "relu",
    seed: int = 0,
    max_train_rows: Optional[int] = None,
    include_protected: bool = False,
) -> TrainResult:
    """Fit the MLP with Adam + binary cross-entropy on the fixed splits.

    ``data`` has the ``adult_income`` schema (features, ``label``, ``split``,
    ``id``). The same ``seed`` fixes the weight init, the mini-batch order, and the
    optional training subsample, so runs that differ only in one hyperparameter
    (e.g. activation) are a controlled comparison.
    """
    numeric, categorical = feature_columns(include_protected)
    columns = numeric + categorical

    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    test = data[data["split"] == "test"]
    if max_train_rows and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=seed)

    y_train = train["label"].to_numpy(dtype=np.float32)
    y_val = val["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    pre = _build_preprocessor(numeric, categorical)
    x_train_t = torch.from_numpy(pre.fit_transform(train[columns]).astype(np.float32))
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    x_val_t = torch.from_numpy(pre.transform(val[columns]).astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1)

    torch.manual_seed(seed)
    in_dim = x_train_t.shape[1]
    model = MLP(in_dim, hidden_dim, n_layers, activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = x_train_t.shape[0]
    loss_history: List[float] = []
    val_loss_history: List[float] = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb, yb = x_train_t[batch_idx], y_train_t[batch_idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)   # forward pass + cost
            loss.backward()                 # backpropagate: dLoss/dW for every W
            optimizer.step()                # update the weights
            epoch_loss += loss.item() * xb.shape[0]
        loss_history.append(epoch_loss / n)
        model.eval()
        with torch.no_grad():
            val_loss_history.append(float(loss_fn(model(x_val_t), y_val_t)))

    model.eval()
    p_train = _predict_proba(pre, model, train[columns])
    p_val = _predict_proba(pre, model, val[columns])
    p_test = _predict_proba(pre, model, test[columns])

    artifact = {
        "pre": pre,
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "activation": activation,
        "numeric": numeric,
        "categorical": categorical,
    }
    model_b64 = base64.b64encode(pickle.dumps(artifact)).decode("ascii")

    status = run_status(loss_history + val_loss_history, p_test)
    sizes = {"n_train": len(train), "n_val": len(val), "n_test": len(test)}
    if status != STATUS_OK:
        return TrainResult(
            status=status, metrics={k: None for k in METRIC_NAMES},
            train_accuracy=None, val_accuracy=None, loss_history=loss_history,
            val_loss_history=val_loss_history, model_b64=model_b64, **sizes,
        )

    pred_test = (p_test >= 0.5).astype(int)
    bins, brier, ece = calibration(y_test, p_test)
    return TrainResult(
        status=status,
        metrics=_metrics(y_test, p_test),
        train_accuracy=float(accuracy_score(y_train.astype(int), (p_train >= 0.5).astype(int))),
        val_accuracy=float(accuracy_score(y_val, (p_val >= 0.5).astype(int))),
        loss_history=loss_history,
        val_loss_history=val_loss_history,
        model_b64=model_b64,
        confusion=confusion(y_test, pred_test),
        class_report=class_report(y_test, pred_test),
        calibration=bins,
        brier=brier,
        ece=ece,
        perm_importance=permutation_importance(pre, model, val[columns], y_val, columns, seed),
        test_predictions=[
            (int(i), int(l), float(p))
            for i, l, p in zip(test["id"].to_numpy(), pred_test, p_test)
        ],
        **sizes,
    )


def load_artifact(model_b64: str) -> Tuple[ColumnTransformer, MLP, List[str]]:
    """Rebuild (preprocessor, network, input columns) from a stored artifact."""
    obj = pickle.loads(base64.b64decode(model_b64))
    # Older artifacts predate n_layers / activation / stored column lists.
    model = MLP(obj["in_dim"], obj["hidden_dim"], obj.get("n_layers", 1),
                obj.get("activation", "relu"))
    model.load_state_dict(obj["state_dict"])
    model.eval()
    columns = obj.get("numeric", []) + obj.get("categorical", [])
    if not columns:  # legacy artifact: recover the column list from the transformer
        columns = [c for _name, _t, cols in obj["pre"].transformers for c in cols]
    return obj["pre"], model, columns


def predict_records(
    model_b64: str | Tuple[ColumnTransformer, MLP, List[str]],
    records: List[Dict[str, object]],
    threshold: float = 0.5,
) -> List[Tuple[int, float]]:
    """Apply a stored artifact to one or more records; return (label, proba).

    The label comes from the threshold function ``label = 1 if p >= threshold``.
    Raising the threshold trades recall for precision on the >50K class.
    ``model_b64`` may also be an already-loaded ``load_artifact`` tuple (cache).
    """
    pre, model, columns = (
        load_artifact(model_b64) if isinstance(model_b64, str) else model_b64
    )
    df = pd.DataFrame(records)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"records are missing required columns: {missing}")
    prob = _predict_proba(pre, model, df[columns])
    return [(int(p >= threshold), float(p)) for p in prob]
