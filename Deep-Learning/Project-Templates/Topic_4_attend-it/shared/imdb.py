"""IMDB Large Movie Review Dataset (Maas et al., 2011) loader.

50,000 English movie reviews from IMDB, labeled by star rating: <= 4/10 is
``negative`` and >= 7/10 is ``positive`` (neutral reviews are excluded). The
official split is 25k train / 25k test, each balanced 50/50.

``scripts/download_imdb.py`` saves the two splits as parquet files with columns
``text`` (str) and ``label`` (0 = negative, 1 = positive)::

    data/imdb/train.parquet
    data/imdb/test.parquet

Rows are returned in file order, so every load returns the same order.
The raw reviews never leave the training machine: Supabase only stores run
metrics, the vocab and the model, plus hashes of user inputs at predict time.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

CLASS_NAMES: List[str] = ["negative", "positive"]
N_CLASSES = len(CLASS_NAMES)
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "imdb"
SPLITS = ("train", "test")


def load_split(split: str, root: Path | str = DEFAULT_ROOT) -> Tuple[List[str], np.ndarray]:
    """Return ``(texts, labels)`` for ``"train"`` or ``"test"``; label 1 = positive."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = Path(root) / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found - run `python -m scripts.download_imdb` first"
        )
    import pyarrow.parquet as pq   # dev-only dependency; the cloud tiers never load IMDB

    table = pq.read_table(path, columns=["text", "label"])
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    if not np.isin(labels, range(N_CLASSES)).all():
        raise ValueError(f"{path} has labels outside 0..{N_CLASSES - 1}")
    return table.column("text").to_pylist(), labels


def train_val_split(n: int, val_fraction: float = 0.1, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Seeded shuffle of ``range(n)`` into ``(train_idx, val_idx)``.

    Validation is carved out of the official *train* split; the official test
    split is only touched for the final reported numbers.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_fraction))
    return idx[n_val:], idx[:n_val]
