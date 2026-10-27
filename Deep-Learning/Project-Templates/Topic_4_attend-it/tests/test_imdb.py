"""Unit tests for shared/imdb.py against tiny parquet fixtures (no download)."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from shared.imdb import CLASS_NAMES, load_split, train_val_split


def _write(path, texts, labels):
    pq.write_table(pa.table({"text": texts, "label": labels}), path)


@pytest.fixture
def fake_root(tmp_path):
    _write(tmp_path / "train.parquet",
           ["Awful.<br />Just awful.", "A masterpiece.", "Loved it!"], [0, 1, 1])
    _write(tmp_path / "test.parquet", ["Boring.", "Pretty good."], [0, 1])
    return tmp_path


def test_class_names():
    assert CLASS_NAMES == ["negative", "positive"]


def test_load_split_returns_texts_and_labels_in_file_order(fake_root):
    texts, labels = load_split("train", fake_root)
    assert texts == ["Awful.<br />Just awful.", "A masterpiece.", "Loved it!"]
    assert labels.dtype.kind == "i" and labels.tolist() == [0, 1, 1]
    texts, labels = load_split("test", fake_root)
    assert texts == ["Boring.", "Pretty good."] and labels.tolist() == [0, 1]


def test_load_split_errors(fake_root, tmp_path):
    with pytest.raises(ValueError):
        load_split("unsupervised", fake_root)
    with pytest.raises(FileNotFoundError):
        load_split("train", tmp_path / "missing")


def test_load_split_rejects_unlabeled_rows(tmp_path):
    # The HF "unsupervised" split uses label -1; it must never slip into training.
    _write(tmp_path / "train.parquet", ["no label"], [-1])
    with pytest.raises(ValueError):
        load_split("train", tmp_path)


def test_train_val_split_is_seeded_disjoint_and_complete():
    tr, va = train_val_split(100, val_fraction=0.2, seed=3)
    assert len(va) == 20 and len(tr) == 80
    assert sorted(tr.tolist() + va.tolist()) == list(range(100))
    tr2, va2 = train_val_split(100, val_fraction=0.2, seed=3)
    assert (tr == tr2).all() and (va == va2).all()
    assert not (train_val_split(100, 0.2, seed=4)[1] == va).all()


def test_train_val_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        train_val_split(10, val_fraction=1.0)
