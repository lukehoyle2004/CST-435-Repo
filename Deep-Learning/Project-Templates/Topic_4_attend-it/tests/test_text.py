"""Unit tests for shared/text.py: tokenization, vocab, padding and masks."""
from __future__ import annotations

import numpy as np
import pytest

from shared.text import (
    PAD_ID,
    SPECIALS,
    UNK_ID,
    Vocab,
    encode_text,
    normalize,
    pad_batch,
    padding_mask,
    tokenize,
)

CORPUS = [
    "This movie was great! Great acting.<br /><br />Loved it.",
    "Not great. The plot isn't good &amp; the acting was bad?",
    "Bad bad bad. I didn't like this movie.",
]


# --- tokenization ----------------------------------------------------------
def test_normalize_strips_br_and_unescapes_entities():
    assert normalize("A<br />B<BR>C &amp; D") == "a b c & d"


def test_tokenize_keeps_contractions_and_tone_punctuation():
    assert tokenize("It ISN'T good, is it?! Don’t go.") == [
        "it", "isn't", "good", "is", "it", "?", "!", "don't", "go",
    ]


def test_tokenize_drops_other_punctuation_and_empty_input():
    assert tokenize("...,;:()\"") == []
    assert tokenize("") == []


# --- vocab -----------------------------------------------------------------
def _vocab(**kw) -> Vocab:
    return Vocab.build((tokenize(t) for t in CORPUS), **kw)


def test_specials_have_fixed_ids():
    v = _vocab()
    assert v.itos[:2] == SPECIALS
    assert v.stoi["<pad>"] == PAD_ID == 0
    assert v.stoi["<unk>"] == UNK_ID == 1


def test_vocab_orders_by_frequency_then_alphabetically():
    v = _vocab(min_freq=1)
    # "bad" appears 4x, then "great" 3x; "acting"/"movie"/"the"/"this"/"was" 2x
    assert v.itos[2:4] == ["bad", "great"]
    assert v.itos[4:9] == ["acting", "movie", "the", "this", "was"]


def test_vocab_is_independent_of_document_order():
    a = Vocab.build(tokenize(t) for t in CORPUS)
    b = Vocab.build(tokenize(t) for t in reversed(CORPUS))
    assert a.itos == b.itos and a.sha256() == b.sha256()


def test_min_freq_and_max_size():
    v = _vocab(min_freq=2)
    assert "loved" not in v and "great" in v
    capped = _vocab(min_freq=1, max_size=4)
    assert len(capped) == 4 and capped.itos == SPECIALS + ["bad", "great"]


def test_unknown_words_map_to_unk_and_decode_round_trips():
    v = _vocab()
    ids = v.encode(["great", "zyzzyva", "movie"])
    assert ids[1] == UNK_ID
    assert v.decode(ids) == ["great", "<unk>", "movie"]


def test_vocab_serialization_round_trip():
    v = _vocab()
    restored = Vocab.from_dict(v.to_dict())
    assert restored.itos == v.itos and restored.sha256() == v.sha256()


def test_vocab_rejects_bad_itos():
    with pytest.raises(ValueError):
        Vocab(["movie", "<pad>", "<unk>"])
    with pytest.raises(ValueError):
        Vocab(SPECIALS + ["a", "a"])


# --- encoding --------------------------------------------------------------
def test_encode_text_truncates_to_max_len():
    tokens, ids = encode_text("bad " * 10, _vocab(), max_len=4)
    assert tokens == ["bad"] * 4 and len(ids) == 4


def test_encode_text_rejects_text_without_tokens():
    with pytest.raises(ValueError):
        encode_text("... ;-)", _vocab())


# --- padding & masks -------------------------------------------------------
def test_pad_batch_right_pads_to_longest():
    ids, lengths = pad_batch([[5, 6, 7], [8], [9, 10]])
    assert ids.dtype == np.int64 and ids.shape == (3, 3)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0], [9, 10, 0]]
    assert lengths.tolist() == [3, 1, 2]


def test_pad_batch_rejects_empty_inputs():
    with pytest.raises(ValueError):
        pad_batch([])
    with pytest.raises(ValueError):
        pad_batch([[3], []])


def test_padding_mask_marks_real_tokens():
    ids, lengths = pad_batch([[5, 6, 7], [8], [9, 10]])
    mask = padding_mask(lengths)
    assert mask.dtype == bool and mask.shape == ids.shape
    assert mask.tolist() == [
        [True, True, True],
        [True, False, False],
        [True, True, False],
    ]
    # Real ids are never PAD, so the mask equals ids != PAD here.
    assert (mask == (ids != PAD_ID)).all()


def test_padding_mask_honours_explicit_width():
    assert padding_mask([2], max_len=4).tolist() == [[True, True, False, False]]


def test_end_to_end_batch_from_text():
    v = _vocab()
    encoded = [encode_text(t, v)[1] for t in CORPUS]
    ids, lengths = pad_batch(encoded)
    mask = padding_mask(lengths)
    assert mask.sum(axis=1).tolist() == [len(e) for e in encoded]
    assert (ids[~mask] == PAD_ID).all()
