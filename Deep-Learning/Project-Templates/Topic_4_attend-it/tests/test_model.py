"""Unit tests for seqsight/model.py: packing, masking and the matched model pair."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from seqsight.model import (
    AttentionLSTMClassifier,
    LSTMClassifier,
    build_model,
    collate,
    count_parameters,
    lengths_to_mask,
    load_exported,
    predict_ids,
)
from shared.text import padding_mask

SMALL = {"vocab_size": 50, "embed_dim": 8, "hidden_dim": 12, "attn_dim": 6, "dropout": 0.0}
BATCH = [[5, 6, 7, 8, 9], [10, 11], [12, 13, 14]]


def _model(kind: str, **kw) -> torch.nn.Module:
    torch.manual_seed(0)
    return build_model({**SMALL, "kind": kind, **kw}).eval()


@pytest.fixture(params=["lstm", "attention"])
def model(request):
    return _model(request.param)


def test_build_model_picks_class_and_rejects_unknown_kind():
    assert isinstance(_model("lstm"), LSTMClassifier)
    assert isinstance(_model("attention"), AttentionLSTMClassifier)
    with pytest.raises(ValueError):
        build_model({"kind": "gru"})


def test_output_shapes(model):
    ids, lengths = collate(BATCH)
    logits, w = model(ids, lengths)
    assert logits.shape == (3, 2)
    if model.kind == "attention":
        assert w.shape == ids.shape
    else:
        assert w is None


def test_padding_does_not_change_the_prediction(model):
    # Each review scored alone (no padding) must match the same review scored
    # inside a padded batch: packing + masking make padding invisible.
    ids, lengths = collate(BATCH)
    with torch.no_grad():
        batch_logits, batch_w = model(ids, lengths)
        for i, seq in enumerate(BATCH):
            solo_logits, solo_w = model(*collate([seq]))
            assert torch.allclose(batch_logits[i], solo_logits[0], atol=1e-5)
            if solo_w is not None:
                assert torch.allclose(batch_w[i, : len(seq)], solo_w[0], atol=1e-5)


def test_tokens_in_padded_slots_are_ignored(model):
    ids, lengths = collate(BATCH)
    junk = ids.clone()
    pad = ~lengths_to_mask(lengths, ids.size(1))
    junk[pad] = 42                            # garbage where padding should be
    with torch.no_grad():
        assert torch.allclose(model(ids, lengths)[0], model(junk, lengths)[0], atol=1e-6)


def test_attention_is_zero_on_padding_and_sums_to_one():
    ids, lengths = collate(BATCH)
    with torch.no_grad():
        _, w = _model("attention")(ids, lengths)
    mask = lengths_to_mask(lengths, ids.size(1))
    assert (w[~mask] == 0).all()
    assert torch.allclose(w.sum(-1), torch.ones(3))


def test_torch_mask_matches_shared_numpy_mask():
    lengths = torch.tensor([5, 2, 3])
    assert (lengths_to_mask(lengths, 5).numpy() == padding_mask([5, 2, 3])).all()


def test_baseline_reads_last_real_token_not_padding():
    # The baseline's final state for a padded sequence must equal its state
    # when run unpadded, i.e. the LSTM stopped at the sequence's own length.
    m = _model("lstm")
    ids, lengths = collate(BATCH)
    with torch.no_grad():
        _, last = m.encoder(ids, lengths)
        for i, seq in enumerate(BATCH):
            _, solo = m.encoder(*collate([seq]))
            assert torch.allclose(last[i], solo[0], atol=1e-6)


def test_bidirectional_encoder_doubles_width():
    m = _model("attention", bidirectional=True)
    logits, w = m(*collate(BATCH))
    assert m.encoder.out_dim == 2 * SMALL["hidden_dim"] and logits.shape == (3, 2)


def test_models_are_matched_except_for_attention():
    lstm, attn = _model("lstm"), _model("attention")
    extra = count_parameters(attn) - count_parameters(lstm)
    d, h = SMALL["attn_dim"], SMALL["hidden_dim"]
    assert extra == h * d + d + d            # key projection (W_k, b_k) + query vector


def test_gradients_reach_every_parameter(model):
    model.train()
    ids, lengths = collate(BATCH)
    logits, _ = model(ids, lengths)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name


def test_attention_model_can_learn_a_toy_rule():
    # Label = 1 iff token 3 appears anywhere; position and length vary.
    rng = np.random.default_rng(0)
    seqs, labels = [], []
    for _ in range(256):
        n = int(rng.integers(4, 15))
        s = rng.integers(4, 50, size=n).tolist()
        y = int(rng.integers(0, 2))
        if y:
            s[int(rng.integers(0, n))] = 3
        seqs.append(s)
        labels.append(y)
    torch.manual_seed(0)
    m = build_model({**SMALL, "kind": "attention"})
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    ids, lengths = collate(seqs)
    y = torch.tensor(labels)
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(ids, lengths)[0], y)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        logits, w = m(ids, lengths)
    assert (logits.argmax(-1) == y).float().mean() > 0.95
    # On positive examples the attention peaks on the token that decides the label.
    pos = [i for i, lab in enumerate(labels) if lab]
    hits = [ids[i, w[i].argmax()].item() == 3 for i in pos]
    assert np.mean(hits) > 0.9


def test_predict_ids_and_export_round_trip(tmp_path):
    m = _model("attention")
    probs, w = predict_ids(m, [5, 6, 7])
    assert len(probs) == 2 and sum(probs) == pytest.approx(1.0)
    assert len(w) == 3 and sum(w) == pytest.approx(1.0)

    stem = tmp_path / "attention"
    torch.save(m.state_dict(), stem.with_suffix(".pt"))
    stem.with_suffix(".json").write_text(json.dumps({**SMALL, "kind": "attention"}))
    loaded, cfg = load_exported(stem)
    assert cfg["kind"] == "attention"
    assert predict_ids(loaded, [5, 6, 7]) == (probs, w)
