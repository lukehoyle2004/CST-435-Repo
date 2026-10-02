"""The matrix-form XOR example: gradients are correct and the network learns XOR."""
from __future__ import annotations

import numpy as np

from scripts.xor_matrix_backprop import X, Y, backward, forward, init_params, numeric_grad, train


def test_analytic_gradients_match_finite_differences():
    p = init_params()
    grads = backward(p, forward(p, X), X, Y)
    for name in ("W1", "b1", "W2", "b2"):
        assert grads[name].shape == p[name].shape
        assert np.allclose(grads[name], numeric_grad(p, name), atol=1e-6)


def test_network_solves_xor():
    trained, losses = train()
    preds = (forward(trained, X)["P"] >= 0.5).astype(int)
    assert (preds == Y).all()
    assert losses[-1] < losses[0]
