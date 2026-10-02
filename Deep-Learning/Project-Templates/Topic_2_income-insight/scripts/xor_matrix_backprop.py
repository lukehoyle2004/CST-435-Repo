"""XOR worked example: forward and backward propagation in matrix form (NumPy).

Run from the project folder:

    python -m scripts.xor_matrix_backprop

A 2-2-1 network (sigmoid hidden layer, sigmoid output, binary cross-entropy)
learns XOR, which no single linear layer can represent. Every forward and
backward step is written as a matrix product, the shape of every matrix is
printed, and the analytic gradients are checked against finite differences.

Notation (one row per example, batch of n = 4):
    X  (n, 2)    inputs              Y  (n, 1)  targets
    W1 (2, h)    b1 (1, h)           Z1 = X W1 + b1     (n, h)   A1 = sigmoid(Z1)
    W2 (h, 1)    b2 (1, 1)           Z2 = A1 W2 + b2    (n, 1)   P  = sigmoid(Z2)
    L  = -(1/n) * sum[ Y log P + (1 - Y) log(1 - P) ]

Backward (chain rule; * is element-wise):
    dZ2 = (P - Y) / n                         (n, 1)  sigmoid + BCE simplify to P - Y
    dW2 = A1^T dZ2                            (h, 1)
    db2 = sum over rows of dZ2                (1, 1)
    dA1 = dZ2 W2^T                            (n, h)
    dZ1 = dA1 * A1 * (1 - A1)                 (n, h)  sigmoid'(Z1) = A1 (1 - A1)
    dW1 = X^T dZ1                             (2, h)
    db1 = sum over rows of dZ1                (1, h)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def init_params(hidden: int = 2, seed: int = 1) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1, (2, hidden)), "b1": np.zeros((1, hidden)),
        "W2": rng.normal(0, 1, (hidden, 1)), "b2": np.zeros((1, 1)),
    }


def forward(p: Dict[str, np.ndarray], x: np.ndarray) -> Dict[str, np.ndarray]:
    z1 = x @ p["W1"] + p["b1"]
    a1 = sigmoid(z1)
    z2 = a1 @ p["W2"] + p["b2"]
    return {"Z1": z1, "A1": a1, "Z2": z2, "P": sigmoid(z2)}


def bce(prob: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)))


def backward(p: Dict[str, np.ndarray], cache: Dict[str, np.ndarray],
             x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    n = x.shape[0]
    dz2 = (cache["P"] - y) / n
    dw2 = cache["A1"].T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * cache["A1"] * (1 - cache["A1"])
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2,
            "dZ2": dz2, "dA1": da1, "dZ1": dz1}


def numeric_grad(p: Dict[str, np.ndarray], name: str, eps: float = 1e-6) -> np.ndarray:
    """Central finite differences of the loss w.r.t. one parameter matrix."""
    grad = np.zeros_like(p[name])
    for idx in np.ndindex(*p[name].shape):
        orig = p[name][idx]
        p[name][idx] = orig + eps
        up = bce(forward(p, X)["P"], Y)
        p[name][idx] = orig - eps
        down = bce(forward(p, X)["P"], Y)
        p[name][idx] = orig
        grad[idx] = (up - down) / (2 * eps)
    return grad


def train(lr: float = 2.0, epochs: int = 5000, hidden: int = 2, seed: int = 1
          ) -> Tuple[Dict[str, np.ndarray], list]:
    p = init_params(hidden, seed)
    losses = []
    for _ in range(epochs):
        cache = forward(p, X)
        losses.append(bce(cache["P"], Y))
        g = backward(p, cache, X, Y)
        for name in ("W1", "b1", "W2", "b2"):
            p[name] -= lr * g[name]          # gradient-descent update
    return p, losses


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    p = init_params()
    cache = forward(p, X)
    grads = backward(p, cache, X, Y)

    print("Shapes (n = 4 examples, h = 2 hidden units):")
    for name, arr in [("X", X), ("Y", Y), ("W1", p["W1"]), ("b1", p["b1"]),
                      ("Z1", cache["Z1"]), ("A1", cache["A1"]), ("W2", p["W2"]),
                      ("b2", p["b2"]), ("Z2", cache["Z2"]), ("P", cache["P"]),
                      ("dZ2", grads["dZ2"]), ("dW2", grads["W2"]), ("db2", grads["b2"]),
                      ("dA1", grads["dA1"]), ("dZ1", grads["dZ1"]), ("dW1", grads["W1"]),
                      ("db1", grads["b1"])]:
        print(f"  {name:>3}: {arr.shape}")

    print("\nGradient check (analytic vs finite differences, max abs difference):")
    for name in ("W1", "b1", "W2", "b2"):
        diff = np.max(np.abs(grads[name] - numeric_grad(p, name)))
        print(f"  d{name}: {diff:.2e}")

    print(f"\nInitial loss: {bce(cache['P'], Y):.4f}")
    trained, losses = train()
    probs = forward(trained, X)["P"]
    print(f"Loss after {len(losses)} epochs: {losses[-1]:.4f}\n")
    print("  x1 x2 | target | P(y=1) | predicted")
    for (x1, x2), y, pr in zip(X.astype(int), Y.astype(int).ravel(), probs.ravel()):
        print(f"   {x1}  {x2} |   {y}    | {pr:.4f} |    {int(pr >= 0.5)}")
    print("\nLearned hidden-layer activations A1 (each row is one input):")
    print(forward(trained, X)["A1"])
    print("The hidden layer maps the 4 points into a space where one line separates "
          "the XOR classes -- the expressive power a single linear layer lacks.")


if __name__ == "__main__":
    main()
