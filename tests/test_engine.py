"""Gradient checking: compare the engine's analytic backward() gradients
against numerical (central-difference) gradients for every op, plus a full
2-layer MLP + cross-entropy forward/backward pass.
"""

import numpy as np

from nn.engine import Tensor
from nn.layers import Linear, ReLU, Sequential, cross_entropy_loss


def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f1 = f(x)
        x[idx] = orig - eps
        f2 = f(x)
        x[idx] = orig
        grad[idx] = (f1 - f2) / (2 * eps)
    return grad


def test_add_broadcast():
    np.random.seed(0)
    a_data = np.random.randn(4, 3)
    b_data = np.random.randn(3)

    numeric = numerical_gradient(lambda a: (a + b_data).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    out = (a + b).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)
    assert np.allclose(b.grad, numerical_gradient(lambda b_: (a_data + b_).sum(), b_data.copy()), atol=1e-5)


def test_mul():
    np.random.seed(1)
    a_data = np.random.randn(3, 3)
    b_data = np.random.randn(3, 3)

    numeric = numerical_gradient(lambda a: (a * b_data).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    out = (a * b).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)


def test_pow():
    np.random.seed(2)
    a_data = np.random.rand(5) + 0.5  # keep positive, avoid 0**negative issues

    numeric = numerical_gradient(lambda a: (a ** 3).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    out = (a ** 3).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)


def test_matmul():
    np.random.seed(3)
    a_data = np.random.randn(4, 5)
    b_data = np.random.randn(5, 2)

    numeric = numerical_gradient(lambda a: (a @ b_data).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    out = (a @ b).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)


def test_relu():
    np.random.seed(4)
    a_data = np.random.randn(6)

    numeric = numerical_gradient(lambda a: np.maximum(0, a).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    out = a.relu().sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)


def test_exp_log():
    np.random.seed(5)
    a_data = np.random.rand(5) + 0.1  # keep positive for log

    numeric = numerical_gradient(lambda a: np.log(np.exp(a)).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    out = a.exp().log().sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)


def test_mean():
    np.random.seed(6)
    a_data = np.random.randn(4, 3)

    numeric = numerical_gradient(lambda a: (a ** 2).mean(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    out = (a ** 2).mean()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)


def test_mlp_cross_entropy_grad_check():
    np.random.seed(7)
    batch, in_dim, hidden, out_dim = 5, 6, 4, 3
    x_data = np.random.randn(batch, in_dim)
    targets = np.random.randint(0, out_dim, size=batch)

    l1 = Linear(in_dim, hidden)
    l2 = Linear(hidden, out_dim)
    model = Sequential(l1, ReLU(), l2)

    def forward_loss(w1):
        h = np.maximum(0, x_data @ w1 + l1.bias.data)
        logits = h @ l2.weight.data + l2.bias.data
        m = logits.max(axis=1, keepdims=True)
        shifted = logits - m
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(batch), targets].mean()

    numeric = numerical_gradient(forward_loss, l1.weight.data.copy())

    out = model(Tensor(x_data))
    loss = cross_entropy_loss(out, targets)
    loss.backward()

    assert np.allclose(l1.weight.grad, numeric, atol=1e-5)
