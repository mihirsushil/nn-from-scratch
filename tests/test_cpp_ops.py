"""Correctness tests for the optional C++ kernels.

These are skipped, not failed, when the extension is not built, so the pure
NumPy path stays the reference and CI can run either way.
"""

import numpy as np
import pytest

from nn import engine
from nn.engine import Tensor

nn_cpp = pytest.importorskip("nn_cpp", reason="C++ extension not built")


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (4, 3, 2), (128, 784, 128), (37, 19, 53)])
def test_matmul_matches_numpy(m, k, n):
    rng = np.random.default_rng(1000 * m + 10 * k + n)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    np.testing.assert_allclose(nn_cpp.matmul(a, b), a @ b, rtol=1e-12, atol=1e-12)


def test_matmul_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        nn_cpp.matmul(np.zeros((3, 4)), np.zeros((5, 2)))


def test_matmul_rejects_non_2d():
    with pytest.raises(ValueError):
        nn_cpp.matmul(np.zeros(4), np.zeros((4, 2)))


def test_softmax_rows_matches_numpy():
    rng = np.random.default_rng(7)
    z = rng.standard_normal((16, 10)) * 3
    e = np.exp(z - z.max(axis=1, keepdims=True))
    expected = e / e.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(nn_cpp.softmax_rows(z), expected, rtol=1e-12, atol=1e-12)


def test_softmax_rows_is_stable_for_large_logits():
    # Naive exp() would overflow to inf/inf -> nan here.
    z = np.array([[1000.0, 1001.0, 1002.0], [-800.0, -799.0, -801.0]])
    out = nn_cpp.softmax_rows(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-12)


def test_engine_dispatches_to_cpp_and_is_numerically_identical(monkeypatch):
    monkeypatch.setattr(engine, "_USE_CPP", True)
    rng = np.random.default_rng(11)
    a, b = rng.standard_normal((32, 64)), rng.standard_normal((64, 8))

    dispatch_data = (Tensor(a) @ Tensor(b)).data

    monkeypatch.setattr(engine, "_USE_CPP", False)
    numpy_data = (Tensor(a) @ Tensor(b)).data

    np.testing.assert_allclose(dispatch_data, numpy_data, rtol=1e-12, atol=1e-12)


def test_gradient_check_still_passes_on_cpp_path(monkeypatch):
    """The backward pass is unchanged NumPy; this proves the C++ forward keeps it correct."""
    monkeypatch.setattr(engine, "_USE_CPP", True)
    rng = np.random.default_rng(3)
    x_data = rng.standard_normal((6, 5))
    w_data = rng.standard_normal((5, 4))
    upstream = rng.standard_normal((6, 4))

    def loss(w):
        return np.sum((x_data @ w) * upstream)

    numeric = np.zeros_like(w_data)
    eps = 1e-6
    for idx in np.ndindex(*w_data.shape):
        orig = w_data[idx]
        w_data[idx] = orig + eps
        f1 = loss(w_data)
        w_data[idx] = orig - eps
        f2 = loss(w_data)
        w_data[idx] = orig
        numeric[idx] = (f1 - f2) / (2 * eps)

    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)
    out = x @ w
    out.grad = upstream
    out._backward()

    np.testing.assert_allclose(w.grad, numeric, atol=1e-6)
