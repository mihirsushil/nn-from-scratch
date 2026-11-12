# nn-from-scratch

A neural network trained on real MNIST digits, built entirely on a
hand-written automatic differentiation engine — no PyTorch, no TensorFlow,
no autograd library. The point was to actually understand backpropagation,
not just call `.fit()`.

## What's in here

- `nn/engine.py` — a tiny reverse-mode autodiff engine. A `Tensor` wraps a
  NumPy array and records how it was produced (which operation, which
  parent tensors). Calling `.backward()` walks that computation graph
  backwards and accumulates gradients via the chain rule. This is the same
  idea as [micrograd](https://github.com/karpathy/micrograd), except it
  operates on NumPy arrays instead of individual scalars, which is what
  makes it fast enough to train on 60,000 real images instead of toy
  examples.
- `nn/layers.py` — `Linear`, `ReLU`, `Sequential`, and a numerically-stable
  `cross_entropy_loss`, all built out of `Tensor` ops so gradients for the
  whole network fall out of the engine automatically — no layer has its own
  hand-derived backward pass.
- `nn/optim.py` — plain SGD with momentum.
- `tests/test_engine.py` — **gradient checking**: for every op (and for a
  full 2-layer MLP + cross-entropy loss), the analytic gradient from
  `.backward()` is compared against a numerical gradient computed by
  finite differences. This is the standard way to convince yourself an
  autodiff implementation is actually correct, not just "runs without
  crashing."
- `train_mnist.py` — loads real MNIST (784-pixel, 10-class) via
  scikit-learn, trains a 784→128→64→10 MLP with mini-batch SGD, and plots
  loss/accuracy curves.
- `csrc/nn_cpp.cpp` + `tests/test_cpp_ops.py` — an optional C++ (pybind11)
  implementation of the engine's two hot forward kernels, matmul and softmax,
  gradient-checked against the NumPy path. Off by default; see
  (see below) for the measurements that explain why.
- `bench/bench_matmul.py` — kernel-level and step-level benchmarks; writes
  `results/matmul_benchmark.md`.

## How backprop works here, briefly

Every `Tensor` op (`+`, `*`, `@`, `relu()`, `exp()`, `log()`, `sum()`, ...)
does two things when called: it computes the forward result, and it stashes
a small closure describing how to turn "the gradient of the loss with
respect to my output" into "the gradient of the loss with respect to my
inputs" (the local application of the chain rule). `backward()` topologically
sorts the graph and calls those closures in reverse order, so each
parameter ends up with `dLoss/dParameter` — exactly what `SGD.step()` needs.

The one non-obvious wrinkle is broadcasting: `x @ W + b` broadcasts a
`(features,)` bias against a `(batch, features)` activation, so its gradient
has to be summed back down to `(features,)` — see `Tensor._unbroadcast`.

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

pytest tests/            # gradient-check the engine
python3 train_mnist.py   # train on MNIST, saves results/training_curves.png
```

> **Note (macOS + python.org installs):** if `train_mnist.py` fails with
> `SSL: CERTIFICATE_VERIFY_FAILED` while downloading MNIST, this Python
> install has no local CA bundle. Fix without touching system Python:
> `pip install certifi` then run with
> `SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())") python3 train_mnist.py`.

## Results

Trained for 15 epochs, batch size 128, SGD with momentum 0.9, lr 0.1, on the
full 70,000-image MNIST set (60k train / 10k test):

- **97.62% test accuracy**, 99.56% train accuracy
- ~0.7s per epoch (pure NumPy, CPU) — the whole run finishes in under 15s
- All 8 gradient-check tests pass (analytic vs. numerical gradients agree
  to 1e-5), so the accuracy is coming from a verifiably correct
  implementation, not a lucky bug that happens to train anyway

![training curves](results/training_curves.png)

## C++ kernels: measured, and deliberately not the default

`csrc/nn_cpp.cpp` implements the engine's two hot forward kernels — 2-D matmul
and row-wise softmax — as a small pybind11 extension, and `Tensor.__matmul__`
dispatches to it when the extension is built and `NN_CPP=1` is set. It is
opt-in on purpose. `bench/bench_matmul.py` is the reason:

| | NumPy | C++ kernel |
|---|---:|---:|
| 128x784 @ 784x128 | 0.09 ms (~285 GFLOP/s) | 2.50 ms (~10 GFLOP/s) |
| 1024x1024 @ 1024x1024 | 8.4 ms (~273 GFLOP/s) | 208 ms (~10 GFLOP/s) |
| full training step (batch 128) | 0.65 ms | 3.37 ms |

Two findings, and the second one is the useful one:

1. **A hand-written loop cannot compete with Accelerate on Apple silicon.** The
   BLAS path reaches ~285 GFLOP/s because Accelerate dispatches to the AMX
   matrix coprocessor; the plain `i-k-j` loop saturates around 10 GFLOP/s. This
   is the expected outcome, and it is measured here rather than assumed.
2. **The kernel was never the bottleneck.** Matmul is only ~19% of a NumPy
   training step; the other ~80% is Python graph construction (Tensor
   allocation, closures, broadcasting). No kernel swap can remove that, so
   making the C++ path the default would have made the project ~5x slower for
   no structural gain — hence opt-in.

The honest answer to "should this engine get a C++ backend?" is: not for speed.
The lever that would actually move step time is graph-level fusion (or a C++
autograd core), not a matmul kernel. The extension stays as a readable
reference implementation and as the artifact behind these numbers.

```bash
pip install pybind11 setuptools
python setup.py build_ext --inplace
NN_CPP=1 pytest tests/              # exercise the C++ path as well as NumPy
python bench/bench_matmul.py        # rewrites results/matmul_benchmark.md
```

Both paths are checked by the same gradient tests: `tests/test_cpp_ops.py`
verifies the C++ kernels against NumPy to 1e-12 and re-runs the finite-difference
gradient check with the C++ forward active, so the backward pass is proven to
stay correct whichever forward is used.
