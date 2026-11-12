// C++ kernels for the from-scratch autodiff engine.
//
// Scope: the two hot forward kernels the MLP spends nearly all of its time in
// -- the 2-D matmul used by every Linear layer, and row-wise softmax. Backward
// stays in NumPy (it is a handful of transposed matmuls and elementwise ops),
// so this file stays deliberately small: a portable, dependency-free forward
// path the Python engine can dispatch to.
//
// Both functions are pure: no global state, no GIL tricks. Inputs are coerced
// to C-contiguous float64, which is what the engine already stores, so in the
// common case the coercion is a no-op view rather than a copy.
//
// Honest performance note: a hand-written loop does not compete with
// Accelerate's BLAS on Apple silicon, which reaches ~285 GFLOP/s through the
// AMX matrix coprocessor versus ~10 GFLOP/s here. bench/bench_matmul.py
// measures both the kernel-level gap and what it does -- and does not -- mean
// for a full training step.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

#if defined(__clang__) || defined(__GNUC__)
#define NN_RESTRICT __restrict__
#else
#define NN_RESTRICT
#endif

using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;

void check_2d(const Matrix& a, const char* name) {
    if (a.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be 2-D, got " +
                                    std::to_string(a.ndim()) + "-D");
    }
}

// C = A @ B with A (M,K), B (K,N), C (M,N).
//
// Loop order i-k-j: the inner loop walks a row of B and a row of C, both
// contiguous, and reuses the single scalar A[i,k] across the whole inner loop.
// NN_RESTRICT lets the compiler assume the three buffers do not alias, which
// is what allows the inner loop to be unrolled/vectorized at all; without it
// every store to C could invalidate a loaded B element.
Matrix matmul(const Matrix& a, const Matrix& b) {
    check_2d(a, "a");
    check_2d(b, "b");
    const py::ssize_t M = a.shape(0);
    const py::ssize_t K = a.shape(1);
    const py::ssize_t Kb = b.shape(0);
    const py::ssize_t N = b.shape(1);
    if (K != Kb) {
        throw std::invalid_argument(
            "shape mismatch: a is (" + std::to_string(M) + "," + std::to_string(K) +
            "), b is (" + std::to_string(Kb) + "," + std::to_string(N) + ")");
    }

    Matrix out({M, N});
    const double* NN_RESTRICT pa = a.data();
    const double* NN_RESTRICT pb = b.data();
    double* NN_RESTRICT pc = out.mutable_data();
    for (py::ssize_t i = 0; i < M; ++i) {
        double* crow = pc + i * N;
        for (py::ssize_t j = 0; j < N; ++j) {
            crow[j] = 0.0;
        }
        for (py::ssize_t k = 0; k < K; ++k) {
            const double aik = pa[i * K + k];
            const double* brow = pb + k * N;
            for (py::ssize_t j = 0; j < N; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
    return out;
}

// Row-wise numerically-stable softmax: subtract the row max before exp so a
// large logit cannot overflow, which is the same trick nn/layers.py uses in
// Python.
Matrix softmax_rows(const Matrix& z) {
    check_2d(z, "z");
    const py::ssize_t M = z.shape(0);
    const py::ssize_t N = z.shape(1);
    Matrix out({M, N});
    const double* pz = z.data();
    double* po = out.mutable_data();

    for (py::ssize_t i = 0; i < M; ++i) {
        const double* row = pz + i * N;
        double* orow = po + i * N;
        double max_val = -std::numeric_limits<double>::infinity();
        for (py::ssize_t j = 0; j < N; ++j) {
            if (row[j] > max_val) max_val = row[j];
        }
        double sum = 0.0;
        for (py::ssize_t j = 0; j < N; ++j) {
            const double e = std::exp(row[j] - max_val);
            orow[j] = e;
            sum += e;
        }
        const double inv = 1.0 / sum;
        for (py::ssize_t j = 0; j < N; ++j) {
            orow[j] *= inv;
        }
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(nn_cpp, m) {
    m.doc() = "C++ forward kernels for the from-scratch engine (matmul, softmax)";
    m.def("matmul", &matmul, py::arg("a"), py::arg("b"),
          "C = A @ B for C-contiguous float64 2-D arrays.");
    m.def("softmax_rows", &softmax_rows, py::arg("z"),
          "Row-wise, numerically-stable softmax of a 2-D array.");
}
