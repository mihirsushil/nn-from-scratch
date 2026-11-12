# Kernel benchmarks

Generated 2026-10-29 02:12 on macOS-26.5.1-arm64-arm-64bit-Mach-O. Median of N calls, milliseconds.

## Matmul: C++ vs NumPy

`numpy (1 thread)` pins Accelerate/OpenBLAS to one thread, so the comparison is kernel-to-kernel rather than kernel-to-BLAS-thread-pool.

| Shape | Where | numpy | numpy (1T) | C++ | C++ GFLOP/s | vs numpy | vs numpy 1T |
|---|---|---:|---:|---:|---:|---:|---:|
| 128x784 @ 784x128 | layer 1, batch 128 | 0.089 | 0.090 | 2.521 | 10.2 | 0.04x | 0.04x |
| 128x128 @ 128x64 | layer 2, batch 128 | 0.009 | 0.009 | 0.158 | 13.3 | 0.06x | 0.06x |
| 128x64 @ 64x10 | layer 3, batch 128 | 0.004 | 0.004 | 0.039 | 4.2 | 0.10x | 0.10x |
| 512x784 @ 784x128 | layer 1, batch 512 | 0.354 | 0.355 | 10.150 | 10.1 | 0.03x | 0.03x |
| 1024x1024 @ 1024x1024 | square reference | 8.180 | 8.820 | 213.000 | 10.1 | 0.04x | 0.04x |

>1 means the C++ kernel won. It does not, and cannot on this hardware: Accelerate
dispatches these shapes to the AMX matrix coprocessor, which a plain loop can't
reach. The interesting number is the gap, not a win.

## Training step: what the kernel is worth

| Path | step (ms) | matmul (ms) | matmul share |
|---|---:|---:|---:|
| NumPy | 0.65 | 0.13 | 19.4% |
| C++ dispatch | 3.37 | 2.78 | 82.4% |

Two findings, and the second is the point. First, the NumPy path is faster end to end:
the C++ dispatch makes a step 5.1x *slower*, because Accelerate's matmul is
~28x faster than the loop. Second, matmul is only 19% of a NumPy step -- the
other ~80% is Python graph construction (Tensor allocation, closures, broadcasting),
which no kernel swap can remove. So the measured answer to "should this engine get a
C++ backend?" is no, not for speed; the extension is kept opt-in (NN_CPP=1) as a
readable reference kernel and as the benchmark artifact that justifies that call.

If this engine were optimised for real, the lever is graph-level fusion (or a C++
autograd core), not the matmul kernel.
