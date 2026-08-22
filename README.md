# SGEMM CUDA

Learning project: implementing and optimizing single-precision GEMM (`C = alpha*A*B + beta*C`) in CUDA, benchmarking each step against the last.

## Structure

```
src/kernels/     naive.cuh, coalesced.cuh, shared_mem.cuh, register_blocked.cuh,
                 register_blocked_2d.cuh, vectorized.cuh — implemented
                 tiled.cuh — unused placeholder
src/main.cu      standalone runner for shared_mem.cuh (M=N=K=512, correctness check only)
include/utils.cuh
benchmark/bench.py
test.cu             benchmark harness: builds/runs whichever kernel header it #includes
test_register.cu    same harness, dedicated to register_blocked.cuh (different launch shape)
test_register_2d.cu same harness, dedicated to register_blocked_2d.cuh (different launch shape)
test_vectorized.cu  same harness, dedicated to vectorized.cuh (different launch shape)
bench_cublas.cu     same harness against cublasSgemm, for a reference/upper-bound comparison
documents/          write-up for each optimization step
```

## Build & run

```
make test        # builds test.exe via nvcc (uses MSVC path set in Makefile)
./test.exe
```

`test.cu` includes one kernel header at a time — swap the `#include "src/kernels/..."` line to benchmark a different kernel.

To build the cuBLAS reference benchmark (needs `-lcublas`):

```
nvcc -ccbin <path-to-cl.exe-dir> -arch=sm_89 -O3 bench_cublas.cu -lcublas -o bench_cublas.exe
```

## Profiling

```
ncu --set full -o profile_name -f .\test.exe
ncu-ui .\profile_name.ncu-rep
```

## Results so far

| Kernel | GFLOPS (1024³, median of 5 trials) | Notes |
|---|---|---|
| naive | ~231 | uncoalesced global loads |
| coalesced | ~783 | ~3.4x from fixing memory access pattern alone |
| shared_mem (tiled) | ~968 | ~1.2x more from reusing global loads via shared memory |
| register_blocked (1D) | ~3350 | ~3.5x more from amortizing each shared-memory load across TM FMAs |
| register_blocked_2d | ~5703 | ~1.7x more from caching both A and B into registers (outer-product FMAs) |
| vectorized (float4) | ~7057 | ~1.24x more from float4 loads/stores — still ~13% behind cuBLAS |
| cuBLAS (`cublasSgemm`) | ~7964 | reference/upper bound |

Each `test_*`/`bench_cublas` binary runs 5 independent timed trials (10 launches each) per process and prints both `min` and `median` — a single trial's number can be skewed by GPU boost-clock state, so don't trust a one-off run; rerun a few times if a result looks surprising (see [documents/03_register_blocking.md](documents/03_register_blocking.md) for a case where that mattered).

See [documents/](documents/) for detailed analysis of each step.
