# SGEMM CUDA

Learning project: implementing and optimizing single-precision GEMM (`C = alpha*A*B + beta*C`) in CUDA, benchmarking each step against the last.

## Structure

```
src/kernels/     naive.cuh, coalesced.cuh, shared_mem.cuh, register_blocked.cuh,
                 register_blocked_2d.cuh, vectorized.cuh, vectorized_padded.cuh,
                 warp_tiled.cuh, warp_tiled_padded.cuh
src/main.cu      standalone runner for shared_mem.cuh (M=N=K=512, correctness check only)
include/utils.cuh
benchmark/bench.py
bench_harness.cuh          shared benchmark logic (malloc/timing/verify) used by every test_*.cu below
test.cu                    thin wrapper: builds/runs whichever kernel header it #includes (naive/coalesced/shared_mem)
test_register.cu           thin wrapper for register_blocked.cuh
test_register_2d.cu        thin wrapper for register_blocked_2d.cuh
test_vectorized.cu         thin wrapper for vectorized.cuh
test_vectorized_padded.cu  thin wrapper for vectorized_padded.cuh
test_warp_tiled.cu         thin wrapper for warp_tiled.cuh
test_warp_tiled_padded.cu  thin wrapper for warp_tiled_padded.cuh
bench_cublas.cu            thin wrapper around cublasSgemm, for a reference/upper-bound comparison
documents/                 write-up for each optimization step
```

## Build & run

```
make test        # builds test.exe via nvcc (uses MSVC path set in Makefile)
./test.exe
```

`test.cu` includes one kernel header at a time — swap the `#include "src/kernels/..."` line to benchmark a different kernel.

Every kernel shares the identical signature `matMult(A, B, C, alpha, beta, M, N, K)`, so `bench_harness.cuh`'s `runBenchmark(label, M, N, K, launch)` handles all the shared benchmark mechanics (malloc, memcpy, warm-up, 5-trial timing with min/median, CPU-reference verification) once; each `test_*.cu` is just its `#include`, a `dim3 block`/`dim3 grid` computed from that kernel's own tile macros, and a one-line lambda calling `matMult<<<grid, block>>>(...)`. To benchmark a new kernel, copy any existing `test_*.cu` and swap those three things.

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
| vectorized (float4) | ~7057 (min ~7485) | ~1.24x more from float4 loads/stores — ~13% behind cuBLAS |
| vectorized_padded | min ~7574 | ~1.2% more from fixing one of two flagged bank conflicts (padding only fixes the inter-row kind) |
| warp_tiled | min ~7995 | structural fix (no padding needed) for store coalescing and one bank conflict — ~7% behind cuBLAS |
| warp_tiled_padded | min ~8174 | + padding fixes the one remaining conflict warp-tiling didn't touch — ~4.6% behind cuBLAS |
| cuBLAS (`cublasSgemm`) | min ~8551 | reference/upper bound |

Each `test_*`/`bench_cublas` binary runs 5 independent timed trials (10 launches each) per process and prints both `min` and `median` — a single trial's number can be skewed by GPU boost-clock state, so don't trust a one-off run; rerun a few times if a result looks surprising (see [documents/03_register_blocking.md](documents/03_register_blocking.md) for a case where that mattered). For `vectorized`/`vectorized_padded` specifically, prefer `min` over `median` — see [documents/06_shared_mem_padding.md](documents/06_shared_mem_padding.md).

See [documents/](documents/) for detailed analysis of each step.
