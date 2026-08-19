# SGEMM CUDA

Learning project: implementing and optimizing single-precision GEMM (`C = alpha*A*B + beta*C`) in CUDA, benchmarking each step against the last.

## Structure

```
src/kernels/     naive.cuh, coalesced.cuh — implemented
                 shared_mem.cuh, tiled.cuh — planned next
src/main.cu      (stub)
include/utils.cuh
benchmark/bench.py
test.cu          benchmark harness: builds/runs whichever kernel header it #includes
documents/       write-up for each optimization step
```

## Build & run

```
make test        # builds test.exe via nvcc (uses MSVC path set in Makefile)
./test.exe
```

`test.cu` includes one kernel header at a time — swap the `#include "src/kernels/..."` line to benchmark a different kernel.

## Profiling

```
ncu --set full -o profile_name -f .\test.exe
ncu-ui .\profile_name.ncu-rep
```

## Results so far

| Kernel | GFLOPS (1024³) | Notes |
|---|---|---|
| naive | ~186 | uncoalesced global loads |
| coalesced | ~730–850 | ~4x from fixing memory access pattern alone |

See [documents/](documents/) for detailed analysis of each step.
