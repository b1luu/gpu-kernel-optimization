# Analysis 1: Naive vs. Coalesced SGEMM

**Setup:** M=N=K=1024, `block(16,16)`, FP32. Timed as a 10-launch average after 3 warm-up launches. Profiled with `ncu --set full`.

## The change

Both kernels are identical except for which thread-index axis drives which output coordinate:

```cpp
// naive.cuh                       // coalesced.cuh
int col = ... + threadIdx.y;       int row = ... + threadIdx.y;
int row = ... + threadIdx.x;       int col = ... + threadIdx.x;
```

`threadIdx.x` is the fastest-varying index within a warp. In `coalesced.cuh` it drives `col`, so `B[k*N + col]` becomes a run of contiguous addresses across the warp — a true coalesced load. In `naive.cuh` it drives `row`, so `A[row*K + k]` becomes 32 addresses each `K` floats apart — a badly strided load.

## Result

| | time/launch | GFLOPS |
|---|---|---|
| naive | ~1.15 ms | ~186 |
| coalesced | ~0.25–0.29 ms | ~730–850 |

**~4x faster** from the index swap alone — no change to the arithmetic.

## Why (from Nsight Compute)

Both kernels are memory-bound (~96–98% of peak memory throughput either way — swapping indices doesn't reduce *how much* data moves, only how efficiently each transaction is used).

| Metric | naive | coalesced |
|---|---|---|
| L1/TEX Hit Rate | 87.63% | 87.63% |
| L2 Hit Rate | 97.86% | 90.73% |
| Global load efficiency | flagged: only 18/32 bytes/sector used | flagged: only 18/32 bytes/sector used |

Both reports flag the same "18 of 32 bytes per sector" warning. `coalesced.cuh` fixed `B`'s access (fully coalesced now), but `A[row*K+k]` is a **broadcast** — 16 threads reading one shared float — which is cheap (no serialization) but still pulls a full 32-byte sector per instruction while using only 4 bytes of it. That's the source of the ~4x measured gain being "only" 4x and not larger, and it's also the headroom that's still on the table.

## Next step

Shared-memory tiling: load a tile of A and B into shared memory once per block, then have all threads reuse it across multiple FMAs instead of re-issuing a global load (even a cheap broadcast one) every iteration. That's the fix for the remaining 18/32-byte inefficiency in both kernels.

## Caveat on profiling numbers

Don't compare raw `Duration` between `.ncu-rep` files as if they were real speed — `--set full` does 45 replay passes per launch with cache flushing between them, which changes runtime behavior. Use `ncu` output for *root cause* (hit rates, sector utilization, stalls); use plain unprofiled `test.exe` runs for actual GFLOPS.
