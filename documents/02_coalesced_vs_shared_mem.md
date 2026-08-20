# Analysis 2: Coalesced vs. Shared-Memory Tiled SGEMM

**Setup:** M=N=K=1024, `block(16,16)`, FP32, `TILE=16`. Timed as a 10-launch average after 3 warm-up launches (`test_coalesced.exe` / `test_sharedmem.exe`, both built from `test.cu` with the include swapped).

## The change

`coalesced.cuh` reads `A[row*K + k]` and `B[k*N + col]` straight from global memory on every iteration of the `k` loop. The `B` access is coalesced across the warp; the `A` access is a broadcast (16 threads reading the same address) — cheap, but it still pulls a full 32-byte sector per instruction while only 4 bytes are used, and it does this fresh every iteration even though many threads in the block will re-read the same data.

`shared_mem.cuh` fixes the *reuse* problem, not the access-pattern problem: each block cooperatively loads a `TILE x TILE` (16x16) tile of `A` and a `TILE x TILE` tile of `B` into `__shared__` arrays once per tile-step, `__syncthreads()`, then every thread does `TILE` FMAs out of shared memory before the block moves to the next tile along `K`. Each global-memory element loaded is now reused `TILE` times from fast on-chip shared memory instead of being re-fetched from global memory (or re-broadcast from L1/L2) on every `k`.

## Result

| | time/launch | GFLOPS |
|---|---|---|
| naive | ~11.24 ms | ~191 |
| coalesced | ~2.69 ms | ~798 |
| shared_mem (tiled) | ~2.23 ms | ~964 |

**~1.2x faster than coalesced, ~5x faster than naive.** The jump is smaller than naive→coalesced because coalescing already fixed the dominant problem (bad access *pattern*); tiling is squeezing the smaller remaining inefficiency (redundant global *traffic*).

## Why (from Nsight Compute)

Profiled with `ncu --set full -o profile_sharedmem test_sharedmem.exe` (GeForce RTX 4060, compute capability 8.9, `block(16,16)`, 38 registers/thread).

| Metric | coalesced | shared_mem |
|---|---|---|
| DRAM Throughput | (memory-bound) | **2.78%** |
| L1/TEX Cache Throughput | — | **96.59%** |
| L1/TEX Hit Rate | 87.63% | **8.08%** |
| L2 Hit Rate | 90.73% | 97.39% |
| Compute (SM) Throughput | — | 96.29% |
| SM Busy / Issue Slots Busy | — | 34.81% / 30.81% (flagged "Low Utilization") |

The bottleneck moved. `coalesced.cuh` was memory-bound on **global memory/DRAM** — every `k` iteration re-issued a global load. `shared_mem.cuh` fixed that: **DRAM Throughput dropped to 2.78%**, confirming the tile is genuinely being reused out of shared memory instead of re-fetched from global memory.

But the bottleneck didn't disappear — it moved into the **shared-memory pipe itself**. On this GPU, shared memory and L1 cache are the same physical unit, so L1/TEX Cache Throughput (96.59%) now mostly reflects `As`/`Bs` shared-memory traffic, not global-load reuse — which is also why L1/TEX Hit Rate *dropped* to 8.08% (there's little redundant global-load traffic left for L1 to catch; the reuse work was handed off to shared memory).

Nsight's own diagnosis: **Mio Throttle Stalls** — each warp averages 19.6 stall cycles waiting for the MIO (memory input/output) instruction queue, ~51% of the 38.3 cycles between issuing two instructions. Its note: *"trying to use fewer but wider loads can reduce this pressure."* That lines up with **SM Busy at only 34.81%** (flagged "Low Utilization," est. local speedup 85.74%) despite 96%+ Speed-of-Light throughput numbers — the SM isn't short on work, it's throttled on issuing the sheer number of small (4-byte) shared-memory load instructions the inner loop generates: two shared loads (`As[ty][k]`, `Bs[k][tx]`) per FMA.

## Next step

The profiling data points at **register blocking**: have each thread compute multiple output elements (e.g. a 4x1 or 4x4 tile per thread) so each shared-memory load is reused across several FMAs instead of one. This directly reduces the shared-memory instruction count causing the MIO throttle stalls above, raising arithmetic intensity per load rather than per global-memory transaction (which tiling already solved). Increasing `TILE` is a secondary lever (more reuse per load, but more shared memory + fewer resident blocks/SM).

## Caveat on profiling numbers

Same caveat as [01](01_naive_vs_coalesced.md#caveat-on-profiling-numbers): don't compare raw `Duration` between `.ncu-rep` files — `--set full` does 45 replay passes with cache flushing between them. Use `ncu` for root-cause metrics; use the unprofiled `test_*.exe` runs above for actual GFLOPS.
