# Analysis 8: Warp Tiling + Padding Combined

**Setup:** M=N=K=1024, FP32. `warp_tiled_padded.cuh` is `warp_tiled.cuh` (doc [07](07_warp_tiling.md)) plus one independent change: `As[BK][BM]`→`As[BK][BM+4]`, the same `+4` padding technique from `vectorized_padded.cuh` (doc [06](06_shared_mem_padding.md)). `Bs` is left unpadded — its bank conflict was already fixed by warp-tiling itself, not padding. All comparisons below are interleaved runs (`warp_tiled` / `warp_tiled_padded` / `cuBLAS` alternating) with the GPU confirmed idle beforehand (`nvidia-smi --query-compute-apps` checked clean, no game or other CUDA process running — see the caveat in doc 07 about why that check matters).

## The change

Doc 07 found that warp-tiling fixed three of four flagged issues (global load coalescing, global store coalescing, shared load bank conflict) purely by restructuring the compute phase's thread-to-output mapping — but left one survivor: a 2.4-way shared **store** bank conflict, because it lives in the block-level cooperative load of `As`, code warp-tiling never touched. That load is structurally identical to `vectorized.cuh`'s, which doc 06 already fixed with `+4` padding. This kernel just applies that same, already-proven, independent fix on top of the warp-tiled structure — testing whether the two fixes compose cleanly rather than assuming they do.

## Result

Interleaved with `warp_tiled` and `cublasSgemm`, 6 rounds, GPU confirmed idle:

| kernel | avg min GFLOPS | avg median GFLOPS |
|---|---|---|
| warp_tiled | ~7995 | ~7447 |
| **warp_tiled_padded** | **~8174** | **~7781** |
| cuBLAS | ~8551 | ~7959 |

Gap to cuBLAS closed from **~7% → ~4.6%** (min) and **~9% → ~2.3%** (median) — the two fixes composed cleanly, each contributing its own independent improvement, with no sign of interference.

## Why (from Nsight Compute)

Profiled with `ncu --set full -o profile_warp_tiled_padded test_warp_tiled_padded.exe` (`block(128,1,1)`, `grid(16,16)`, 66 registers/thread — unchanged from `warp_tiled.cuh`).

| Metric | warp_tiled | warp_tiled_padded |
|---|---|---|
| Registers/thread | 66 | 66 (unchanged — padding adds ~120 bytes of shared memory, doesn't touch register allocation) |
| Theoretical / Achieved Occupancy | 58.3% / 48.0% | 58.3% / 48.2% (unchanged) |
| Shared store bank conflict | flagged, 2.4-way (Est. Speedup 26.4%) | **entire Memory Workload Analysis Tables section is empty — nothing flagged** |
| Compute (SM) Throughput | 58.59% | 60.31% |
| Memory Throughput | 76.66% | 75.70% |

Every memory-access-pattern issue Nsight was willing to flag in `warp_tiled.cuh` is gone. Registers and occupancy are identical, confirming the improvement came specifically from removing the conflict, not from some occupancy side-effect of the extra shared memory. The tail-effect finding (`Est. Speedup 50%`, same `grid(16,16)` / `1.52` waves) is also unchanged — expected, since padding doesn't touch grid shape.

## What's left

With every profiler-flagged issue now resolved, the remaining ~4.6%/2.3% gap to cuBLAS is most plausibly the tail effect (`Est. Speedup 50%` — `256` blocks against `24` SMs leaves a `1.52`-wave grid with a costly partial wave) and the 58.3%-capped occupancy (still register-limited at 66/thread) — neither of which has a small, targeted fix the way the last three rounds did. Closing those would mean either a substantially different tile-size search (trading occupancy against per-thread work) or the other untouched lever from doc 03: double buffering, which targets the `__syncthreads()` bubble at each tile boundary rather than anything visible in this profile.

Given the series has gone from ~186 GFLOPS (naive) to within ~2–5% of a production BLAS library through seven concrete, individually-profiled steps, this is a reasonable place to consider the core learning arc complete.

## Caveat on profiling numbers

Same caveat as [01](01_naive_vs_coalesced.md#caveat-on-profiling-numbers)–[07](07_warp_tiling.md#caveat-on-profiling-numbers): don't compare raw `Duration` between `.ncu-rep` files — `--set full` does 45 replay passes with cache flushing between them. Use `ncu` for root-cause metrics; use the unprofiled min/median numbers above for actual GFLOPS, and check `nvidia-smi --query-compute-apps` before trusting a surprising comparison.
