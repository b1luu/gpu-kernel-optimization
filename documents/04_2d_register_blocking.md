# Analysis 4: 2D Register Blocking

**Setup:** M=N=K=1024, FP32. `register_blocked.cuh` (1D): `BM=BN=64`, `BK=8`, `TM=8`, 512 threads/block. `register_blocked_2d.cuh`: `BM=BN=64`, `BK=8`, `TM=TN=4`, 256 threads/block, `grid(N/BN, M/BM)` = `(16,16)`. Timed via `test_register.exe` / `test_register_2d.exe` — 5 trials of a 10-launch average each, min/median reported (see [03](03_register_blocking.md#a-benchmarking-pitfall-worth-writing-down) for why that matters).

## The change

`register_blocked.cuh` only caches `B`'s value in a register per reduction step (`tmpB`) and reuses it across `TM` FMAs — but every one of those FMAs still re-reads `As` fresh from shared memory. `register_blocked_2d.cuh` caches *both* sides: each thread computes a `TM x TN` tile of outputs instead of `TM x 1`. Per `dotIdx`, it pulls `TM` values from `As` and `TN` values from `B` into registers (`regM`, `regN`) once, then does an **outer product** — `TM*TN` FMAs from only `TM+TN` shared-memory reads, instead of `TM` FMAs from `TM+1` reads. Arithmetic intensity per shared-memory instruction goes up substantially.

Loading also changes shape: `BM*BK` (512) and `BK*BN` (512) no longer equal the thread count (256), so each thread loads 2 elements per array via a strided loop instead of 1:1.

## Result

| kernel | median GFLOPS (1024³) |
|---|---|
| register_blocked (1D) | ~3350 |
| register_blocked_2d | **~5703** |
| cuBLAS (`cublasSgemm`) | ~7893 |

**~1.7x faster than 1D blocking** — the single biggest jump since coalescing. The gap to cuBLAS shrank from ~2.4x to **~1.4x**.

## Why (from Nsight Compute)

Profiled with `ncu --set full -o profile_register_2d test_register_2d.exe` (`block(256,1,1)`, `grid(16,16)`, 56 registers/thread).

| Metric | register_blocked (1D) | register_blocked_2d |
|---|---|---|
| Registers/thread | 44 | **56** |
| Theoretical Occupancy | 66.7% | 66.7% (same bucket, still register-limited) |
| Achieved Occupancy | (not separately measured) | **58.93%** — see tail effect below |
| Compute Throughput | 78.5% | 57.0% |
| Memory Throughput | 78.5% | **75.1%** — now the dominant bottleneck ("Memory is more heavily utilized than Compute") |
| DRAM Throughput | 2.78% (shared_mem baseline) | 21.3% |
| Warp cycles per issued instruction | ~20.8 avg (1D's total-stall figure) | **13.61** — fewer, more expensive cycles-per-instruction bottleneck moved from issue-rate to something else |

Two new, specifically-named inefficiencies showed up that weren't the story in the 1D kernel:

**1. Uncoalesced global stores — Est. Speedup 56.33%, the largest single flagged issue.** Nsight reports only 8.0 of 32 bytes used per store sector. The cause is the write-back indexing: `col = blockCol + threadCol*TN + j`. Adjacent threads (`threadCol`, `threadCol+1`) are `TN=4` floats (16 bytes) apart in memory, not 1 float apart — so a warp's stores land in a strided pattern instead of one contiguous burst. This didn't exist in the 1D kernel because there `col = blockCol + threadCol` (no `*TN` factor) kept adjacent threads writing adjacent columns.

**2. Tail effect from occupancy — Est. Speedup 33.33%.** `grid(16,16)` = 256 blocks; at 66.7% theoretical occupancy this device fits 2 full "waves" of blocks plus a partial wave of 64 blocks. Nsight's model: under uniform block duration, that partial wave alone could account for up to a third of total runtime. This is a new cost introduced by this kernel's specific block/grid/occupancy combination — none of the earlier kernels' grid shapes happened to land on a partial wave this badly.

Register pressure also climbed again (44 → 56/thread) from `regM[TM]`, `regN[TN]`, and the `TM*TN=16`-element `threadResults` array — as predicted in [03](03_register_blocking.md#next-step). Theoretical occupancy stayed at the same 66.7% bucket by coincidence, but achieved occupancy (58.93%) is now explicitly below theoretical, which the tail-effect finding above likely explains.

## Next step

The store-coalescing issue is the highest-leverage, most concrete fix available (56% estimated speedup, larger than any other flagged issue in this report): **vectorized stores** (`float4`) would write 4 contiguous output floats in one 128-bit transaction instead of 4 separate 4-byte ones, which both fixes the stride-4 access pattern and cuts store instruction count 4x. The same `float4` treatment applied to the `As`/`Bs` loads would help the smaller-but-still-flagged 6.26%-speedup load-coalescing issue too. This is the natural next kernel: same `TMxTN` structure, but reinterpret the shared/global buffers as `float4` where the layout allows it.

The tail-effect finding is a secondary, size-specific issue — it's a property of `grid(16,16)` against this GPU's SM count and this kernel's occupancy, not something to redesign the algorithm around. Different `BM`/`BN` choices would shift it, but chasing it before fixing the store-coalescing issue would be solving a smaller problem first.

## Caveat on profiling numbers

Same caveat as [01](01_naive_vs_coalesced.md#caveat-on-profiling-numbers)–[03](03_register_blocking.md#caveat-on-profiling-numbers): don't compare raw `Duration` between `.ncu-rep` files — `--set full` does 45 replay passes with cache flushing between them (this run showed ~5 GFLOPS profiled vs. ~5700 GFLOPS unprofiled — expected, not a regression). Use `ncu` for root-cause metrics; use the unprofiled `test_register_2d.exe` min/median numbers above for actual GFLOPS.
