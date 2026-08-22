# Analysis 5: Vectorized (float4) Loads and Stores

**Setup:** M=N=K=1024, FP32. `vectorized.cuh`: `BM=BN=128`, `BK=8`, `TM=TN=8`, 256 threads/block, `grid(N/BN, M/BM)` = `(8,8)`. Compared against `register_blocked_2d.cuh` (`BM=BN=64`, `TM=TN=4`, 256 threads, `grid(16,16)`). Timed via `test_register_2d.exe` / `test_vectorized.exe` — 5 trials of a 10-launch average each, min/median reported.

## The change

`register_blocked_2d.cuh`'s profile ([04](04_2d_register_blocking.md)) flagged uncoalesced global stores as its single largest issue (Est. Speedup 56%): adjacent threads wrote 4-byte values 16 bytes apart, wasting 24 of every 32 bytes moved. `vectorized.cuh` fixes the mechanism directly — every global load and the final store use `float4` (128-bit) transactions instead of scalar ones, cutting memory-instruction count roughly 4x and packing each thread's output into contiguous 16-byte writes.

Two structural changes came with it: `As` is now stored **transposed** (`As[BK][BM]`) so `regM`'s `TM` consecutive M-values are contiguous in shared memory (required for a vectorized read), and the tile size grew to `BM=BN=128, TM=TN=8` — not an arbitrary choice, but the smallest config where `(BM*BK)/4` and `(BK*BN)/4` divide evenly by the 256-thread block, so every thread loads exactly one `float4` with no stride loop. The first config tried (`BM=BN=64, TM=TN=4`, matching `register_blocked_2d.cuh`'s tile size) failed this check — it would have left half the threads idle during the tile load.

## Result

| kernel | median GFLOPS (1024³) |
|---|---|
| register_blocked_2d | ~5703 |
| **vectorized** | **~7057** (avg median across 5 separate process launches) |
| cuBLAS (`cublasSgemm`) | ~7964 (avg median across 5 separate process launches) |

**~1.24x faster than 2D blocking**, and **~13% behind cuBLAS** — not the ~5% first reported here. That number was a same-batch mismatch: comparing `vectorized`'s *min* (7482) against an *older cuBLAS median* (7893) from a different run. Rerunning both 5x as separate processes (the same pitfall-check as [03](03_register_blocking.md#a-benchmarking-pitfall-worth-writing-down)) showed cuBLAS ahead in every single run, with no overlap between the two distributions — median-to-median and min-to-min both land at roughly the same ~12–13% gap. Always compare the same statistic from the same batch; a mismatched comparison across two different runs can look better (or worse) than reality even when neither number is individually wrong.

## Why (from Nsight Compute)

Profiled with `ncu --set full -o profile_vectorized test_vectorized.exe` (`block(256,1,1)`, `grid(8,8)`, 105 registers/thread).

| Metric | register_blocked_2d | vectorized |
|---|---|---|
| Registers/thread | 56 | **105** |
| Theoretical Occupancy | 66.7% | **33.3%** — collapsed; only 2 blocks/SM fit (register-limited) |
| Achieved Occupancy | 58.93% | 28.18% |
| Waves per SM / tail effect | 2.67 waves, Est. Speedup 33.3% | **1.33 waves, Est. Speedup 50%** — 1 full wave + a 16-block partial wave |
| Uncoalesced global stores | 8.0/32 bytes/sector, Est. Speedup 56.3% | **16.0/32 bytes/sector, Est. Speedup 34.0%** |
| Uncoalesced global loads | 29.3/32 bytes/sector, Est. Speedup 6.26% | 30.2/32 bytes/sector, Est. Speedup 3.78% |
| Shared memory bank conflicts | not flagged | **new: 5-way on loads (Est. 31.6%), 2.4-way on stores (Est. 26.4%)** |
| Compute / Memory Throughput | 57.0% / 75.1% | 50.7% / 68.0% |

Two things got measurably *worse* here — register pressure and occupancy both cratered (105 regs/thread from the `threadResults[64]` accumulator plus `regM[8]`/`regN[8]`; only 33.3% theoretical occupancy; a much costlier 50%-estimated tail effect from the smaller 64-block grid) — and the kernel still got 1.3x faster. The reason: total instruction count dropped enough (fewer, wider memory transactions) that raw throughput won out over parallelism/occupancy. Worth remembering as a general lesson: occupancy and tail-effect percentages are not simply monotonic with wall-clock performance.

**Store coalescing improved but isn't fully fixed.** Bytes-per-sector doubled (8→16 of 32) — consistent with the write size doubling from 4 bytes/thread (scalar) to 16 bytes/thread (`float4`) — but it's stuck at 50%, not 100%. The cause: `col = blockCol + threadCol*TN + v*4` with `TN=8` means adjacent threads (`threadCol`, `threadCol+1`) start 8 floats (32 bytes) apart, exactly one full sector — so each 32-byte sector is serviced by only one thread's 16-byte `float4`, not two adjacent threads' `float4`s packed together. The gap between them *is* filled, but by the same thread's own `v=1` iteration a moment later, not by a neighboring thread in the same instruction — which is what the coalescing metric actually measures (bytes used per sector *per transaction*).

**New finding: shared memory bank conflicts.** Not visible in any earlier kernel. Most likely source: `regN`'s read, `Bs[dotIdx][threadCol*TN + v*4]` — with `TN=8`, consecutive threads are 8 floats apart, and `8 * 4 threads = 32` lands exactly on the 32-bank period, so every 4th thread's `float4` read hits the same bank group. This is the textbook bank-conflict trigger (a stride that's a multiple of the bank count) and the standard fix is padding the shared array (e.g. `Bs[BK][BN+1]`) to break the alignment.

## Next step

Two concrete, well-understood fixes remain, in order of likely impact:

1. **Pad the shared memory arrays** (`As[BK][BM+1]`, `Bs[BK][BN+1]`) to break the stride-32 bank-conflict alignment — cheap, mechanical, and directly targets the two new bank-conflict findings (Est. Speedup ~26–32% combined).
2. **Restructure the store-coalescing pattern** — the remaining 50%-utilized stores need adjacent threads' `float4`s to be address-adjacent, which the current `threadCol*TN` blocked mapping doesn't provide. This is what warp-tiling schemes (used in cuBLAS/CUTLASS) solve properly, but it's a bigger redesign than anything done so far in this series.

Given the kernel is already within ~5% of cuBLAS, this is a reasonable place to treat the learning series as complete unless you want to chase that last gap — double buffering (overlapping the next tile's load with the current tile's compute) is the other classic lever nothing here has touched yet.

## Caveat on profiling numbers

Same caveat as [01](01_naive_vs_coalesced.md#caveat-on-profiling-numbers)–[04](04_2d_register_blocking.md#caveat-on-profiling-numbers): don't compare raw `Duration` between `.ncu-rep` files — `--set full` does 45 replay passes with cache flushing between them (this run showed ~5 GFLOPS profiled vs. ~7471 GFLOPS unprofiled — expected, not a regression). Use `ncu` for root-cause metrics; use the unprofiled `test_vectorized.exe` min/median numbers above for actual GFLOPS.
