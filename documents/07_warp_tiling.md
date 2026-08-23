# Analysis 7: Warp Tiling

**Setup:** M=N=K=1024, FP32. `warp_tiled.cuh`: `BM=BN=64, BK=8` (block tile), `WARPS_M=WARPS_N=2` (4 warps/block, 128 threads/block), `WM=WN=32` (warp tile), `TM=8, TN=4` (per-thread tile). Compared against `vectorized.cuh` (`BM=BN=128, TM=TN=8, 256 threads`) and `cublasSgemm`. All numbers below are from interleaved runs (alternating kernel/cuBLAS process launches) after confirming the GPU was otherwise idle — an earlier attempt was contaminated by a running game contending for the GPU and had to be discarded (see the caveat at the end).

## The change

Every kernel through `vectorized_padded.cuh` had one flat mapping from `threadIdx.x` straight to an output position within the *entire* `BM x BN` block tile. That's the root cause of two problems `vectorized_padded.cuh` only half-fixed:

- **Store coalescing**: `threadCol*TN` with `TN=8` put adjacent threads 32 bytes apart — a full sector, so each store only used half of it.
- **Shared-memory bank conflicts**: `Bs`'s row was `BN=128` elements wide, four full 32-bank periods, so the `threadCol*TN` stride pattern repeated within a warp's own read and collided with itself.

`warp_tiled.cuh` fixes the actual structure instead of patching around it: an explicit warp-level tile sits between the block tile and the per-thread tile. Each of a block's 4 warps owns a fixed, contiguous `WM=32 x WN=32` region — chosen deliberately to be exactly one 32-bank period wide, so a warp's own shared-memory access pattern never repeats within itself. `TN=4` was chosen to exactly match `float4`'s width, so adjacent lanes' output writes land in exactly-adjacent 16-byte chunks instead of leaving a gap. Both fixes fall directly out of the tile-size choice — no padding trick required.

## Result

Interleaved with `cublasSgemm`, 6 rounds, GPU otherwise idle:

| kernel | avg min GFLOPS | avg median GFLOPS |
|---|---|---|
| vectorized | ~7485 | ~7057 |
| **warp_tiled** | **~7993** | **~7558** |
| cuBLAS | ~8547 | ~8246 |

**warp_tiled is ~7% behind cuBLAS on min, ~9% on median** — a real improvement over `vectorized`'s ~13% gap, with zero overlap between the `warp_tiled` and `cuBLAS` distributions across all 6 rounds (warp_tiled min always 7965–8005, cuBLAS min always 8529–8560).

## Why (from Nsight Compute)

Profiled with `ncu --set full -o profile_warp_tiled test_warp_tiled.exe` (`block(128,1,1)`, `grid(16,16)`, 66 registers/thread).

| Metric | vectorized | vectorized_padded | warp_tiled |
|---|---|---|---|
| Registers/thread | 105 | 105 | **66** |
| Theoretical Occupancy | 33.3% | 33.3% | **58.3%** |
| Achieved Occupancy | 28.2% | 28.3% | 48.0% |
| Global load coalescing | flagged (3.8%) | flagged (3.1%) | **not flagged — fixed** |
| Global store coalescing | flagged (34.0%) | flagged (28.3%) | **not flagged — fixed** |
| Shared load bank conflict | flagged, 5-way (31.6%) | flagged, 5-way (30.0%) | **not flagged — fixed** |
| Shared store bank conflict | flagged, 2.4-way (26.4%) | not flagged (fixed by padding) | flagged, 2.4-way (26.4%) — **carried over, unaddressed** |

Three of the four flagged issues from the vectorized kernels are completely gone — not reduced, gone from the report entirely — exactly matching what the `WM=WN=32` and `TN=4` choices were designed to do. Register pressure dropped by more than a third (105→66/thread, `threadResults[TM*TN]=32` instead of `64`), nearly doubling theoretical occupancy (33.3%→58.3%).

The one surviving issue — the shared store bank conflict — is *not* a warp-tiling failure; it's untouched code. The block-level cooperative load of `As` (all 128 threads scatter-storing a `float4` global read into the transposed shared tile) is identical in structure to `vectorized.cuh`'s, and warp-tiling only restructured the *compute* phase's thread-to-output mapping, not this load phase. `vectorized_padded.cuh` fixed this specific conflict with `+4` padding; that fix was never ported into `warp_tiled.cuh`, so the same 2.4-way conflict is still there, at the identical magnitude.

A tail-effect finding (`Est. Speedup 50%`, grid `256` blocks vs `24` SMs, `1.52` waves) is also present, similar in kind to earlier kernels' — a property of this specific grid/occupancy combination rather than something warp-tiling addresses.

## Next step

The cheapest remaining move: **port the `+4` padding fix from `vectorized_padded.cuh` into the `As` shared array here.** It's an orthogonal, independent fix — nothing about warp-tiling's benefits should regress — and doc 06 already confirmed it eliminates exactly this class of conflict without hurting alignment. Given `vectorized_padded.cuh` only gained ~1.2% from fixing this same conflict in isolation, expect a similarly modest but real further improvement here, likely landing the gap to cuBLAS somewhere in the 5–8% range rather than 7–9%.

Past that, the standard remaining lever nothing in this series has touched is **double buffering** (see [03](03_register_blocking.md#next-step)) — overlapping the next tile's global load with the current tile's compute, which targets the `__syncthreads()` bubble at each tile boundary rather than anything measured in this report.

## A benchmarking caveat worth repeating

The first attempt at this comparison was run while a game (VALORANT) was actively using the GPU in the background. Both kernels' numbers swung wildly (roughly 3400–8300 GFLOPS) with no consistent pattern tied to which kernel was running — `nvidia-smi --query-compute-apps` showing `VALORANT-Win64-Shipping.exe` in the process list confirmed it, and utilization/clocks normalized once the game closed. Same lesson as [03](03_register_blocking.md#a-benchmarking-pitfall-worth-writing-down), one level up: it's not just separate process launches of *your own* binaries that need checking — anything else contending for the GPU (games, other CUDA processes, browser GPU compositing) invalidates a comparison just as thoroughly. `nvidia-smi --query-compute-apps=pid,process_name --format=csv` is the fast way to check before trusting a surprising number.

## Caveat on profiling numbers

Same caveat as [01](01_naive_vs_coalesced.md#caveat-on-profiling-numbers)–[06](06_shared_mem_padding.md#caveat-on-profiling-numbers): don't compare raw `Duration` between `.ncu-rep` files — `--set full` does 45 replay passes with cache flushing between them. Use `ncu` for root-cause metrics; use the unprofiled min/median numbers above for actual GFLOPS.
