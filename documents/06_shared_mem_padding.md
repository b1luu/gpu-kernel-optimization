# Analysis 6: Shared Memory Padding — A Partial Fix

**Setup:** M=N=K=1024, FP32, same `BM=BN=128, BK=8, TM=TN=8` config as `vectorized.cuh`. `vectorized_padded.cuh` is identical except `As[BK][BM]`→`As[BK][BM+4]`, `Bs[BK][BN]`→`Bs[BK][BN+4]` (padded by 4 floats, not 1 — see below). Timed via `test_vectorized.exe` / `test_vectorized_padded.exe`.

## The change, and a correctness trap along the way

`vectorized.cuh`'s profile ([05](05_vectorized.md)) flagged two shared-memory bank-conflict issues that hadn't appeared in any earlier kernel: 5-way conflicts on shared loads (Est. Speedup 31.6%) and 2.4-way on shared stores (Est. Speedup 26.4%). The classic fix is padding — add extra columns to each row so the row stride is no longer a multiple of 32 (the shared-memory bank count), which breaks the address pattern that causes different rows to collide on the same bank.

The naive version of that fix — pad by exactly 1 float — is actually broken here. `vectorized.cuh` reads/writes `As`/`Bs` via `reinterpret_cast<float4*>`, which requires 16-byte-aligned addresses. Padding `BM=128` to `129` makes the row stride `516` bytes, not a multiple of 16 — every 4th row's `float4` accesses become misaligned, which is undefined behavior for a raw pointer cast on shared memory. The fix actually used pads by **4 floats** (16 bytes) instead: `BM+4=132`. This still breaks the stride-32-elements periodicity (`132` isn't a multiple of `32`) while keeping every row's start float4-aligned.

## Did it work? Partially — and the profiler shows exactly which half

| Bank conflict | vectorized.cuh | vectorized_padded.cuh |
|---|---|---|
| Shared **stores** | 2.4-way, Est. Speedup 26.4% | **gone — not flagged at all** |
| Shared **loads** | 5.0-way, Est. Speedup 31.6% | **unchanged — still 5.0-way, Est. Speedup 29.97%**, same 4,194,304 conflicts |

The store conflict (from `As`'s scalar scatter-store during tile load) was **inter-row** — different rows landing on the same bank because `BM` was a multiple of 32 — and padding is exactly the right tool for that. It's fixed.

The load conflict (from `regN`'s read, `Bs[dotIdx][threadCol*TN + v*4]`) is **intra-row** — for one fixed `dotIdx`, different threads' `threadCol*8` offsets collide on the same 4 banks (since `8 * 4 threads = 32`, exactly one bank-count period). Row padding changes the gap *between* rows, not offsets *within* one row, so it does nothing for this conflict — which is exactly what the unchanged profiler numbers confirm. This was flagged as a real possibility before testing (see [05](05_vectorized.md#next-step)) rather than assumed away, and the profile settled it.

## Result

Median GFLOPS was noisy between runs for both kernels (occasional single-trial outliers within a 5-trial batch), so **min** — consistently clean across 6 interleaved runs of each binary with zero overlap between the two — is the reliable number here:

| kernel | min GFLOPS (1024³, avg of 6 interleaved runs) |
|---|---|
| vectorized | ~7485 |
| vectorized_padded | ~7574 |

**~1.2% faster.** Small, but real and reproducible — proportional to fixing only one of the two flagged conflicts. Registers/thread (105), theoretical occupancy (33.3%), and the tail-effect finding (Est. Speedup 50%) were all unchanged — the extra padding added negligible shared memory (8.19KB → 8.45KB static/block) and didn't shift the register-limited occupancy ceiling at all.

## Next step

Fixing the remaining load conflict needs a different mapping than padding can provide: the intra-row `threadCol*TN` stride is a property of how columns are assigned to threads, not of the array's row layout. That's the same underlying issue as the still-unresolved store-coalescing gap from [05](05_vectorized.md#why-from-nsight-compute) — both come from `threadCol` owning a contiguous `TN`-wide block instead of being interleaved across the tile. Warp-tiling (restructuring so a warp's 32 threads cooperate on one contiguous region) is the standard fix for this whole family of problems at once, rather than patching each symptom individually.

## Caveat on profiling numbers

Same caveat as [01](01_naive_vs_coalesced.md#caveat-on-profiling-numbers)–[05](05_vectorized.md#caveat-on-profiling-numbers): don't compare raw `Duration` between `.ncu-rep` files — `--set full` does 45 replay passes with cache flushing between them. Use `ncu` for root-cause metrics; use the unprofiled min/median numbers above for actual GFLOPS — and per this analysis, prefer **min** over median when a kernel's median is noisy across trials.
