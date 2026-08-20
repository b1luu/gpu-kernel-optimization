# Analysis 3: Register Blocking, and a Benchmarking Pitfall

**Setup:** M=N=K=1024, FP32. `shared_mem.cuh`: `block(16,16)`, `TILE=16`. `register_blocked.cuh`: `BM=BN=64`, `BK=8`, `TM=8`, 512 threads/block (1D), `grid(N/BN, M/BM)`. Timed via `test_sharedmem.exe` / `test_register.exe` / `bench_cublas.exe` — 5 trials of a 10-launch average each, reporting min/median (see the pitfall below for why).

## The change

`shared_mem.cuh`'s profiling (see [02](02_coalesced_vs_shared_mem.md)) found the kernel bound on **Mio Throttle Stalls**: each `k` iteration issues two shared-memory loads (`As[ty][k]`, `Bs[k][tx]`) to produce one FMA — the shared-memory instruction queue, not compute, was the bottleneck.

`register_blocked.cuh` amortizes that: each thread now owns `TM=8` output rows instead of 1. Inside the reduction loop, one `Bs[dotIdx][threadCol]` load is pulled into a register (`tmpB`) once and then reused across all `TM` FMAs against different rows of `As`. Same number of `Bs` loads, `TM`x fewer `Bs` loads *per unit of work* — directly targeting the instruction-issue bottleneck the profiler found, rather than the memory-traffic problem tiling already solved.

Block/grid shape also changes: instead of one thread per output element (`16x16` threads = `16x16` outputs), each block is `BM x BN = 64x64` outputs computed by `BM*BN/TM = 512` threads — a 1D launch, since thread-to-output mapping is no longer 1:1.

## Result

| kernel | median GFLOPS (1024³) |
|---|---|
| shared_mem (tiled) | ~968 |
| register_blocked | ~3350 |
| cuBLAS (`cublasSgemm`) | ~7893 |

**~3.5x faster than shared_mem**, closing more of the gap to cuBLAS than any step so far — but still ~2.4x behind it.

## A benchmarking pitfall worth writing down

The first cuBLAS number measured (in isolation, a single process launch) was **2920 GFLOPS** — which made `register_blocked` (3101 GFLOPS in that same isolated comparison) look like it had *beaten* cuBLAS. Rerunning both binaries 5x back-to-back told a different story: `register_blocked` stayed in the 2946–3663 GFLOPS band, but cuBLAS jumped to 6199–8540 GFLOPS. The original 2920 number was an outlier, most likely from GPU boost-clock state — a freshly-launched process starting from an idle/downclocked GPU measures slower than one launched right after other GPU work, even with in-process warm-up iterations, because the warm-up absorbs *driver/JIT* costs but not *clock ramp-up across the whole process's brief lifetime*.

**Lesson:** never trust a single-process-launch benchmark number, especially when comparing two binaries whose results were captured in different runs. All benchmark harnesses in this repo (`test.cu`, `test_register.cu`, `bench_cublas.cu`) now run 5 independent timed trials per launch and report both `min` and `median` for exactly this reason.

## Why (from Nsight Compute)

Profiled with `ncu --set full -o profile_register test_register.exe` (same GPU, `block(512,1,1)`, 44 registers/thread).

| Metric | shared_mem | register_blocked |
|---|---|---|
| Registers/thread | 38 | **44** |
| Theoretical Occupancy | not register-limited | **66.7%** — capped at 8.00/12 warps per scheduler by register count |
| Mio Throttle Stalls (cycles/warp) | 19.6 | **6.6** |
| Mio Throttle, % of avg cycles between issues | 51.3% (of 38.3 avg) | 31.9% (of 20.8 avg) |
| Compute Throughput | 96.29% | 78.5% |
| Memory Throughput | 96.29% | 78.5% |

The fix worked on its own terms: MIO throttle stall cycles roughly halved (19.6 → 6.6), and the total average cycles between issuing two instructions dropped from 38.3 to 20.8 — clear evidence that amortizing each `Bs` load across `TM=8` FMAs reduced instruction-issue pressure, exactly as intended.

But it traded one bottleneck for another. `threadResults[TM]` plus the loop's live state pushed register usage from 38 to 44 per thread, and Nsight now flags **register count as the occupancy-limiting resource** — only 66.7% of the hardware's max warps-per-scheduler can be resident. With fewer warps in flight, there's less independent work available to hide *any* stall (including what MIO throttling remains), which is the likely reason both Compute and Memory Throughput dropped from ~96% to ~78.5% even though the kernel is objectively doing less wasteful instruction issuing than before.

## Next step

The remaining ~2.4x gap to cuBLAS is where production kernels add: 2D register blocking (each thread computes a `TMxTN` tile instead of `TMx1`, amortizing *both* `As` and `Bs` loads), vectorized loads (`float4`) to cut instruction count further, and double-buffering shared memory to overlap the next tile's load with the current tile's compute.

Given this profile, any further widening (bigger `TM`/`TN`) needs to be weighed against the register-pressure/occupancy tradeoff just observed — more per-thread accumulator state will push registers/thread up further, and 2D blocking's `TMxTN` accumulator array is larger than 1D's `TM`. Worth checking `--ptxas-options=-v` or `__launch_bounds__` to see/cap register usage before assuming a wider tile is a free win.

## Caveat on profiling numbers

Same caveat as [01](01_naive_vs_coalesced.md#caveat-on-profiling-numbers) and [02](02_coalesced_vs_shared_mem.md#why-from-nsight-compute): don't compare raw `Duration` between `.ncu-rep` files — `--set full` does 45 replay passes with cache flushing between them. Use `ncu` for root-cause metrics; use the unprofiled `test_*.exe`/`bench_cublas.exe` min/median numbers above for actual GFLOPS.
