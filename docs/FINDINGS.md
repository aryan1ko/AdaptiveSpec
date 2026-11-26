# Experimental Log

## EXP-001: Measured roofline, Apple M3
**Date:** 2026-11-26 · **Artifact:** `results/roofline_m3_2026-11-26.txt` ·
**Code:** `src/roofline_probe.mm` · **Harness:** `include/aspec/bench.hpp`

### Method
Streaming-read bandwidth from a 512 MB private buffer, swept across footprint with grid
fixed at 65536 threads (256 TG x 256) so occupancy is constant and only footprint varies.
Footprint order randomized (`aspec::shuffled_order`) so thermal drift cannot impersonate a
footprint trend. 3 warmup + 15 timed reps per point; GPU-timestamped
(`GPUEndTime - GPUStartTime`), not wall clock. FP32 peak from 8 independent register-
resident FMA chains with multiplier/addend supplied through a buffer so the recurrence
cannot be constant-folded and a runtime trip count so it cannot be unrolled away.
Thermal state `nominal` before and after.

### Results

| Footprint | reads/thread | GB/s | instability | Regime |
|---|---|---|---|---|
| 1 MiB | 1 | 1276.0 | 0.1% | **artifact** - rejected |
| 2 MiB | 2 | 678.8 | 61.2% | **artifact** - rejected |
| 4 MiB | 4 | 672.1 | 40.6% | **artifact** - rejected |
| 8 MiB | 8 | 912.7 | 13.6% | SLC-resident |
| 16 MiB | 16 | 346.9 | 3.3% | transition |
| 32 MiB | 32 | 121.8 | 18.9% | transition |
| 64 MiB | 64 | 79.6 | 33.3% | DRAM |
| 128 MiB | 128 | 91.0 | 15.1% | DRAM |
| 256 MiB | 256 | 87.9 | 19.2% | DRAM |
| 512 MiB | 512 | 92.3 | 4.2% | DRAM |

- **Streaming bandwidth: 89.4 GB/s** (median of DRAM plateau) = **89% of the advertised
  100 GB/s**. A credible achievable fraction.
- **Resident bandwidth: 912.7 GB/s** (8 MiB, the smallest footprint with enough per-thread
  work to be trustworthy).
- **Peak FP32: 2865.9 GFLOP/s**, instability 0.1%. Roughly 82% of the ~3.5 TFLOP/s
  implied by 10 cores x 128 FP32 lanes x 2 FLOP x ~1.38 GHz.

Methodological note: with grid fixed, footprints below 8 MiB degenerate into every thread
re-reading a single address, which measures L1/register reuse rather than any level's
bandwidth. Those points are detected (`reads/thread < 8`) and excluded rather than
silently averaged in. The first run of this experiment reported 1265 GB/s as "peak
bandwidth" by taking a max over all footprints; that was wrong and produced a nonsense
ridge point of 2.3 FLOP/byte. Characterizing the resident regime properly needs a
dedicated experiment with grid scaled to footprint.

### Derived roofline

Q4_0 arithmetic intensity: 18 B per 32 weights = 0.5625 B/weight. A GEMV with `R` output
columns loads each weight once and does `2R` FLOPs with it, so AI = `3.56 * R` FLOP/byte.

| Regime | Bandwidth | Ridge point | Q4_0 crossover |
|---|---|---|---|
| Streaming (DRAM) | 89.4 GB/s | **32.0 FLOP/byte** | **9.0 columns (gamma = 8.0)** |
| Resident (SLC) | 912.7 GB/s | **3.1 FLOP/byte** | 0.9 columns - compute-bound at ne11=1 |

### Finding 1: the shipped constant is right, not mistuned
Prediction going in was that llama.cpp's `ne11 <= 8` ceiling (and CUDA's
`MMVQ_MAX_BATCH_SIZE = 8`) would prove mistuned on M3. **It does not.** The measured
streaming crossover is 9.0 columns, i.e. gamma = 8.0. The "`~8`" the maintainer wrote with
an explicit guess marker is, on this hardware, essentially the roofline-correct answer.
This partially falsifies ADR-001's framing and must be stated as such.

### Finding 2: the ridge point moves 10x within one model, so no constant can be right
The two regimes differ by **10.3x** in ridge point. Qwen2.5-3B Q4_0 decode tensors
(hidden 2048, intermediate 11008, 36 layers, GQA 16/2 heads):

| Tensor | K x N | MiB | Regime |
|---|---|---|---|
| q_proj | 2048 x 2048 | 2.25 | SLC-resident |
| k_proj, v_proj | 2048 x 256 | 0.28 | SLC-resident |
| o_proj | 2048 x 2048 | 2.25 | SLC-resident |
| gate_proj, up_proj | 2048 x 11008 | 12.09 | exceeds SLC |
| down_proj | 11008 x 2048 | 12.09 | exceeds SLC |

Per layer 41.3 MiB; **1.45 GiB streamed per token** across 36 layers. Footprints span
**43x** (0.28 -> 12.09 MiB) and straddle the 8 MB SLC boundary *within a single layer*.
In the resident regime Q4_0 GEMV is already compute-bound at `ne11 = 1`, so extra columns
cost full price; in the streaming regime they are nearly free up to 9. A single global
`ne11` ceiling cannot be correct for both, and the shipped code keys its dispatch only on
divisibility (`ne00 % 256`, `ne00 % 128`) and type - **never on footprint**. That is the
real gap, and it is better founded than "the constant is mistuned."

Open question this raises: during real decode all 1.45 GiB streams per token and evicts
everything, so whether the small tensors are *effectively* resident depends on eviction
across a forward pass, not on their size in isolation. Microbenchmark and end-to-end
behaviour may differ sharply here. This must be measured, not assumed.

### Finding 3: a measurement trap, caught before building on it
A GEMV microbenchmark that loops on one q_proj tensor (2.25 MiB) measures the resident
regime at ~900 GB/s and would overstate decode throughput by **~10x**. Every kernel sweep
in this project must either use a working set >> 8 MB or rotate across tensors to mimic a
forward pass. Recorded as a hard harness requirement.

### Derived ceiling for week 4
At 89.4 GB/s streaming, 1.45 GiB/token gives **17.4 ms/token = 57.4 tok/s** as the
weights-only batch-1 decode ceiling for Qwen2.5-3B Q4_0 on this machine. Ignores KV cache
traffic, attention, embeddings, sampling and all dispatch overhead, so real throughput
must be below it. This is the number llama.cpp's measured tok/s should be compared against.

### Prediction status
| ID | Status |
|---|---|
| P1 (`T(ne11)` non-monotonic) | untested - needs the GEMV sweep |
| P2 (discontinuity at ne11=9) | untested |
| P3 (`nsg=2` suboptimal) | untested - roofline says nothing about simdgroup count |
| P4 (`ne11 < 3` gate on `nxpsg=16` too conservative) | **testable as specified**: all 7 Qwen2.5-3B decode tensors have K divisible by 256, so that gate is the only thing blocking `nxpsg=16` for the entire model |
