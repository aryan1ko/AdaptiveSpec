# Experiment log

## EXP-001: Measured roofline, Apple M3
2026-11-26 · `src/roofline_probe.mm` · output in `results/roofline_m3_2026-11-26.txt`

### Method
Streaming reads from a 512 MB private buffer, swept across footprint with the grid fixed
at 65536 threads so occupancy stays constant and only footprint varies. Footprint order
randomized. 3 warmup and 15 timed reps per point, GPU-timestamped. FP32 peak from 8
independent register-resident FMA chains, with the multiplier and addend passed through a
buffer so the recurrence can't be constant-folded and a runtime trip count so the loop
can't be unrolled away. Thermal state nominal before and after.

### Results

| Footprint | reads/thread | GB/s | instability | |
|---|---|---|---|---|
| 1 MiB | 1 | 1276.0 | 0.1% | rejected |
| 2 MiB | 2 | 678.8 | 61.2% | rejected |
| 4 MiB | 4 | 672.1 | 40.6% | rejected |
| 8 MiB | 8 | 912.7 | 13.6% | resident |
| 16 MiB | 16 | 346.9 | 3.3% | transition |
| 32 MiB | 32 | 121.8 | 18.9% | transition |
| 64 MiB | 64 | 79.6 | 33.3% | DRAM |
| 128 MiB | 128 | 91.0 | 15.1% | DRAM |
| 256 MiB | 256 | 87.9 | 19.2% | DRAM |
| 512 MiB | 512 | 92.3 | 4.2% | DRAM |

Streaming bandwidth 89.4 GB/s (median of the DRAM plateau), 89% of the advertised 100.
Resident 912.7 GB/s. Peak FP32 2865.9 GFLOP/s at 0.1% instability, about 82% of the
~3.5 TFLOP/s implied by 10 cores x 128 lanes x 2 FLOP x ~1.38 GHz.

With the grid fixed, footprints under 8 MiB degenerate into every thread re-reading one
address, which measures L1 reuse rather than bandwidth. Those points are detected
(`reads/thread < 8`) and excluded. The first version of this experiment took a max over
all footprints, picked up the 1 MiB number as "peak bandwidth" and produced a nonsense
ridge point of 2.3 FLOP/byte. Characterizing the resident regime properly needs a
separate experiment with the grid scaled to footprint.

### Derived

Q4_0 is 18 bytes per 32 weights, so 0.5625 B/weight. A GEMV with `R` columns loads each
weight once and does `2R` FLOPs with it, giving AI = `3.56R` FLOP/byte.

| Regime | Bandwidth | Ridge point | Q4_0 crossover |
|---|---|---|---|
| Streaming | 89.4 GB/s | 32.0 FLOP/byte | 9.0 columns (γ = 8.0) |
| Resident | 912.7 GB/s | 3.1 FLOP/byte | 0.9 columns, compute-bound at `ne11`=1 |

Run-to-run the DRAM plateau lands between about 89 and 94 GB/s, moving the crossover
between roughly 8.6 and 9.0 columns. The plateau points carry 15-33% instability, so the
single figures above are from the recorded run, not a tight estimate. Tightening this
needs more reps.

### Finding 1: the shipped constant is about right
I predicted `ne11 <= 8` and `MMVQ_MAX_BATCH_SIZE = 8` would be mistuned on M3. The
measured streaming crossover is 9.0 columns, so they are close to the roofline answer
here. The `~8` written with a guess marker in the source is roughly correct.

### Finding 2: the ridge point moves 10x inside one model
The two regimes differ 10.3x in ridge point. Qwen2.5-3B Q4_0 decode tensors (hidden 2048,
intermediate 11008, 36 layers, GQA 16/2):

| Tensor | K x N | MiB | |
|---|---|---|---|
| `q_proj`, `o_proj` | 2048 x 2048 | 2.25 | resident |
| `k_proj`, `v_proj` | 2048 x 256 | 0.28 | resident |
| `gate_proj`, `up_proj` | 2048 x 11008 | 12.09 | exceeds SLC |
| `down_proj` | 11008 x 2048 | 12.09 | exceeds SLC |

41.3 MiB per layer, 1.45 GiB streamed per token. Footprints span 43x and straddle the
8 MB SLC boundary within one layer. Resident GEMV is compute-bound at `ne11`=1 so extra
columns cost full price; streaming GEMV gets them nearly free up to 9. A single global
ceiling can't be right for both, and the dispatch code never consults footprint.

Open question: during real decode all 1.45 GiB streams per token and evicts everything,
so whether the small tensors are effectively resident depends on eviction across a
forward pass, not on size in isolation. Microbenchmark and end-to-end behaviour may
differ here. Needs measuring.

### Finding 3: a measurement trap
A microbenchmark looping on one `q_proj` tensor (2.25 MiB) measures the resident regime
at ~900 GB/s and overstates decode throughput by about 10x. Sweeps must either use a
working set well over 8 MB or rotate across tensors.

### Ceiling for integration
At 89.4 GB/s, 1.45 GiB per token is 17.4 ms/token, so 57.4 tok/s weights-only for
Qwen2.5-3B Q4_0. Ignores KV cache, attention, embeddings, sampling and dispatch overhead,
so measured throughput has to be lower.

### Prediction status
| | |
|---|---|
| P1 `T(ne11)` not monotonic | untested, needs the GEMV sweep |
| P2 discontinuity at `ne11`=9 | untested |
| P3 `nsg`=2 too low | untested, the roofline says nothing about simdgroup count |
| P4 `ne11 < 3` gate too conservative | testable as written: all 7 decode tensors have K divisible by 256, so that gate is the only thing blocking `nxpsg`=16 for the whole model |
