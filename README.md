# AdaptiveSpec

**Hardware-aware dispatch for small-batch quantized matmul in LLM inference.**

Production inference engines choose between a matrix-vector kernel and a tiled
matrix-matrix kernel using hand-tuned static constants. In `llama.cpp` those constants
are, verbatim from the source, acknowledged as unverified:

```cpp
// ggml/src/ggml-metal/ggml-metal-ops.cpp:2782
// first try to use small-batch mat-mv kernels
// these should be efficient for BS [2, ~8]
...
// TODO: determine the optimal parameters based on grid utilization
//       I still don't know why we should not always use the maximum available threads
//       ... need to confirm this
const int nsg = 2;
...
// note: not sure how optimal are those across all different hardware.
//       there might be something cleverer
```

The CUDA backend has the same shape of problem — a single global
`MMVQ_MAX_BATCH_SIZE = 8` picks GEMV vs. tensor-core MMQ — and
[an open PR](https://github.com/ggml-org/llama.cpp/pull/28613) is currently retuning
those thresholds for RDNA3.5, which is direct evidence they are architecture-dependent.

**Speculative decoding drives the batch dimension straight through that range.** Verifying
γ draft tokens makes the decode matmul `ne11 = γ+1` columns wide. So the draft length a
speculative decoder should choose and the kernel the engine should dispatch are the same
decision — and today both sides of it are constants someone typed in.

This project measures that cost surface and replaces the constants with a model.

---

## Status

**Week 1 of 4. Active investigation, not a finished result.** What exists today:

| | |
|---|---|
| Measurement harness | done — `include/aspec/bench.hpp` |
| Measured roofline for Apple M3 | done — `docs/FINDINGS.md` EXP-001 |
| Q4_0 reference path | **spec + tests written, implementation in progress** |
| Parameterized Metal kernel family | not started |
| Cost model + γ controller | not started |
| llama.cpp integration, end-to-end tok/s | not started |

## Measured results so far

Apple M3 (10-core GPU, 8 GB unified memory), `results/roofline_m3_2026-11-26.txt`.
Measured, not quoted from a spec sheet:

| Quantity | Measured |
|---|---|
| Streaming read bandwidth (DRAM) | **89.4 GB/s** — 89% of the advertised 100 |
| Cache-resident read bandwidth (8 MiB) | **912.7 GB/s** |
| Peak FP32 FMA throughput | **2865.9 GFLOP/s** |
| Streaming ridge point | **32.0 FLOP/byte** |
| Resident ridge point | **3.1 FLOP/byte** |

### Finding 1 — the shipped constant is right, and that was not the prediction

The project's opening hypothesis was that `ne11 <= 8` would prove mistuned on this
hardware. It does not. Q4_0 weights cost 0.5625 B/weight, so a GEMV with `R` columns has
arithmetic intensity `3.56 R` FLOP/byte, and the measured streaming ridge point of 32.0
puts the memory→compute crossover at **9.0 columns (γ = 8.0)**. The `~8` written as an
admitted guess is essentially the roofline-correct answer here. Recorded in
`docs/DECISIONS.md` as ADR-002, which withdraws the original framing.

### Finding 2 — the ridge point moves 10× within a single model

Streaming and cache-resident regimes differ by **10.3×** in ridge point, and in the
resident regime a Q4_0 GEMV is *already compute-bound at `ne11 = 1`*. Qwen2.5-3B's decode
weight tensors span **43× in footprint inside one layer**:

| Tensor | K × N | MiB (Q4_0) | vs. 8 MB SLC |
|---|---|---|---|
| `k_proj`, `v_proj` | 2048 × 256 | 0.28 | resident |
| `q_proj`, `o_proj` | 2048 × 2048 | 2.25 | resident |
| `gate_proj`, `up_proj` | 2048 × 11008 | 12.09 | exceeds |
| `down_proj` | 11008 × 2048 | 12.09 | exceeds |

41.3 MiB per layer × 36 layers = **1.45 GiB streamed per token**. No single global `ne11`
ceiling can be correct on both sides of that boundary, and the shipped dispatch keys on
divisibility (`ne00 % 256`, `ne00 % 128`) and quantization type — **never on footprint**.
That is the gap this project targets.

### Finding 3 — a benchmarking trap, documented so others avoid it

A GEMV microbenchmark that loops on one 2.25 MiB `q_proj` tensor measures the resident
regime at ~900 GB/s and **overstates decode throughput by roughly 10×**. Every sweep here
must defeat the 8 MB system-level cache. The harness treats this as a hard requirement.

### Derived ceiling

At 89.4 GB/s streaming, 1.45 GiB/token gives **17.4 ms/token = 57.4 tok/s** as the
weights-only batch-1 decode ceiling for Qwen2.5-3B Q4_0 on this machine. It ignores KV
cache traffic, attention, embeddings and dispatch overhead, so real throughput must fall
below it. This is the reference for end-to-end numbers later.

---

## Method

Benchmarking rules enforced by `include/aspec/bench.hpp`, because the target machine is a
fanless-class 8 GB laptop where naive measurement produces fiction:

- **GPU-timestamped** timing (`GPUEndTime - GPUStartTime`), not wall clock, so CPU-side
  encode and submission latency stay out of kernel numbers.
- **Randomized trial order** for every sweep. Measuring `ne11 = 1,2,…,16` in order on a
  thermally-throttling machine manufactures a fake upward trend.
- **Median / p95 / p99**, never a bare mean — one thermal excursion moves a mean and
  leaves a median alone.
- An **`instability()` metric** that marks a measurement untrustworthy rather than
  reporting it. In EXP-001 it rejected three footprints whose numbers looked plausible.
- Warmup excluded; shader compilation excluded from steady-state timing; seeds fixed;
  thermal state logged before and after.

Negative results and withdrawn hypotheses are kept in `docs/`, not deleted. ADR-002
exists because ADR-001 was partly wrong.

## Build

Requires macOS with Xcode **Command Line Tools only** — no full Xcode. Metal shaders are
compiled at runtime via `newLibraryWithSource`, so the offline `metal` compiler is not
needed.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

./build/roofline_probe     # measure this machine's roofline
./build/test_q4_0          # Q4_0 reference correctness (currently red by design)
```

## Layout

```
include/aspec/bench.hpp    measurement harness: stats, randomized ordering, instability
include/aspec/q4_0.hpp     GGUF Q4_0 format spec, bit-compatible with ggml
src/roofline_probe.mm      bandwidth-vs-footprint and FP32 peak microbenchmarks
src/q4_0.cpp               CPU reference path — the oracle every GPU kernel is checked against
tests/test_q4_0.cpp        correctness tests, incl. exact byte layout vs hand-computed blocks
docs/DECISIONS.md          architecture decision records, including withdrawn ones
docs/FINDINGS.md           experimental log with method, raw numbers and caveats
results/                   raw measurement output
```

## References

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) — Metal and CUDA dispatch paths under study
- [MLX PR #4787](https://github.com/ml-explore/mlx/pull/4787) — `affine qmv_mma` for rows 3–16, merged 2026-11-02; reports M2 Pro batch-8 decode 70.0 → 202.1 tok/s, and 2.09–2.84× from dispatch-threshold changes alone
- [llama.cpp PR #28613](https://github.com/ggml-org/llama.cpp/pull/28613) — retuning MMVQ batch thresholds for RDNA3.5
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths](https://arxiv.org/abs/2405.19715) (COLM 2025) — learned acceptance prediction for candidate length
- [SpecSearch](https://arxiv.org/abs/2611.22296) — roofline-based draft length, but for datacenter GPUs at high concurrency; does not address single-user unified-memory hardware

## License

MIT
