# AdaptiveSpec

Measuring how small-batch quantized matmul should be dispatched on Apple Silicon, and
using that to pick the draft length for speculative decoding.

## Problem

llama.cpp picks between a mat-vec and a mat-mat kernel using hardcoded constants. From
`ggml/src/ggml-metal/ggml-metal-ops.cpp`:

```cpp
// these should be efficient for BS [2, ~8]
const int nsg = 2;
// TODO: determine the optimal parameters based on grid utilization
// note: not sure how optimal are those across all different hardware.
```

The CUDA backend makes the same choice with a single `MMVQ_MAX_BATCH_SIZE = 8`.

Speculative decoding makes the decode matmul `ne11 = γ+1` columns wide, so it runs right
through the range those constants govern. Picking the draft length and picking the kernel
are the same decision, and both are currently constants someone typed in.

## Status

Week 1 of 4.

- [x] Measurement harness
- [x] Measured roofline (M3)
- [ ] Q4_0 reference path — spec and tests done, implementation in progress
- [ ] Parameterized Metal kernel variants
- [ ] Cost model and draft-length controller
- [ ] llama.cpp integration, end-to-end tok/s

## Measured

Apple M3, 10-core GPU, 8 GB unified memory. Raw output in `results/`.

| | |
|---|---|
| Streaming read bandwidth | 89.4 GB/s (89% of the advertised 100) |
| Cache-resident, 8 MiB | 912.7 GB/s |
| Peak FP32 FMA | 2866 GFLOP/s |
| Ridge point, streaming / resident | 32.0 / 3.1 FLOP/byte |

Q4_0 costs 0.5625 bytes per weight, so a GEMV with `R` columns has arithmetic intensity
`3.56R` FLOP/byte. Against the streaming ridge point the memory-to-compute crossover
lands around 9 columns, so γ ≈ 8. Run to run it moves between 8.6 and 9.0, since the
DRAM plateau measurements carry 15-33% instability.

So `ne11 <= 8` is roughly right on this machine. I expected to find it mistuned.

The useful result is that the ridge point moves 10x between regimes, and Qwen2.5-3B's
decode tensors straddle the boundary inside a single layer:

| Tensor | K × N | MiB (Q4_0) | Fits 8 MB SLC |
|---|---|---|---|
| `k_proj`, `v_proj` | 2048 × 256 | 0.28 | yes |
| `q_proj`, `o_proj` | 2048 × 2048 | 2.25 | yes |
| `gate_proj`, `up_proj` | 2048 × 11008 | 12.09 | no |
| `down_proj` | 11008 × 2048 | 12.09 | no |

In the resident regime a Q4_0 GEMV is already compute-bound at `ne11 = 1`. In the
streaming regime extra columns are nearly free up to 9. No single constant is right on
both sides, and the dispatch code only checks divisibility and quantization type, never
footprint.

1.45 GiB streams per token, which at 89.4 GB/s gives 57 tok/s as the weights-only
ceiling for Qwen2.5-3B Q4_0 here. Ignores KV cache, attention and dispatch overhead, so
real decode comes in under it.

A sweep that reuses one 2.25 MiB tensor measures the resident regime and overstates
decode throughput by ~10x, so sweeps have to exceed the 8 MB SLC.

## Build

macOS with Command Line Tools. Full Xcode is not needed — Metal shaders compile at
runtime via `newLibraryWithSource`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

./build/roofline_probe    # measure this machine
./build/test_q4_0         # red until the reference path is implemented
```

## Method

The target is a fanless 8 GB laptop, so measurement needs care:

- GPU timestamps, not wall clock
- Randomized sweep order, since thermal drift otherwise looks like a trend
- Median / p95 / p99, and an instability metric that flags a bad measurement instead of
  reporting it (it rejected three footprints in EXP-001)
- Warmup and shader compilation excluded, seeds fixed, thermal state logged

## Layout

`include/aspec/` harness and Q4_0 format · `src/roofline_probe.mm` microbenchmarks ·
`src/q4_0.cpp` CPU reference, the oracle for GPU kernels · `tests/` correctness ·
`docs/DECISIONS.md` why it's built this way · `docs/FINDINGS.md` experiment log

## References

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the dispatch paths under study
- [MLX #4787](https://github.com/ml-explore/mlx/pull/4787) — `qmv_mma` for 3–16 rows; reports 2.09–2.84x from dispatch-threshold changes alone
- [llama.cpp #28613](https://github.com/ggml-org/llama.cpp/pull/28613) — retuning MMVQ thresholds per architecture
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [SpecDec++](https://arxiv.org/abs/2405.19715), [SpecSearch](https://arxiv.org/abs/2611.22296) — adaptive draft length, learned and roofline-based

## License

MIT
