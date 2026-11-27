# Decisions

Working notes on why the project is built this way.

## D1: What it optimizes
2026-11-26

Replace the hardcoded constants that pick between mat-vec and mat-mat kernels for
small-batch quantized matmul with a measured cost model, and use that model to pick the
speculative decoding draft length.

The gap, from reading the source:

`ggml/src/ggml-metal/ggml-metal-ops.cpp:2782` gates a small-batch quantized mat-vec path
on `ne11` in [2,8] for Q4_0-family types and [4,8] for K-quants, with `ne00 % 128 == 0`.
Inside it, `nsg = 2` is hardcoded alongside a TODO saying the maintainer doesn't know why
it isn't the maximum available threads and hasn't confirmed the tail-effect hypothesis.
`nxpsg` comes from a divisibility cascade (16 if `ne00 % 256 == 0 && ne11 < 3`, else 8 if
`ne00 % 128 == 0`, else 4) and `r1ptg` from a switch on `ne11` that isn't monotonic
(2→2, 3→3, 4→4, 5→5, 6→3, 7→4, 8→4), with a comment saying it isn't known to be optimal
across hardware.

`ggml/src/ggml-cuda/mmvq.cu` makes the same choice with one global
`MMVQ_MAX_BATCH_SIZE = 8`, overridable at runtime by `GGML_CUDA_MMVQ_MAX`. AMD MFMA
already carries per-type overrides (Q4_K ≤ 3, Q6_K ≤ 5), and PR #28613 is retuning these
per architecture, which shows the constants are architecture-dependent.

Scope:
- Primary: Apple M3 / Metal. Own the kernel variants and the cost surface.
- Secondary: a T4 through Colab. `GGML_CUDA_MMVQ_MAX` is an env var, so the CUDA-side
  crossover sweep needs no CUDA code. Writing a CUDA kernel is a stretch goal.
- Weight format: GGUF Q4_0, 32 weights per block, one fp16 scale, 18 bytes.
- Weeks 1–3 use synthetic matrices at Qwen2.5-3B shapes, so no model weights are needed
  until integration.

Alternatives not taken:
- Kernel variants with no controller. Lower risk, but thin — MLX already merged a tuned
  multi-row quantized kernel (#4787, Nov 2026).
- Adaptive draft length from confidence or a learned head. Confidence truncation already
  ships as `--spec-draft-p-min`, and the learned version is published (SpecDec++).
- KV-length adaptivity in attention. Real, but crowded.
- Writing an inference engine from scratch. No credible baseline to compare against.

## D2: Revision after EXP-001
2026-11-26

D1 assumed `ne11 <= 8` would turn out mistuned on M3. Measured crossover is 8.6-9.0
columns across runs, so the constant is approximately correct. That framing is withdrawn.

What replaced it: the ridge point moves 10.3x between cache-resident and streaming, and
Qwen2.5-3B's decode tensors span 43x in footprint inside a single layer, straddling the
8 MB SLC. So dispatch should depend on footprint, which the shipped heuristics never
check — they only look at divisibility and quantization type.

This rests on a measured 10x effect rather than on an assumption that someone else's
constant was wrong, and it survives their constant being right.

## Predictions
Written before measuring, so the data can disagree.

- P1: `T(ne11)` isn't monotonic; `ne11` = 6 and 7 cost more than 5, since `r1ptg` drops
  to 3 and 4 and needs two passes.
- P2: hard discontinuity at `ne11` = 9, where the op leaves `mul_mv_ext`.
- P3: `nsg` = 2 is too low on M3 for large `ne01`; more simdgroups should win once there
  are enough rows to hide the tail effect.
- P4: the `ne11 < 3` gate on `nxpsg` = 16 is too conservative. All seven Qwen2.5-3B
  decode tensors have K divisible by 256, so that gate is the only thing blocking
  `nxpsg` = 16 for the whole model.

Decision rule for end of week 2:
- knees present and moving with shape → build the controller
- flat to `ne11` = 16 → just max the draft length, drop the controller
- linear → speculation doesn't pay on this hardware, and that's the result

## D3: Plan-level optimization rejected; the bottleneck is dequantization
2026-11-26 · supersedes the direction implied by D2

An expansion was considered: a cost-based execution plan optimizer over a transformer
subgraph, enumerating fusion boundaries, intermediate layouts and kernel configurations
jointly rather than per operator. EXP-002 was run as a gate on it.

The gate failed. Measured non-additivity is **+4.5% at R=1 and +1.2% at R=4**, at 0-2%
instability. Meanwhile kernel-level choices measured on the same hardware:

| Effect | Size |
|---|---|
| lane mapping inside one kernel | **2.9x** |
| cost of R=1 -> R=4 vs the roofline ideal | **3.1x vs 1.0x** |
| plan-level operator interaction | **1-5%** |

Kernel-level leverage is roughly 60x larger than anything plan selection could recover.
A plan optimizer would also be untestable until the kernel is memory-bound, which is the
kernel work anyway, so it is gated behind that regardless.

Also rejected on grounds of prior art: ggml already has a graph optimizer doing the
standard fusions; TASO (SOSP'19) and Mirage (OSDI'25) already do cost-based graph
substitution with far more sophistication than four weeks allows.

### Revised direction
EXP-002 Results 3 and 4 point somewhere better. The common claim that decode is
memory-bandwidth-bound holds here only for tensors that exceed the 8 MB system cache. For
cache-resident tensors, which are four of the seven matmuls in every Qwen2.5-3B layer,
Q4_0 GEMV is compute-bound on dequantization and sits at 8-11% of its achievable
bandwidth. Consequently extra columns cost 77-90% of full price instead of being nearly
free, which undercuts the premise speculative decoding depends on.

The problem is therefore: close the dequantization gap for cache-resident quantized GEMV,
and measure whether that restores the near-free scaling in R that makes speculation pay.
The adaptive component survives in a narrower and now measured form: which regime a
tensor is in is a function of its footprint, and the right kernel and the right R differ
between regimes inside a single layer.
