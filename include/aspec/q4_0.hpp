#pragma once
//
// GGUF Q4_0 quantization, bit-compatible with ggml.
//
// Layout (ggml/src/ggml-common.h:194):
//     #define QK4_0 32
//     typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;   // 18 bytes
//
// The nibble packing is the part that trips people up. Byte `qs[j]` does NOT hold
// weights 2j and 2j+1. It holds:
//     low  nibble of qs[j]  ->  weight j
//     high nibble of qs[j]  ->  weight j + 16
// i.e. the block is split in half and the halves are interleaved at nibble granularity.
// ggml chose this so a SIMD unpack can produce 16 consecutive weights with one mask and
// one shift of the same 16-byte register, rather than a shuffle. We must match it exactly
// or we cannot load real GGUF files in week 4.
//
// Quantization (ggml/src/ggml-quants.c, quantize_row_q4_0_ref):
//     find the element of largest magnitude; `max` is its SIGNED value, `amax` its
//     absolute value
//     d  = max / -8            <- note: NEGATIVE when max is positive
//     id = d ? 1/d : 0
//     q  = MIN(15, (int8_t)(x*id + 8.5f))
// Dequantization: value = (q - 8) * d
//
// Two consequences worth understanding before you implement:
//   - The element of largest magnitude always maps to q = 0 and dequantizes to
//     -8*d == max, so it round-trips exactly up to the fp16 rounding of d.
//   - The cast `(int8_t)(x*id + 8.5f)` truncates. Since x*id is in [-8, 8] by
//     construction the argument is always in [0.5, 16.5], so truncation of (v + 8.5)
//     is round-to-nearest of (v + 8). There is deliberately no lower clamp.
//
#include <cstddef>
#include <cstdint>

namespace aspec {

using half_t = __fp16;   // native ARM64 fp16; matches ggml_half bit-for-bit

constexpr int kQK4_0 = 32;

struct BlockQ4_0 {
    half_t  d;                   // scale
    uint8_t qs[kQK4_0 / 2];      // packed nibbles
};
static_assert(sizeof(BlockQ4_0) == 18, "block_q4_0 must be 18 bytes to match ggml");

inline std::size_t q4_0_blocks(std::size_t k) { return k / kQK4_0; }

// Quantize `k` contiguous floats into `k/32` blocks. `k` must be a multiple of 32.
// Must be bit-identical to ggml's quantize_row_q4_0_ref.
void quantize_q4_0(const float* src, BlockQ4_0* dst, std::size_t k);

// Inverse of the above. Writes `k` floats.
void dequantize_q4_0(const BlockQ4_0* src, float* dst, std::size_t k);

// Reference quantized GEMV / thin GEMM, in ggml's index convention:
//
//     W : quantized weights, logical shape [K, N] -- K is the contraction dim and is
//         contiguous, so row n occupies blocks [n*K/32, (n+1)*K/32)
//     X : fp32 activations, logical shape [K, R] -- column r occupies [r*K, (r+1)*K)
//     Y : fp32 output,      logical shape [N, R]
//
//     Y[r*N + n] = sum over k of  W[n*K + k] * X[r*K + k]
//
// R is the column count: 1 for ordinary single-token decode, gamma+1 under speculative
// verification. This is the quantity llama.cpp calls `ne11` and gates on.
//
// Correctness only -- this is the oracle every Metal kernel is checked against, so keep
// it obvious rather than fast. Accumulate in float (not double): the GPU kernels will
// accumulate in float, and an oracle that is more accurate than the thing under test
// makes tolerances harder to reason about, not easier.
void gemv_q4_0_ref(const BlockQ4_0* W, const float* X, float* Y,
                   std::size_t K, std::size_t N, std::size_t R);

}  // namespace aspec
