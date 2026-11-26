#pragma once
//
// GGUF Q4_0, bit-compatible with ggml so week 4 can load real GGUF weights.
//
// Layout (ggml/src/ggml-common.h):
//     typedef struct { ggml_half d; uint8_t qs[16]; } block_q4_0;   // 18 bytes
//
// The packing is the part that catches people. Byte qs[j] does NOT hold weights 2j and
// 2j+1. It holds weight j in the low nibble and weight j+16 in the high nibble, so the
// block is split in half and the halves interleave at nibble granularity. ggml did this
// so a SIMD unpack yields 16 consecutive weights from one mask and one shift of the same
// register instead of a shuffle.
//
// Quantize (ggml/src/ggml-quants.c, quantize_row_q4_0_ref): find the element of largest
// magnitude, take its signed value as `max`, then
//     d = max / -8          (negative when max is positive)
//     q = MIN(15, (int8_t)(x/d + 8.5f))
// Dequantize: value = (q - 8) * d
//
// Two consequences worth knowing before implementing:
//   - The largest-magnitude element always maps to q=0 and dequantizes to -8d == max,
//     so it round-trips exactly apart from fp16 rounding of d.
//   - That cast truncates, but x/d is in [-8,8] by construction so the argument is
//     always positive and truncating (v + 8.5) is round-to-nearest of (v + 8). There is
//     deliberately no lower clamp.
//
#include <cstddef>
#include <cstdint>

namespace aspec {

using half_t = __fp16;   // native ARM64 fp16, matches ggml_half bit-for-bit

constexpr int kQK4_0 = 32;

struct BlockQ4_0 {
    half_t  d;
    uint8_t qs[kQK4_0 / 2];
};
static_assert(sizeof(BlockQ4_0) == 18, "block_q4_0 must be 18 bytes to match ggml");

inline std::size_t q4_0_blocks(std::size_t k) { return k / kQK4_0; }

// k must be a multiple of 32. Must be bit-identical to quantize_row_q4_0_ref.
void quantize_q4_0(const float* src, BlockQ4_0* dst, std::size_t k);
void dequantize_q4_0(const BlockQ4_0* src, float* dst, std::size_t k);

// Reference quantized GEMV / thin GEMM, in ggml's index convention:
//
//     W  quantized weights, [K, N], K contiguous: row n is blocks [n*K/32, (n+1)*K/32)
//     X  fp32 activations,  [K, R], column r is [r*K, (r+1)*K)
//     Y  fp32 output,       [N, R]
//
//     Y[r*N + n] = sum_k W[n*K + k] * X[r*K + k]
//
// R is the column count: 1 for plain decode, gamma+1 under speculative verification.
// This is what llama.cpp calls ne11 and gates on.
//
// Correctness only. This is the oracle for the Metal kernels, so keep it obvious rather
// than fast. Accumulate in float, not double: the GPU kernels accumulate in float, and
// an oracle more accurate than the thing under test makes tolerances harder to reason
// about rather than easier.
void gemv_q4_0_ref(const BlockQ4_0* W, const float* X, float* Y,
                   std::size_t K, std::size_t N, std::size_t R);

}  // namespace aspec
