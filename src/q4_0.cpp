#include "aspec/q4_0.hpp"

// Implement against the spec in include/aspec/q4_0.hpp and the tests in
// tests/test_q4_0.cpp:
//
//     cmake --build build && ./build/test_q4_0
//
// Order: quantize, dequantize, gemv. test_layout passing proves the nibble
// interleaving is right, which is the only part with a real gotcha.

namespace aspec {

void quantize_q4_0(const float* src, BlockQ4_0* dst, std::size_t k) {
    (void)src; (void)dst; (void)k;
    // TODO
}

void dequantize_q4_0(const BlockQ4_0* src, float* dst, std::size_t k) {
    (void)src; (void)dst; (void)k;
    // TODO
}

void gemv_q4_0_ref(const BlockQ4_0* W, const float* X, float* Y,
                   std::size_t K, std::size_t N, std::size_t R) {
    (void)W; (void)X; (void)Y; (void)K; (void)N; (void)R;
    // TODO
}

}  // namespace aspec
