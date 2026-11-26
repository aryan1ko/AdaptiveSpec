//
// Correctness tests for the Q4_0 reference path. test_layout matters most: it pins exact
// bytes against hand-computed expectations from ggml's quantize_row_q4_0_ref, so passing
// it means real GGUF weights will load.
//
#include "aspec/q4_0.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int g_failures = 0;
int g_checks   = 0;
int g_printed  = 0;
constexpr int kMaxPrint = 6;   // keep a red run readable

void begin_test(const char* name) { g_printed = 0; std::printf("%s\n", name); }

// Returns true if this failure should be printed, false once the per-test cap is hit.
bool want_print() {
    if (g_printed < kMaxPrint) { ++g_printed; return true; }
    if (g_printed == kMaxPrint) { ++g_printed; std::printf("  ... further failures suppressed\n"); }
    return false;
}

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) { ++g_failures; if (want_print()) std::printf("  FAIL  %s\n", what); }
}

void check_eq_u8(uint8_t got, uint8_t want, const char* what, int idx) {
    ++g_checks;
    if (got != want) {
        ++g_failures;
        if (want_print())
            std::printf("  FAIL  %s[%d]: got 0x%02X want 0x%02X\n", what, idx, got, want);
    }
}

void check_near(float got, float want, float tol, const char* what, int idx) {
    ++g_checks;
    if (!(std::fabs(got - want) <= tol)) {
        ++g_failures;
        if (want_print())
            std::printf("  FAIL  %s[%d]: got %.9g want %.9g (tol %.3g)\n",
                        what, idx, got, want, tol);
    }
}

// ---------------------------------------------------------------------------
// Test 1: exact bit layout for one hand-computed block.
//
// Input: x[0] = -8, x[3] = 4, x[19] = -4, all other elements 0.
//   element of largest magnitude is x[0], so max = -8, amax = 8
//   d  = max / -8 = 1.0,  id = 1.0
//
//   j=0 : low  <- x[0]  = -8 -> (int8_t)(-8 + 8.5) = (int8_t)0.5  = 0
//         high <- x[16] =  0 -> (int8_t)( 0 + 8.5) = (int8_t)8.5  = 8
//         qs[0] = 0 | (8 << 4) = 0x80
//   j=3 : low  <- x[3]  =  4 -> (int8_t)( 4 + 8.5) = (int8_t)12.5 = 12
//         high <- x[19] = -4 -> (int8_t)(-4 + 8.5) = (int8_t)4.5  = 4
//         qs[3] = 12 | (4 << 4) = 0x4C
//   else: both halves 0 -> 8 | (8 << 4) = 0x88
//
// This also catches the common mistake of packing weights 2j/2j+1 into byte j instead
// of weights j/j+16.
// ---------------------------------------------------------------------------
void test_layout() {
    begin_test("test_layout");
    std::vector<float> x(aspec::kQK4_0, 0.0f);
    x[0]  = -8.0f;
    x[3]  =  4.0f;
    x[19] = -4.0f;

    aspec::BlockQ4_0 b{};
    aspec::quantize_q4_0(x.data(), &b, x.size());

    check_near(static_cast<float>(b.d), 1.0f, 1e-6f, "d", 0);
    for (int j = 0; j < aspec::kQK4_0 / 2; ++j) {
        uint8_t want = (j == 0) ? 0x80 : (j == 3) ? 0x4C : 0x88;
        check_eq_u8(b.qs[j], want, "qs", j);
    }

    // Round-trip the same block: the three nonzero inputs are exactly representable.
    std::vector<float> y(aspec::kQK4_0, 999.0f);
    aspec::dequantize_q4_0(&b, y.data(), y.size());
    for (int j = 0; j < aspec::kQK4_0; ++j) {
        float want = (j == 0) ? -8.0f : (j == 3) ? 4.0f : (j == 19) ? -4.0f : 0.0f;
        check_near(y[j], want, 1e-6f, "roundtrip", j);
    }
}

// ---------------------------------------------------------------------------
// Test 2: round-trip error bound on random data.
//
// |d| = amax/8 and q-8 = round(x/d), so a typical element errs by at most |d|/2 =
// amax/16. The asymmetric range (-8..7) means an element with magnitude amax but sign
// opposite to `max` saturates at q=15 and can reach amax/8, so amax/8 is the hard bound.
// The mean check is what actually catches a subtly wrong implementation; the hard bound
// alone is too loose.
// ---------------------------------------------------------------------------
void test_roundtrip_bound() {
    begin_test("test_roundtrip_bound");
    constexpr std::size_t K = 4096;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x(K);
    for (auto& v : x) v = dist(rng);

    std::vector<aspec::BlockQ4_0> q(aspec::q4_0_blocks(K));
    std::vector<float> y(K);
    aspec::quantize_q4_0(x.data(), q.data(), K);
    aspec::dequantize_q4_0(q.data(), y.data(), K);

    const float fp16_slack = 1e-3f;   // d is stored as fp16: ~2^-11 relative
    bool  hard_ok = true;
    double err_sum = 0.0;
    double amax_sum = 0.0;

    for (std::size_t b = 0; b < q.size(); ++b) {
        float amax = 0.0f;
        for (int j = 0; j < aspec::kQK4_0; ++j)
            amax = std::max(amax, std::fabs(x[b * aspec::kQK4_0 + j]));
        for (int j = 0; j < aspec::kQK4_0; ++j) {
            const std::size_t i = b * aspec::kQK4_0 + j;
            const double e = std::fabs(double(y[i]) - double(x[i]));
            if (e > amax * (1.0 / 8.0 + fp16_slack)) hard_ok = false;
            err_sum += e;
        }
        amax_sum += amax;
    }
    check(hard_ok, "every element within amax/8");

    const double mean_err  = err_sum / double(K);
    const double mean_amax = amax_sum / double(q.size());
    check(mean_err < mean_amax / 16.0, "mean error below amax/16");
    std::printf("  (mean |err| = %.5f, mean amax = %.5f, ratio = %.3f of amax)\n",
                mean_err, mean_amax, mean_err / mean_amax);
}

// ---------------------------------------------------------------------------
// Test 3: the largest-magnitude element maps to q=0 and dequantizes to -8d == max, so
// it must survive the round trip apart from fp16 rounding of d.
// ---------------------------------------------------------------------------
void test_max_element_exact() {
    begin_test("test_max_element_exact");
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    for (int trial = 0; trial < 64; ++trial) {
        std::vector<float> x(aspec::kQK4_0);
        for (auto& v : x) v = dist(rng);

        std::size_t arg = 0;
        for (std::size_t j = 1; j < x.size(); ++j)
            if (std::fabs(x[j]) > std::fabs(x[arg])) arg = j;

        aspec::BlockQ4_0 b{};
        std::vector<float> y(aspec::kQK4_0);
        aspec::quantize_q4_0(x.data(), &b, x.size());
        aspec::dequantize_q4_0(&b, y.data(), y.size());

        // fp16 has ~11 mantissa bits; allow a relative slack on the scale.
        const float tol = std::fabs(x[arg]) * 1e-3f + 1e-6f;
        check_near(y[arg], x[arg], tol, "max element", int(trial));
    }
}

// ---------------------------------------------------------------------------
// Test 4: gemv_q4_0_ref agrees with dequantize-then-plain-fp32-GEMV.
//
// K = 2048 is Qwen2.5-3B's hidden size; N is kept small so the test stays fast. R=1 is
// plain decode, R>1 is what speculative verification produces (llama.cpp's ne11).
// ---------------------------------------------------------------------------
void test_gemv_matches_dequantized() {
    begin_test("test_gemv_matches_dequantized");
    constexpr std::size_t K = 2048, N = 128;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> Wf(K * N);
    for (auto& v : Wf) v = dist(rng);

    // Quantize row by row: each row of N is K contiguous values.
    std::vector<aspec::BlockQ4_0> Wq(N * aspec::q4_0_blocks(K));
    for (std::size_t n = 0; n < N; ++n)
        aspec::quantize_q4_0(&Wf[n * K], &Wq[n * aspec::q4_0_blocks(K)], K);

    // The oracle must see the DEQUANTIZED weights, not the originals: we are testing
    // the GEMV, not the quantizer.
    std::vector<float> Wdq(K * N);
    for (std::size_t n = 0; n < N; ++n)
        aspec::dequantize_q4_0(&Wq[n * aspec::q4_0_blocks(K)], &Wdq[n * K], K);

    for (std::size_t R = 1; R <= 5; ++R) {
        std::vector<float> X(K * R);
        for (auto& v : X) v = dist(rng);

        std::vector<float> Y(N * R, 0.0f), Yref(N * R, 0.0f);
        aspec::gemv_q4_0_ref(Wq.data(), X.data(), Y.data(), K, N, R);

        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < K; ++k) acc += Wdq[n * K + k] * X[r * K + k];
                Yref[r * N + n] = acc;
            }

        float max_ref = 0.0f, max_diff = 0.0f;
        for (std::size_t i = 0; i < Y.size(); ++i) {
            max_ref  = std::max(max_ref, std::fabs(Yref[i]));
            max_diff = std::max(max_diff, std::fabs(Y[i] - Yref[i]));
        }
        // Non-degeneracy first: with dequantize stubbed, Wdq and Yref are all-zero and
        // a max_diff==0 comparison "passes" while proving nothing.
        ++g_checks;
        if (!(max_ref > 1e-3f)) {
            ++g_failures;
            if (want_print())
                std::printf("  FAIL  R=%zu: reference output is degenerate (max_ref %.3g)"
                            " -- dequantize_q4_0 likely unimplemented\n", R, max_ref);
            continue;
        }

        // Tolerance allows a different summation order over K=2048 float terms.
        const bool ok = max_diff <= 1e-4f * max_ref;
        ++g_checks;
        if (!ok) {
            ++g_failures;
            if (want_print())
                std::printf("  FAIL  R=%zu: max_diff %.6g vs max_ref %.6g (rel %.3g)\n",
                            R, max_diff, max_ref, max_diff / max_ref);
        } else {
            std::printf("  R=%zu ok (rel err %.3g)\n", R, max_diff / max_ref);
        }
    }
}

}  // namespace

int main() {
    test_layout();
    test_roundtrip_bound();
    test_max_element_exact();
    test_gemv_matches_dequantized();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) std::printf("ALL PASS\n");
    return g_failures == 0 ? 0 : 1;
}
