#pragma once
//
// Measurement harness. Every number this project reports flows through here, so the
// rigor requirements live in one place:
//   - warmup iterations excluded from statistics
//   - many timed reps, reported as median / p95 / p99 / min (never mean alone: a single
//     thermal excursion or scheduler hiccup moves a mean and leaves a median alone)
//   - randomized trial order for sweeps, so a monotonic thermal drift cannot masquerade
//     as a monotonic trend in the independent variable
//   - fixed seeds
//
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace aspec {

// ---------------------------------------------------------------------------- Stats

struct Stats {
    double min = 0, median = 0, mean = 0, p95 = 0, p99 = 0, stddev = 0;
    std::size_t n = 0;

    // xs is taken by value: we sort it.
    static Stats from(std::vector<double> xs) {
        Stats s;
        if (xs.empty()) return s;
        std::sort(xs.begin(), xs.end());
        s.n      = xs.size();
        s.min    = xs.front();
        s.median = quantile_sorted(xs, 0.50);
        s.p95    = quantile_sorted(xs, 0.95);
        s.p99    = quantile_sorted(xs, 0.99);
        s.mean   = std::accumulate(xs.begin(), xs.end(), 0.0) / double(s.n);
        double acc = 0.0;
        for (double x : xs) acc += (x - s.mean) * (x - s.mean);
        s.stddev = (s.n > 1) ? std::sqrt(acc / double(s.n - 1)) : 0.0;
        return s;
    }

    // Spread of the middle of the distribution, as a fraction of the median. A large
    // value means the measurement is not steady-state and the number should not be
    // quoted. Deliberately uses p95 rather than max: we expect rare OS-scheduling
    // outliers and do not want them to condemn an otherwise clean run.
    double instability() const { return median > 0 ? (p95 - min) / median : 0.0; }

private:
    // Linear interpolation between order statistics (same convention as numpy's
    // default 'linear' method) on an already-sorted vector.
    static double quantile_sorted(const std::vector<double>& sorted, double q) {
        if (sorted.size() == 1) return sorted[0];
        const double pos = q * double(sorted.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
        const double frac = pos - double(lo);
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }
};

// ------------------------------------------------------------------------- Timing

// Calls fn() `warmup` times discarding results, then `reps` times collecting them.
// fn must return the duration of one iteration in seconds. Returning the duration
// rather than measuring around fn lets callers report GPU-timestamp time (from
// MTLCommandBuffer GPUStartTime/GPUEndTime) instead of wall clock, which includes
// CPU-side encode and dispatch latency.
inline std::vector<double> timed_reps(int warmup, int reps,
                                      const std::function<double()>& fn) {
    for (int i = 0; i < warmup; ++i) (void)fn();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) out.push_back(fn());
    return out;
}

// ------------------------------------------------------------- Sweep ordering

// A shuffled visiting order over [0, n). Use this for every parameter sweep: measuring
// ne11 = 1,2,3,...,16 in order on a thermally-throttling laptop manufactures a fake
// upward trend. Seeded, so a sweep is reproducible.
inline std::vector<std::size_t> shuffled_order(std::size_t n, std::uint64_t seed = 0xA5EC) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::mt19937_64 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    return idx;
}

// ------------------------------------------------------------------- Derived units

inline double gib_per_s(double bytes, double seconds) {
    return bytes / seconds / (1024.0 * 1024.0 * 1024.0);
}
inline double gb_per_s(double bytes, double seconds) {
    return bytes / seconds / 1e9;   // vendor bandwidth specs are decimal GB/s
}
inline double gflop_per_s(double flops, double seconds) {
    return flops / seconds / 1e9;
}

}  // namespace aspec
