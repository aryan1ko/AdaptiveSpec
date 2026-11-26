#pragma once
//
// Measurement harness. Every number in this project goes through here.
//
// The target is a fanless 8 GB laptop, so the defaults exist for specific reasons:
// warmup is excluded, stats are reported as median/p95/p99 rather than a mean (one
// thermal excursion moves a mean and leaves a median alone), and sweeps visit their
// parameter in randomized order so thermal drift can't look like a real trend.
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

struct Stats {
    double min = 0, median = 0, mean = 0, p95 = 0, p99 = 0, stddev = 0;
    std::size_t n = 0;

    static Stats from(std::vector<double> xs) {   // by value: we sort it
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

    // Spread of the middle of the distribution as a fraction of the median. High means
    // the measurement isn't steady-state and shouldn't be quoted. Uses p95 rather than
    // max so a rare scheduling outlier doesn't condemn a clean run.
    double instability() const { return median > 0 ? (p95 - min) / median : 0.0; }

private:
    static double quantile_sorted(const std::vector<double>& sorted, double q) {
        if (sorted.size() == 1) return sorted[0];
        const double pos = q * double(sorted.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
        const double frac = pos - double(lo);
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }
};

// fn returns the duration of one iteration in seconds. Returning it rather than timing
// around fn lets callers report GPU-timestamped time instead of wall clock.
inline std::vector<double> timed_reps(int warmup, int reps,
                                      const std::function<double()>& fn) {
    for (int i = 0; i < warmup; ++i) (void)fn();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) out.push_back(fn());
    return out;
}

// Shuffled visiting order over [0, n). Seeded, so sweeps stay reproducible.
inline std::vector<std::size_t> shuffled_order(std::size_t n, std::uint64_t seed = 0xA5EC) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::mt19937_64 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    return idx;
}

inline double gib_per_s(double bytes, double seconds) {
    return bytes / seconds / (1024.0 * 1024.0 * 1024.0);
}
inline double gb_per_s(double bytes, double seconds) {
    return bytes / seconds / 1e9;   // vendor specs are decimal GB/s
}
inline double gflop_per_s(double flops, double seconds) {
    return flops / seconds / 1e9;
}

}  // namespace aspec
