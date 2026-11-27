//
// Non-additivity gate.
//
// Plan selection is only worth building if cost(A then B) != cost(A) + cost(B). If costs
// are additive, optimizing each operator independently is provably optimal and a plan
// optimizer is decoration. This measures whether they are.
//
// The mechanism under test is the 8 MB system-level cache. EXP-001 measured a 10x
// bandwidth difference between cache-resident and streaming reads, and one Qwen2.5-3B
// layer streams 41.3 MiB of weights, so what ran before an operator should change what
// that operator costs.
//
// METHOD, and why it is built this way. A first version timed single operators in their
// own command buffers under different preconditions. That produced nonsense: ratios below
// 1.0 (cold apparently faster than warm) and instability up to 85%. The cause was GPU
// clock state, not cache state. A 64 MB flush kernel leaves the GPU at high clocks, so
// whatever runs next looks fast, while a small precondition lets clocks sag. Separate
// command buffers per measurement added idle gaps that made it worse, and the 5.5 us
// per-command-buffer floor exceeded the cost of the smaller operators outright.
//
// So instead: encode a SEQUENCE repeated M times inside ONE command buffer and time the
// whole thing, with M chosen so each measurement runs ~30 ms. That pins clocks for the
// duration, amortizes the command-buffer floor to nothing, and never compares across
// power states. Non-additivity then falls out of differences between sequences:
//
//     gap = T([A,B]) - ( T([A]) + T([B]) )
//
// T([A]) is A run back-to-back, so A's weights stay resident if they fit in the SLC.
// That is exactly what a per-operator microbenchmark reports, and therefore what a greedy
// optimizer calibrated on per-operator numbers would predict. T([A,B]) is each operator
// seeing the other's footprint. The difference is the error greedy makes.
//
// Weights use a split Q4_0 layout: quants and scales in separate arrays. Footprint is
// identical to packed Q4_0 at 0.5625 B/weight, but both streams stay aligned and
// vectorizable, so bandwidth is not limited by unaligned 18-byte block reads.
//
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "aspec/bench.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr const char* kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint R_CONST   [[function_constant(0)]];  // columns; compile-time so acc[] stays in registers
constant uint SG_PER_TG [[function_constant(1)]];  // simdgroups per threadgroup (llama.cpp's nsg)
constant uint LANE_MAP  [[function_constant(2)]];  // 0 = block-per-lane, 1 = weight-per-lane

struct GemvArgs { uint K; uint N; uint nblk; };

// One simdgroup per output row, SG_PER_TG simdgroups per threadgroup. Lane t walks
// blocks strided by 32. SG_PER_TG is the occupancy knob: at 1 the GPU cannot hide the
// nibble-unpack latency and the kernel becomes ALU-bound regardless of footprint.
kernel void gemv_q4_0_split(
    device const uint*  qs     [[buffer(0)]],   // N * K/8 uints  (K/2 bytes per row)
    device const half*  scales [[buffer(1)]],   // N * nblk
    device const float* X      [[buffer(2)]],   // K * R
    device float*       Y      [[buffer(3)]],   // N * R
    constant GemvArgs&  a      [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint n = tgid * SG_PER_TG + sgid;
    if (n >= a.N) return;

    float acc[8];
    for (uint r = 0; r < R_CONST; ++r) acc[r] = 0.0f;

    const uint qrow = n * (a.K / 8);
    const uint srow = n * a.nblk;

    if (LANE_MAP == 0) {
        // BLOCK-PER-LANE. Lane t owns whole 32-weight blocks, striding by 32 blocks.
        // One 4-byte load feeds 8 weights, so unpack work is amortised 8x. The cost is
        // that X[k] across the 32 lanes is stride-128-bytes, i.e. uncoalesced.
        for (uint b = lane; b < a.nblk; b += 32) {
            const float d  = float(scales[srow + b]);
            const uint  q0 = qrow + b * 4;
            const uint  k0 = b * 32;
            for (uint r = 0; r < R_CONST; ++r) {
                const device float* xr = X + r * a.K;
                float sum = 0.0f;
                for (uint q = 0; q < 4; ++q) {
                    const uint packed = qs[q0 + q];
                    for (uint t = 0; t < 4; ++t) {
                        const uint byte = (packed >> (8u * t)) & 0xFFu;
                        const uint j    = q * 4 + t;
                        sum = fma(float(int(byte & 0x0Fu) - 8), xr[k0 + j],     sum);
                        sum = fma(float(int(byte >> 4)   - 8), xr[k0 + 16 + j], sum);
                    }
                }
                acc[r] = fma(sum, d, acc[r]);
            }
        }
    } else {
        // WEIGHT-PER-LANE. Lane t owns weight k = b*32 + t of every block, so X[k] across
        // the 32 lanes is 32 consecutive floats (fully coalesced) and all lanes broadcast
        // the same 16 weight bytes. The cost is one load/shift/mask per SINGLE weight
        // instead of per eight, which is 8x more unpack work.
        const uint jbyte = lane & 15u;
        const bool hi    = lane >= 16u;
        for (uint b = 0; b < a.nblk; ++b) {
            const uint  packed = qs[qrow + b * 4 + (jbyte >> 2)];
            const uint  byte   = (packed >> (8u * (jbyte & 3u))) & 0xFFu;
            const uint  nib    = hi ? (byte >> 4) : (byte & 0x0Fu);
            const float wd     = (float(nib) - 8.0f) * float(scales[srow + b]);
            const uint  k      = b * 32u + lane;
            for (uint r = 0; r < R_CONST; ++r)
                acc[r] = fma(wd, X[r * a.K + k], acc[r]);
        }
    }

    for (uint r = 0; r < R_CONST; ++r) {
        const float tot = simd_sum(acc[r]);
        if (lane == 0) Y[r * a.N + n] = tot;
    }
}

// Streams a buffer far larger than the SLC, to evict everything.
kernel void flush(device const float4* src  [[buffer(0)]],
                  device float*        sink [[buffer(1)]],
                  constant uint&       n4   [[buffer(2)]],
                  uint gid [[thread_position_in_grid]],
                  uint gsz [[threads_per_grid]])
{
    float4 acc = float4(0.0f);
    for (uint i = gid; i < n4; i += gsz) acc += src[i];
    sink[gid] = acc.x + acc.y + acc.z + acc.w;
}

kernel void nullk(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out[0] = 1.0f;
}
)METAL";

struct GemvArgs { uint32_t K, N, nblk; };

struct Tensor {
    const char* name;
    uint32_t K, N;
    id<MTLBuffer> qs, scales, X, Y;
    double mib() const { return double(K) * N * 0.5625 / (1024.0 * 1024.0); }
};

void die(const char* what, NSError* e) {
    std::fprintf(stderr, "%s: %s\n", what, e ? [[e localizedDescription] UTF8String] : "?");
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
 @autoreleasepool {
    const uint32_t R = (argc > 1) ? uint32_t(atoi(argv[1])) : 1;
    if (R < 1 || R > 8) { std::fprintf(stderr, "R must be 1..8\n"); return 1; }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* err = nil;

    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                          options:nil error:&err];
    if (!lib) die("compile", err);

    // R and SG_PER_TG as function constants: one kernel source, many specialized
    // pipelines, built at pipeline-creation time. R must be compile-time or the
    // accumulator array spills to memory and we measure the spill, not the cache.
    auto makeGemv = [&](uint32_t sgPerTg, uint32_t laneMap = 0) -> id<MTLComputePipelineState> {
        MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
        [fc setConstantValue:&R type:MTLDataTypeUInt atIndex:0];
        [fc setConstantValue:&sgPerTg type:MTLDataTypeUInt atIndex:1];
        [fc setConstantValue:&laneMap type:MTLDataTypeUInt atIndex:2];
        NSError* e2 = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@"gemv_q4_0_split"
                                      constantValues:fc error:&e2];
        if (!fn) die("specialize", e2);
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e2];
        if (!p) die("pso gemv", e2);
        return p;
    };
    uint32_t sgPerTg = 8;                       // overridden by the sweep below
    id<MTLComputePipelineState> psoGemv = makeGemv(sgPerTg);
    id<MTLComputePipelineState> psoFlush = [dev newComputePipelineStateWithFunction:
                                            [lib newFunctionWithName:@"flush"] error:&err];
    id<MTLComputePipelineState> psoNull  = [dev newComputePipelineStateWithFunction:
                                            [lib newFunctionWithName:@"nullk"] error:&err];
    if (!psoFlush || !psoNull) die("pso aux", err);

    std::printf("device  : %s\n", [[dev name] UTF8String]);
    std::printf("R       : %u column%s\n", R, R == 1 ? "" : "s");
    std::printf("SLC     : 8 MB (M3)\n\n");

    // ---------------------------------------------------------------- tensors
    // Qwen2.5-3B: hidden 2048, intermediate 11008, GQA 16/2 heads.
    std::vector<Tensor> ts = {
        {"q_proj",    2048,  2048},
        {"k_proj",    2048,   256},
        {"v_proj",    2048,   256},
        {"o_proj",    2048,  2048},
        {"gate_proj", 2048, 11008},
        {"up_proj",   2048, 11008},
        {"down_proj",11008,  2048},
    };
    enum { Q=0, K=1, V=2, O=3, G=4, U=5, D=6 };

    for (auto& t : ts) {
        const size_t qsBytes = size_t(t.K) * t.N / 2;              // 4 bits per weight
        const size_t nblk    = t.K / 32;
        const size_t scBytes = size_t(nblk) * t.N * sizeof(uint16_t);
        t.qs     = [dev newBufferWithLength:qsBytes options:MTLResourceStorageModeShared];
        t.scales = [dev newBufferWithLength:scBytes options:MTLResourceStorageModeShared];
        t.X      = [dev newBufferWithLength:size_t(t.K) * R * sizeof(float)
                                    options:MTLResourceStorageModeShared];
        t.Y      = [dev newBufferWithLength:size_t(t.N) * R * sizeof(float)
                                    options:MTLResourceStorageModeShared];
        if (!t.qs || !t.scales || !t.X || !t.Y) { std::fprintf(stderr, "alloc failed\n"); return 1; }

        // Known pattern so the result is analytic: every byte 0x99 means both nibbles are
        // 9, so every weight is 9-8 = 1. With scale 1 and x 1, each output must equal K.
        memset([t.qs contents], 0x99, qsBytes);
        uint16_t one_h = 0x3C00;                                    // fp16 1.0
        uint16_t* sc = (uint16_t*)[t.scales contents];
        for (size_t i = 0; i < nblk * t.N; ++i) sc[i] = one_h;
        float* x = (float*)[t.X contents];
        for (size_t i = 0; i < size_t(t.K) * R; ++i) x[i] = 1.0f;
    }

    // Sustained load used only to pin GPU clocks high before a measurement block.
    constexpr size_t kLoadBytes = 64ull * 1024 * 1024;
    id<MTLBuffer> loadSrc  = [dev newBufferWithLength:kLoadBytes
                                             options:MTLResourceStorageModePrivate];
    id<MTLBuffer> loadSink = [dev newBufferWithLength:65536 * sizeof(float)
                                             options:MTLResourceStorageModePrivate];

    // ------------------------------------------------------- encode helpers
    auto encodeGemv = [&](id<MTLComputeCommandEncoder> e, const Tensor& t) {
        GemvArgs a{t.K, t.N, t.K / 32};
        [e setComputePipelineState:psoGemv];
        const uint32_t tgCount = (t.N + sgPerTg - 1) / sgPerTg;
        [e setBuffer:t.qs offset:0 atIndex:0];
        [e setBuffer:t.scales offset:0 atIndex:1];
        [e setBuffer:t.X offset:0 atIndex:2];
        [e setBuffer:t.Y offset:0 atIndex:3];
        [e setBytes:&a length:sizeof(a) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(tgCount, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * sgPerTg, 1, 1)];
    };

    // Encode `seq` repeated `reps` times into a single serial compute encoder.
    auto runSeq = [&](const std::vector<int>& seq, int reps, bool timed) -> double {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];   // serial by default
        for (int i = 0; i < reps; ++i)
            for (int idx : seq) encodeGemv(e, ts[idx]);
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "dispatch failed\n"); std::exit(1);
        }
        return timed ? ([cb GPUEndTime] - [cb GPUStartTime]) : 0.0;
    };

    auto pinClocks = [&]{
        uint32_t n4 = uint32_t(kLoadBytes / 16);
        for (int i = 0; i < 3; ++i) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:psoFlush];
            [e setBuffer:loadSrc offset:0 atIndex:0];
            [e setBuffer:loadSink offset:0 atIndex:1];
            [e setBytes:&n4 length:sizeof(n4) atIndex:2];
            [e dispatchThreads:MTLSizeMake(65536,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }
    };

    // ---------------------------------------------------------- correctness
    std::printf("--- correctness (every weight = 1, x = 1, so Y must equal K) ---\n");
    bool allok = true;
    for (size_t i = 0; i < ts.size(); ++i) {
        runSeq({int(i)}, 1, false);
        const float* y = (const float*)[ts[i].Y contents];
        double worst = 0.0;
        for (uint32_t r = 0; r < R; ++r)
            for (uint32_t n = 0; n < ts[i].N; ++n)
                worst = std::fmax(worst, std::fabs(double(y[r * ts[i].N + n]) - double(ts[i].K)));
        const bool ok = worst <= 1e-3 * ts[i].K;
        allok &= ok;
        std::printf("  %-10s K=%5u N=%5u  %6.2f MiB  max|err| %.3g  %s\n",
                    ts[i].name, ts[i].K, ts[i].N, ts[i].mib(), worst, ok ? "PASS" : "FAIL");
    }
    if (!allok) { std::fprintf(stderr, "\nkernel is wrong; not reporting timings\n"); return 1; }

    // ------------------------------------------------- measure one sequence
    constexpr double kTargetSec = 0.030;   // per measurement, so the 5.5 us floor is ~0.02%
    struct Result { double perIter; double instab; int reps; };

    auto measure = [&](const std::vector<int>& seq) -> Result {
        pinClocks();
        const double one = runSeq(seq, 1, true);                 // rough estimate
        int reps = std::max(4, int(kTargetSec / std::fmax(one, 1e-6)));
        reps = std::min(reps, 20000);
        runSeq(seq, reps, false);                                // warm up at this size
        auto st = aspec::Stats::from(aspec::timed_reps(2, 15, [&]{ return runSeq(seq, reps, true); }));
        return {st.median / reps, st.instability(), reps};
    };

    auto name = [&](const std::vector<int>& seq) {
        std::string s2;
        for (size_t i = 0; i < seq.size(); ++i) { if (i) s2 += "+"; s2 += ts[seq[i]].name; }
        return s2;
    };

    // Per-operator costs, measured back-to-back. This is the greedy optimizer's input.
    std::printf("\n--- per-operator cost, measured in isolation (what greedy is calibrated on) ---\n");
    std::printf("%-10s %8s %10s %8s %9s %12s\n",
                "tensor", "MiB", "us/op", "reps", "instab", "eff GB/s");
    std::vector<Result> solo(ts.size());
    uint32_t soloSg = 8;
    for (size_t oi : aspec::shuffled_order(ts.size())) {
        solo[oi] = measure({int(oi)});
        }
    for (size_t i = 0; i < ts.size(); ++i) {
        const double bytes = ts[i].mib() * 1024.0 * 1024.0;
        std::printf("%-10s %8.2f %10.2f %8d %8.0f%% %12.1f\n",
                    ts[i].name, ts[i].mib(), solo[i].perIter * 1e6, solo[i].reps,
                    100.0 * solo[i].instab, bytes / solo[i].perIter / 1e9);
    }

    // ------------------------------- lane mapping x occupancy sweep (P3)
    // Cache interference can only be observed if the kernel is memory-bound. Report
    // effective bandwidth on the weight stream against the two reference points from
    // EXP-001: the 89 GB/s DRAM plateau and the 913 GB/s cache-resident figure.
    std::printf("\n--- effective GB/s on weights: lane mapping x simdgroups-per-threadgroup ---\n");
    const uint32_t sgOpts[] = {1, 2, 4, 8, 16, 32};
    constexpr size_t kNSg = sizeof(sgOpts)/sizeof(sgOpts[0]);
    id<MTLComputePipelineState> psos[2][kNSg];
    for (uint32_t lm = 0; lm < 2; ++lm)
        for (size_t j = 0; j < kNSg; ++j) psos[lm][j] = makeGemv(sgOpts[j], lm);

    uint32_t bestSg[16] = {0}, bestLm[16] = {0};
    double   bestGbs[16] = {0};
    for (uint32_t lm = 0; lm < 2; ++lm) {
        std::printf("  %s\n", lm == 0 ? "block-per-lane (8 weights per load, uncoalesced X)"
                                       : "weight-per-lane (1 weight per load, coalesced X)");
        std::printf("    %-10s %8s", "tensor", "MiB");
        for (uint32_t sg : sgOpts) std::printf(" %8u", sg);
        std::printf("\n");
        for (size_t i = 0; i < ts.size(); ++i) {
            const double bytes = ts[i].mib() * 1024.0 * 1024.0;
            std::printf("    %-10s %8.2f", ts[i].name, ts[i].mib());
            for (size_t j = 0; j < kNSg; ++j) {
                sgPerTg = sgOpts[j]; psoGemv = psos[lm][j];
                const double gbs = bytes / measure({int(i)}).perIter / 1e9;
                std::printf(" %8.1f", gbs);
                if (gbs > bestGbs[i]) { bestGbs[i] = gbs; bestSg[i] = sgOpts[j]; bestLm[i] = lm; }
            }
            std::printf("\n");
        }
    }
    std::printf("  (llama.cpp hardcodes nsg=2 for every tensor)\n");

    std::printf("\n--- per-operator cost at each tensor's best configuration ---\n");
    std::printf("%-10s %8s %10s %9s %10s %9s  %s\n",
                "tensor", "MiB", "us/op", "instab", "eff GB/s", "best cfg", "regime");
    for (size_t i = 0; i < ts.size(); ++i) {
        for (size_t j = 0; j < kNSg; ++j)
            if (sgOpts[j] == bestSg[i]) { sgPerTg = sgOpts[j]; psoGemv = psos[bestLm[i]][j]; }
        solo[i] = measure({int(i)});
        const double gbs = ts[i].mib() * 1024.0 * 1024.0 / solo[i].perIter / 1e9;
        char cfg[32]; snprintf(cfg, sizeof(cfg), "lm%u/nsg%u", bestLm[i], bestSg[i]);
        std::printf("%-10s %8.2f %10.2f %8.0f%% %10.1f %9s  %s\n",
                    ts[i].name, ts[i].mib(), solo[i].perIter * 1e6,
                    100.0 * solo[i].instab, gbs, cfg,
                    gbs > 200.0 ? "memory-bound, cache-resident"
                                : (gbs > 80.0 ? "near the DRAM roof" : "ALU-BOUND"));
    }

    // Sequences run at the configuration that was best for the first tensor in them.
    for (size_t j = 0; j < kNSg; ++j)
        if (sgOpts[j] == bestSg[G]) { sgPerTg = sgOpts[j]; psoGemv = psos[bestLm[G]][j]; }

    // Sequences, and what greedy would have predicted for each.
    // Each sequence runs at a single occupancy, so pick the one that is best for the
    // largest tensor in it; mixing per-op pipelines inside a sequence is what the real
    // plan optimizer would do, and is a later experiment.
    std::printf("\n--- additivity: measured sequence vs sum of isolated costs ---\n");
    std::vector<std::vector<int>> seqs = {
        {G, U, D},                  // the FFN block
        {Q, K, V},                  // attention projections, all SLC-resident alone
        {Q, G},                     // small + large, the eviction case
        {Q, K, V, O, G, U, D},      // a whole layer
    };
    std::printf("%-42s %11s %11s %9s %8s\n",
                "sequence", "measured", "greedy sum", "gap", "instab");
    double worstGap = 0.0; std::string worstSeq;
    for (const auto& sq : seqs) {
        const Result r = measure(sq);
        double sum = 0.0;
        for (int idx : sq) sum += solo[idx].perIter;
        const double gap = 100.0 * (r.perIter - sum) / sum;
        if (std::fabs(gap) > std::fabs(worstGap)) { worstGap = gap; worstSeq = name(sq); }
        std::printf("%-42s %9.2f us %9.2f us %+8.1f%% %7.0f%%\n",
                    name(sq).c_str(), r.perIter * 1e6, sum * 1e6, gap, 100.0 * r.instab);
    }

    // --------------------------------------------------------------- verdict
    double maxInstab = 0.0;
    for (const auto& s2 : solo) maxInstab = std::fmax(maxInstab, s2.instab);
    std::printf("\n--- verdict ---\n");
    std::printf("  largest additivity gap : %+.1f%%  (%s)\n", worstGap, worstSeq.c_str());
    std::printf("  worst per-op instability: %.0f%%\n", 100.0 * maxInstab);
    if (maxInstab > 0.25) {
        std::printf("  => MEASUREMENT NOT TRUSTWORTHY; instability too high to conclude\n");
    } else if (std::fabs(worstGap) > 10.0) {
        std::printf("  => cost is NOT additive: execution history changes operator cost by\n");
        std::printf("     more than 10%%, so plan selection has something real to exploit\n");
    } else {
        std::printf("  => cost is approximately additive on these shapes; per-operator\n");
        std::printf("     optimization is close to optimal and plan selection buys little\n");
    }
 }
 return 0;
}
