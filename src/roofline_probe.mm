//
// Measure this machine's two roofline parameters instead of quoting them:
//   1. streaming-read bandwidth, swept across footprint so the SLC cliff is visible
//   2. peak FP32 FMA throughput, register-resident so no memory traffic intrudes
//
// Their ratio is the ridge point, which gives the column count at which a Q4_0 GEMV
// stops being memory-bound. A Qwen2.5-3B weight tensor is only a few MB, so whether a
// decode GEMV streams from DRAM or hits the SLC changes how every later number reads.
//
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "aspec/bench.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

// Q4_0: 32 quantized weights per block, 16 B of packed nibbles + one fp16 scale.
constexpr double kQ4_0_BytesPerWeight = 18.0 / 32.0;   // = 0.5625

constexpr const char* kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct BwArgs  { uint n4; uint stride; uint passes; };
struct FmaArgs { uint iters; float b; float c; };

// gid-derived values, so a constant-valued buffer can't let the memory system do
// anything clever.
kernel void bw_fill(device float4*   dst  [[buffer(0)]],
                    uint             gid  [[thread_position_in_grid]])
{
    dst[gid] = float4(float(gid), float(gid) + 1.0f, float(gid) + 2.0f, float(gid) + 3.0f);
}

// Consecutive threads touch consecutive float4s, so each simdgroup issues one coalesced
// 512 B transaction per step. acc is written out so the loads can't be eliminated; that
// write is grid-sized, under 1% of the read volume.
kernel void bw_read(device const float4*  src  [[buffer(0)]],
                    device float*         sink [[buffer(1)]],
                    constant BwArgs&      a    [[buffer(2)]],
                    uint                  gid  [[thread_position_in_grid]])
{
    float4 acc = float4(0.0f);
    for (uint p = 0; p < a.passes; ++p) {
        for (uint i = gid; i < a.n4; i += a.stride) {
            acc += src[i];
        }
    }
    sink[gid] = acc.x + acc.y + acc.z + acc.w;
}

// Eight independent FMA chains: enough ILP to keep the FP32 pipeline fed while each
// chain stays a real loop-carried dependency. b and c come through a buffer so the
// recurrence can't be constant-folded, and the trip count is runtime so the loop can't
// be unrolled away.
kernel void fma_peak(device float*     out  [[buffer(0)]],
                     constant FmaArgs& a    [[buffer(1)]],
                     uint              gid  [[thread_position_in_grid]])
{
    const float b = a.b, c = a.c;
    float a0 = float(gid) * 1e-7f;
    float a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
    float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;

    for (uint i = 0; i < a.iters; ++i) {
        a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
)METAL";

struct BwArgs  { uint32_t n4; uint32_t stride; uint32_t passes; };
struct FmaArgs { uint32_t iters; float b; float c; };

const char* thermal_name(NSProcessInfoThermalState s) {
    switch (s) {
        case NSProcessInfoThermalStateNominal:  return "nominal";
        case NSProcessInfoThermalStateFair:     return "fair";
        case NSProcessInfoThermalStateSerious:  return "SERIOUS";
        case NSProcessInfoThermalStateCritical: return "CRITICAL";
    }
    return "unknown";
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> dev, id<MTLLibrary> lib,
                                          NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { std::fprintf(stderr, "missing kernel %s\n", [name UTF8String]); std::exit(1); }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        std::fprintf(stderr, "pipeline %s: %s\n", [name UTF8String],
                     [[err localizedDescription] UTF8String]);
        std::exit(1);
    }
    return pso;
}

// One dispatch, returning its GPU-timestamped duration in seconds. GPU timestamps
// exclude CPU-side encode and submission, which is what we want when characterizing a
// kernel rather than a pipeline.
double run_once(id<MTLCommandQueue> q, id<MTLComputePipelineState> pso,
                NSUInteger grid, NSUInteger tg,
                const std::vector<std::pair<NSUInteger, id<MTLBuffer>>>& buffers,
                const void* args, NSUInteger args_len, NSUInteger args_idx) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (const auto& b : buffers) [enc setBuffer:b.second offset:0 atIndex:b.first];
    if (args) [enc setBytes:args length:args_len atIndex:args_idx];
    [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] != MTLCommandBufferStatusCompleted) {
        std::fprintf(stderr, "dispatch failed (status %ld)\n", (long)[cb status]);
        std::exit(1);
    }
    return [cb GPUEndTime] - [cb GPUStartTime];
}

}  // namespace

int main() {
 @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::fprintf(stderr, "no Metal device\n"); return 1; }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                          options:nil error:&err];
    if (!lib) {
        std::fprintf(stderr, "compile failed: %s\n", [[err localizedDescription] UTF8String]);
        return 1;
    }
    auto pso_fill = make_pipeline(dev, lib, @"bw_fill");
    auto pso_read = make_pipeline(dev, lib, @"bw_read");
    auto pso_fma  = make_pipeline(dev, lib, @"fma_peak");

    std::printf("device               : %s\n", [[dev name] UTF8String]);
    std::printf("threadExecutionWidth : %lu (bw_read)\n",
                (unsigned long)[pso_read threadExecutionWidth]);
    std::printf("thermal state (pre)  : %s\n",
                thermal_name([[NSProcessInfo processInfo] thermalState]));

    // ---------------------------------------------------------------- bandwidth
    //
    // One 512 MB allocation read back at a range of footprints. Grid fixed at 65536
    // threads so occupancy stays constant and only footprint varies, which puts the
    // smallest measurable footprint at 65536 * 16 B = 1 MiB.
    constexpr NSUInteger kGrid = 65536, kTG = 256;
    constexpr std::size_t kMaxBytes = 512ull * 1024 * 1024;
    constexpr double kTargetBytes = 512.0 * 1024 * 1024;   // work per rep, for timer resolution

    id<MTLBuffer> src  = [dev newBufferWithLength:kMaxBytes
                                          options:MTLResourceStorageModePrivate];
    id<MTLBuffer> sink = [dev newBufferWithLength:kGrid * sizeof(float)
                                          options:MTLResourceStorageModePrivate];
    if (!src || !sink) { std::fprintf(stderr, "buffer alloc failed\n"); return 1; }

    run_once(queue, pso_fill, kMaxBytes / sizeof(float) / 4, kTG, {{0, src}}, nullptr, 0, 0);

    const std::vector<std::size_t> footprints_mib = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    constexpr uint32_t kMinReadsPerThread = 8;

    std::printf("\n--- streaming read bandwidth vs footprint ---\n");
    std::printf("%10s  %8s  %10s  %10s  %9s\n",
                "footprint", "rd/thr", "GB/s", "GiB/s", "instab");

    // Two bandwidths matter and conflating them is a trap. Qwen2.5-3B Q4_0 tensors run
    // 0.28 to 12.09 MiB, so a benchmark that hammers one tensor measures cache-resident
    // bandwidth and overstates decode by ~10x. A real forward pass streams 1.45 GiB per
    // token and evicts everything, so the streaming plateau governs decode. Report both.

    // Randomized order, so thermal drift can't impersonate a footprint trend.
    const auto order = aspec::shuffled_order(footprints_mib.size());
    std::vector<double> bw_result(footprints_mib.size(), 0.0);
    std::vector<double> bw_instab(footprints_mib.size(), 0.0);
    std::vector<uint32_t> bw_rdthr(footprints_mib.size(), 0);

    for (std::size_t oi : order) {
        const std::size_t bytes = footprints_mib[oi] * 1024 * 1024;
        BwArgs a{};
        a.n4     = static_cast<uint32_t>(bytes / 16);
        a.stride = static_cast<uint32_t>(kGrid);
        a.passes = static_cast<uint32_t>(std::max(1.0, kTargetBytes / double(bytes)));

        const double total_bytes = double(bytes) * double(a.passes);
        auto samples = aspec::timed_reps(3, 15, [&] {
            return run_once(queue, pso_read, kGrid, kTG, {{0, src}, {1, sink}}, &a, sizeof(a), 2);
        });
        const auto st = aspec::Stats::from(samples);
        bw_result[oi] = total_bytes / st.median;
        bw_instab[oi] = st.instability();
        bw_rdthr[oi]  = a.n4 / static_cast<uint32_t>(kGrid);
    }
    for (std::size_t i = 0; i < footprints_mib.size(); ++i) {
        std::printf("%6zu MiB  %8u  %10.1f  %10.1f  %8.1f%%%s\n", footprints_mib[i],
                    bw_rdthr[i], bw_result[i] / 1e9,
                    bw_result[i] / (1024.0 * 1024 * 1024), bw_instab[i] * 100.0,
                    bw_rdthr[i] < kMinReadsPerThread ? "   <- too few reads/thread" : "");
    }

    // ------------------------------------------------------------------- FLOP/s
    constexpr NSUInteger kFmaGrid = 65536;
    FmaArgs fa{};
    fa.iters = 4096;
    fa.b = 1.0000001f;
    fa.c = 0.9999999f;

    id<MTLBuffer> fout = [dev newBufferWithLength:kFmaGrid * sizeof(float)
                                          options:MTLResourceStorageModePrivate];
    auto fsamples = aspec::timed_reps(3, 15, [&] {
        return run_once(queue, pso_fma, kFmaGrid, kTG, {{0, fout}}, &fa, sizeof(fa), 1);
    });
    const auto fst = aspec::Stats::from(fsamples);
    // 8 FMA per inner iteration, 2 FLOP per FMA.
    const double flops = double(kFmaGrid) * double(fa.iters) * 8.0 * 2.0;
    const double peak_flops_per_s = flops / fst.median;

    std::printf("\n--- FP32 FMA throughput ---\n");
    std::printf("  measured           : %.1f GFLOP/s  (instab %.1f%%)\n",
                peak_flops_per_s / 1e9, fst.instability() * 100.0);

    // --------------------------------------------------------------- ridge point
    //
    // Streaming = median of the DRAM plateau. Resident = best clean sub-SLC point,
    // excluding anything the instability metric distrusts.
    std::vector<double> plateau, resident;
    for (std::size_t i = 0; i < footprints_mib.size(); ++i) {
        if (bw_instab[i] > 0.50) continue;                       // distrust noisy points
        if (bw_rdthr[i] < kMinReadsPerThread) continue;           // distrust artifacts
        if (footprints_mib[i] >= 64)     plateau.push_back(bw_result[i]);
        else if (footprints_mib[i] <= 8) resident.push_back(bw_result[i]);
    }
    const double bw_stream   = aspec::Stats::from(plateau).median;
    const double bw_resident = resident.empty() ? 0.0
                             : *std::max_element(resident.begin(), resident.end());

    // A Q4_0 GEMV with R output columns (R = gamma+1 under speculative verification)
    // loads each weight once and performs 2R FLOPs with it.
    const double ai_per_col   = 2.0 / kQ4_0_BytesPerWeight;        // ~3.56 FLOP/byte/col
    const double ridge_stream = peak_flops_per_s / bw_stream;
    const double cols_stream  = ridge_stream / ai_per_col;

    std::printf("\n--- measured roofline ---\n");
    std::printf("  streaming bandwidth: %.1f GB/s   (DRAM plateau, >=64 MiB)\n",
                bw_stream / 1e9);
    std::printf("  resident bandwidth : %.1f GB/s   (best clean footprint <=8 MiB)\n",
                bw_resident / 1e9);
    std::printf("  peak FP32 compute  : %.1f GFLOP/s\n", peak_flops_per_s / 1e9);
    std::printf("  Q4_0 GEMV AI       : %.2f FLOP/byte per output column\n", ai_per_col);
    std::printf("\n  STREAMING regime (weights from DRAM -- governs real decode):\n");
    std::printf("    ridge point      : %.1f FLOP/byte\n", ridge_stream);
    std::printf("    crossover        : %.1f columns  (gamma = %.1f)\n",
                cols_stream, cols_stream - 1.0);
    if (bw_resident > 0.0) {
        const double ridge_res = peak_flops_per_s / bw_resident;
        const double cols_res  = ridge_res / ai_per_col;
        std::printf("\n  RESIDENT regime (tensor fits in SLC -- what a naive microbenchmark\n");
        std::printf("  measures if it reuses one tensor):\n");
        std::printf("    ridge point      : %.1f FLOP/byte\n", ridge_res);
        std::printf("    crossover        : %.1f columns  (gamma = %.1f)\n",
                    cols_res, cols_res - 1.0);
        std::printf("    -> at %.0f%% of streaming ridge, a cache-resident Q4_0 GEMV is\n",
                    100.0 * ridge_res / ridge_stream);
        std::printf("       %s at ne11=1.\n",
                    ai_per_col >= ridge_res ? "ALREADY COMPUTE-BOUND" : "still memory-bound");
    }
    std::printf("\n  Shipped constants to compare against: llama.cpp Metal gates mul_mv_ext\n");
    std::printf("  at ne11 in [2,8] (Q4_0) / [4,8] (K-quants); CUDA uses\n");
    std::printf("  MMVQ_MAX_BATCH_SIZE = 8.\n");

    std::printf("\nthermal state (post) : %s\n",
                thermal_name([[NSProcessInfo processInfo] thermalState]));
 }
 return 0;
}
