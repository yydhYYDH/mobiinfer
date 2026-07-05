//
//  GemvW4A8W8A8BWTest.cpp
//  MNNTests
//
//  Decode-shaped HybridConv bandwidth benchmark for W8A8 and W4A8.
//

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include "CommonOpCreator.hpp"
#include "MNNTestSuite.h"

using namespace MNN;
using namespace MNN::Express;

namespace {

using Clock = std::chrono::high_resolution_clock;

static double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static double measureMemcpyPeakGBs(size_t bytes, int threads, int repeats) {
    std::vector<uint8_t> src(bytes), dst(bytes);
    std::memset(src.data(), 0xa5, bytes);
    std::memset(dst.data(), 0x00, bytes);
    std::memcpy(dst.data(), src.data(), bytes);

    double best = 0.0;
    for (int r = 0; r < repeats; ++r) {
        auto start = Clock::now();
        std::vector<std::thread> workers;
        size_t chunk = bytes / threads;
        for (int t = 0; t < threads; ++t) {
            size_t off = (size_t)t * chunk;
            size_t len = (t == threads - 1) ? (bytes - off) : chunk;
            workers.emplace_back([&, off, len]() { std::memcpy(dst.data() + off, src.data() + off, len); });
        }
        for (auto& th : workers) {
            th.join();
        }
        double dt = secondsSince(start);
        best = std::max(best, (2.0 * bytes) / dt / 1.0e9);
    }
    if (dst[0] == 0x12) {
        std::printf("?");
    }
    return best;
}

struct Shape {
    int oc;
    int ic;
    const char* name;
};

struct Result {
    int bits;
    int oc;
    int ic;
    int threads;
    double avgUs;
    double weightBytes;
    double effGBs;
    double gops;
};

static Result benchOne(const Shape& shape, int bits, int blockSize, int precision, int threads, int iters,
                       MNNForwardType forwardType) {
    BackendConfig config;
    config.precision = (BackendConfig::PrecisionMode)precision;
    config.memory = BackendConfig::Memory_Low;
    auto executor = Executor::newExecutor(forwardType, config, threads);
    ExecutorScope scope(executor);

    int oc = shape.oc;
    int ic = shape.ic;
    int bs = blockSize;
    if (bs <= 0 || ic % bs != 0) {
        bs = ic;
    }
    int blockNum = ic / bs;

    std::vector<float> weight(oc * ic);
    std::vector<float> scale(2 * oc * blockNum);
    std::vector<float> bias(oc, 0.0f);
    for (int o = 0; o < oc; ++o) {
        for (int i = 0; i < ic; ++i) {
            weight[o * ic + i] = ((float)((o * 131 + i * 17) % 255) - 127.0f) * 0.003f;
        }
        for (int b = 0; b < blockNum; ++b) {
            scale[2 * (o * blockNum + b) + 0] = -0.5f;
            scale[2 * (o * blockNum + b) + 1] = 0.01f;
        }
    }

    auto x = _Input({1, ic, 1, 1}, NCHW, halide_type_of<float>());
    auto xPtr = x->writeMap<float>();
    for (int i = 0; i < ic; ++i) {
        xPtr[i] = (float)((i % 31) - 15) * 0.125f;
    }
    x = _Convert(x, NC4HW4);
    x->writeScaleMap(1.0f, 0.0f);

    auto y = _HybridConv(weight, std::move(bias), scale, x, {ic, oc}, {1, 1}, PaddingMode::CAFFE, {1, 1}, {1, 1}, 1,
                         {0, 0}, false, false, bits, true);
    x.fix(VARP::INPUT);

    x->writeMap<float>();
    y->readMap<float>();

    std::vector<uint8_t> flushBuf(64 * 1024 * 1024, 1);
    auto flushCache = [&]() {
        volatile uint64_t sink = 0;
        for (size_t i = 0; i < flushBuf.size(); i += 64) {
            sink += flushBuf[i];
        }
        (void)sink;
    };

    double bestUs = 1.0e100;
    const int outer = 3;
    for (int r = 0; r < outer; ++r) {
        double total = 0.0;
        for (int i = 0; i < iters; ++i) {
            flushCache();
            auto start = Clock::now();
            x->writeMap<float>();
            y->readMap<float>();
            total += secondsSince(start);
        }
        bestUs = std::min(bestUs, total * 1.0e6 / iters);
    }

    double pureWeight = (double)oc * ic * bits / 8.0;
    double scaleZp = (double)oc * blockNum * 2.0 * 2.0;
    double weightBytes = pureWeight + scaleZp;
    double sec = bestUs / 1.0e6;

    Result res;
    res.bits = bits;
    res.oc = oc;
    res.ic = ic;
    res.threads = threads;
    res.avgUs = bestUs;
    res.weightBytes = weightBytes;
    res.effGBs = weightBytes / sec / 1.0e9;
    res.gops = (2.0 * oc * ic) / sec / 1.0e9;
    return res;
}

} // namespace

class GemvW4A8W8A8BWTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto& status = MNNTestSuite::get()->pStaus;
        int threads = 4;
        auto forwardType = (MNNForwardType)status.forwardType;
        if (forwardType != MNN_FORWARD_CPU) {
            std::printf("This benchmark is CPU-only. Please run with forward=0.\n");
            return false;
        }

        std::vector<Shape> shapes = {
            {1024, 1024, "small"},
            {2048, 2048, "2k_square"},
            {4096, 2560, "attn_o/up"},
            {2560, 4096, "attn_qkv/down"},
            {9728, 2560, "ffn_up"},
            {2560, 9728, "ffn_down"},
            {4096, 14336, "llama_ffn"},
        };

        const int iters = 100;
        double peak = measureMemcpyPeakGBs((size_t)256 << 20, threads, 5);

        std::printf("\n## W8A8 / W4A8 decode GEMV bandwidth\n");
        std::printf("MNN op path: _HybridConv -> Convolution low-memory weight-only quant\n");
        std::printf("Execution: FP activation -> dynamic INT8 activation quant, W8/W4 packed weight, INT8 dot, INT32 accumulate, dequant output\n");
        std::printf("CPU only, backend=%d precision=%d memory=low threads=%d\n", status.forwardType, precision, threads);
        std::printf("memcpy peak: %.1f GB/s @ %d threads\n\n", peak, threads);
        std::printf("shape      | type | thr |    OC |    IC |   us/iter |  W MiB | bytes/elem | eff GB/s | %%peak |   GOPS | op name\n");
        std::printf("-----------|------|----:|------:|------:|----------:|-------:|-----------:|---------:|------:|-------:|-------------------------------\n");

        for (const auto& shape : shapes) {
            for (int bits : {8, 4}) {
                int blockSize = bits == 4 ? 64 : shape.ic;
                Result r = benchOne(shape, bits, blockSize, precision, threads, iters, forwardType);
                double bpe = r.weightBytes / ((double)r.oc * r.ic);
                double pct = 100.0 * r.effGBs / peak;
                const char* opName = bits == 8 ? "HybridConv W8A8 dynamic-quant GEMV"
                                               : "HybridConv W4A8 dynamic-quant GEMV";
                std::printf("%-10s | w%-4d | %3d | %5d | %5d | %9.1f | %6.1f | %10.4f | %8.1f | %5.1f | %6.1f | %s\n",
                            shape.name, bits, r.threads, r.oc, r.ic, r.avgUs, r.weightBytes / (1024.0 * 1024.0),
                            bpe, r.effGBs, pct, r.gops, opName);
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(GemvW4A8W8A8BWTest, "speed/GemvW4A8W8A8BW");
