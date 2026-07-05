//
//  GemvW8A8Speed.cpp
//  MNNTests
//
//  Decode-shaped w8a8 GEMV speed test for ARM CPU.
//

#include <chrono>
#include <cstdio>
#include <cstring>
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

static double elapsedUs(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

struct Shape {
    int m;
    int k;
    int n;
    const char* name;
};

static VARP buildW8A8Conv1x1(VARP x, int ic, int oc) {
    const int nbit = 8;
    const int blockSize = ic;
    const int blockNum = 1;
    std::vector<float> weight(oc * ic);
    std::vector<float> scale(2 * oc * blockNum);
    std::vector<float> bias(oc, 0.0f);

    for (int o = 0; o < oc; ++o) {
        for (int i = 0; i < ic; ++i) {
            weight[o * ic + i] = ((float)((o * 131 + i * 17) % 255) - 127.0f) * 0.003f;
        }
        scale[2 * o + 0] = -0.5f;
        scale[2 * o + 1] = 0.01f;
    }
    return _HybridConv(weight, std::move(bias), scale, x, {ic, oc}, {1, 1}, PaddingMode::CAFFE, {1, 1}, {1, 1}, 1,
                       {0, 0}, false, false, nbit, true);
}

static void runShape(const Shape& shape, int precision, int threads, MNNForwardType forwardType) {
    BackendConfig config;
    config.precision = (BackendConfig::PrecisionMode)precision;
    config.memory = BackendConfig::Memory_Low;
    auto executor = Executor::newExecutor(forwardType, config, threads);
    ExecutorScope scope(executor);

    auto x = _Input({1, shape.k, 1, shape.m}, NCHW, halide_type_of<float>());
    auto xPtr = x->writeMap<float>();
    for (int i = 0; i < shape.k * shape.m; ++i) {
        xPtr[i] = (float)((i % 31) - 15) * 0.125f;
    }
    x = _Convert(x, NC4HW4);
    x->writeScaleMap(1.0f, 0.0f);

    auto y = buildW8A8Conv1x1(x, shape.k, shape.n);
    x.fix(VARP::INPUT);

    for (int i = 0; i < 5; ++i) {
        x->writeMap<float>();
        y->readMap<float>();
    }

    int loops = 100;
    if ((int64_t)shape.m * shape.k * shape.n > 80LL * 1000 * 1000) {
        loops = 30;
    }
    if ((int64_t)shape.m * shape.k * shape.n > 200LL * 1000 * 1000) {
        loops = 10;
    }

    double bestUs = 1e100;
    double sumUs = 0.0;
    const int repeats = 5;
    for (int r = 0; r < repeats; ++r) {
        auto start = Clock::now();
        for (int i = 0; i < loops; ++i) {
            x->writeMap<float>();
            y->readMap<float>();
        }
        double avgUs = elapsedUs(start) / loops;
        bestUs = std::min(bestUs, avgUs);
        sumUs += avgUs;
    }

    double avgUs = sumUs / repeats;
    double ops = 2.0 * (double)shape.m * (double)shape.k * (double)shape.n;
    double bestGops = ops / bestUs / 1000.0;
    double weightMiB = (double)shape.k * (double)shape.n / (1024.0 * 1024.0);
    double weightGBs = ((double)shape.k * (double)shape.n) / (bestUs * 1.0e-6) / 1.0e9;

    std::printf("%-18s | %3d | %5d | %5d | %5d | %8.3f | %8.3f | %8.2f | %8.2f | %8.1f\n",
                shape.name, threads, shape.m, shape.k, shape.n, bestUs, avgUs, bestGops, weightMiB, weightGBs);
}

} // namespace

class GemvW8A8Speed : public MNNTestCase {
public:
    bool run(int precision) override {
        auto& status = MNNTestSuite::get()->pStaus;
        int threads = 4;
        auto forwardType = (MNNForwardType)status.forwardType;

        std::vector<Shape> shapes = {
            {1, 1024, 1024, "small"},
            {1, 2048, 2048, "2k_square"},
            {1, 2560, 4096, "attn_o/up"},
            {1, 4096, 2560, "attn_qkv/down"},
            {1, 2560, 9728, "ffn_up"},
            {1, 9728, 2560, "ffn_down"},
            {4, 2560, 4096, "m4_up"},
            {8, 2560, 4096, "m8_up"},
        };

        std::printf("\n## W8A8 decode-shaped HybridConv GEMV/GEMM\n");
        std::printf("backend=%d precision=%d memory=low threads=%d\n", status.forwardType, precision, threads);
        std::printf("name               | thr |     M |     K |     N |  best us |   avg us |     GOPS |    W MiB |  W GB/s\n");
        std::printf("-------------------|----:|------:|------:|------:|---------:|---------:|---------:|---------:|--------:\n");
        for (const auto& shape : shapes) {
            runShape(shape, precision, threads, forwardType);
        }
        return true;
    }
};

MNNTestSuiteRegister(GemvW8A8Speed, "speed/GemvW8A8Sizes");
