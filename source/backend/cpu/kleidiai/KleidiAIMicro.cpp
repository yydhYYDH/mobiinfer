//
//  KleidiAIMicro.cpp
//  MNN
//
//  Created by MNN on 2026/06/25.
//

#include <MNN/KleidiAIMicro.h>

#ifdef MNN_KLEIDIAI_ENABLED

#include "backend/cpu/CPURuntime.hpp"
#include "backend/cpu/kleidiai/mnn_kleidiai.h"

#include <MNN/AutoTime.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace {

struct MNNKleidiAIQ4F16Handle {
    MNN::KleidiAI* kai = nullptr;
    MNN::KleidiAI::AccelType accelType = MNN::KleidiAI::AccelType::ACC_TYPE_ERROR;
    int n = 0;
    int k = 0;
    int blockSize = 0;
    std::vector<uint8_t> rhsPacked;
    std::vector<uint8_t> lhsPacked;
    double rhsPackMs = 0.0;
};

static double _elapsedMs(MNN::Timer& timer) {
    return static_cast<double>(timer.durationInUs()) / 1000.0;
}

static void _fillProfile(const MNNKleidiAIQ4F16Handle* handle, int m, int iterations, uint64_t lhsPackedBytes,
                         double lhsPackMs, double matmulMs, MNNKleidiAIQ4F16Profile* profile) {
    if (profile == nullptr) {
        return;
    }
    profile->m = m;
    profile->n = handle ? handle->n : 0;
    profile->k = handle ? handle->k : 0;
    profile->block_size = handle ? handle->blockSize : 0;
    profile->iterations = iterations;
    profile->rhs_packed_bytes = handle ? static_cast<uint64_t>(handle->rhsPacked.size()) : 0;
    profile->lhs_packed_bytes = lhsPackedBytes;
    profile->rhs_pack_ms = handle ? handle->rhsPackMs : 0.0;
    profile->lhs_pack_ms = lhsPackMs;
    profile->matmul_ms = matmulMs;
    profile->total_ms = lhsPackMs + matmulMs;
}

} // namespace

extern "C" void* MNNKleidiAIQ4F16Create(int n, int k, int block_size, const uint8_t* weight_int4,
                                        const float* scale, const float* zero, const float* bias,
                                        MNNKleidiAIQ4F16Profile* profile) {
    if (profile != nullptr) {
        ::memset(profile, 0, sizeof(MNNKleidiAIQ4F16Profile));
    }
    if (n <= 0 || k <= 0 || block_size <= 0 || weight_int4 == nullptr || scale == nullptr || zero == nullptr ||
        bias == nullptr) {
        return nullptr;
    }
    if ((k % block_size) != 0 || (block_size % 32) != 0) {
        return nullptr;
    }

    auto handle = new MNNKleidiAIQ4F16Handle;
    handle->kai = &MNN::KleidiAI::getInstance(*MNNGetCPUInfo());
    handle->accelType = MNN::KleidiAI::getQIntAccelType(4, true, block_size, 2);
    if (!handle->kai->canAccelerate(handle->accelType)) {
        delete handle;
        return nullptr;
    }
    if (!handle->kai->isLoaded(handle->accelType)) {
        handle->kai->setLoaded(handle->accelType);
        handle->kai->printInfo(handle->accelType);
    }
    handle->n = n;
    handle->k = k;
    handle->blockSize = block_size;

    auto rhsPackedSize = handle->kai->getRhsPackedSize(handle->accelType, n, k, block_size);
    handle->rhsPacked.resize(rhsPackedSize);
    MNN::Timer timer;
    handle->kai->runRhsPack(handle->accelType, 1, n, k, block_size, 0, weight_int4, scale, zero, bias,
                            handle->rhsPacked.data());
    handle->rhsPackMs = _elapsedMs(timer);

    _fillProfile(handle, 0, 0, 0, 0.0, 0.0, profile);
    return handle;
}

extern "C" int MNNKleidiAIQ4F16Run(void* rawHandle, const uint16_t* lhs_fp16, uint16_t* dst_fp16, int m,
                                   int iterations, MNNKleidiAIQ4F16Profile* profile) {
    auto handle = reinterpret_cast<MNNKleidiAIQ4F16Handle*>(rawHandle);
    if (handle == nullptr || handle->kai == nullptr || lhs_fp16 == nullptr || dst_fp16 == nullptr || m <= 0) {
        return -1;
    }
    if (iterations <= 0) {
        iterations = 1;
    }

    auto lhsPackedSize = handle->kai->getLhsQuantedPackedSize(handle->accelType, m, handle->k, handle->blockSize);
    handle->lhsPacked.resize(lhsPackedSize);

    double lhsPackMs = 0.0;
    double matmulMs = 0.0;
    auto scalarMin = -std::numeric_limits<float>::infinity();
    auto scalarMax = std::numeric_limits<float>::infinity();
    auto mr = handle->kai->getMr(handle->accelType, m);
    for (int i = 0; i < iterations; ++i) {
        MNN::Timer lhsTimer;
        handle->kai->runLhsQuantPack(handle->accelType, m, handle->k, handle->blockSize, mr, lhs_fp16,
                                     handle->lhsPacked.data());
        lhsPackMs += _elapsedMs(lhsTimer);

        MNN::Timer matmulTimer;
        handle->kai->runMatmul(handle->accelType, m, handle->n, handle->k, handle->blockSize,
                               handle->lhsPacked.data(), handle->rhsPacked.data(), dst_fp16,
                               static_cast<size_t>(handle->n) * sizeof(uint16_t), sizeof(uint16_t), scalarMax,
                               scalarMin);
        matmulMs += _elapsedMs(matmulTimer);
    }

    _fillProfile(handle, m, iterations, static_cast<uint64_t>(lhsPackedSize), lhsPackMs, matmulMs, profile);
    return 0;
}

extern "C" void MNNKleidiAIQ4F16Destroy(void* rawHandle) {
    auto handle = reinterpret_cast<MNNKleidiAIQ4F16Handle*>(rawHandle);
    delete handle;
}

#else

extern "C" void* MNNKleidiAIQ4F16Create(int, int, int, const uint8_t*, const float*, const float*, const float*,
                                        MNNKleidiAIQ4F16Profile* profile) {
    if (profile != nullptr) {
        ::memset(profile, 0, sizeof(MNNKleidiAIQ4F16Profile));
    }
    return nullptr;
}

extern "C" int MNNKleidiAIQ4F16Run(void*, const uint16_t*, uint16_t*, int, int,
                                   MNNKleidiAIQ4F16Profile* profile) {
    if (profile != nullptr) {
        ::memset(profile, 0, sizeof(MNNKleidiAIQ4F16Profile));
    }
    return -1;
}

extern "C" void MNNKleidiAIQ4F16Destroy(void*) {
}

#endif
