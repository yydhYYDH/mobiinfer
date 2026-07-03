//
//  MatMulSpeed.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/IDSTEncoder.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN::Express;

static void fillFloat(float* dst, int h, int w, float offset = 0.0f) {
    for (int y = 0; y < h; ++y) {
        auto dstY = dst + w * y;
        for (int x = 0; x < w; ++x) {
            dstY[x] = ((float)x * 0.1f + (float)y + offset) / 10000.0f;
        }
    }
}

static void _originMatMul(float* C, const float* A, const float* B, int e, int l, int h) {
    for (int y = 0; y < e; ++y) {
        auto AY = A + l * y;
        auto CY = C + h * y;
        for (int x = 0; x < h; ++x) {
            auto BX        = B + x;
            float expected = 0.0f;
            for (int k = 0; k < l; ++k) {
                expected += AY[k] * BX[k * h];
            }
            CY[x] = expected;
        }
    }
}

static std::vector<std::vector<int>> _getComputeSize() {
    std::vector<std::vector<int>> elh {
        {540, 540, 320},
        {1024, 1024, 1024},
        {5, 1024, 1024},
        {1024, 1024, 5},
        {1024, 5, 1024},
        {8, 1024, 1024},
        {10, 1024, 1024},
        {1024, 1024, 10},
        {1024, 10, 1024},
        {1, 1024, 1024},
        {1024, 1024, 1},
        {1024, 1, 1024},
        {128, 128, 3072},
        {128, 3072, 128},
    };
    return elh;
}
class BatchMatMulSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<std::vector<int>> elh = _getComputeSize();
        for (auto& iter : elh) {
            int e = iter[0];
            int h = iter[2];
            int l = iter[1];
            auto res = _run(e, h, l);
            if (!res) {
                return false;
            }
        }
        return true;
    }

    bool _run(int e, int h, int l) {
        {
            int batch = 20;
            // Test MatMul
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type                = MNN::OpType_MatMul;
            op->main.type           = MNN::OpParameter_MatMul;
            op->main.value          = new MNN::MatMulT;
            auto matmulParam        = op->main.AsMatMul();
            matmulParam->transposeA = false;
            matmulParam->transposeB = false;

            auto x0 = _Input({}, NHWC, halide_type_of<float>());
            auto x1 = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({batch, e, l});
            x1->resize({batch, l, h});
            auto y = Variable::create(Expr::create(op.get(), {x0, x1}));
            Variable::prepareCompute({y});
            ::memset(x0->writeMap<float>(), 0, x0->getInfo()->size * sizeof(float));
            ::memset(x1->writeMap<float>(), 0, x1->getInfo()->size * sizeof(float));
            y->readMap<float>();

            const auto time = 5;
            MNN::Timer _t;
            for (int t = 0; t < time; ++t) {
                x0->writeMap<float>();
                x1->writeMap<float>();
                y->readMap<float>();
            }
            float timeCost = _t.durationInUs() / 1000.0f / (float)time;
            float flops = (float)batch * (float)e * (float)l * (float)h / timeCost / 1000.0f / 1000.0f;
            MNN_PRINT("[%d, %d, %d], Avg time: %f ms , flops: %f G\n", e, l, h, timeCost, flops);
        }
        return true;
    }
};

class MatMulSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<std::vector<int>> elh = _getComputeSize();
        for (auto& iter : elh) {
            int e = iter[0];
            int h = iter[2];
            int l = iter[1];
            auto res = _run(e, h, l);
            if (!res) {
                return false;
            }
        }
        return true;
    }

    bool _run(int e, int h, int l) {
        {
            // Test MatMul
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type                = MNN::OpType_MatMul;
            op->main.type           = MNN::OpParameter_MatMul;
            op->main.value          = new MNN::MatMulT;
            auto matmulParam        = op->main.AsMatMul();
            matmulParam->transposeA = false;
            matmulParam->transposeB = false;

            auto x0 = _Input({}, NHWC, halide_type_of<float>());
            auto x1 = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({e, l});
            x1->resize({l, h});
            auto y = Variable::create(Expr::create(op.get(), {x0, x1}));
            Variable::prepareCompute({y});
            auto dstY = _Input({e, h}, NHWC, halide_type_of<float>());
            fillFloat(x0->writeMap<float>(), e, l);
            fillFloat(x1->writeMap<float>(), l, h);
            _originMatMul(dstY->writeMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h);

            auto absMaxV = _ReduceMax(_Abs(dstY));
            auto diffV   = _ReduceMax(_Abs(dstY - y));
            Variable::prepareCompute({absMaxV, diffV}, true);

            auto absMax = absMaxV->readMap<float>()[0];
            MNN_ASSERT(absMax != 0.0f);
            auto diff = diffV->readMap<float>()[0];

            bool res  = false;
            if (diff < 0.01f * absMax) {
                res = true;
            }
            if (!res) {
                MNN_PRINT("%f error larger than %f * 0.001f\n", diff, absMax);
                //                return false;
            }
            const auto time = 100;
            MNN::Timer _t;
            for (int t = 0; t < time; ++t) {
                x0->writeMap<float>();
                x1->writeMap<float>();
                y->readMap<float>();
            }
            float timeCost = _t.durationInUs() / 1000.0f / (float)time;
            float flops = (float)e * (float)l * (float)h / timeCost / 1000.0f / 1000.0f;
            MNN_PRINT("[%d, %d, %d], Avg time: %f ms , flops: %f G\n", e, l, h, timeCost, flops);
        }
        return true;
    }
};

class MatMulSpeedConstTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<std::vector<int>> elh = _getComputeSize();
        for (auto& iter : elh) {
            int e = iter[0];
            int h = iter[2];
            int l = iter[1];
            auto res = _runConst(e, h, l);
            if (!res) {
                return false;
            }
        }
        return true;
    }

    bool _runConst(int e, int h, int l) {
        {
            // Use Conv1x1 instead of MatMul
            auto x0 = _Input({1, l, e, 1}, NC4HW4, halide_type_of<float>());
            auto y = _Conv(0.0f, 0.0f, x0, {l, h}, {1, 1});
            Variable::prepareCompute({y});
            int time = 100;
            if (e < 100 || l < 100 || h < 100) {
                time = 1000;
            }
            MNN_PRINT("MatMul B Const (Conv1x1): [%d, %d, %d], run %d\n", e, l, h, time);
            {
                AUTOTIME;
                //Prepare
                x0->writeMap<float>();
                y->readMap<float>();
            }
            MNN::Timer _t;
            for (int t = 0; t < time; ++t) {
                x0->writeMap<float>();
                y->readMap<float>();
            }
            float timeCost = _t.durationInUs() / 1000.0f / (float)time;
            float flops = (float)e * (float)l * (float)h / timeCost / 1000.0f / 1000.0f;
            MNN_PRINT("[%d, %d, %d], Avg time: %f ms , flops: %f G\n", e, l, h, timeCost, flops);
        }
        return true;
    }
};
class Conv1x1E1E2Row0Test : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int ic = 2048;
        const int oc = 2048;
        std::vector<float> input0(ic);
        std::vector<float> input1(ic);
        std::vector<float> weight(ic * oc);
        std::vector<float> bias(oc);
        for (int i = 0; i < ic; ++i) {
            input0[i] = std::sin((float)i * 0.017f) * 0.13f + std::cos((float)i * 0.031f) * 0.07f;
            input1[i] = std::sin((float)i * 0.023f + 0.5f) * 0.11f;
        }
        for (int i = 0; i < ic * oc; ++i) {
            weight[i] = std::sin((float)i * 0.0017f) * 0.015f + std::cos((float)i * 0.0009f) * 0.01f;
        }
        for (int i = 0; i < oc; ++i) {
            bias[i] = std::sin((float)i * 0.019f) * 0.003f;
        }
        loadRealQProjInputs(input0, input1, ic);

        bool ok = true;
        ok = runFloatConv("FP32/fp-precision", input0, input1, weight, bias, ic, oc, precision) && ok;
        ok = runHybridConv("HybridInt8Weight-nbits8", input0, input1, weight, bias, ic, oc, 8) && ok;
        ok = runHybridConv("HybridInt8Weight-nbits4", input0, input1, weight, bias, ic, oc, 4) && ok;
        ok = runBlockW4A8Conv("QProjLike-w4g32-a8", input0, input1, weight, bias, ic, oc, 32) && ok;
        ok = runRealQProjConv("RealQProj-layer1-w4g32-a8", input0, input1, ic, oc) && ok;
        return ok;
    }

private:
    struct RealQProjWeight {
        bool valid = false;
        std::vector<int8_t> buffer;
        std::vector<float> alpha;
        std::vector<float> bias;
    };

    static bool compareRow0(const char* title, VARP y1, VARP y2, int oc, float threshold) {
        const float* y1Ptr = y1->readMap<float>();
        const float* y2Ptr = y2->readMap<float>();
        float maxAbsDiff = 0.0f;
        float maxRelDiff = 0.0f;
        int maxIndex = 0;
        double sumDiff = 0.0;
        for (int i = 0; i < oc; ++i) {
            float diff = std::fabs(y1Ptr[i] - y2Ptr[i]);
            float denom = std::max(std::fabs(y1Ptr[i]), 1.0e-8f);
            float rel = diff / denom;
            sumDiff += diff;
            if (diff > maxAbsDiff) {
                maxAbsDiff = diff;
                maxRelDiff = rel;
                maxIndex = i;
            }
        }
        MNN_PRINT("%s Conv1x1 E1/E2 row0: maxAbsDiff=%g maxRelDiff=%g meanAbsDiff=%g index=%d y1=%g y2=%g\n",
                  title, maxAbsDiff, maxRelDiff, sumDiff / oc, maxIndex, y1Ptr[maxIndex], y2Ptr[maxIndex]);
        MNN_PRINT("%s first6 E1: %g,%g,%g,%g,%g,%g\n", title, y1Ptr[0], y1Ptr[1], y1Ptr[2], y1Ptr[3],
                  y1Ptr[4], y1Ptr[5]);
        MNN_PRINT("%s first6 E2 row0: %g,%g,%g,%g,%g,%g\n", title, y2Ptr[0], y2Ptr[1], y2Ptr[2], y2Ptr[3],
                  y2Ptr[4], y2Ptr[5]);
        if (maxAbsDiff > threshold) {
            MNN_PRINT("%s maxAbsDiff=%g exceeds threshold=%g\n", title, maxAbsDiff, threshold);
            return false;
        }
        return true;
    }

    static void compareRemoteQProjDump(const char* title, VARP y1, VARP y2, int oc) {
        const char* dirEnv = std::getenv("MNN_QPROJ_REAL_INPUT_DIR");
        if (dirEnv == nullptr || dirEnv[0] == '\0') {
            return;
        }
        std::string dir(dirEnv);
        std::string seq1Path =
            dir + "/layers.1_self_attn_q_proj_Linear/seq1_ord57__layers_1_self_attn_q_proj_Linear_full.txt";
        std::string seq2Path =
            dir + "/layers.1_self_attn_q_proj_Linear/seq2_ord57__layers_1_self_attn_q_proj_Linear_full.txt";
        std::vector<float> seq1;
        std::vector<float> seq2;
        if (!readDumpValues(seq1Path, seq1) || !readDumpValues(seq2Path, seq2)) {
            return;
        }
        if (seq1.size() < static_cast<size_t>(oc) || seq2.size() < static_cast<size_t>(oc)) {
            return;
        }

        auto report = [&](const char* tag, const float* local, const std::vector<float>& remote, int remoteOffset) {
            float maxAbsDiff = 0.0f;
            double sumDiff = 0.0;
            int maxIndex = 0;
            for (int i = 0; i < oc; ++i) {
                float diff = std::fabs(local[i] - remote[remoteOffset + i]);
                sumDiff += diff;
                if (diff > maxAbsDiff) {
                    maxAbsDiff = diff;
                    maxIndex = i;
                }
            }
            MNN_PRINT("%s local-vs-remote %s: maxAbsDiff=%g meanAbsDiff=%g index=%d local=%g remote=%g\n",
                      title, tag, maxAbsDiff, sumDiff / oc, maxIndex, local[maxIndex],
                      remote[remoteOffset + maxIndex]);
        };
        const float* y1Ptr = y1->readMap<float>();
        const float* y2Ptr = y2->readMap<float>();
        report("seq1", y1Ptr, seq1, 0);
        report("seq2row0", y2Ptr, seq2, 0);
    }

    static void fillInputs(VARP x1, VARP x2, const std::vector<float>& input0, const std::vector<float>& input1) {
        auto x1Ptr = x1->writeMap<float>();
        std::copy(input0.begin(), input0.end(), x1Ptr);
        x1->unMap();
        auto x2Ptr = x2->writeMap<float>();
        std::copy(input0.begin(), input0.end(), x2Ptr);
        std::copy(input1.begin(), input1.end(), x2Ptr + input0.size());
        x2->unMap();
    }

    static bool readDumpValues(const std::string& path, std::vector<float>& values) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            return false;
        }
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            values.push_back(std::stof(line));
        }
        return true;
    }

    template <typename T>
    static bool readBinaryValues(const std::string& path, std::vector<T>& values) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            return false;
        }
        ifs.seekg(0, std::ios::end);
        auto bytes = ifs.tellg();
        if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
            return false;
        }
        values.resize(static_cast<size_t>(bytes) / sizeof(T));
        ifs.seekg(0, std::ios::beg);
        if (!values.empty()) {
            ifs.read(reinterpret_cast<char*>(values.data()), bytes);
        }
        return true;
    }

    static void loadRealQProjInputs(std::vector<float>& input0, std::vector<float>& input1, int ic) {
        const char* dirEnv = std::getenv("MNN_QPROJ_REAL_INPUT_DIR");
        if (dirEnv == nullptr || dirEnv[0] == '\0') {
            return;
        }
        std::string dir(dirEnv);
        std::string seq1Path =
            dir + "/blocks.1_input_layernorm/seq1_ord55__blocks_1_input_layernorm_Mul_1_output_0_full.txt";
        std::string seq2Path =
            dir + "/blocks.1_input_layernorm/seq2_ord55__blocks_1_input_layernorm_Mul_1_output_0_full.txt";
        std::vector<float> seq1;
        std::vector<float> seq2;
        if (!readDumpValues(seq1Path, seq1) || !readDumpValues(seq2Path, seq2)) {
            MNN_PRINT("MNN_QPROJ_REAL_INPUT_DIR is set but q_proj input dump files cannot be read: %s\n",
                      dir.c_str());
            return;
        }
        if (seq1.size() < static_cast<size_t>(ic) || seq2.size() < static_cast<size_t>(2 * ic)) {
            MNN_PRINT("MNN_QPROJ_REAL_INPUT_DIR dump size mismatch: seq1=%d seq2=%d expected=%d/%d\n",
                      static_cast<int>(seq1.size()), static_cast<int>(seq2.size()), ic, 2 * ic);
            return;
        }
        std::copy(seq1.begin(), seq1.begin() + ic, input0.begin());
        std::copy(seq2.begin() + ic, seq2.begin() + 2 * ic, input1.begin());
        double row1Abs = 0.0;
        for (int i = 0; i < ic; ++i) {
            row1Abs += std::fabs(input1[i]);
        }
        MNN_PRINT("Loaded real q_proj logical inputs from %s: row0_first6=%g,%g,%g,%g,%g,%g row1_abs=%g\n",
                  dir.c_str(), input0[0], input0[1], input0[2], input0[3], input0[4], input0[5], row1Abs);
    }

    static RealQProjWeight loadRealQProjWeight() {
        RealQProjWeight result;
        const char* dirEnv = std::getenv("MNN_QPROJ_REAL_WEIGHT_DIR");
        if (dirEnv == nullptr || dirEnv[0] == '\0') {
            return result;
        }
        std::string dir(dirEnv);
        std::string prefix = dir + "/layer1_q_proj";
        if (!readBinaryValues(prefix + "_buffer.bin", result.buffer) ||
            !readBinaryValues(prefix + "_alpha.bin", result.alpha)) {
            MNN_PRINT("MNN_QPROJ_REAL_WEIGHT_DIR is set but real q_proj weight files cannot be read: %s\n",
                      dir.c_str());
            return result;
        }
        readBinaryValues(prefix + "_bias.bin", result.bias);
        if (result.buffer.empty() || result.alpha.empty()) {
            MNN_PRINT("MNN_QPROJ_REAL_WEIGHT_DIR has empty real q_proj weight data: %s\n", dir.c_str());
            return result;
        }
        if (result.bias.empty()) {
            result.bias.resize(2048, 0.0f);
        }
        result.valid = true;
        MNN_PRINT("Loaded real layer1 q_proj weight from %s: buffer=%d alpha=%d bias=%d alpha_first=%g,%g\n",
                  dir.c_str(), static_cast<int>(result.buffer.size()), static_cast<int>(result.alpha.size()),
                  static_cast<int>(result.bias.size()), result.alpha[0], result.alpha.size() > 1 ? result.alpha[1] : 0.0f);
        return result;
    }

    static bool runFloatConv(const char* title, const std::vector<float>& input0, const std::vector<float>& input1,
                             const std::vector<float>& weight, const std::vector<float>& bias, int ic, int oc,
                             int precision) {
        auto x1 = _Input({1, ic, 1, 1}, NCHW, halide_type_of<float>());
        auto x2 = _Input({2, ic, 1, 1}, NCHW, halide_type_of<float>());
        auto y1 = _Convert(_Conv(std::vector<float>(weight), std::vector<float>(bias), _Convert(x1, NC4HW4),
                                 {ic, oc}, {1, 1}, VALID, {1, 1}, {1, 1}, 1, {0, 0, 0, 0}),
                           NCHW);
        auto y2 = _Convert(_Conv(std::vector<float>(weight), std::vector<float>(bias), _Convert(x2, NC4HW4),
                                 {ic, oc}, {1, 1}, VALID, {1, 1}, {1, 1}, 1, {0, 0, 0, 0}),
                           NCHW);
        Variable::prepareCompute({y1, y2});
        fillInputs(x1, x2, input0, input1);
        float threshold = precision >= MNN::BackendConfig::Precision_Low ? 2.0e-3f : 1.0e-4f;
        return compareRow0(title, y1, y2, oc, threshold);
    }

    static bool runHybridConv(const char* title, const std::vector<float>& input0, const std::vector<float>& input1,
                              const std::vector<float>& weight, const std::vector<float>& bias, int ic, int oc,
                              int nbits) {
        auto x1 = _Input({1, ic, 1, 1}, NCHW, halide_type_of<float>());
        auto x2 = _Input({2, ic, 1, 1}, NCHW, halide_type_of<float>());
        auto y1 = _Convert(_HybridInt8Conv(std::vector<float>(weight), std::vector<float>(bias), _Convert(x1, NC4HW4),
                                           {ic, oc}, {1, 1}, VALID, {1, 1}, {1, 1}, 1, {0, 0, 0, 0}, false, false,
                                           nbits, 0),
                           NCHW);
        auto y2 = _Convert(_HybridInt8Conv(std::vector<float>(weight), std::vector<float>(bias), _Convert(x2, NC4HW4),
                                           {ic, oc}, {1, 1}, VALID, {1, 1}, {1, 1}, 1, {0, 0, 0, 0}, false, false,
                                           nbits, 0),
                           NCHW);
        Variable::prepareCompute({y1, y2});
        fillInputs(x1, x2, input0, input1);
        return compareRow0(title, y1, y2, oc, 2.0e-3f);
    }

    static VARP blockQuantConv1x1(const std::vector<float>& weight, const std::vector<float>& bias, VARP x, int ic,
                                  int oc, int nbits, int blockSize) {
        MNN_ASSERT(ic % blockSize == 0);
        const int blockNum = ic / blockSize;
        const int qmax = (1 << (nbits - 1)) - 1;
        const int qmin = -(1 << (nbits - 1));
        std::vector<float> alpha(2 * oc * blockNum, 0.0f);
        for (int o = 0; o < oc; ++o) {
            for (int b = 0; b < blockNum; ++b) {
                float minV = std::numeric_limits<float>::max();
                float maxV = -std::numeric_limits<float>::max();
                const float* block = weight.data() + o * ic + b * blockSize;
                for (int i = 0; i < blockSize; ++i) {
                    minV = std::min(minV, block[i]);
                    maxV = std::max(maxV, block[i]);
                }
                int index = o * blockNum + b;
                alpha[2 * index] = minV;
                alpha[2 * index + 1] = std::max((maxV - minV) / static_cast<float>(qmax - qmin), 1.0e-8f);
            }
        }

        std::unique_ptr<OpT> convOp(new OpT);
        convOp->type = OpType_Convolution;
        convOp->main.type = OpParameter_Convolution2D;
        convOp->main.value = new Convolution2DT;
        auto conv2D = convOp->main.AsConvolution2D();
        conv2D->common.reset(new Convolution2DCommonT);
        conv2D->common->padMode = PadMode_CAFFE;
        conv2D->common->padX = 0;
        conv2D->common->padY = 0;
        conv2D->common->kernelX = 1;
        conv2D->common->kernelY = 1;
        conv2D->common->strideX = 1;
        conv2D->common->strideY = 1;
        conv2D->common->dilateX = 1;
        conv2D->common->dilateY = 1;
        conv2D->common->group = 1;
        conv2D->common->outputCount = oc;
        conv2D->common->inputCount = ic;
        conv2D->common->relu = false;
        conv2D->common->relu6 = false;
        conv2D->quanParameter = IDSTEncoder::encode(weight.data(), alpha, blockSize, oc * blockNum,
                                                    /*asymmetricQuantFlag=*/true, /*quantWeightPtr=*/nullptr,
                                                    /*clampMin=*/qmin, {nbits, false});
        conv2D->weight.clear();
        conv2D->bias = bias;
        return Variable::create(Expr::create(convOp.get(), {x}));
    }

    static VARP realQProjConv1x1(const RealQProjWeight& weight, VARP x, int ic, int oc) {
        std::unique_ptr<OpT> convOp(new OpT);
        convOp->type = OpType_Convolution;
        convOp->main.type = OpParameter_Convolution2D;
        convOp->main.value = new Convolution2DT;
        auto conv2D = convOp->main.AsConvolution2D();
        conv2D->common.reset(new Convolution2DCommonT);
        conv2D->common->padMode = PadMode_CAFFE;
        conv2D->common->padX = 0;
        conv2D->common->padY = 0;
        conv2D->common->kernelX = 1;
        conv2D->common->kernelY = 1;
        conv2D->common->strideX = 1;
        conv2D->common->strideY = 1;
        conv2D->common->dilateX = 1;
        conv2D->common->dilateY = 1;
        conv2D->common->group = 1;
        conv2D->common->outputCount = oc;
        conv2D->common->inputCount = ic;
        conv2D->common->relu = false;
        conv2D->common->relu6 = false;
        conv2D->quanParameter.reset(new IDSTQuanT);
        conv2D->quanParameter->buffer = weight.buffer;
        conv2D->quanParameter->alpha = weight.alpha;
        conv2D->quanParameter->type = 1;
        conv2D->quanParameter->quantScale = 1.0f;
        conv2D->quanParameter->scaleIn = 0.0f;
        conv2D->quanParameter->scaleOut = 0.0f;
        conv2D->quanParameter->aMaxOrBits = 4;
        conv2D->quanParameter->aMin = 1;
        conv2D->quanParameter->readType = 131072;
        conv2D->quanParameter->has_scaleInt = false;
        conv2D->quanParameter->shapeInt32 = false;
        conv2D->quanParameter->scaleStorage = ScaleStorageType_FP32;
        conv2D->bias = weight.bias;
        return Variable::create(Expr::create(convOp.get(), {x}));
    }

    static VARP qProjLikeGraph(VARP x, const std::vector<float>& weight, const std::vector<float>& bias, int ic, int oc,
                               int seq, int blockSize) {
        // Match exported q_proj:
        // [seq, hidden] -> [seq, hidden, 1, 1] -> NC4HW4 1x1 Convolution -> NCHW -> [1, seq, hidden].
        auto convIn = _Convert(_Reshape(x, {seq, ic, 1, 1}, NCHW), NC4HW4);
        auto convOut = blockQuantConv1x1(weight, bias, convIn, ic, oc, 4, blockSize);
        return _Reshape(_Convert(convOut, NCHW), {1, seq, oc}, NCHW);
    }

    static bool runBlockW4A8Conv(const char* title, const std::vector<float>& input0,
                                 const std::vector<float>& input1, const std::vector<float>& weight,
                                 const std::vector<float>& bias, int ic, int oc, int blockSize) {
        auto x1 = _Input({1, ic}, NCHW, halide_type_of<float>());
        auto x2 = _Input({2, ic}, NCHW, halide_type_of<float>());
        auto y1 = qProjLikeGraph(x1, weight, bias, ic, oc, 1, blockSize);
        auto y2 = qProjLikeGraph(x2, weight, bias, ic, oc, 2, blockSize);
        Variable::prepareCompute({y1, y2});
        fillInputs(x1, x2, input0, input1);
        return compareRow0(title, y1, y2, oc, 2.0e-3f);
    }

    static bool runRealQProjConv(const char* title, const std::vector<float>& input0,
                                 const std::vector<float>& input1, int ic, int oc) {
        auto realWeight = loadRealQProjWeight();
        if (!realWeight.valid) {
            return true;
        }
        auto x1 = _Input({1, ic}, NCHW, halide_type_of<float>());
        auto x2 = _Input({2, ic}, NCHW, halide_type_of<float>());
        auto y1 = _Reshape(_Convert(realQProjConv1x1(realWeight, _Convert(_Reshape(x1, {1, ic, 1, 1}, NCHW), NC4HW4),
                                                     ic, oc),
                                    NCHW),
                           {1, 1, oc}, NCHW);
        auto y2 = _Reshape(_Convert(realQProjConv1x1(realWeight, _Convert(_Reshape(x2, {2, ic, 1, 1}, NCHW), NC4HW4),
                                                     ic, oc),
                                    NCHW),
                           {1, 2, oc}, NCHW);
        Variable::prepareCompute({y1, y2});
        fillInputs(x1, x2, input0, input1);
        bool ok = compareRow0(title, y1, y2, oc, 2.0e-3f);
        compareRemoteQProjDump(title, y1, y2, oc);
        return ok;
    }
};
MNNTestSuiteRegister(MatMulSpeedTest, "speed/MatMulTest");
MNNTestSuiteRegister(BatchMatMulSpeedTest, "speed/MatMulBatchTest");
MNNTestSuiteRegister(MatMulSpeedConstTest, "speed/MatMulBConstTest");
MNNTestSuiteRegister(Conv1x1E1E2Row0Test, "speed/Conv1x1E1E2Row0");
