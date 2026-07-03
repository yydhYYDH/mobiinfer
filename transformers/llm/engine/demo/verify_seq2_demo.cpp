//
//  verify_seq2_demo.cpp
//
//  Minimal target-only check for speculative verify equivalence:
//  AR decode with seqLen=1 should match verify decode logits[0] with seqLen=2.
//

#include <MNN/expr/ExecutorScope.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../src/kvmeta.hpp"
#include "../src/llmconfig.hpp"

#define private public
#define protected public
#include "llm/llm.hpp"
#undef protected
#undef private

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Transformer;

static std::string readFile(const std::string& path) {
    std::ifstream fs(path);
    std::ostringstream os;
    os << fs.rdbuf();
    return os.str();
}

static VARP makeVerifyMask(int pastLen, int seqLen) {
    auto mask = _Input({1, 1, seqLen, pastLen + seqLen}, NCHW, halide_type_of<float>());
    auto ptr = mask->writeMap<float>();
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < pastLen + seqLen; ++j) {
            ptr[i * (pastLen + seqLen) + j] =
                (j <= pastLen + i) ? 0.0f : std::numeric_limits<float>::lowest();
        }
    }
    return mask;
}

static VARP makeVerifyPos(Llm* llm, int pastLen, int seqLen) {
    if (llm->mConfig->is_mrope()) {
        auto pos = _Input({3, seqLen}, NCHW, halide_type_of<int>());
        auto ptr = pos->writeMap<int>();
        for (int i = 0; i < seqLen; ++i) {
            int v = pastLen + i;
            ptr[i] = v;
            ptr[seqLen + i] = v;
            ptr[2 * seqLen + i] = v;
        }
        return pos;
    }
    auto pos = _Input({1, seqLen}, NCHW, halide_type_of<int>());
    auto ptr = pos->writeMap<int>();
    for (int i = 0; i < seqLen; ++i) {
        ptr[i] = pastLen + i;
    }
    return pos;
}

static int argmaxAt(VARP logits, int row) {
    auto info = logits->getInfo();
    if (info == nullptr || info->dim.empty()) {
        return -1;
    }
    int vocab = info->dim.back();
    auto ptr = logits->readMap<float>() + row * vocab;
    return static_cast<int>(std::max_element(ptr, ptr + vocab) - ptr);
}

static std::vector<int> topKAt(VARP logits, int row, int k) {
    auto info = logits->getInfo();
    int vocab = info->dim.back();
    auto ptr = logits->readMap<float>() + row * vocab;
    std::vector<int> ids(vocab);
    for (int i = 0; i < vocab; ++i) {
        ids[i] = i;
    }
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) {
        return ptr[a] > ptr[b];
    });
    ids.resize(k);
    return ids;
}

static float maxAbsDiffAt(VARP a, int rowA, VARP b, int rowB) {
    auto info = a->getInfo();
    int vocab = info->dim.back();
    auto pa = a->readMap<float>() + rowA * vocab;
    auto pb = b->readMap<float>() + rowB * vocab;
    float diff = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        diff = std::max(diff, std::fabs(pa[i] - pb[i]));
    }
    return diff;
}

static void printTopK(Llm* llm, const char* label, VARP logits, int row, int k) {
    std::cout << label << " argmax=" << argmaxAt(logits, row)
              << " text=" << llm->tokenizer_decode(argmaxAt(logits, row)) << "\n";
    auto ids = topKAt(logits, row, k);
    std::cout << label << " top" << k << ":";
    auto info = logits->getInfo();
    int vocab = info->dim.back();
    auto ptr = logits->readMap<float>() + row * vocab;
    for (int id : ids) {
        std::cout << " " << id << "(" << llm->tokenizer_decode(id) << "," << ptr[id] << ")";
    }
    std::cout << "\n";
}

static VARP prefillAndGetLogits(Llm* llm, const std::string& prompt) {
    std::cout << "[probe] prefill begin" << std::endl;
    llm->reset();
    llm->generate_init(nullptr, "\n");
    auto ids = llm->tokenizer_encode(prompt);
    std::cout << "[probe] tokenized ids=" << ids.size() << std::endl;
    auto embeds = llm->embedding(ids);
    if (embeds.get() == nullptr || embeds->getInfo() == nullptr) {
        std::cerr << "[probe] embedding failed" << std::endl;
        return nullptr;
    }
    std::cout << "[probe] embeds seq=" << embeds->getInfo()->dim[llm->mSeqLenIndex] << std::endl;
    auto outputs = llm->forwardVec(embeds);
    if (outputs.empty() || outputs[0].get() == nullptr) {
        std::cerr << "[probe] prefill forward failed" << std::endl;
        return nullptr;
    }
    llm->updateContext(static_cast<int>(embeds->getInfo()->dim[llm->mSeqLenIndex]), 0);
    std::cout << "prompt_tokens=" << ids.size() << " all_seq_len=" << llm->mContext->all_seq_len << "\n";
    return outputs[0];
}

static void markDecodeStarted(Llm* llm, int token) {
    // Real generation samples the first token from prefill logits, appends it to
    // output history, then enters decode. The token is not in KV until the next
    // forward consumes it, but gen_seq_len must be non-zero for Omni decode
    // paths such as Qwen3-VL deepstack zeroing.
    llm->mContext->current_token = token;
    llm->mContext->gen_seq_len = 1;
    llm->mContext->output_tokens.push_back(token);
    llm->mContext->history_tokens.push_back(token);
}

static std::unique_ptr<Llm> createLoadedLlm(const std::string& configPath) {
    std::unique_ptr<Llm> llm(Llm::createLLM(configPath));
    llm->set_config("{\"tmp_path\":\"tmp\",\"sampler_type\":\"greedy\",\"top_k\":0,\"top_p\":1.0,\"temperature\":1.0}");
    if (!llm->load()) {
        return nullptr;
    }
    return llm;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " config.json prompt.txt [token_a] [token_b1] [token_b2]\n";
        return 1;
    }
    std::string configPath = argv[1];
    std::string prompt = readFile(argv[2]);
    auto llmProbe = createLoadedLlm(configPath);
    if (!llmProbe) {
        std::cerr << "load failed\n";
        return 2;
    }

    VARP prefillLogits = prefillAndGetLogits(llmProbe.get(), prompt);
    if (prefillLogits.get() == nullptr) {
        return 3;
    }
    int tokenA = argc > 3 ? std::atoi(argv[3]) : argmaxAt(prefillLogits, 0);
    int tokenB1 = argc > 4 ? std::atoi(argv[4]) : 1839;   // success in the pruned tokenizer used in debugging.
    int tokenB2 = argc > 5 ? std::atoi(argv[5]) : 11031;  // arbitrary in-vocab token.
    std::cout << "A=" << tokenA << " text=" << llmProbe->tokenizer_decode(tokenA) << "\n";
    std::cout << "B1=" << tokenB1 << " text=" << llmProbe->tokenizer_decode(tokenB1) << "\n";
    std::cout << "B2=" << tokenB2 << " text=" << llmProbe->tokenizer_decode(tokenB2) << "\n";

    // AR: prefill prompt, then decode one token [A].
    auto llmAr = createLoadedLlm(configPath);
    if (!llmAr) {
        std::cerr << "load AR llm failed\n";
        return 4;
    }
    prefillAndGetLogits(llmAr.get(), prompt);
    markDecodeStarted(llmAr.get(), tokenA);
    std::cout << "[probe] AR forward [A]" << std::endl;
    auto arLogits = llmAr->forward({tokenA}, false);
    if (arLogits.get() == nullptr) {
        std::cerr << "AR forward failed" << std::endl;
        return 4;
    }
    printTopK(llmAr.get(), "AR(seq1)", arLogits, 0, 5);

    // Verify [A, B1]: prefill prompt, then run target verify with explicit causal mask.
    auto llmB1 = createLoadedLlm(configPath);
    if (!llmB1) {
        std::cerr << "load B1 llm failed\n";
        return 5;
    }
    prefillAndGetLogits(llmB1.get(), prompt);
    markDecodeStarted(llmB1.get(), tokenA);
    int pastLen = llmB1->mContext->all_seq_len;
    std::cout << "[probe] verify B1 pastLen=" << pastLen << std::endl;
    auto verifyEmbeds1 = llmB1->embedding({tokenA, tokenB1});
    llmB1->mMeta->add = 2;
    auto verifyOutputs1 =
        llmB1->forwardRaw(verifyEmbeds1, makeVerifyMask(pastLen, 2), makeVerifyPos(llmB1.get(), pastLen, 2));
    if (verifyOutputs1.empty()) {
        std::cerr << "verify B1 failed" << std::endl;
        return 5;
    }
    auto verifyLogits1 = verifyOutputs1[0];
    printTopK(llmB1.get(), "VERIFY_B1(row0)", verifyLogits1, 0, 5);
    printTopK(llmB1.get(), "VERIFY_B1(row1)", verifyLogits1, 1, 5);

    // Verify [A, B2]: same prompt and current token, different future token.
    auto llmB2 = createLoadedLlm(configPath);
    if (!llmB2) {
        std::cerr << "load B2 llm failed\n";
        return 6;
    }
    prefillAndGetLogits(llmB2.get(), prompt);
    markDecodeStarted(llmB2.get(), tokenA);
    pastLen = llmB2->mContext->all_seq_len;
    std::cout << "[probe] verify B2 pastLen=" << pastLen << std::endl;
    auto verifyEmbeds2 = llmB2->embedding({tokenA, tokenB2});
    llmB2->mMeta->add = 2;
    auto verifyOutputs2 =
        llmB2->forwardRaw(verifyEmbeds2, makeVerifyMask(pastLen, 2), makeVerifyPos(llmB2.get(), pastLen, 2));
    if (verifyOutputs2.empty()) {
        std::cerr << "verify B2 failed" << std::endl;
        return 6;
    }
    auto verifyLogits2 = verifyOutputs2[0];
    printTopK(llmB2.get(), "VERIFY_B2(row0)", verifyLogits2, 0, 5);
    printTopK(llmB2.get(), "VERIFY_B2(row1)", verifyLogits2, 1, 5);

    std::cout << "diff AR vs VERIFY_B1(row0): " << maxAbsDiffAt(arLogits, 0, verifyLogits1, 0) << "\n";
    std::cout << "diff VERIFY_B1(row0) vs VERIFY_B2(row0): " << maxAbsDiffAt(verifyLogits1, 0, verifyLogits2, 0) << "\n";
    return 0;
}
