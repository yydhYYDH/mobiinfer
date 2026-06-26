//
//  kvcache_demo.cpp
//

#include "llm/llm.hpp"

#include <MNN/AutoTime.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using namespace MNN::Transformer;

static std::string read_prompt_file(const std::string& prompt_file) {
    std::ifstream prompt_fs(prompt_file);
    std::ostringstream temp_os;
    temp_os << prompt_fs.rdbuf();
    return temp_os.str();
}

static std::vector<int> encode_raw_prompt(Llm* llm, const std::string& prompt) {
    return llm->tokenizer_encode(prompt);
}

static int dump_kvcache(Llm* llm, const std::string& cache_name, const std::string& prompt_file) {
    auto prompt = read_prompt_file(prompt_file);
    if (prompt.empty()) {
        MNN_ERROR("Prompt file is empty: %s\n", prompt_file.c_str());
        return 1;
    }
    auto input_ids = encode_raw_prompt(llm, prompt);
    if (input_ids.empty()) {
        MNN_ERROR("Prompt encodes to empty token list: %s\n", prompt_file.c_str());
        return 1;
    }
    if (!llm->dumpPromptKVCache(input_ids, cache_name)) {
        MNN_ERROR("Failed to dump KV cache: %s\n", cache_name.c_str());
        return 1;
    }
    auto context = llm->getContext();
    MNN_PRINT("KV cache dump done. prompt tokens=%d prefill time=%.2f s\n",
              context->prompt_len, context->prefill_us / 1e6);
    return 0;
}

static int load_kvcache(Llm* llm, const std::string& cache_name, const std::string& prompt_file,
                        int max_token_number) {
    auto prompt = read_prompt_file(prompt_file);
    if (prompt.empty()) {
        MNN_ERROR("Prompt file is empty: %s\n", prompt_file.c_str());
        return 1;
    }
    auto input_ids = encode_raw_prompt(llm, prompt);
    if (input_ids.empty()) {
        MNN_ERROR("Prompt encodes to empty token list: %s\n", prompt_file.c_str());
        return 1;
    }

    std::vector<int> suffix_ids;
    std::ostringstream answer;
    llm->generate_init(&answer, "\n");
    if (!llm->loadPromptKVCache(input_ids, cache_name, &suffix_ids)) {
        MNN_ERROR("Failed to load KV cache: %s\n", cache_name.c_str());
        return 1;
    }
    auto context = llm->getContext();
    if (!llm->continueFromCachedToken(context->current_token, max_token_number)) {
        MNN_ERROR("Error: Generation failed due to internal error\n");
        return 1;
    }

    MNN_PRINT("%s", answer.str().c_str());
    MNN_PRINT("\n#################################\n");
    MNN_PRINT("cached prompt tokens num = %d\n", (int)input_ids.size());
    MNN_PRINT("suffix tokens num = %zu\n", suffix_ids.size());
    MNN_PRINT("decode tokens num = %d\n", context->gen_seq_len);
    MNN_PRINT("prefill time = %.2f s\n", context->prefill_us / 1e6);
    MNN_PRINT("decode time = %.2f s\n", context->decode_us / 1e6);
    MNN_PRINT("##################################\n");
    return 0;
}

static int load_prefix_step(Llm* llm, const std::string& cache_name, const std::string& prefix_file,
                            const std::string& variable_file, int max_token_number) {
    auto prefix = read_prompt_file(prefix_file);
    if (prefix.empty()) {
        MNN_ERROR("Prefix file is empty: %s\n", prefix_file.c_str());
        return 1;
    }
    auto variable = read_prompt_file(variable_file);
    if (variable.empty()) {
        MNN_ERROR("Variable file is empty: %s\n", variable_file.c_str());
        return 1;
    }
    auto prefix_ids = encode_raw_prompt(llm, prefix);
    auto variable_ids = encode_raw_prompt(llm, variable);
    if (prefix_ids.empty() || variable_ids.empty()) {
        MNN_ERROR("Prefix or variable encodes to empty token list.\n");
        return 1;
    }

    std::ostringstream answer;
    llm->generate_init(&answer, "\n");
    if (!llm->loadPromptKVCachePrefixOnly(prefix_ids, cache_name)) {
        MNN_ERROR("Failed to load prefix-only KV cache: %s\n", cache_name.c_str());
        return 1;
    }
    llm->generate(variable_ids, max_token_number);
    auto context = llm->getContext();
    if (context->status == LlmStatus::INTERNAL_ERROR) {
        MNN_ERROR("Error: Generation failed due to internal error\n");
        return 1;
    }

    MNN_PRINT("%s", answer.str().c_str());
    MNN_PRINT("\n#################################\n");
    MNN_PRINT("cached prefix tokens num = %d\n", (int)prefix_ids.size());
    MNN_PRINT("variable tokens num = %d\n", (int)variable_ids.size());
    MNN_PRINT("decode tokens num = %d\n", context->gen_seq_len);
    MNN_PRINT("prefill time = %.2f s\n", context->prefill_us / 1e6);
    MNN_PRINT("decode time = %.2f s\n", context->decode_us / 1e6);
    MNN_PRINT("##################################\n");
    return 0;
}

static int split_step(Llm* llm, const std::string& prefix_file, const std::string& variable_file,
                      int max_token_number) {
    auto prefix = read_prompt_file(prefix_file);
    if (prefix.empty()) {
        MNN_ERROR("Prefix file is empty: %s\n", prefix_file.c_str());
        return 1;
    }
    auto variable = read_prompt_file(variable_file);
    if (variable.empty()) {
        MNN_ERROR("Variable file is empty: %s\n", variable_file.c_str());
        return 1;
    }
    auto prefix_ids = encode_raw_prompt(llm, prefix);
    auto variable_ids = encode_raw_prompt(llm, variable);
    if (prefix_ids.empty() || variable_ids.empty()) {
        MNN_ERROR("Prefix or variable encodes to empty token list.\n");
        return 1;
    }

    std::ostringstream answer;
    llm->generate_init(&answer, "\n");
    llm->generate(prefix_ids, 0);
    llm->generate(variable_ids, max_token_number);
    auto context = llm->getContext();
    if (context->status == LlmStatus::INTERNAL_ERROR) {
        MNN_ERROR("Error: Generation failed due to internal error\n");
        return 1;
    }

    MNN_PRINT("%s", answer.str().c_str());
    MNN_PRINT("\n#################################\n");
    MNN_PRINT("split prefix tokens num = %d\n", (int)prefix_ids.size());
    MNN_PRINT("variable tokens num = %d\n", (int)variable_ids.size());
    MNN_PRINT("decode tokens num = %d\n", context->gen_seq_len);
    MNN_PRINT("prefill time = %.2f s\n", context->prefill_us / 1e6);
    MNN_PRINT("decode time = %.2f s\n", context->decode_us / 1e6);
    MNN_PRINT("##################################\n");
    return 0;
}

static int raw_generate(Llm* llm, const std::string& prompt_file, int max_token_number) {
    auto prompt = read_prompt_file(prompt_file);
    if (prompt.empty()) {
        MNN_ERROR("Prompt file is empty: %s\n", prompt_file.c_str());
        return 1;
    }
    auto input_ids = encode_raw_prompt(llm, prompt);
    if (input_ids.empty()) {
        MNN_ERROR("Prompt encodes to empty token list: %s\n", prompt_file.c_str());
        return 1;
    }

    std::ostringstream answer;
    llm->generate_init(&answer, "\n");
    llm->generate(input_ids, 0);
    auto context = llm->getContext();
    while (!llm->stoped() && context->gen_seq_len < max_token_number) {
        llm->generate(1);
        if (context->status == LlmStatus::INTERNAL_ERROR) {
            MNN_ERROR("Error: Generation failed due to internal error\n");
            return 1;
        }
    }

    MNN_PRINT("%s", answer.str().c_str());
    MNN_PRINT("\n#################################\n");
    MNN_PRINT("prompt tokens num = %d\n", context->prompt_len);
    MNN_PRINT("decode tokens num = %d\n", context->gen_seq_len);
    MNN_PRINT("prefill time = %.2f s\n", context->prefill_us / 1e6);
    MNN_PRINT("decode time = %.2f s\n", context->decode_us / 1e6);
    MNN_PRINT("##################################\n");
    return 0;
}

static void print_usage(const char* name) {
    MNN_PRINT("Usage:\n");
    MNN_PRINT("  %s config.json --raw prompt.txt [max_tokens]\n", name);
    MNN_PRINT("  %s config.json --dump <cache_name> prompt.txt\n", name);
    MNN_PRINT("  %s config.json --load <cache_name> prompt.txt [max_tokens]\n", name);
    MNN_PRINT("  %s config.json --split-step prefix.txt variable.txt [max_tokens]\n", name);
    MNN_PRINT("  %s config.json --load-prefix-step <cache_name> prefix.txt variable.txt [max_tokens]\n", name);
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 0;
    }

    std::string config_path = argv[1];
    std::string mode = argv[2];

    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    if (!llm) {
        MNN_ERROR("createLLM failed\n");
        return 1;
    }
    llm->set_config(R"({"async":false,"kvcache_mmap":true})");
    {
        AUTOTIME;
        bool res = llm->load();
        if (!res) {
            MNN_ERROR("LLM init error\n");
            return 1;
        }
    }

    if (mode == "--dump") {
        if (argc < 5) {
            print_usage(argv[0]);
            return 1;
        }
        std::string cache_name = argv[3];
        std::string prompt_file = argv[4];
        return dump_kvcache(llm.get(), cache_name, prompt_file);
    }
    if (mode == "--load") {
        if (argc < 5) {
            print_usage(argv[0]);
            return 1;
        }
        std::string cache_name = argv[3];
        std::string prompt_file = argv[4];
        int max_token_number = -1;
        if (argc >= 6) {
            std::istringstream os(argv[5]);
            os >> max_token_number;
        }
        return load_kvcache(llm.get(), cache_name, prompt_file, max_token_number);
    }
    if (mode == "--raw") {
        std::string prompt_file = argv[3];
        int max_token_number = -1;
        if (argc >= 5) {
            std::istringstream os(argv[4]);
            os >> max_token_number;
        }
        return raw_generate(llm.get(), prompt_file, max_token_number);
    }
    if (mode == "--load-prefix-step") {
        if (argc < 6) {
            print_usage(argv[0]);
            return 1;
        }
        std::string cache_name = argv[3];
        std::string prefix_file = argv[4];
        std::string variable_file = argv[5];
        int max_token_number = -1;
        if (argc >= 7) {
            std::istringstream os(argv[6]);
            os >> max_token_number;
        }
        return load_prefix_step(llm.get(), cache_name, prefix_file, variable_file, max_token_number);
    }
    if (mode == "--split-step") {
        if (argc < 5) {
            print_usage(argv[0]);
            return 1;
        }
        std::string prefix_file = argv[3];
        std::string variable_file = argv[4];
        int max_token_number = -1;
        if (argc >= 6) {
            std::istringstream os(argv[5]);
            os >> max_token_number;
        }
        return split_step(llm.get(), prefix_file, variable_file, max_token_number);
    }

    print_usage(argv[0]);
    return 1;
}
