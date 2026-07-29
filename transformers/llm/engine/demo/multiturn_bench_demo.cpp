//
//  multiturn_bench_demo.cpp
//
// Replay a multi-turn GUI-agent benchmark with either full-prompt prefill or
// cached fixed/growing history prefill.
//

#include "llm/llm.hpp"

#include <MNN/AutoTime.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN::Transformer;

struct StepRow {
    std::string trajectory;
    std::string step;
    int history_count = 0;
    std::string prompt_path;
    std::string image_path;
    std::string task;
    std::string assistant;
};

struct PromptParts {
    std::string base;
    std::string history;
    std::string dynamic;
    bool ok = false;
};

struct Stats {
    int trajectories = 0;
    int steps = 0;
    int logical_prompt_tokens = 0;
    int actual_prefill_tokens = 0;
    int decode_tokens = 0;
    int64_t vision_us = 0;
    int64_t prefill_us = 0;
    int64_t decode_us = 0;
    int64_t sample_us = 0;
    int64_t tokenize_us = 0;
    int64_t wall_us = 0;
    float pixels_mp = 0.0f;
    int errors = 0;
    int token_boundary_fallbacks = 0;
    int lookahead_steps = 0;
    int lookahead_spec_steps = 0;
    int lookahead_ar_steps = 0;
    int lookahead_draft_tokens = 0;
    int lookahead_accepted_draft_tokens = 0;
    int lookahead_full_accept_steps = 0;
    int lookahead_accepted_tokens = 0;
};

static std::string read_file(const std::string& path) {
    std::ifstream fs(path);
    std::ostringstream os;
    os << fs.rdbuf();
    return os.str();
}

static std::string ensure_chat_template(std::string prompt) {
    if (prompt.compare(0, 12, "<|im_start|>") == 0) {
        return prompt;
    }
    const std::string user_marker = "\n### Current Task";
    auto user_pos = prompt.find(user_marker);
    if (user_pos == std::string::npos) {
        return prompt;
    }
    std::string system = prompt.substr(0, user_pos + 1);
    std::string user = prompt.substr(user_pos + 1);
    while (!user.empty() && (user.back() == '\n' || user.back() == '\r')) {
        user.pop_back();
    }
    return "<|im_start|>system\n" + system +
           "<|im_end|>\n<|im_start|>user\n" + user +
           "<|im_end|>\n<|im_start|>assistant\n";
}

static std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == '\t') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

static std::vector<StepRow> read_manifest(const std::string& path) {
    std::vector<StepRow> rows;
    std::ifstream fs(path);
    std::string line;
    bool first = true;
    while (std::getline(fs, line)) {
        if (first) {
            first = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }
        auto fields = split_tsv(line);
        if (fields.size() < 7) {
            continue;
        }
        StepRow row;
        row.trajectory = fields[0];
        row.step = fields[1];
        row.history_count = std::atoi(fields[2].c_str());
        row.prompt_path = fields[3];
        row.image_path = fields[4];
        row.task = fields[5];
        row.assistant = fields[6];
        rows.push_back(row);
    }
    return rows;
}

static std::string join_path(const std::string& root, const std::string& rel) {
    if (rel.empty()) {
        return root;
    }
    if (!rel.empty() && rel[0] == '/') {
        return rel;
    }
    if (!root.empty() && root.back() == '/') {
        return root + rel;
    }
    return root + "/" + rel;
}

static PromptParts split_prompt(const std::string& prompt) {
    PromptParts parts;
    const std::string marker = "### Action History\nThe sequence of actions you have already taken:\n";
    auto marker_pos = prompt.find(marker);
    auto image_pos = prompt.find("<img>");
    if (marker_pos == std::string::npos || image_pos == std::string::npos || image_pos < marker_pos) {
        return parts;
    }
    auto history_start = marker_pos + marker.size();
    parts.base = prompt.substr(0, history_start);
    parts.history = prompt.substr(history_start, image_pos - history_start);
    parts.dynamic = prompt.substr(image_pos);
    while (!parts.history.empty() && (parts.history.back() == '\n' || parts.history.back() == '\r')) {
        parts.dynamic.insert(parts.dynamic.begin(), parts.history.back());
        parts.history.pop_back();
    }
    parts.ok = true;
    return parts;
}

static int64_t now_us() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

static bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

static bool starts_with_tokens(const std::vector<int>& value, const std::vector<int>& prefix) {
    if (value.size() < prefix.size()) {
        return false;
    }
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (value[i] != prefix[i]) {
            return false;
        }
    }
    return true;
}

static bool encode_delta_after_prefix(Llm* llm,
                                      const std::string& prefix,
                                      const std::string& delta,
                                      const std::vector<int>& prefix_ids,
                                      std::vector<int>* delta_ids,
                                      int* full_tokens,
                                      int64_t* tokenize_us,
                                      Stats* stats = nullptr,
                                      bool count_vision = true) {
    auto context = llm->getContext();
    auto vision_before = context->vision_us;
    auto pixels_before = context->pixels_mp;
    auto empty_prefix_ids = llm->tokenizer_encode("");
    auto t0 = now_us();
    auto encoded_full = llm->tokenizer_encode(prefix + delta);
    auto t1 = now_us();
    *tokenize_us += (t1 - t0);
    if (full_tokens) {
        *full_tokens = static_cast<int>(encoded_full.size());
    }
    if (!starts_with_tokens(encoded_full, prefix_ids)) {
        size_t char_pos = 0;
        size_t token_split = empty_prefix_ids.size();
        for (size_t i = empty_prefix_ids.size(); i < encoded_full.size(); ++i) {
            std::string piece = llm->tokenizer_decode(encoded_full[i]);
            if (char_pos + piece.size() > prefix.size()) {
                break;
            }
            char_pos += piece.size();
            token_split++;
        }
        if (token_split >= encoded_full.size()) {
            delta_ids->clear();
        } else {
            delta_ids->assign(encoded_full.begin() + token_split, encoded_full.end());
        }
        llm->trimMultiModalPositionIds(token_split);
        if (stats) {
            stats->token_boundary_fallbacks++;
        }
        if (stats && count_vision) {
            stats->vision_us += (context->vision_us - vision_before);
            stats->pixels_mp += (context->pixels_mp - pixels_before);
        }
        return !delta_ids->empty();
    }
    delta_ids->assign(encoded_full.begin() + prefix_ids.size(), encoded_full.end());
    llm->trimMultiModalPositionIds(prefix_ids.size());
    if (stats && count_vision) {
        stats->vision_us += (context->vision_us - vision_before);
        stats->pixels_mp += (context->pixels_mp - pixels_before);
    }
    return true;
}

static bool run_tokens(Llm* llm,
                       const std::vector<int>& ids,
                       int max_new_tokens,
                       bool decode,
                       Stats* stats,
                       const std::string& output_label = "") {
    auto context = llm->getContext();
    std::ostringstream sink;
    auto vision_before = context->vision_us;
    auto pixels_before = context->pixels_mp;
    auto wall0 = now_us();
    if (decode) {
        llm->response(ids, &sink, nullptr, 0);
        while (!llm->stoped() && context->gen_seq_len < max_new_tokens) {
            llm->generate(1);
            if (context->status == LlmStatus::INTERNAL_ERROR) {
                return false;
            }
        }
    } else {
        llm->response(ids, &sink, nullptr, 0);
    }
    auto wall1 = now_us();
    stats->wall_us += (wall1 - wall0);
    stats->vision_us += (context->vision_us - vision_before);
    stats->pixels_mp += (context->pixels_mp - pixels_before);
    stats->prefill_us += context->prefill_us;
    stats->decode_us += context->decode_us;
    stats->sample_us += context->sample_us;
    stats->actual_prefill_tokens += static_cast<int>(ids.size());
    if (decode) {
        stats->decode_tokens += context->gen_seq_len;
        stats->lookahead_steps += context->lookahead_steps;
        stats->lookahead_spec_steps += context->lookahead_spec_steps;
        stats->lookahead_ar_steps += context->lookahead_ar_steps;
        stats->lookahead_draft_tokens += context->lookahead_draft_tokens;
        stats->lookahead_accepted_draft_tokens += context->lookahead_accepted_draft_tokens;
        stats->lookahead_full_accept_steps += context->lookahead_full_accept_steps;
        stats->lookahead_accepted_tokens += context->lookahead_accepted_tokens;
        const char* print_output = std::getenv("MOBIINFER_PRINT_OUTPUT");
        if (print_output != nullptr && std::string(print_output) != "0") {
            std::string text = sink.str();
            if (text.size() > 1200) {
                text = text.substr(0, 1200) + "\n...[truncated]";
            }
            std::cerr << "\n[MODEL_OUTPUT_BEGIN] " << output_label << "\n"
                      << text
                      << "\n[MODEL_OUTPUT_END] " << output_label << "\n";
        }
    }
    return context->status != LlmStatus::INTERNAL_ERROR;
}

static bool run_text(Llm* llm,
                     const std::string& text,
                     int max_new_tokens,
                     bool decode,
                     Stats* stats,
                     const std::string& output_label = "") {
    auto context = llm->getContext();
    std::ostringstream sink;
    auto vision_before = context->vision_us;
    auto pixels_before = context->pixels_mp;
    auto wall0 = now_us();
    if (decode) {
        llm->response(text, &sink, nullptr, 0);
        while (!llm->stoped() && context->gen_seq_len < max_new_tokens) {
            llm->generate(1);
            if (context->status == LlmStatus::INTERNAL_ERROR) {
                return false;
            }
        }
    } else {
        llm->response(text, &sink, nullptr, 0);
    }
    auto wall1 = now_us();
    stats->wall_us += (wall1 - wall0);
    stats->vision_us += (context->vision_us - vision_before);
    stats->pixels_mp += (context->pixels_mp - pixels_before);
    stats->prefill_us += context->prefill_us;
    stats->decode_us += context->decode_us;
    stats->sample_us += context->sample_us;
    stats->actual_prefill_tokens += context->prompt_len;
    if (decode) {
        stats->decode_tokens += context->gen_seq_len;
        const char* print_output = std::getenv("MOBIINFER_PRINT_OUTPUT");
        if (print_output != nullptr && std::string(print_output) != "0") {
            std::string out = sink.str();
            if (out.size() > 1200) {
                out = out.substr(0, 1200) + "\n...[truncated]";
            }
            std::cerr << "\n[MODEL_OUTPUT_BEGIN] " << output_label << "\n"
                      << out
                      << "\n[MODEL_OUTPUT_END] " << output_label << "\n";
        }
    }
    return context->status != LlmStatus::INTERNAL_ERROR;
}

static bool run_raw(Llm* llm, const std::string& bench_root, const std::vector<StepRow>& rows,
                    int max_new_tokens, Stats* stats) {
    std::string last_traj;
    for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
        const auto& row = rows[row_index];
        if (row.trajectory != last_traj) {
            stats->trajectories++;
            last_traj = row.trajectory;
        }
        auto prompt = ensure_chat_template(read_file(join_path(bench_root, row.prompt_path)));
        llm->reset();
        auto context = llm->getContext();
        auto vision_before = context->vision_us;
        auto pixels_before = context->pixels_mp;
        auto t0 = now_us();
        auto ids = llm->tokenizer_encode(prompt);
        auto t1 = now_us();
        stats->tokenize_us += (t1 - t0);
        stats->vision_us += (context->vision_us - vision_before);
        stats->pixels_mp += (context->pixels_mp - pixels_before);
        stats->logical_prompt_tokens += static_cast<int>(ids.size());
        if (!run_tokens(llm, ids, max_new_tokens, true, stats,
                        row.trajectory + "/" + row.step)) {
            stats->errors++;
        }
        stats->steps++;
        std::cerr << "[raw] finished step " << stats->steps << "/" << rows.size()
                  << " trajectory=" << row.trajectory << " step=" << row.step << std::endl;
    }
    return stats->errors == 0;
}

static bool run_cached_base(Llm* llm, const std::string& bench_root,
                            const std::vector<StepRow>& rows,
                            int max_new_tokens, Stats* stats) {
    std::string current_traj;
    std::string cached_base;
    size_t base_pos = 0;
    bool have_step_suffix = false;

    for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
        const auto& row = rows[row_index];
        auto prompt = ensure_chat_template(read_file(join_path(bench_root, row.prompt_path)));
        auto parts = split_prompt(prompt);
        if (!parts.ok) {
            MNN_ERROR("Cannot split prompt: %s\n", row.prompt_path.c_str());
            stats->errors++;
            continue;
        }

        if (row.trajectory != current_traj) {
            current_traj = row.trajectory;
            stats->trajectories++;
            cached_base = parts.base;
            llm->reset();
            if (!run_text(llm, cached_base, 0, false, stats)) {
                stats->errors++;
            }
            base_pos = llm->getCurrentHistory();
            have_step_suffix = false;
        } else if (parts.base != cached_base) {
            MNN_ERROR("Base prompt changed inside trajectory %s\n", row.trajectory.c_str());
            stats->errors++;
            continue;
        }

        if (have_step_suffix) {
            llm->eraseHistory(base_pos, 0);
        }

        std::string variable_text = parts.history + parts.dynamic;
        stats->logical_prompt_tokens += static_cast<int>(base_pos); // approximate, avoids extra image encode
        if (!run_text(llm, variable_text, max_new_tokens, true, stats,
                      row.trajectory + "/" + row.step)) {
            stats->errors++;
        }
        have_step_suffix = true;
        stats->steps++;
        std::cerr << "[cached-base] finished step " << stats->steps << "/" << rows.size()
                  << " trajectory=" << row.trajectory << " step=" << row.step << std::endl;
    }
    return stats->errors == 0;
}

static bool run_cached_history(Llm* llm, const std::string& bench_root,
                               const std::vector<StepRow>& rows,
                               int max_new_tokens, Stats* stats) {
    std::string current_traj;
    std::vector<int> cached_prefix_ids;
    bool have_previous_step = false;

    for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
        const auto& row = rows[row_index];
        auto prompt = ensure_chat_template(read_file(join_path(bench_root, row.prompt_path)));
        auto image_pos = prompt.find("<img>");
        if (image_pos == std::string::npos) {
            MNN_ERROR("Cannot find image boundary: %s\n", row.prompt_path.c_str());
            stats->errors++;
            continue;
        }

        if (row.trajectory != current_traj) {
            current_traj = row.trajectory;
            stats->trajectories++;
            cached_prefix_ids.clear();
            have_previous_step = false;
            llm->reset();
        }

        auto context = llm->getContext();
        auto vision_before = context->vision_us;
        auto pixels_before = context->pixels_mp;
        auto tokenize_start = now_us();
        auto prefix_ids = llm->tokenizer_encode(prompt.substr(0, image_pos));
        auto full_ids = llm->tokenizer_encode(prompt);
        auto tokenize_end = now_us();
        stats->tokenize_us += tokenize_end - tokenize_start;
        stats->vision_us += context->vision_us - vision_before;
        stats->pixels_mp += context->pixels_mp - pixels_before;
        stats->logical_prompt_tokens += static_cast<int>(full_ids.size());

        if (!starts_with_tokens(full_ids, prefix_ids)) {
            MNN_ERROR("Image boundary is not token-prefix stable: %s\n", row.prompt_path.c_str());
            stats->errors++;
            continue;
        }

        size_t common_tokens = 0;
        if (have_previous_step) {
            size_t common_limit = std::min(cached_prefix_ids.size(), prefix_ids.size());
            while (common_tokens < common_limit &&
                   cached_prefix_ids[common_tokens] == prefix_ids[common_tokens]) {
                common_tokens++;
            }
            if (llm->getCurrentHistory() <= common_tokens) {
                MNN_ERROR("Invalid KV rollback boundary in trajectory %s step %s\n",
                          row.trajectory.c_str(), row.step.c_str());
                stats->errors++;
                continue;
            }
            llm->eraseHistory(common_tokens, 0);
        }

        llm->trimMultiModalPositionIds(common_tokens);
        std::vector<int> delta_ids(full_ids.begin() + common_tokens, full_ids.end());
        if (!run_tokens(llm, delta_ids, max_new_tokens, true, stats,
                        row.trajectory + "/" + row.step)) {
            stats->errors++;
        }
        cached_prefix_ids = std::move(prefix_ids);
        have_previous_step = true;
        stats->steps++;
        std::cerr << "[cached-history] finished step " << stats->steps << "/" << rows.size()
                  << " trajectory=" << row.trajectory << " step=" << row.step << std::endl;
    }
    return stats->errors == 0;
}

static void print_stats(const std::string& mode, const Stats& stats) {
    double vision_s = stats.vision_us / 1e6;
    double prefill_s = stats.prefill_us / 1e6;
    double decode_s = stats.decode_us / 1e6;
    double tokenize_s = stats.tokenize_us / 1e6;
    double wall_s = stats.wall_us / 1e6;
    MNN_PRINT("\n#################################\n");
    MNN_PRINT("mode = %s\n", mode.c_str());
    MNN_PRINT("trajectories num = %d\n", stats.trajectories);
    MNN_PRINT("steps num = %d\n", stats.steps);
    MNN_PRINT("logical prompt tokens num = %d\n", stats.logical_prompt_tokens);
    MNN_PRINT("actual prefill tokens num = %d\n", stats.actual_prefill_tokens);
    MNN_PRINT("decode tokens num = %d\n", stats.decode_tokens);
    MNN_PRINT("tokenize time = %.4f s\n", tokenize_s);
    MNN_PRINT("vision time = %.4f s\n", vision_s);
    MNN_PRINT("pixels_mp = %.4f MP\n", stats.pixels_mp);
    MNN_PRINT("vision speed = %.2f MP/s\n",
              vision_s > 0.0 ? stats.pixels_mp / vision_s : 0.0);
    MNN_PRINT("prefill time = %.4f s\n", prefill_s);
    MNN_PRINT("decode time = %.4f s\n", decode_s);
    MNN_PRINT("sample time = %.4f s\n", stats.sample_us / 1e6);
    MNN_PRINT("wall generate time = %.4f s\n", wall_s);
    MNN_PRINT("effective prefill speed = %.2f tok/s\n",
              prefill_s > 0.0 ? stats.logical_prompt_tokens / prefill_s : 0.0);
    MNN_PRINT("actual prefill speed = %.2f tok/s\n",
              prefill_s > 0.0 ? stats.actual_prefill_tokens / prefill_s : 0.0);
    MNN_PRINT("decode speed = %.2f tok/s\n",
              decode_s > 0.0 ? stats.decode_tokens / decode_s : 0.0);
    MNN_PRINT("prefill token reduction = %.4f\n",
              stats.logical_prompt_tokens > 0
                  ? 1.0 - (double)stats.actual_prefill_tokens / stats.logical_prompt_tokens
                  : 0.0);
    MNN_PRINT("lookahead steps num = %d\n", stats.lookahead_steps);
    MNN_PRINT("lookahead spec steps num = %d\n", stats.lookahead_spec_steps);
    MNN_PRINT("lookahead ar steps num = %d\n", stats.lookahead_ar_steps);
    MNN_PRINT("lookahead draft tokens num = %d\n", stats.lookahead_draft_tokens);
    MNN_PRINT("lookahead accepted draft tokens num = %d\n", stats.lookahead_accepted_draft_tokens);
    MNN_PRINT("lookahead draft accept rate = %.2f%%\n",
              stats.lookahead_draft_tokens > 0
                  ? 100.0 * stats.lookahead_accepted_draft_tokens / stats.lookahead_draft_tokens
                  : 0.0);
    MNN_PRINT("lookahead full accept steps num = %d\n", stats.lookahead_full_accept_steps);
    MNN_PRINT("lookahead full accept rate = %.2f%%\n",
              stats.lookahead_spec_steps > 0
                  ? 100.0 * stats.lookahead_full_accept_steps / stats.lookahead_spec_steps
                  : 0.0);
    MNN_PRINT("lookahead accepted tokens num = %d\n", stats.lookahead_accepted_tokens);
    MNN_PRINT("token boundary fallbacks num = %d\n", stats.token_boundary_fallbacks);
    MNN_PRINT("errors num = %d\n", stats.errors);
    MNN_PRINT("##################################\n");
}

static void usage(const char* name) {
    MNN_PRINT("Usage: %s config.json bench_root [raw|raw-ar|cached-base|cached-history|cached-history-ar] [max_new_tokens] [limit_steps]\n", name);
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    std::string config_path = argv[1];
    std::string bench_root = argv[2];
    std::string mode = argv[3];
    int max_new_tokens = 192;
    if (argc >= 5) {
        max_new_tokens = std::atoi(argv[4]);
    }
    int limit_steps = -1;
    if (argc >= 6) {
        limit_steps = std::atoi(argv[5]);
    }

    auto rows = read_manifest(join_path(bench_root, "manifest.tsv"));
    if (limit_steps > 0 && static_cast<int>(rows.size()) > limit_steps) {
        rows.resize(limit_steps);
    }
    if (rows.empty()) {
        MNN_ERROR("No manifest rows found under %s\n", bench_root.c_str());
        return 1;
    }

    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    if (!llm) {
        MNN_ERROR("createLLM failed\n");
        return 1;
    }
    llm->set_config(R"({"async":false,"use_template":false,"reuse_kv":true,"tmp_path":"tmp"})");
    if (mode == "raw-ar" || mode == "cached-history-ar") {
        llm->set_config(R"({"speculative_type":"","lookahead_debug_stats":false,"lookahead_debug_stats_detail":false})");
    }
    {
        AUTOTIME;
        if (!llm->load()) {
            MNN_ERROR("LLM init error\n");
            return 1;
        }
    }

    Stats stats;
    bool ok = false;
    if (mode == "raw" || mode == "raw-ar") {
        llm->set_config(R"({"reuse_kv":false,"use_template":false})");
        if (mode == "raw") {
            llm->set_config(R"({"lookahead_debug_stats":true,"lookahead_debug_stats_detail":false})");
        }
        ok = run_raw(llm.get(), bench_root, rows, max_new_tokens, &stats);
    } else if (mode == "cached-history") {
        llm->set_config(R"({"reuse_kv":true,"use_template":false})");
        llm->set_config(R"({"lookahead_debug_stats":true,"lookahead_debug_stats_detail":false})");
        ok = run_cached_history(llm.get(), bench_root, rows, max_new_tokens, &stats);
    } else if (mode == "cached-history-ar") {
        llm->set_config(R"({"reuse_kv":true,"use_template":false})");
        ok = run_cached_history(llm.get(), bench_root, rows, max_new_tokens, &stats);
    } else if (mode == "cached-base") {
        llm->set_config(R"({"reuse_kv":true,"use_template":false})");
        ok = run_cached_base(llm.get(), bench_root, rows, max_new_tokens, &stats);
    } else {
        usage(argv[0]);
        return 1;
    }
    print_stats(mode, stats);
    return ok ? 0 : 1;
}
