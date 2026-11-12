// serve_bench_main.cpp - what batching buys: C requests decoded together against C requests one after the other.
//
//   gi_serve_bench --model DIR_OR_FILE [--concurrency 1,2,4,8] [--prompt-tokens 16] [--decode-tokens 64]
//                  [--runs 3] [--max-seq-len N] [--threads N] [--weights native|f32|int8] [--block-size B]
//
// For every concurrency level C the same C synthetic requests (random prompt tokens, greedy sampling, no end token,
// exactly --decode-tokens generated tokens each) are run through the Engine with C requests allowed to run at
// once. Decoding is limited by the speed at which the weights stream from memory; with C requests in one pass every
// weight is read once for C tokens, so the aggregate tokens/s should grow with C until the arithmetic becomes the limit.
// The C = 1 line is the unbatched baseline (the same code path with a single row per pass). The tokens of every
// request are compared with the C = 1 run of the same request, which must be identical.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "loader/checkpoint.h"
#include "model/transformer.h"
#include "runtime/engine.h"
#include "tokenizer/text_tokenizer.h"

using namespace geneinfer;

namespace {

// Synthetic requests have no text: no token ends a generation and pieces are empty.
class NullTokenizer : public TextTokenizer {
public:
    explicit NullTokenizer(int vocab) : vocab_(vocab) {}
    std::string kind() const override { return "null"; }
    int vocab_size() const override { return vocab_; }
    std::vector<int> encode_prompt(const std::string&) const override { return {0}; }
    std::string decode_piece(int, int) const override { return {}; }
    std::string decode(const std::vector<int>&) const override { return {}; }
    bool is_end_token(int) const override { return false; }

private:
    int vocab_;
};

[[noreturn]] void usage_error(const std::string& msg) {
    std::fprintf(stderr, "gi_serve_bench: %s\n", msg.c_str());
    std::fprintf(stderr,
                 "usage: gi_serve_bench --model M [--concurrency 1,2,4,8] [--prompt-tokens N] [--decode-tokens N] [--runs N]\n"
                 "                      [--max-seq-len N] [--threads N] [--weights native|f32|int8] [--block-size B]\n");
    std::exit(2);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, weights_arg, conc_arg = "1,2,4,8";
    int prompt_tokens = 16, decode_tokens = 64, runs = 3, max_seq_len = 0, n_threads = 0, block_size = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage_error("missing value for " + f);
            return argv[++i];
        };
        if (f == "--model") model_path = value();
        else if (f == "--concurrency") conc_arg = value();
        else if (f == "--prompt-tokens") prompt_tokens = std::atoi(value().c_str());
        else if (f == "--decode-tokens") decode_tokens = std::atoi(value().c_str());
        else if (f == "--runs") runs = std::atoi(value().c_str());
        else if (f == "--max-seq-len") max_seq_len = std::atoi(value().c_str());
        else if (f == "--threads") n_threads = std::atoi(value().c_str());
        else if (f == "--weights") weights_arg = value();
        else if (f == "--block-size") block_size = std::atoi(value().c_str());
        else usage_error("unknown option " + f);
    }
    if (model_path.empty()) usage_error("--model is required");
    if (prompt_tokens < 1 || decode_tokens < 1 || runs < 1) usage_error("--prompt-tokens, --decode-tokens and --runs must be >= 1");
    std::vector<int> levels;
    {
        std::stringstream ss(conc_arg);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const int c = std::atoi(item.c_str());
            if (c < 1) usage_error("--concurrency needs positive integers");
            levels.push_back(c);
        }
        if (levels.empty()) usage_error("--concurrency is empty");
    }

    try {
        LoadOptions lo = LoadOptions::from_environment();
        if (weights_arg == "f32") { lo.keep_bf16 = false; lo.int8 = false; }
        else if (weights_arg == "int8") { lo.int8 = true; }
        else if (weights_arg == "native") { lo.keep_bf16 = true; lo.int8 = false; }
        else if (!weights_arg.empty()) usage_error("--weights must be native, f32 or int8");
        const Checkpoint ck = Checkpoint::load(model_path, lo);
        const int vocab = ck.config().vocab_size;
        const int total = prompt_tokens + decode_tokens;
        const int cmax = *std::max_element(levels.begin(), levels.end());

        // A pool for the largest level; the context is only as long as one request needs.
        if (max_seq_len == 0) max_seq_len = total;
        if (max_seq_len < total) usage_error("--max-seq-len is shorter than prompt + decode tokens");
        const int bs = block_size > 0 ? block_size : 16;
        const int per_request = (total + bs - 1) / bs;
        Transformer model(ck, max_seq_len, n_threads, 0, bs, cmax * per_request + 1);
        NullTokenizer tok(vocab);

        std::mt19937 rng(7);
        std::vector<EngineRequest> reqs(static_cast<std::size_t>(cmax));
        for (EngineRequest& r : reqs) {
            for (int j = 0; j < prompt_tokens; ++j) r.prompt.push_back(static_cast<int>(rng() % static_cast<unsigned>(vocab)));
            r.max_positions = total - 1;   // prompt_tokens + decode_tokens - 1 positions give decode_tokens generated tokens
            r.sampler.temperature = 0.0f;
        }

        std::printf("model: %s | %d threads | prompt %d + decode %d tokens per request | block size %d | max rows per pass %d\n",
                    model_path.c_str(), model.threads(), prompt_tokens, decode_tokens, bs, model.max_batch());
        std::printf("%-11s %-10s %-12s %-14s %-12s %-10s %s\n", "concurrent", "wall (s)", "tokens/s", "tok/s/request", "rows/pass", "speedup", "outputs");

        std::vector<std::vector<int>> baseline(static_cast<std::size_t>(cmax));   // from the C = 1 style run, per request
        double base_tps = 0.0;
        bool all_same = true;
        // Baseline outputs: every request alone.
        for (int r = 0; r < cmax; ++r) {
            Engine e(model, tok, EngineConfig{1, 0, 0});
            const int id = e.submit(reqs[static_cast<std::size_t>(r)]);
            e.run_until_idle();
            baseline[static_cast<std::size_t>(r)] = e.result(id).generated_tokens;
            if (static_cast<int>(baseline[static_cast<std::size_t>(r)].size()) != decode_tokens) throw std::runtime_error("unexpected token count");
        }
        for (const int c : levels) {
            std::vector<double> walls;
            double rows_per_pass = 0.0;
            bool same = true;
            for (int run = -1; run < runs; ++run) {   // run -1 is the unmeasured warm-up
                EngineConfig ec;
                ec.max_running = c;
                Engine e(model, tok, ec);
                std::vector<int> ids;
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < c; ++r) ids.push_back(e.submit(reqs[static_cast<std::size_t>(r)]));
                e.run_until_idle();
                const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (run >= 0) walls.push_back(wall);
                for (int r = 0; r < c; ++r) same = same && e.result(ids[static_cast<std::size_t>(r)]).generated_tokens == baseline[static_cast<std::size_t>(r)];
                rows_per_pass = static_cast<double>(e.stats().rows) / static_cast<double>(e.stats().passes);
            }
            all_same = all_same && same;
            const double wall = median(walls);
            const double tps = static_cast<double>(c) * decode_tokens / wall;
            if (base_tps == 0.0) base_tps = tps;
            std::printf("%-11d %-10.3f %-12.1f %-14.1f %-12.2f %-10.2f %s\n", c, wall, tps, tps / c, rows_per_pass, tps / base_tps,
                        same ? "identical to running alone" : "DIFFER");
        }
        std::printf("(tokens/s counts generated tokens only; wall time includes the prompts.)\n");
        return all_same ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
