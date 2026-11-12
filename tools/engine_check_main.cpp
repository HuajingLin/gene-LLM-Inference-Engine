// engine_check_main.cpp - the Engine (request queue + continuous batching) against generate() run alone.
//
//   gi_engine_check <hf model dir with tokenizer.json> --max-seq-len N [--seed S] [--requests R] [--block-size B]
//                   [--kv-blocks K] [--max-running M] [--step-rows W] [--chunk C] [--staggered] [--oversize] [--wide-eos] [--shared-prefix L] [--no-prefix-cache]
//
// R random requests (prompt lengths, position limits and sampler settings are drawn from the seed) are first run
// one after the other with generate() on a model with a roomy pool (the reference), then all together through an
// Engine on a model with the requested pool size. The generated tokens, text and stop reason of every request must
// be identical. With --staggered the requests are submitted a few at a time while the engine is already running.
// With --oversize the pool must be smaller than a context: one request that can never fit is submitted next to a
// small one; the big one must fail with an error text, the small one must still match generate().
// Prints "ok ..." lines with statistics and "PASS", or "FAIL ..." (exit code 1).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <random>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/transformer.h"
#include "runtime/engine.h"
#include "runtime/generator.h"
#include "tokenizer/text_tokenizer.h"

using namespace geneinfer;

namespace {
struct Failure {
    std::string what;
};

// With --wide-eos about one token in eleven ends a generation (so the end-token rule is really exercised), and every
// piece carries the id of the token before it (so a wrong "previous token" is visible in the text).
class WideTokenizer : public TextTokenizer {
public:
    WideTokenizer(std::unique_ptr<TextTokenizer> base, bool wide) : base_(std::move(base)), wide_(wide) {}
    std::string kind() const override { return base_->kind(); }
    int vocab_size() const override { return base_->vocab_size(); }
    std::vector<int> encode_prompt(const std::string& t) const override { return base_->encode_prompt(t); }
    std::string decode_piece(int prev, int token) const override {
        return base_->decode_piece(prev, token) + (wide_ ? "[" + std::to_string(prev) + "]" : std::string());
    }
    std::string decode(const std::vector<int>& t) const override { return base_->decode(t); }
    bool is_end_token(int token) const override { return base_->is_end_token(token) || (wide_ && token % 11 == 3); }

private:
    std::unique_ptr<TextTokenizer> base_;
    bool wide_;
};
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model dir> --max-seq-len N [options]\n", argv[0]);
        return 2;
    }
    try {
        int ctx = 0, bs = 0, nblocks = 0, nreq = 6, max_running = 4, step_rows = 0, chunk = 0;
        unsigned seed = 1;
        bool staggered = false, oversize = false, wide = false, prefix_cache = true;
        int shared_prefix = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--staggered") {
                staggered = true;
                continue;
            }
            if (a == "--no-prefix-cache") {
                prefix_cache = false;
                continue;
            }
            if (a == "--wide-eos") {
                wide = true;
                continue;
            }
            if (a == "--oversize") {
                oversize = true;
                continue;
            }
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            const std::string v = argv[++i];
            if (a == "--max-seq-len") ctx = std::stoi(v);
            else if (a == "--block-size") bs = std::stoi(v);
            else if (a == "--kv-blocks") nblocks = std::stoi(v);
            else if (a == "--seed") seed = static_cast<unsigned>(std::stoul(v));
            else if (a == "--requests") nreq = std::stoi(v);
            else if (a == "--max-running") max_running = std::stoi(v);
            else if (a == "--step-rows") step_rows = std::stoi(v);
            else if (a == "--chunk") chunk = std::stoi(v);
            else if (a == "--shared-prefix") shared_prefix = std::stoi(v);
            else throw std::runtime_error("unknown option " + a);
        }
        const Checkpoint ck = Checkpoint::load(argv[1]);
        const auto tok = std::make_unique<WideTokenizer>(load_hf_tokenizer(argv[1], false), wide);
        Transformer ref_model(ck, ctx);                       // roomy pool: one full context
        Transformer model(ck, ctx, 0, 0, bs, nblocks);        // the pool under test
        const int vocab = model.config().vocab_size;
        std::mt19937 rng(seed);

        const std::vector<int> base_a = [&] { std::vector<int> v; for (int i = 0; i < std::max(shared_prefix, 1); ++i) v.push_back(static_cast<int>(rng() % static_cast<unsigned>(vocab))); return v; }();
        const std::vector<int> base_b = [&] { std::vector<int> v; for (int i = 0; i < std::max(shared_prefix, 1); ++i) v.push_back(static_cast<int>(rng() % static_cast<unsigned>(vocab))); return v; }();
        std::vector<EngineRequest> reqs;
        std::vector<GenerateResult> refs;
        for (int r = 0; r < nreq; ++r) {
            EngineRequest q;
            const int plen = 1 + static_cast<int>(rng() % static_cast<unsigned>(std::max(1, ctx / 2)));
            for (int j = 0; j < plen; ++j) q.prompt.push_back(static_cast<int>(rng() % static_cast<unsigned>(vocab)));
            if (shared_prefix > 0) {   // start with (a part of) one of two common prefixes, then continue on its own
                const std::vector<int>& base = (rng() & 1u) ? base_a : base_b;
                const int keep = std::min<int>(plen, static_cast<int>(rng() % static_cast<unsigned>(shared_prefix + 1)));
                for (int j = 0; j < keep; ++j) q.prompt[static_cast<std::size_t>(j)] = base[static_cast<std::size_t>(j)];
            }
            q.max_positions = (rng() % 3 == 0) ? 0 : (rng() % 5 == 0) ? 1 + static_cast<int>(rng() % static_cast<unsigned>(plen)) : plen + static_cast<int>(rng() % 24);
            q.sampler.temperature = (rng() % 2 == 0) ? 0.0f : 0.5f + static_cast<float>(rng() % 10) / 10.0f;
            q.sampler.top_k = (rng() % 3 == 0) ? 0 : 1 + static_cast<int>(rng() % 20);
            q.sampler.top_p = (rng() % 3 == 0) ? 1.0f : 0.5f + static_cast<float>(rng() % 50) / 100.0f;
            q.sampler.seed = 1 + rng() % 1000;
            Sampler sampler(q.sampler, vocab);
            refs.push_back(generate(ref_model, *tok, sampler, q.prompt, q.max_positions));
            reqs.push_back(q);
        }

        if (oversize) {
            const int capacity = model.kv_pool().n_blocks() * model.kv_pool().block_size();
            if (capacity >= ctx) throw std::runtime_error("--oversize needs a pool smaller than the context");
            EngineRequest big, small = reqs[0];
            for (int j = 0; j < capacity + 1; ++j) big.prompt.push_back(j % vocab);
            if (static_cast<int>(small.prompt.size()) > capacity / 2) small.prompt.resize(static_cast<std::size_t>(capacity / 2));
            small.max_positions = std::min(small.max_positions > 0 ? small.max_positions : capacity, capacity);
            Sampler sampler(small.sampler, vocab);
            const GenerateResult want = generate(ref_model, *tok, sampler, small.prompt, small.max_positions);
            Engine e(model, *tok, EngineConfig{});
            const int big_id = e.submit(big), small_id = e.submit(small);
            e.run_until_idle();
            if (e.result(big_id).error.find("too small") == std::string::npos) throw Failure{"the oversized request did not fail with an error"};
            if (!e.result(small_id).error.empty() || e.result(small_id).generated_tokens != want.generated_tokens) {
                throw Failure{"the small request next to an oversized one differs from generate() or failed: " + e.result(small_id).error};
            }
            e.clear_prefix_cache();
            if (model.kv_pool().free_blocks() != model.kv_pool().n_blocks()) throw Failure{"blocks leaked"};
            std::printf("ok oversize: the request that cannot fit failed cleanly, its neighbour was unaffected\n");
            std::printf("PASS\n");
            return 0;
        }
        EngineConfig ec;
        ec.max_running = max_running;
        ec.step_rows = step_rows;
        ec.prefill_chunk = chunk;
        ec.prefix_cache = prefix_cache;
        Engine engine(model, *tok, ec);
        std::vector<int> ids;
        std::size_t submitted = 0;
        while (submitted < reqs.size() || !engine.idle()) {
            const std::size_t burst = staggered ? 1 + rng() % 3 : reqs.size();
            for (std::size_t b = 0; b < burst && submitted < reqs.size(); ++b) ids.push_back(engine.submit(reqs[submitted++]));
            if (staggered) {
                const int n = 1 + static_cast<int>(rng() % 4);
                for (int s = 0; s < n && engine.step(); ++s) {
                }
            } else {
                engine.run_until_idle();
            }
        }
        long expected_rows = 0;
        long total_generated = 0;
        for (std::size_t r = 0; r < reqs.size(); ++r) {
            const EngineResult& got = engine.result(ids[r]);
            if (!got.finished) throw Failure{"request " + std::to_string(r) + " did not finish"};
            if (!got.error.empty()) throw Failure{"request " + std::to_string(r) + " failed: " + got.error};
            if (got.generated_tokens != refs[r].generated_tokens) {
                std::size_t k = 0;
                while (k < got.generated_tokens.size() && k < refs[r].generated_tokens.size() && got.generated_tokens[k] == refs[r].generated_tokens[k]) ++k;
                throw Failure{"request " + std::to_string(r) + ": generated tokens differ from generate() at index " + std::to_string(k) + " (engine " +
                              std::to_string(got.generated_tokens.size()) + " tokens, generate " + std::to_string(refs[r].generated_tokens.size()) +
                              "; prompt " + std::to_string(reqs[r].prompt.size()) + ", max_positions " + std::to_string(reqs[r].max_positions) +
                              ", temperature " + std::to_string(reqs[r].sampler.temperature) + ")"};
            }
            if (got.text != refs[r].text) throw Failure{"request " + std::to_string(r) + ": text differs from generate()"};
            if (got.stopped_by_end_token != refs[r].stopped_by_end_token) throw Failure{"request " + std::to_string(r) + ": stop reason differs"};
            total_generated += static_cast<long>(got.generated_tokens.size());
            // positions fed to the model: the last sampled token is never fed; a request that stopped inside its prompt fed `steps` rows
            const long g = static_cast<long>(got.generated_tokens.size());
            const long plen = static_cast<long>(reqs[r].prompt.size());
            const long steps = reqs[r].max_positions <= 0 ? ctx : std::min<long>(reqs[r].max_positions, ctx);
            expected_rows += g > 0 ? plen + g - 1 : std::min(steps, plen);
        }
        if (model.kv_pool().free_blocks() + engine.cached_blocks() != model.kv_pool().n_blocks()) throw Failure{"blocks leaked after all requests finished"};
        engine.clear_prefix_cache();
        if (model.kv_pool().free_blocks() != model.kv_pool().n_blocks()) throw Failure{"blocks still in use after clearing the prefix cache"};
        const EngineStats& st = engine.stats();
        const int rows_limit = (step_rows > 0 && step_rows < model.max_batch()) ? step_rows : model.max_batch();
        if (st.preemptions == 0 && st.rows + st.prefix_tokens != expected_rows) throw Failure{"the engine processed " + std::to_string(st.rows) + " rows and reused " + std::to_string(st.prefix_tokens) + " cached positions, expected " + std::to_string(expected_rows) + " in total (no preemptions)"};
        if (!prefix_cache && (st.prefix_tokens != 0 || st.prefix_hits != 0)) throw Failure{"prefix cache statistics are not zero with the cache disabled"};
        if (st.max_rows_in_pass > rows_limit) throw Failure{"a pass had more rows than --step-rows allows"};
        if (chunk > 0 && st.max_chunk > chunk) throw Failure{"a request added more prompt rows to a pass than --chunk allows"};
        if (st.max_running_seen > max_running) throw Failure{"more requests ran at once than --max-running allows"};
        std::printf("ok engine: %d requests, %ld generated tokens, %ld passes, %ld rows (max %d per pass), %ld preemptions, up to %d running, "
                    "prefix cache: %ld positions reused in %ld admissions, %ld blocks evicted\n",
                    nreq, total_generated, st.passes, st.rows, st.max_rows_in_pass, st.preemptions, st.max_running_seen, st.prefix_tokens,
                    st.prefix_hits, st.evicted_blocks);
        std::printf("PASS\n");
        return 0;
    } catch (const Failure& f) {
        std::printf("FAIL %s\n", f.what.c_str());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
