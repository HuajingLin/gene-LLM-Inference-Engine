// bench_main.cpp - throughput and memory baseline for the engine.
//
//   gi_bench --model <model.bin> [options]
//
// Options:
//   --prompt-tokens N   positions processed as "prompt" (default 32)
//   --decode-tokens N   positions processed one by one with sampling (default 128)
//   --runs N            measured runs (default 5)
//   --max-seq-len N     cap the context length / KV cache (default: the model's)
//   --warmup N          unmeasured runs first, to fault in the memory-mapped weights (default 1)
//   --weights KIND      weight storage: native (default), f32 or int8 (same as the GI_WEIGHTS environment variable)
//   --threads N         threads sharing the matrix products (default: GI_THREADS or 1)
//   --isa NAME          force the instruction set of the kernels: scalar, avx2 or avx512 (default: the best one)
//   --profile           after the timed runs, one more decode pass reports where the time of a token goes
//   --label TEXT        free-text tag stored in the report (e.g. "scalar baseline")
//   --json FILE         also write the report as JSON
//
// The token ids are synthetic (a fixed pseudo-random sequence) and decoding is greedy, so every
// run does exactly the same work. Prompt + decode positions are clamped to the context length.
//
// What is measured:
//   prefill  = forward passes over the prompt (logits only for the last position)
//   decode   = forward pass + greedy sampling per generated token, timed individually
//   memory   = size of the weights file, KV cache, scratch buffers, resident set size
//
// Derived numbers: every decoded token has to read (almost) all weights once, so
//   effective bandwidth = weight bytes / time per token,
//   effective GFLOP/s   = 2 * (weight bytes / 4) / time per token   (one multiply-add per weight).
// If bandwidth is far below what the memory can deliver, the code is compute-bound.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "loader/checkpoint.h"
#include "model/transformer.h"
#include "sampler/sampler.h"
#include "util/process_memory.h"

using namespace geneinfer;
using Clock = std::chrono::steady_clock;

namespace {

[[noreturn]] void usage_error(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::fprintf(stderr,
                 "usage: gi_bench --model M [--prompt-tokens N] [--decode-tokens N] [--runs N]\n"
                 "                [--warmup N] [--max-seq-len N] [--isa NAME] [--threads N] [--weights KIND] [--profile] [--label TEXT] [--json FILE]\n");
    std::exit(2);
}

long long parse_int(const std::string& flag, const std::string& v) {
    char* end = nullptr;
    const long long n = std::strtoll(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0') usage_error("invalid integer for " + flag + ": " + v);
    return n;
}

double percentile(std::vector<double> v, double q) {  // q in [0, 1], nearest-rank
    std::sort(v.begin(), v.end());
    const std::size_t i = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5);
    return v[i];
}

double mib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

std::string json_escape(const std::string& s) {
    std::string out;
    for (const char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else if (static_cast<unsigned char>(c) < 0x20) out += ' ';
        else out += c;
    }
    return out;
}

std::string compiler_string() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, label, json_path, isa_name_arg;
    bool profile = false;
    int prompt_tokens = 32, decode_tokens = 128, runs = 5, warmup = 1, max_seq_len = 0, n_threads = 0;
    std::string weights_arg;

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage_error("missing value for " + flag);
            return argv[++i];
        };
        if (flag == "--model") model_path = value();
        else if (flag == "--prompt-tokens") prompt_tokens = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--decode-tokens") decode_tokens = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--runs") runs = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--warmup") warmup = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--max-seq-len") max_seq_len = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--isa") isa_name_arg = value();
        else if (flag == "--weights") weights_arg = value();
        else if (flag == "--threads") n_threads = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--profile") profile = true;
        else if (flag == "--label") label = value();
        else if (flag == "--json") json_path = value();
        else usage_error("unknown option " + flag);
    }
    if (model_path.empty()) usage_error("--model is required");
    if (!isa_name_arg.empty()) {
        bool known = false;
        for (const geneinfer::kernels::Isa i : {geneinfer::kernels::Isa::Scalar, geneinfer::kernels::Isa::Avx2, geneinfer::kernels::Isa::Avx512}) {
            if (isa_name_arg != geneinfer::kernels::isa_name(i)) continue;
            known = true;
            if (!geneinfer::kernels::set_isa(i)) usage_error("--isa " + isa_name_arg + " is not supported on this machine");
        }
        if (!known) usage_error("--isa must be scalar, avx2 or avx512");
    }
    if (prompt_tokens < 1 || decode_tokens < 1 || runs < 1 || warmup < 0) {
        usage_error("--prompt-tokens, --decode-tokens and --runs must be >= 1, --warmup >= 0");
    }

    try {
        const std::size_t rss_start = current_rss_bytes();
        geneinfer::LoadOptions load_options = geneinfer::LoadOptions::from_environment();
        if (weights_arg == "f32") { load_options.keep_bf16 = false; load_options.int8 = false; }
        else if (weights_arg == "int8") { load_options.int8 = true; }
        else if (weights_arg == "native") { load_options.keep_bf16 = true; load_options.int8 = false; }
        else if (!weights_arg.empty()) usage_error("--weights must be native, f32 or int8");
        const Checkpoint ck = Checkpoint::load(model_path, load_options);
        const Config& cfg = ck.config();
        const std::size_t rss_after_load = current_rss_bytes();

        Transformer model(ck, max_seq_len, n_threads);
        SamplerConfig greedy;
        greedy.temperature = 0.0f;
        Sampler sampler(greedy, cfg.vocab_size);

        // Clamp the workload to the context length.
        const int ctx = model.max_seq_len();
        const int n_prompt = std::min(prompt_tokens, ctx - 1);
        const int n_decode = std::min(decode_tokens, ctx - n_prompt);

        // Fixed pseudo-random prompt (ids >= 3 so no special tokens), starting with BOS.
        std::vector<int> prompt(static_cast<std::size_t>(n_prompt));
        unsigned long long lcg = 12345;
        for (int i = 0; i < n_prompt; ++i) {
            lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
            prompt[static_cast<std::size_t>(i)] = (i == 0) ? 1 : 3 + static_cast<int>((lcg >> 33) % static_cast<unsigned long long>(cfg.vocab_size - 3));
        }

        std::vector<double> prefill_ms_runs, decode_step_ms, decode_total_ms_runs;
        for (int run = -warmup; run < runs; ++run) {
            const bool measured = run >= 0;

            // ---- prefill ----
            const auto t0 = Clock::now();
            const float* logits = nullptr;
            for (int pos = 0; pos < n_prompt; ++pos) {
                logits = model.forward(prompt[static_cast<std::size_t>(pos)], pos, pos == n_prompt - 1);
            }
            int token = sampler.sample(logits);
            const double prefill_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            // ---- decode: every step timed on its own ----
            double decode_total = 0.0;
            for (int step = 0; step < n_decode; ++step) {
                const auto s0 = Clock::now();
                const float* l = model.forward(token, n_prompt + step);
                token = sampler.sample(l);
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - s0).count();
                decode_total += ms;
                if (measured) decode_step_ms.push_back(ms);
            }
            if (measured) {
                prefill_ms_runs.push_back(prefill_ms);
                decode_total_ms_runs.push_back(decode_total);
            }
        }

        // ---- derived numbers ----
        const double prefill_tps = 1000.0 * n_prompt / percentile(prefill_ms_runs, 0.5);
        double total_decode_ms = 0.0;
        for (const double v : decode_total_ms_runs) total_decode_ms += v;
        const double decode_tps_mean = 1000.0 * static_cast<double>(runs) * n_decode / total_decode_ms;
        const double ms_median = percentile(decode_step_ms, 0.5);
        const double ms_p95 = percentile(decode_step_ms, 0.95);
        const double ms_min = *std::min_element(decode_step_ms.begin(), decode_step_ms.end());
        const double ms_max = *std::max_element(decode_step_ms.begin(), decode_step_ms.end());

        // ---- optional profile: one more decode pass with the section timers attached ----
        ForwardProfile prof;
        if (profile) {
            for (int pos = 0; pos < n_prompt; ++pos) {
                model.forward(prompt[static_cast<std::size_t>(pos)], pos, pos == n_prompt - 1);
            }
            model.set_profile(&prof);
            int token = 1;
            for (int step = 0; step < n_decode; ++step) {
                token = sampler.sample(model.forward(token, n_prompt + step));
            }
            model.set_profile(nullptr);
        }

        // Bytes of weights read per decoded token (everything except the unused RoPE tables;
        // an unshared embedding table contributes a single row).
        std::size_t weight_bytes = 0, weight_count = 0;
        const bool int8 = ck.has_int8_weights();
        for (const TensorView& t : ck.tensors()) {
            if (t.name.rfind("freq_cis", 0) == 0) continue;
            const bool embedding = t.name == "token_embedding" || t.name == "model.embed_tokens.weight";
            if (embedding && (int8 || !cfg.shared_classifier)) {
                weight_bytes += static_cast<std::size_t>(cfg.dim) * t.element_bytes();   // one row is looked up
                weight_count += static_cast<std::size_t>(cfg.dim);
            } else if (int8 && t.shape.size() >= 2 && t.name.rfind("rms", 0) != 0) {
                // a matrix: counted below from the loader (its storage may differ from the file)
                weight_count += t.numel;
            } else {
                weight_bytes += t.numel * t.element_bytes();
                weight_count += t.numel;
            }
        }
        if (int8) {
            weight_bytes += ck.matrix_bytes_per_token();
            if (cfg.shared_classifier) weight_count += static_cast<std::size_t>(cfg.vocab_size) * static_cast<std::size_t>(cfg.dim);
        }
        const double gbps = static_cast<double>(weight_bytes) / (ms_median / 1000.0) / 1e9;
        const double gflops = 2.0 * static_cast<double>(weight_count) / (ms_median / 1000.0) / 1e9;   // 2 flops per weight

        const std::size_t kv_bytes = model.kv_cache_bytes();
        const std::size_t scratch = model.scratch_bytes();
        const std::size_t peak = peak_rss_bytes();

#ifdef NDEBUG
        const char* build = "NDEBUG (optimized build)";
#else
        const char* build = "assertions on (Debug build? results are not representative)";
#endif

        std::printf("model: %s%s%s\n", model_path.c_str(), label.empty() ? "" : "   label: ", label.c_str());
        std::printf("  dim %d, layers %d, heads %d/%d, vocab %d, context %d | %s | %s | kernels: %s | threads: %d\n", cfg.dim,
                    cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, ctx, compiler_string().c_str(), build,
                    geneinfer::kernels::isa_name(geneinfer::kernels::active_isa()), model.threads());
        if (ck.has_int8_weights() && !ck.q8_skip_used().empty()) {
            std::printf("  kept in original format (GI_Q8_SKIP / q8_skip.txt): %s\n", ck.q8_skip_used().c_str());
        }
        std::printf("  weights in memory: %s\n", ck.has_int8_weights() ? "int8 matrices, bfloat16 scales per 64 values (quantized at load)"
                                                  : ck.has_bf16_weights() ? "bfloat16 matrices (widened in the kernels)" : "float32");
        std::printf("workload: %d prompt + %d decode positions, %d measured runs (+%d warmup)\n\n",
                    n_prompt, n_decode, runs, warmup);
        std::printf("speed\n");
        std::printf("  prefill                 %10.1f tokens/s   (median of runs)\n", prefill_tps);
        std::printf("  decode                  %10.1f tokens/s   (mean over all runs)\n", decode_tps_mean);
        std::printf("  decode latency per token  median %.3f ms | p95 %.3f ms | min %.3f ms | max %.3f ms\n",
                    ms_median, ms_p95, ms_min, ms_max);
        std::printf("  weights read per token  %10.1f MiB  ->  %.2f GB/s, %.2f GFLOP/s (at median latency)\n",
                    mib(weight_bytes), gbps, gflops);
        if (profile && prof.forward_calls > 0) {
            double total = 0.0, matvec_total = 0.0;
            for (int i = 0; i < ForwardProfile::kSections; ++i) {
                total += prof.seconds[i];
                if (ForwardProfile::is_matvec(i)) matvec_total += prof.seconds[i];
            }
            const double calls = static_cast<double>(prof.forward_calls);
            std::printf("profile (one extra decode pass, %ld tokens, per-token averages)\n", prof.forward_calls);
            for (int i = 0; i < ForwardProfile::kSections; ++i) {
                std::printf("  %-34s %8.3f ms %5.1f %%%s\n", ForwardProfile::name(i), 1000.0 * prof.seconds[i] / calls,
                            100.0 * prof.seconds[i] / total, ForwardProfile::is_matvec(i) ? "   (weights streamed)" : "");
            }
            std::printf("  %-34s %8.3f ms\n", "total", 1000.0 * total / calls);
            std::printf("  weight matrix products %.3f ms (%.1f %%), everything else %.3f ms (%.1f %%)\n",
                        1000.0 * matvec_total / calls, 100.0 * matvec_total / total,
                        1000.0 * (total - matvec_total) / calls, 100.0 * (total - matvec_total) / total);
        }
        std::printf("memory\n");
        std::printf("  weights file (mmap)     %10.1f MiB\n", mib(ck.file_size()));
        std::printf("  KV cache                %10.1f MiB\n", mib(kv_bytes));
        std::printf("  scratch buffers         %10.1f MiB\n", mib(scratch));
        std::printf("  resident, before / after loading the checkpoint   %.1f / %.1f MiB\n", mib(rss_start), mib(rss_after_load));
        std::printf("  peak resident (whole run)                         %.1f MiB\n\n", mib(peak));
        std::printf("| label | prefill tok/s | decode tok/s | median ms/token | GFLOP/s | peak RSS MiB |\n");
        std::printf("|---|---|---|---|---|---|\n");
        std::printf("| %s | %.1f | %.1f | %.3f | %.2f | %.1f |\n", label.empty() ? "-" : label.c_str(),
                    prefill_tps, decode_tps_mean, ms_median, gflops, mib(peak));

        if (!json_path.empty()) {
            std::FILE* f = std::fopen(json_path.c_str(), "w");
            if (!f) throw std::runtime_error("cannot write " + json_path);
            std::fprintf(f,
                "{\n"
                "  \"isa\": \"%s\",\n  \"threads\": %d,\n  \"bf16_weights\": %s,\n  \"int8_weights\": %s,\n  \"label\": \"%s\",\n  \"model\": \"%s\",\n  \"compiler\": \"%s\",\n  \"optimized_build\": %s,\n"
                "  \"dim\": %d,\n  \"n_layers\": %d,\n  \"n_heads\": %d,\n  \"n_kv_heads\": %d,\n  \"vocab_size\": %d,\n  \"context\": %d,\n"
                "  \"prompt_tokens\": %d,\n  \"decode_tokens\": %d,\n  \"runs\": %d,\n  \"warmup\": %d,\n"
                "  \"prefill_tokens_per_s\": %.3f,\n  \"decode_tokens_per_s\": %.3f,\n"
                "  \"decode_ms_median\": %.6f,\n  \"decode_ms_p95\": %.6f,\n  \"decode_ms_min\": %.6f,\n  \"decode_ms_max\": %.6f,\n"
                "  \"weight_bytes_per_token\": %zu,\n  \"effective_gb_per_s\": %.4f,\n  \"effective_gflops\": %.4f,\n"
                "  \"file_bytes\": %zu,\n  \"kv_cache_bytes\": %zu,\n  \"scratch_bytes\": %zu,\n"
                "  \"rss_before_load_bytes\": %zu,\n  \"rss_after_load_bytes\": %zu,\n  \"peak_rss_bytes\": %zu\n"
                "}\n",
                geneinfer::kernels::isa_name(geneinfer::kernels::active_isa()), model.threads(), ck.has_bf16_weights() ? "true" : "false", ck.has_int8_weights() ? "true" : "false",
                json_escape(label).c_str(), json_escape(model_path).c_str(), json_escape(compiler_string()).c_str(),
#ifdef NDEBUG
                "true",
#else
                "false",
#endif
                cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, ctx, n_prompt, n_decode, runs, warmup,
                prefill_tps, decode_tps_mean, ms_median, ms_p95, ms_min, ms_max, weight_bytes, gbps, gflops,
                ck.file_size(), kv_bytes, scratch, rss_start, rss_after_load, peak);
            std::fclose(f);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
