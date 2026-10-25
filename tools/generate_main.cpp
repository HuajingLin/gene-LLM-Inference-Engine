// generate_main.cpp - text generation from the command line.
//
//   gi_generate --model <model.bin> --tokenizer <tokenizer.bin> [options]
//
// Options:
//   --prompt TEXT        text to continue (default: empty, i.e. start from BOS)
//   --steps N            maximum number of positions including the prompt (default: model context)
//   --temperature F      0 = greedy (default 0.8)
//   --top-k N            keep only the N most likely tokens (default 0 = off)
//   --top-p F            nucleus sampling threshold in (0, 1] (default 0.9; 1 = off)
//   --seed N             random seed (default: derived from the clock; printed in the stats)
//   --no-echo            do not print the prompt, only the generated text
//   --quiet              do not print the statistics line (written to stderr)
//   --dump-tokens FILE   write prompt token ids (line 1) and generated token ids (line 2)
//
// The text goes to stdout as it is generated; statistics go to stderr.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "loader/checkpoint.h"
#include "model/transformer.h"
#include "runtime/generator.h"
#include "sampler/sampler.h"
#include "tokenizer/tokenizer.h"

using namespace geneinfer;

namespace {

[[noreturn]] void usage_error(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::fprintf(stderr,
                 "usage: gi_generate --model M --tokenizer T [--prompt TEXT] [--steps N]\n"
                 "                   [--temperature F] [--top-k N] [--top-p F] [--seed N]\n"
                 "                   [--no-echo] [--quiet] [--dump-tokens FILE]\n");
    std::exit(2);
}

double parse_double(const std::string& flag, const std::string& v) {
    char* end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0') usage_error("invalid number for " + flag + ": " + v);
    return d;
}

long long parse_int(const std::string& flag, const std::string& v) {
    char* end = nullptr;
    const long long n = std::strtoll(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0') usage_error("invalid integer for " + flag + ": " + v);
    return n;
}

std::uint64_t parse_seed(const std::string& v) {
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0') usage_error("invalid seed: " + v);
    return static_cast<std::uint64_t>(n);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, tokenizer_path, prompt, dump_path;
    int steps = 0;
    bool echo = true, quiet = false, seed_given = false;
    SamplerConfig scfg;
    scfg.temperature = 0.8f;
    scfg.top_p = 0.9f;

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage_error("missing value for " + flag);
            return argv[++i];
        };
        if (flag == "--model") model_path = value();
        else if (flag == "--tokenizer") tokenizer_path = value();
        else if (flag == "--prompt") prompt = value();
        else if (flag == "--steps") steps = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--temperature") scfg.temperature = static_cast<float>(parse_double(flag, value()));
        else if (flag == "--top-k") scfg.top_k = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--top-p") scfg.top_p = static_cast<float>(parse_double(flag, value()));
        else if (flag == "--seed") { scfg.seed = parse_seed(value()); seed_given = true; }
        else if (flag == "--dump-tokens") dump_path = value();
        else if (flag == "--no-echo") echo = false;
        else if (flag == "--quiet") quiet = true;
        else usage_error("unknown option " + flag);
    }
    if (model_path.empty() || tokenizer_path.empty()) usage_error("--model and --tokenizer are required");
    if (!seed_given) {
        scfg.seed = static_cast<std::uint64_t>(
            std::chrono::system_clock::now().time_since_epoch().count());
    }

    try {
        const Checkpoint ck = Checkpoint::load(model_path);
        const Tokenizer tok = Tokenizer::load(tokenizer_path, ck.config().vocab_size);
        Transformer model(ck);
        Sampler sampler(scfg, ck.config().vocab_size);

        const std::vector<int> ids = tok.encode(prompt, /*bos=*/true, /*eos=*/false);
        if (echo) {
            const std::string text = tok.decode(ids);
            std::fwrite(text.data(), 1, text.size(), stdout);
            std::fflush(stdout);
        }

        const GenerateResult r = generate(model, tok, sampler, ids, steps, [](const std::string& piece) {
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);  // stream: show every token as soon as it exists
        });

        if (!dump_path.empty()) {
            std::ofstream out(dump_path);
            if (!out) throw std::runtime_error("cannot write " + dump_path);
            for (std::size_t i = 0; i < r.prompt_tokens.size(); ++i) out << (i ? " " : "") << r.prompt_tokens[i];
            out << "\n";
            for (std::size_t i = 0; i < r.generated_tokens.size(); ++i) out << (i ? " " : "") << r.generated_tokens[i];
            out << "\n";
        }

        if (!quiet) {
            const double prefill_tps = r.prefill_ms > 0 ? 1000.0 * static_cast<double>(r.prompt_tokens.size()) / r.prefill_ms : 0.0;
            const double decode_tps = r.decode_ms > 0 ? 1000.0 * r.decode_steps / r.decode_ms : 0.0;
            std::fprintf(stderr,
                         "\n[prompt: %zu tokens, prefill %.1f ms (%.1f tok/s) | generated: %zu tokens, "
                         "decode %.1f tok/s | temperature %.2f, top-k %d, top-p %.2f, seed %llu%s]\n",
                         r.prompt_tokens.size(), r.prefill_ms, prefill_tps, r.generated_tokens.size(),
                         decode_tps, static_cast<double>(scfg.temperature), scfg.top_k,
                         static_cast<double>(scfg.top_p), static_cast<unsigned long long>(scfg.seed),
                         r.stopped_by_terminator ? ", ended by end-of-sequence token" : "");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
