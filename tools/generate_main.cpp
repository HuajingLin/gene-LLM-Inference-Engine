// generate_main.cpp - text generation from the command line.
//
//   gi_generate --model <model.bin> --tokenizer <tokenizer.bin> [options]     (llama2.c checkpoint)
//   gi_generate --model <model directory> [options]                           (Hugging Face Llama)
//
// A directory given to --model holds config.json, model.safetensors and tokenizer.json; the tokenizer is
// read from the same directory unless --tokenizer names another one.
//
// Options:
//   --prompt TEXT        text to continue (default: empty, i.e. start from BOS)
//   --steps N            maximum number of positions including the prompt (default: model context)
//   --weights KIND       native (default), f32 or int8: how the weight matrices are held in memory
//   --threads N          threads sharing the matrix products (default: GI_THREADS or 1; same output)
//   --max-seq-len N      cap the context length, i.e. the KV cache memory (default: the model's)
//   --temperature F      0 = greedy (default 0.8)
//   --top-k N            keep only the N most likely tokens (default 0 = off)
//   --top-p F            nucleus sampling threshold in (0, 1] (default 0.9; 1 = off)
//   --seed N             random seed (default: derived from the clock; printed in the stats)
//   --bos                Hugging Face models only: put the BOS token in front of the prompt
//                        (like transformers, nothing is added by default)
//   --no-echo            do not print the prompt, only the generated text
//   --quiet              do not print the statistics line (written to stderr)
//   --dump-tokens FILE   write prompt token ids (line 1) and generated token ids (line 2)
//
// The text goes to stdout as it is generated; statistics go to stderr.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "loader/checkpoint.h"
#include "model/transformer.h"
#include "runtime/generator.h"
#include "sampler/sampler.h"
#include "tokenizer/text_tokenizer.h"

using namespace geneinfer;

namespace {

[[noreturn]] void usage_error(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::fprintf(stderr,
                 "usage: gi_generate --model M [--tokenizer T] [--prompt TEXT] [--steps N] [--max-seq-len N] [--threads N] [--weights KIND]\n"
                 "                   [--temperature F] [--top-k N] [--top-p F] [--seed N]\n"
                 "                   [--bos] [--no-echo] [--quiet] [--dump-tokens FILE]\n"
                 "  M is a llama2.c .bin file (then --tokenizer is required) or a Hugging Face model directory.\n");
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
    int steps = 0, max_seq_len = 0, n_threads = 0;
    std::string weights_arg;
    bool echo = true, quiet = false, seed_given = false, bos = false;
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
        else if (flag == "--max-seq-len") max_seq_len = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--weights") weights_arg = value();
        else if (flag == "--threads") n_threads = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--temperature") scfg.temperature = static_cast<float>(parse_double(flag, value()));
        else if (flag == "--top-k") scfg.top_k = static_cast<int>(parse_int(flag, value()));
        else if (flag == "--top-p") scfg.top_p = static_cast<float>(parse_double(flag, value()));
        else if (flag == "--seed") { scfg.seed = parse_seed(value()); seed_given = true; }
        else if (flag == "--dump-tokens") dump_path = value();
        else if (flag == "--bos") bos = true;
        else if (flag == "--no-echo") echo = false;
        else if (flag == "--quiet") quiet = true;
        else usage_error("unknown option " + flag);
    }
    if (model_path.empty()) usage_error("--model is required");
    if (!seed_given) {
        scfg.seed = static_cast<std::uint64_t>(
            std::chrono::system_clock::now().time_since_epoch().count());
    }

    try {
        std::error_code ec;
        const bool is_hf = std::filesystem::is_directory(std::filesystem::path(model_path), ec);
        LoadOptions load_options = LoadOptions::from_environment();
        if (weights_arg == "f32") { load_options.keep_bf16 = false; load_options.int8 = false; }
        else if (weights_arg == "int8") { load_options.int8 = true; }
        else if (weights_arg == "native") { load_options.keep_bf16 = true; load_options.int8 = false; }
        else if (!weights_arg.empty()) { std::fprintf(stderr, "error: --weights must be native, f32 or int8\n"); return 2; }
        const Checkpoint ck = Checkpoint::load(model_path, load_options);
        std::unique_ptr<TextTokenizer> tok_ptr;
        if (is_hf) {
            tok_ptr = load_hf_tokenizer(tokenizer_path.empty() ? model_path : tokenizer_path, bos);
        } else {
            if (tokenizer_path.empty()) usage_error("a llama2.c model needs --tokenizer");
            if (bos) usage_error("--bos only applies to Hugging Face models (llama2.c always starts with BOS)");
            tok_ptr = load_llama2c_tokenizer(tokenizer_path, ck.config().vocab_size);
        }
        const TextTokenizer& tok = *tok_ptr;
        if (tok.bos_by_default_not_added() && !quiet) {
            std::fprintf(stderr, "note: this tokenizer puts the BOS token in front of prompts by default (Hugging Face does); "
                                 "add --bos to do the same\n");
        }
        if (tok.vocab_size() > ck.config().vocab_size) {
            throw std::runtime_error("the tokenizer has " + std::to_string(tok.vocab_size()) +
                                     " tokens but the model only " + std::to_string(ck.config().vocab_size) +
                                     ": they do not belong together");
        }
        Transformer model(ck, max_seq_len, n_threads);
        Sampler sampler(scfg, ck.config().vocab_size);

        const std::vector<int> ids = tok.encode_prompt(prompt);
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
                         "\n[%s | prompt: %zu tokens, prefill %.1f ms (%.1f tok/s) | generated: %zu tokens, "
                         "decode %.1f tok/s | temperature %.2f, top-k %d, top-p %.2f, seed %llu%s]\n",
                         tok.kind().c_str(), r.prompt_tokens.size(), r.prefill_ms, prefill_tps, r.generated_tokens.size(),
                         decode_tps, static_cast<double>(scfg.temperature), scfg.top_k,
                         static_cast<double>(scfg.top_p), static_cast<unsigned long long>(scfg.seed),
                         r.stopped_by_end_token ? ", ended by an end token" : "");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
