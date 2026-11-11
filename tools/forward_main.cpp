// forward_main.cpp - feeds a token sequence through the model and dumps the results.
//
//   gi_forward <model> <out_dir> [--max-seq-len N] [--prefill P] [--head H] [--max-batch B] <token_id> [<token_id> ...]
//
// <model> is a llama2.c file or a Hugging Face model directory. --max-seq-len caps the context
// (and with it the KV cache memory); it must be at least the number of tokens.
//
// Token i is processed at position i. Files written to <out_dir> (raw float32):
//   logits.f32                 [num_tokens, vocab]  logits after every position
//   embedding.f32              activations of the LAST position (for layer-by-layer debugging):
//   layerNN_attn_out.f32       residual stream after the attention block of layer NN
//   layerNN_ffn_out.f32        residual stream after the feed-forward block of layer NN
//   final_norm.f32             after the final RMSNorm
// --prefill P feeds the first P tokens through forward_batch() (in pieces of at most B tokens, --max-batch) and the
// rest one by one; with --head H the first H of those P tokens are fed one by one before the batch, which then
// starts at position H (tests a batch on top of an existing KV cache). The logits file then holds the rows of positions P-1 .. n-1 (batch processing only returns
// the logits of its last token), and no activation files are written.
// It also prints the average time per forward() call.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "model/transformer.h"

using namespace geneinfer;
namespace fs = std::filesystem;

namespace {

void write_floats(const fs::path& path, const float* data, std::size_t count) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model> <out_dir> [--max-seq-len N] <token_id> [<token_id> ...]\n", argv[0]);
        return 2;
    }

    try {
        const Checkpoint ck = Checkpoint::load(argv[1]);
        const fs::path out_dir = argv[2];
        fs::create_directories(out_dir);

        std::vector<int> tokens;
        int max_seq_len = 0, prefill = 0, max_batch = 0, head = 0, pos_shift = 0;
        bool traced = false;
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--max-seq-len" || a == "--prefill" || a == "--max-batch" || a == "--head" || a == "--pos-shift") {
                if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
                const int v = std::atoi(argv[++i]);
                (a == "--max-seq-len" ? max_seq_len : a == "--prefill" ? prefill : a == "--head" ? head : a == "--pos-shift" ? pos_shift : max_batch) = v;
            } else if (a == "--traced") {
                traced = true;   // test hook: a trace callback makes forward_batch take its token-by-token path
            } else {
                tokens.push_back(std::atoi(argv[i]));
            }
        }
        const int n = static_cast<int>(tokens.size());
        if (n == 0) throw std::runtime_error("no token ids given");

        if (head < 0 || (head > 0 && head >= prefill)) throw std::runtime_error("--head must be smaller than --prefill");
        if (prefill < 0 || prefill > n) throw std::runtime_error("--prefill must be between 0 and the number of tokens");
        Transformer model(ck, max_seq_len, 0, max_batch);
        if (n > model.max_seq_len()) {
            throw std::runtime_error("sequence of " + std::to_string(n) +
                                     " tokens exceeds the context length " +
                                     std::to_string(model.max_seq_len()));
        }

        bool capture = false;  // only the last position is traced
        if (prefill == 0 || traced) model.set_trace([&](const char* tag, int layer, const float* data, int size) {
            if (!capture) return;
            std::string name = tag;
            if (layer >= 0) {
                char buf[32];
                std::snprintf(buf, sizeof buf, "layer%02d_", layer);
                name = buf + name;
            }
            write_floats(out_dir / (name + ".f32"), data, static_cast<std::size_t>(size));
        });

        const std::size_t vocab = static_cast<std::size_t>(ck.config().vocab_size);
        std::ofstream logits_file(out_dir / "logits.f32", std::ios::binary);
        if (!logits_file) throw std::runtime_error("cannot write logits.f32");

        const auto start = std::chrono::steady_clock::now();
        int first = 0;
        if (prefill > 0) {
            for (int pos = 0; pos < head; ++pos) model.forward(tokens[static_cast<std::size_t>(pos)], pos, false);
            const float* logits = model.forward_batch(tokens.data() + head, prefill - head, head + pos_shift, true);
            logits_file.write(reinterpret_cast<const char*>(logits), static_cast<std::streamsize>(vocab * sizeof(float)));
            first = prefill;
        }
        for (int pos = first; pos < n; ++pos) {
            capture = (pos == n - 1);
            const float* logits = model.forward(tokens[static_cast<std::size_t>(pos)], pos);
            logits_file.write(reinterpret_cast<const char*>(logits),
                              static_cast<std::streamsize>(vocab * sizeof(float)));
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start).count();

        std::printf("positions=%d vocab=%zu avg_ms_per_token=%.3f\n", n, vocab, ms / n);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
