// forward_main.cpp - feeds a token sequence through the model and dumps the results.
//
//   gi_forward <model.bin> <out_dir> <token_id> [<token_id> ...]
//
// Token i is processed at position i. Files written to <out_dir> (raw float32):
//   logits.f32                 [num_tokens, vocab]  logits after every position
//   embedding.f32              activations of the LAST position (for layer-by-layer debugging):
//   layerNN_attn_out.f32       residual stream after the attention block of layer NN
//   layerNN_ffn_out.f32        residual stream after the feed-forward block of layer NN
//   final_norm.f32             after the final RMSNorm
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
        std::fprintf(stderr, "usage: %s <model.bin> <out_dir> <token_id> [<token_id> ...]\n", argv[0]);
        return 2;
    }

    try {
        const Checkpoint ck = Checkpoint::load(argv[1]);
        const fs::path out_dir = argv[2];
        fs::create_directories(out_dir);

        std::vector<int> tokens;
        for (int i = 3; i < argc; ++i) tokens.push_back(std::atoi(argv[i]));
        const int n = static_cast<int>(tokens.size());
        if (n > ck.config().seq_len) {
            throw std::runtime_error("sequence of " + std::to_string(n) +
                                     " tokens exceeds the model context length " +
                                     std::to_string(ck.config().seq_len));
        }

        Transformer model(ck);

        bool capture = false;  // only the last position is traced
        model.set_trace([&](const char* tag, int layer, const float* data, int size) {
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
        for (int pos = 0; pos < n; ++pos) {
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
