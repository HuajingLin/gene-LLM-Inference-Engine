// quant_dump_main.cpp - writes the int8 matrices produced by the loader (tests/verify_quantize.py checks them).
//
//   gi_quant_dump <model> <out_dir>
//
// Loads the model with int8 quantization and writes, for every layer matrix and for the classifier:
//   layer<L>_<name>.q   raw int8 values [rows, cols]            (name: wq wk wv wo w1 w2 w3)
//   layer<L>_<name>.s   raw 16-bit bfloat16 scales [rows, ceil(cols / 64)]
//   wcls.q / wcls.s     the classifier
// and prints "owned_bytes native <n> int8 <m>" (memory owned by the loader without and with quantization).
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

#include "common/q8.h"
#include "loader/checkpoint.h"

using namespace geneinfer;
namespace fs = std::filesystem;

namespace {

void write_bytes(const fs::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

void dump(const fs::path& dir, const std::string& name, const Mat& m, int rows, int cols) {
    if (!m.is_q8()) throw std::runtime_error(name + " is not quantized");
    write_bytes(dir / (name + ".q"), m.q8, static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    write_bytes(dir / (name + ".s"), m.q8_scale,
                static_cast<std::size_t>(rows) * static_cast<std::size_t>(q8_groups(cols)) * sizeof(std::uint16_t));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model> <out_dir>\n", argv[0]);
        return 2;
    }
    try {
        LoadOptions options = LoadOptions::from_environment();
        options.int8 = false;
        const std::size_t owned_native = Checkpoint::load(argv[1], options).owned_bytes();
        options.int8 = true;
        const Checkpoint ck = Checkpoint::load(argv[1], options);
        const Config& c = ck.config();
        const fs::path dir = argv[2];
        fs::create_directories(dir);
        const Weights& w = ck.weights();
        for (int l = 0; l < c.n_layers; ++l) {
            const LayerWeights& lw = w.layers[static_cast<std::size_t>(l)];
            const std::string p = "layer" + std::to_string(l) + "_";
            dump(dir, p + "wq", lw.wq, c.attn_dim(), c.dim);
            dump(dir, p + "wk", lw.wk, c.kv_dim(), c.dim);
            dump(dir, p + "wv", lw.wv, c.kv_dim(), c.dim);
            dump(dir, p + "wo", lw.wo, c.dim, c.attn_dim());
            dump(dir, p + "w1", lw.w1, c.hidden_dim, c.dim);
            dump(dir, p + "w2", lw.w2, c.dim, c.hidden_dim);
            dump(dir, p + "w3", lw.w3, c.hidden_dim, c.dim);
        }
        dump(dir, "wcls", w.wcls, c.vocab_size, c.dim);
        std::printf("owned_bytes native %zu int8 %zu\n", owned_native, ck.owned_bytes());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
