// inspect_main.cpp - loads a checkpoint and prints the config plus a fingerprint of every tensor
// (offset / element count / first 4 values / sum). The output is tab-separated lines: readable for humans
// and easy for tests/verify_loader.py to compare against NumPy.
#include <cstddef>
#include <cstdio>
#include <exception>
#include <string>

#include "loader/checkpoint.h"

using namespace geneinfer;

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--f32-weights")) {
        std::fprintf(stderr, "usage: %s <model.bin | model directory> [--f32-weights]\n", argv[0]);
        return 2;
    }

    try {
        LoadOptions options = LoadOptions::from_environment();
        if (argc == 3) options.keep_bf16 = false;   // expand bfloat16 matrices to float32 at load time
        const Checkpoint ck = Checkpoint::load(argv[1], options);
        const Config& c = ck.config();

        std::printf("config\tdim\t%d\n", c.dim);
        std::printf("config\thidden_dim\t%d\n", c.hidden_dim);
        std::printf("config\tn_layers\t%d\n", c.n_layers);
        std::printf("config\tn_heads\t%d\n", c.n_heads);
        std::printf("config\tn_kv_heads\t%d\n", c.n_kv_heads);
        std::printf("config\tvocab_size\t%d\n", c.vocab_size);
        std::printf("config\tseq_len\t%d\n", c.seq_len);
        std::printf("config\tshared_classifier\t%d\n", c.shared_classifier ? 1 : 0);
        std::printf("config\thead_size\t%d\n", c.head_size());
        std::printf("config\tkv_dim\t%d\n", c.kv_dim());
        std::printf("config\tfile_size\t%zu\n", ck.file_size());
        std::printf("config\towned_bytes\t%zu\n", ck.owned_bytes());
        std::printf("config\tattn_dim\t%d\n", c.attn_dim());
        std::printf("config\thead_dim\t%d\n", c.head_dim);
        std::printf("config\trope_theta\t%.9g\n", static_cast<double>(c.rope_theta));
        std::printf("config\trms_norm_eps\t%.9g\n", static_cast<double>(c.rms_norm_eps));
        std::printf("config\trope_style\t%s\n", c.rope_style == RopeStyle::HalfSplit ? "half_split" : "interleaved");

        // tensor <name> <byte_offset> <numel> <shape> <first4> <sum> <storage: f32|bf16>
        for (const TensorView& t : ck.tensors()) {
            std::string shape = "[";
            for (std::size_t i = 0; i < t.shape.size(); ++i) {
                if (i) shape += ",";
                shape += std::to_string(t.shape[i]);
            }
            shape += "]";

            std::string first4;
            for (std::size_t i = 0; i < 4 && i < t.numel; ++i) {
                char buf[32];
                std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(t.value(i)));
                if (i) first4 += ",";
                first4 += buf;
            }

            double sum = 0.0;  // accumulate in double so float rounding does not disturb the comparison
            for (std::size_t i = 0; i < t.numel; ++i) sum += static_cast<double>(t.value(i));

            std::printf("tensor\t%s\t%zu\t%zu\t%s\t%s\t%.17g\t%s\n", t.name.c_str(), t.byte_offset,
                        t.numel, shape.c_str(), first4.c_str(), sum, t.storage == Storage::BF16 ? "bf16" : "f32");
        }

        // role <layer> <member> <tensor name> <element offset inside that tensor>
        // Shows which file tensor ended up in which slot of the weights structure (layer -1 = global).
        auto locate = [&](const char* member, int layer, const void* p, Storage storage) {
            const char* addr = static_cast<const char*>(p);
            for (const TensorView& t : ck.tensors()) {
                const char* begin = static_cast<const char*>(t.address());
                if (t.storage == storage && addr >= begin && addr < begin + t.numel * t.element_bytes()) {
                    std::printf("role\t%d\t%s\t%s\t%td\n", layer, member, t.name.c_str(),
                                (addr - begin) / static_cast<std::ptrdiff_t>(t.element_bytes()));
                    return;
                }
            }
            std::printf("role\t%d\t%s\t<not inside any tensor>\t0\n", layer, member);
        };
        auto locate_mat = [&](const char* member, int layer, const Mat& m) {
            locate(member, layer, m.address(), m.is_bf16() ? Storage::BF16 : Storage::F32);
        };
        const Weights& wt = ck.weights();
        locate_mat("token_embedding", -1, wt.token_embedding);
        locate("rms_final", -1, wt.rms_final, Storage::F32);
        locate_mat("wcls", -1, wt.wcls);
        for (std::size_t l = 0; l < wt.layers.size(); ++l) {
            const LayerWeights& lw = wt.layers[l];
            const int li = static_cast<int>(l);
            locate("rms_att", li, lw.rms_att, Storage::F32);
            locate_mat("wq", li, lw.wq);
            locate_mat("wk", li, lw.wk);
            locate_mat("wv", li, lw.wv);
            locate_mat("wo", li, lw.wo);
            locate("rms_ffn", li, lw.rms_ffn, Storage::F32);
            locate_mat("w1", li, lw.w1);
            locate_mat("w2", li, lw.w2);
            locate_mat("w3", li, lw.w3);
        }

        // with a shared classifier, wcls must be the same memory as token_embedding
        std::printf("config\twcls_is_embedding\t%d\n",
                    ck.weights().wcls == ck.weights().token_embedding ? 1 : 0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
