// inspect_main.cpp - loads a checkpoint and prints the config plus a fingerprint of every tensor
// (offset / element count / first 4 values / sum). The output is tab-separated lines: readable for humans
// and easy for tests/verify_loader.py to compare against NumPy.
#include <cstdio>
#include <exception>
#include <string>

#include "loader/checkpoint.h"

using namespace geneinfer;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        return 2;
    }

    try {
        const Checkpoint ck = Checkpoint::load(argv[1]);
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
        std::printf("config\tattn_dim\t%d\n", c.attn_dim());
        std::printf("config\thead_dim\t%d\n", c.head_dim);
        std::printf("config\trope_theta\t%.9g\n", static_cast<double>(c.rope_theta));
        std::printf("config\trms_norm_eps\t%.9g\n", static_cast<double>(c.rms_norm_eps));
        std::printf("config\trope_style\t%s\n", c.rope_style == RopeStyle::HalfSplit ? "half_split" : "interleaved");

        // tensor <name> <byte_offset> <numel> <shape> <first4> <sum>
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
                std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(t.data[i]));
                if (i) first4 += ",";
                first4 += buf;
            }

            double sum = 0.0;  // accumulate in double so float rounding does not disturb the comparison
            for (std::size_t i = 0; i < t.numel; ++i) sum += static_cast<double>(t.data[i]);

            std::printf("tensor\t%s\t%zu\t%zu\t%s\t%s\t%.17g\n", t.name.c_str(), t.byte_offset,
                        t.numel, shape.c_str(), first4.c_str(), sum);
        }

        // role <layer> <member> <tensor name> <element offset inside that tensor>
        // Shows which file tensor ended up in which slot of the weights structure (layer -1 = global).
        auto locate = [&](const char* member, int layer, const float* p) {
            for (const TensorView& t : ck.tensors()) {
                if (p >= t.data && p < t.data + t.numel) {
                    std::printf("role\t%d\t%s\t%s\t%td\n", layer, member, t.name.c_str(), p - t.data);
                    return;
                }
            }
            std::printf("role\t%d\t%s\t<not inside any tensor>\t0\n", layer, member);
        };
        const Weights& wt = ck.weights();
        locate("token_embedding", -1, wt.token_embedding);
        locate("rms_final", -1, wt.rms_final);
        locate("wcls", -1, wt.wcls);
        for (std::size_t l = 0; l < wt.layers.size(); ++l) {
            const LayerWeights& lw = wt.layers[l];
            const int li = static_cast<int>(l);
            locate("rms_att", li, lw.rms_att);
            locate("wq", li, lw.wq);
            locate("wk", li, lw.wk);
            locate("wv", li, lw.wv);
            locate("wo", li, lw.wo);
            locate("rms_ffn", li, lw.rms_ffn);
            locate("w1", li, lw.w1);
            locate("w2", li, lw.w2);
            locate("w3", li, lw.w3);
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
