// inspect_main.cpp — Loads a checkpoint and prints the configuration along with a fingerprint 
// for each tensor (offset, number of elements, first 4 values, and sum).

// The output consists of tab-separated lines, designed to be both human-readable and suitable 
// for element-wise comparison against results from NumPy via `tests/verify_loader.py`.
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

            double sum = 0.0;  // Use `double` for accumulation to avoid precision errors—inherent to `float`—from interfering with comparisons.
            for (std::size_t i = 0; i < t.numel; ++i) sum += static_cast<double>(t.data[i]);

            std::printf("tensor\t%s\t%zu\t%zu\t%s\t%s\t%.17g\n", t.name.c_str(), t.byte_offset,
                        t.numel, shape.c_str(), first4.c_str(), sum);
        }

        // When sharing the classification head, `wcls` should point to the same memory as `token_embedding`.
        std::printf("config\twcls_is_embedding\t%d\n",
                    ck.weights().wcls == ck.weights().token_embedding ? 1 : 0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
