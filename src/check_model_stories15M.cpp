#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>

// Corresponds to the `Config` structure in `llama2.c` (7 `int`s, totaling 28 bytes)
struct Config {
    int dim;        // The dimensionality of all vectors
    int hidden_dim; // Hidden layer dimension
    int n_layers;   // Number of Layers
    int n_heads;    // Number of Attention Heads
    int n_kv_heads; // Number of Key-Value Attention Heads
    int vocab_size; // Vocabulary Size
    int seq_len;    // Maximum Sequence Length
};

// Define a `Weights` structure to store pointers to the model's weights.
struct Weights {
    const float* token_embedding_table; // [vocab_size, dim]
    const float* rms_att_weight;        // [n_layers, dim]
    const float* wq;                    // [n_layers, dim, (n_heads * head_size)]
    const float* wk;                    // [n_layers, dim, (n_kv_heads * head_size)]
    const float* wv;                    // [n_layers, dim, (n_kv_heads * head_size)]
    const float* wo;                    // [n_layers, (n_heads * head_size), dim]
    const float* rms_ffn_weight;        // [n_layers, dim]
    const float* w1;                    // [n_layers, dim, hidden_dim]
    const float* w2;                    // [n_layers, hidden_dim, dim]
    const float* w3;                    // [n_layers, dim, hidden_dim]
    const float* rms_final_weight;      // [dim]
    const float* freq_cis_real;         // [seq_len, head_size / 2]
    const float* freq_cis_imag;         // [seq_len, head_size / 2]
    const float* wcls;                  // [vocab_size, dim] (只有 shared_weights = false 時存在)
};

int main(int argc, char* argv[]) {
    std::string filename = "models/stories15M/stories15M.bin";
    if (argc > 1) {
        filename = argv[1];
    }

    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Unable to open file:" << filename << std::endl;
        return 1;
    }

    //Get actual file size (bytes)
    std::streamsize actual_file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 1. read Header (28 bytes)
    Config config;
    if (!file.read(reinterpret_cast<char*>(&config), sizeof(Config))) {
        std::cerr << "Failed to read Header!" << std::endl;
        return 1;
    }

    if (config.dim % config.n_heads != 0) {
        std::cerr << "Error: dim is not divisible by n_heads" << std::endl;
        return 1;
    }
    std::cout << "=== step 0: read Header ===" << std::endl;
    std::cout << "dim        : " << config.dim << std::endl;
    std::cout << "hidden_dim : " << config.hidden_dim << std::endl;
    std::cout << "n_layers   : " << config.n_layers << std::endl;
    std::cout << "n_heads    : " << config.n_heads << std::endl;
    std::cout << "n_kv_heads : " << config.n_kv_heads << std::endl;
    std::cout << "vocab_size : " << config.vocab_size << std::endl;
    std::cout << "seq_len    : " << config.seq_len << std::endl;
    std::cout << "------------------------" << std::endl;

    int abs_vocab_size = std::abs(config.vocab_size);
    bool shared_weights = (config.vocab_size > 0);
    int head_size = config.dim / config.n_heads;

    // Calculate the total number of float elements
    std::streamsize raw_data_bytes = actual_file_size - sizeof(Config);
    size_t total_floats = raw_data_bytes / sizeof(float);

    std::cout << "=== step 1: read all weights into memory ===" << std::endl;
    std::cout << "weight block size: " << raw_data_bytes << " bytes (" 
              << (raw_data_bytes / (1024.0 * 1024.0)) << " MB), total elements: " 
              << total_floats << " float32" << std::endl;

    //Read the weights following the header directly into a `vector<float>`.
    std::vector<float> weight_buffer(total_floats);
    if (!file.read(reinterpret_cast<char*>(weight_buffer.data()), raw_data_bytes)) {
        std::cerr << "Failed to read weight data!" << std::endl;
        return 1;
    }

    //2. Map the weights metrics and perform an offset check.
    std::cout << "\n=== step 2: map weight pointers & offset checks ===" << std::endl;
    Weights w;
    const float* ptr = weight_buffer.data();

    // Lambda function to advance the pointer and return the current position
    auto advance_ptr = [&ptr](size_t count) -> const float* {
        const float* current = ptr;
        ptr += count;
        return current;
    };

    w.token_embedding_table = advance_ptr((size_t)abs_vocab_size * config.dim);
    w.rms_att_weight        = advance_ptr((size_t)config.n_layers * config.dim);
    w.wq                    = advance_ptr((size_t)config.n_layers * config.dim * (config.n_heads * head_size));
    w.wk                    = advance_ptr((size_t)config.n_layers * config.dim * (config.n_kv_heads * head_size));
    w.wv                    = advance_ptr((size_t)config.n_layers * config.dim * (config.n_kv_heads * head_size));
    w.wo                    = advance_ptr((size_t)config.n_layers * config.dim * config.dim);
    w.rms_ffn_weight        = advance_ptr((size_t)config.n_layers * config.dim);
    w.w1                    = advance_ptr((size_t)config.n_layers * config.dim * config.hidden_dim);
    w.w2                    = advance_ptr((size_t)config.n_layers * config.hidden_dim * config.dim);
    w.w3                    = advance_ptr((size_t)config.n_layers * config.dim * config.hidden_dim);
    w.rms_final_weight      = advance_ptr((size_t)config.dim);
    w.freq_cis_real         = advance_ptr((size_t)config.seq_len * (head_size / 2));
    w.freq_cis_imag         = advance_ptr((size_t)config.seq_len * (head_size / 2));
    w.wcls                  = shared_weights ? w.token_embedding_table : advance_ptr((size_t)abs_vocab_size * config.dim);

    const float* buffer_end = weight_buffer.data() + weight_buffer.size();

    // Check 1: Advance the pointer to the end and see if it matches the buffer end
    std::cout << "Buffer start address: " << weight_buffer.data() << std::endl;
    std::cout << "Pointer after advancing: " << ptr << std::endl;
    std::cout << "Expected buffer end: " << buffer_end << std::endl;

    if (ptr == buffer_end) {
        std::cout << "[Check 1 Passed] The pointer advancement aligns perfectly with the end of the memory block!" << std::endl;
    } else {
        std::cerr << "[Check 1 Failed] Pointer misalignment! Difference: " << (ptr - buffer_end) << " float elements." << std::endl;
        return 1;
    }

    // 3. Perform numerical checks (NaN / Inf) and verify the value range of RMS weights.
    std::cout << "\n=== step 3: Sanity Check (NaN/Inf detection & RMS weight sampling) ===" << std::endl;

    // Check 2: Scan all weights, ensuring no NaN or Inf values
    size_t nan_count = 0;
    size_t inf_count = 0;
    for (size_t i = 0; i < weight_buffer.size(); ++i) {
        if (std::isnan(weight_buffer[i])) nan_count++;
        if (std::isinf(weight_buffer[i])) inf_count++;
    }

    if (nan_count == 0 && inf_count == 0) {
        std::cout << "[Check 2 Passed] Scanned " << weight_buffer.size() << " weights, no NaN or Inf found." << std::endl;
    } else {
        std::cerr << "[Check 2 Failed] Abnormal values detected! NaN count: " << nan_count << ", Inf count: " << inf_count << std::endl;
        return 1;
    }

    // Check 3: Print out the first few values of rms_att and rms_final
    std::cout << "\n[Check 3] Print out the value range of RMSNorm weights:" << std::endl;

    // Scan the entire rms_final_weight block (total config.dim values)
    float rms_min = std::numeric_limits<float>::max();
    float rms_max = std::numeric_limits<float>::lowest();
    for (int i = 0; i < config.dim; ++i) {
        float val = w.rms_final_weight[i];
        rms_min = std::min(rms_min, val);
        rms_max = std::max(rms_max, val);
    }

    std::cout << "[Check 3] rms_final_weight block statistics (" << config.dim << " values):" << std::endl;
    std::cout << "  - Min value: " << rms_min << std::endl;
    std::cout << "  - Max value: " << rms_max << std::endl;

    bool rope_ok = true;
    for (int i = 0; i < head_size / 2; ++i) {
        if (std::fabs(w.freq_cis_real[i] - 1.0f) > 1e-6f ||
            std::fabs(w.freq_cis_imag[i]) > 1e-6f) {
            rope_ok = false;
            break;
        }
    }
    std::cout << "[Check 4] RoPE position 0 check: " << (rope_ok ? "pass" : "fail") << std::endl;

    return 0;
}