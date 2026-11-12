// hf_checkpoint.cpp - loading a Hugging Face Llama model directory (config.json + model.safetensors).
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "loader/checkpoint.h"
#include "loader/safetensors.h"
#include "util/json.h"

namespace geneinfer {

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("checkpoint: " + msg); }

std::string read_text_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int required_int(const JsonValue& cfg, const char* key) {
    const JsonValue* v = cfg.find(key);
    if (!v) fail(std::string("config.json has no '") + key + "'");
    const std::int64_t n = v->as_int();
    if (n <= 0 || n > 100000000) fail(std::string("config.json: '") + key + "' has an invalid value");
    return static_cast<int>(n);
}

int optional_int(const JsonValue& cfg, const char* key, int fallback) {
    const JsonValue* v = cfg.find(key);
    if (!v || v->is_null()) return fallback;
    const std::int64_t n = v->as_int();
    if (n <= 0 || n > 100000000) fail(std::string("config.json: '") + key + "' has an invalid value");
    return static_cast<int>(n);
}

double optional_number(const JsonValue& cfg, const char* key, double fallback) {
    const JsonValue* v = cfg.find(key);
    return (!v || v->is_null()) ? fallback : v->as_number();
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) s += (i ? "," : "") + std::to_string(shape[i]);
    return s + "]";
}

// Translates config.json into a Config, refusing anything this engine cannot run correctly.
Config parse_config(const JsonValue& j) {
    if (!j.is_object()) fail("config.json is not a JSON object");

    if (const JsonValue* t = j.find("model_type")) {
        if (t->is_string() && t->as_string() != "llama" && t->as_string() != "qwen3") {
            fail("unsupported model_type '" + t->as_string() + "' (only 'llama' and 'qwen3' are implemented)");
        }
    }
    if (const JsonValue* s = j.find("rope_scaling")) {
        if (!s->is_null()) fail("rope_scaling is not supported yet");
    }
    if (const JsonValue* a = j.find("hidden_act")) {
        if (a->is_string() && a->as_string() != "silu") fail("unsupported hidden_act '" + a->as_string() + "'");
    }
    for (const char* bias : {"attention_bias", "mlp_bias"}) {
        if (const JsonValue* b = j.find(bias)) {
            if (b->is_bool() && b->as_bool()) fail(std::string(bias) + " = true is not supported");
        }
    }

    Config c;
    if (const JsonValue* t = j.find("model_type")) c.qk_norm = t->is_string() && t->as_string() == "qwen3";
    c.dim = required_int(j, "hidden_size");
    c.hidden_dim = required_int(j, "intermediate_size");
    c.n_layers = required_int(j, "num_hidden_layers");
    c.n_heads = required_int(j, "num_attention_heads");
    c.n_kv_heads = optional_int(j, "num_key_value_heads", c.n_heads);
    c.vocab_size = required_int(j, "vocab_size");
    c.seq_len = optional_int(j, "max_position_embeddings", 2048);
    c.head_dim = optional_int(j, "head_dim", 0);
    if (c.head_dim == 0) {
        if (c.dim % c.n_heads != 0) fail("hidden_size is not divisible by num_attention_heads");
        c.head_dim = c.dim / c.n_heads;
    }
    if (c.n_heads % c.n_kv_heads != 0) fail("num_attention_heads must be divisible by num_key_value_heads");
    if (c.head_dim % 2 != 0) fail("head_dim must be even (RoPE rotates pairs)");

    c.rope_theta = static_cast<float>(optional_number(j, "rope_theta", 10000.0));
    c.rms_norm_eps = static_cast<float>(optional_number(j, "rms_norm_eps", 1e-6));
    c.rope_style = RopeStyle::HalfSplit;  // Hugging Face Llama always rotates (i, i + head_dim/2)

    const JsonValue* tie = j.find("tie_word_embeddings");
    c.shared_classifier = tie && tie->is_bool() && tie->as_bool();
    return c;
}

}  // namespace

Checkpoint Checkpoint::load_hf(const std::string& directory, const LoadOptions& options) {
    const fs::path dir(directory);
    const fs::path config_path = dir / "config.json";
    const fs::path weights_path = dir / "model.safetensors";

    if (!fs::exists(config_path)) fail("no config.json in " + directory);
    if (!fs::exists(weights_path)) {
        if (fs::exists(dir / "model.safetensors.index.json")) {
            fail("sharded checkpoints (model.safetensors.index.json) are not supported yet");
        }
        fail("no model.safetensors in " + directory);
    }

    Checkpoint ck;
    try {
        ck.cfg_ = parse_config(parse_json(read_text_file(config_path)));
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.rfind("checkpoint:", 0) == 0) throw;
        fail("config.json: " + msg);
    }
    const Config& c = ck.cfg_;

    ck.safetensors_ = std::make_unique<SafetensorsFile>(SafetensorsFile::open(weights_path.string()));
    const SafetensorsFile& st = *ck.safetensors_;

    const std::int64_t D = c.dim, H = c.hidden_dim, V = c.vocab_size;
    const std::int64_t A = c.attn_dim(), KV = c.kv_dim();

    ck.converted_.reserve(static_cast<std::size_t>(c.n_layers) * 11 + 3);
    ck.tensors_.reserve(static_cast<std::size_t>(c.n_layers) * 11 + 3);

    // Returns a view of the tensor `name`, after checking existence, dtype and shape.
    // Float32 data is used in place. bfloat16 MATRICES (`matrix` = true) are used in place too, as 16-bit
    // values, when options.keep_bf16 is set and the data is 2-byte aligned. Everything else (F16, bfloat16
    // norm vectors, misaligned data) is converted into float32 memory owned by `ck`.
    auto fetch = [&](const std::string& name, std::vector<std::int64_t> shape, bool matrix) -> Mat {
        const SafetensorsEntry* e = st.find(name);
        if (!e) fail("tensor '" + name + "' not found in model.safetensors");
        if (e->shape != shape) {
            fail("tensor '" + name + "' has shape " + shape_string(e->shape) + ", expected " + shape_string(shape));
        }
        if (e->dtype == DType::Other) {
            fail("tensor '" + name + "' has unsupported dtype " + e->dtype_name + " (supported: F32, F16, BF16)");
        }

        const std::size_t numel = e->numel();
        const unsigned char* raw = st.data(*e);
        TensorView tv;
        if (e->dtype == DType::F32 && reinterpret_cast<std::uintptr_t>(raw) % alignof(float) == 0) {
            tv.data = reinterpret_cast<const float*>(raw);  // zero-copy
        } else if (e->dtype == DType::BF16 && matrix && options.keep_bf16 &&
                   reinterpret_cast<std::uintptr_t>(raw) % alignof(std::uint16_t) == 0) {
            tv.storage = Storage::BF16;
            tv.bf16 = reinterpret_cast<const std::uint16_t*>(raw);  // zero-copy, widened inside the kernels
        } else {
            ck.converted_.emplace_back(numel);
            convert_to_f32(e->dtype, raw, ck.converted_.back().data(), numel);
            tv.data = ck.converted_.back().data();
        }

        tv.name = name;
        tv.numel = numel;
        tv.byte_offset = e->begin;
        tv.shape = std::move(shape);
        ck.tensors_.push_back(std::move(tv));
        const TensorView& added = ck.tensors_.back();
        return added.bf16 ? Mat(added.bf16) : Mat(added.data);
    };
    auto vec = [&](const std::string& name, std::vector<std::int64_t> shape) -> const float* {
        return fetch(name, std::move(shape), false).f32;
    };

    Weights& w = ck.w_;
    w.token_embedding = fetch("model.embed_tokens.weight", {V, D}, true);

    w.layers.resize(static_cast<std::size_t>(c.n_layers));
    for (int l = 0; l < c.n_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        LayerWeights& lw = w.layers[static_cast<std::size_t>(l)];
        lw.rms_att = vec(p + "input_layernorm.weight", {D});
        lw.wq = fetch(p + "self_attn.q_proj.weight", {A, D}, true);
        lw.wk = fetch(p + "self_attn.k_proj.weight", {KV, D}, true);
        lw.wv = fetch(p + "self_attn.v_proj.weight", {KV, D}, true);
        lw.wo = fetch(p + "self_attn.o_proj.weight", {D, A}, true);
        if (c.qk_norm) {
            lw.q_norm = vec(p + "self_attn.q_norm.weight", {c.head_dim});
            lw.k_norm = vec(p + "self_attn.k_norm.weight", {c.head_dim});
        }
        lw.rms_ffn = vec(p + "post_attention_layernorm.weight", {D});
        lw.w1 = fetch(p + "mlp.gate_proj.weight", {H, D}, true);
        lw.w3 = fetch(p + "mlp.up_proj.weight", {H, D}, true);
        lw.w2 = fetch(p + "mlp.down_proj.weight", {D, H}, true);
    }
    w.rms_final = vec("model.norm.weight", {D});
    w.wcls = c.shared_classifier ? w.token_embedding : fetch("lm_head.weight", {V, D}, true);

    return ck;
}

}  // namespace geneinfer
