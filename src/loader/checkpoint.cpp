#include "checkpoint.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace geneinfer {

namespace {

constexpr std::size_t kHeaderInts = 7;
constexpr std::size_t kHeaderBytes = kHeaderInts * sizeof(std::int32_t);

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("checkpoint: " + msg); }

std::size_t product(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (auto s : shape)
        n *= static_cast<std::size_t>(s);
    return n;
}

void validate(const Config& c) {
    if (c.dim <= 0 || c.hidden_dim <= 0 || c.n_layers <= 0 || c.n_heads <= 0 ||
        c.n_kv_heads <= 0 || c.vocab_size <= 0 || c.seq_len <= 0) {
        std::ostringstream os;
        os << "invalid header (not a llama2.c float32 checkpoint?): dim=" << c.dim
           << " hidden=" << c.hidden_dim << " layers=" << c.n_layers << " heads=" << c.n_heads
           << " kv_heads=" << c.n_kv_heads << " vocab=" << c.vocab_size << " seq_len=" << c.seq_len;
        fail(os.str());
    }
    if (c.dim % c.n_heads != 0) fail("dim must be divisible by n_heads");
    if (c.n_heads % c.n_kv_heads != 0) fail("n_heads must be divisible by n_kv_heads");
    if (c.head_size() % 2 != 0) fail("head_size must be even (RoPE rotates pairs)");
}

}  // namespace

Checkpoint Checkpoint::load(const std::string& path) {
    Checkpoint ck;
    ck.file_ = MappedFile(path);

    const unsigned char* base = ck.file_.data();
    const std::size_t file_size = ck.file_.size();
    if (file_size < kHeaderBytes) fail("file too small to contain a header");

    // ---- 1. header ----------------------------------------------------------
    // Use `memcpy` for reading to avoid type-punning (which leads to undefined behavior) on unaligned or aliased memory.
    std::int32_t h[kHeaderInts];
    std::memcpy(h, base, kHeaderBytes);

    Config& c = ck.cfg_;
    c.dim = h[0];
    c.hidden_dim = h[1];
    c.n_layers = h[2];
    c.n_heads = h[3];
    c.n_kv_heads = h[4];
    c.shared_classifier = h[5] > 0;
    c.vocab_size = std::abs(h[5]);
    c.seq_len = h[6];
    validate(c);

    // ---- 2. Split tensors in a fixed order ---------------------------------------
    const std::int64_t L = c.n_layers, D = c.dim, H = c.hidden_dim, V = c.vocab_size;
    const std::int64_t KV = c.kv_dim(), HS = c.head_size(), S = c.seq_len;

    std::size_t offset = kHeaderBytes;
    auto take = [&](const char* name, std::vector<std::int64_t> shape) -> const float* {
        const std::size_t numel = product(shape);
        const std::size_t bytes = numel * sizeof(float);
        if (offset + bytes > file_size) {
            fail(std::string("file truncated while reading tensor '") + name + "'");
        }
        TensorView tv;
        tv.name = name;
        tv.data = reinterpret_cast<const float*>(base + offset);
        tv.numel = numel;
        tv.byte_offset = offset;
        tv.shape = std::move(shape);
        ck.tensors_.push_back(std::move(tv));
        offset += bytes;
        return ck.tensors_.back().data;
    };

    Weights& w = ck.w_;
    w.token_embedding = take("token_embedding", {V, D});
    w.rms_att = take("rms_att", {L, D});
    w.wq = take("wq", {L, D, D});     // n_heads * head_size == dim
    w.wk = take("wk", {L, KV, D});
    w.wv = take("wv", {L, KV, D});
    w.wo = take("wo", {L, D, D});
    w.rms_ffn = take("rms_ffn", {L, D});
    w.w1 = take("w1", {L, H, D});
    w.w2 = take("w2", {L, D, H});
    w.w3 = take("w3", {L, H, D});
    w.rms_final = take("rms_final", {D});
    // Legacy RoPE precomputation table. Neither `run.c` nor we use it 
    // (as `cos`/`sin` values ​​are calculated on the fly during runtime), 
    //but it must be skipped over.
    take("freq_cis_real", {S, HS / 2});
    take("freq_cis_imag", {S, HS / 2});
    w.wcls = c.shared_classifier ? w.token_embedding : take("wcls", {V, D});

    // ---- 3. Sizes must match exactly: expose format/version issues early --------------
    if (offset != file_size) {
        std::ostringstream os;
        os << "size mismatch: header implies " << offset << " bytes, file has " << file_size
           << " bytes (wrong format/version, e.g. int8-quantized or v1+ export?)";
        fail(os.str());
    }

    return ck;
}

}  // namespace geneinfer
