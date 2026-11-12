#include "model/transformer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/q8.h"
#include "kernels/kernels.h"

namespace geneinfer {

namespace {

inline std::size_t sz(int v) { return static_cast<std::size_t>(v); }

// out = W * x for a float32 or bfloat16 weight matrix.
inline void matvec(float* out, const Mat& W, const float* x, int rows, int cols) {
    if (W.is_q8()) kernels::matvec_q8(out, W.q8, W.q8_scale, x, rows, cols);
    else if (W.is_bf16()) kernels::matvec_bf16(out, W.bf16, x, rows, cols);
    else kernels::matvec(out, W.f32, x, rows, cols);
}

// Rows [begin, end) of the product, written to out[begin..end).
inline void matvec_rows(float* out, const Mat& W, const float* x, int begin, int end, int cols) {
    const std::size_t offset = static_cast<std::size_t>(begin) * static_cast<std::size_t>(cols);
    if (W.is_q8()) {
        const std::size_t soffset = static_cast<std::size_t>(begin) * static_cast<std::size_t>(q8_groups(cols));
        kernels::matvec_q8(out + begin, W.q8 + offset, W.q8_scale + soffset, x, end - begin, cols);
    } else if (W.is_bf16()) kernels::matvec_bf16(out + begin, W.bf16 + offset, x, end - begin, cols);
    else kernels::matvec(out + begin, W.f32 + offset, x, end - begin, cols);
}

long env_long(const char* name, long fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (*end != '\0' || n < 0) {
        std::fprintf(stderr, "warning: %s=%s is not a non-negative integer; using %ld\n", name, v, fallback);
        return fallback;
    }
    return n;
}

// Attributes the time since the previous checkpoint to `section`.
inline void lap(ForwardProfile* p, ForwardProfile::Section section) {
    if (!p) return;
    const auto now = std::chrono::steady_clock::now();
    p->seconds[section] += std::chrono::duration<double>(now - p->last).count();
    p->last = now;
}

}  // namespace

const char* ForwardProfile::name(int s) {
    static const char* const names[kSections] = {
        "embedding lookup", "rmsnorm", "q/k/v projections", "rope", "attention (scores, softmax, mix)",
        "output projection", "ffn gate+up", "silu * up", "ffn down", "residual adds", "classifier (logits)"};
    return names[s];
}

Transformer::Transformer(const Checkpoint& checkpoint, int max_seq_len, int n_threads, int max_batch, int block_size,
                         int kv_blocks)
    : cfg_(checkpoint.config()), w_(checkpoint.weights()) {
    if (max_seq_len < 0 || max_seq_len > cfg_.seq_len) {
        throw std::invalid_argument("max_seq_len must be in [0, " + std::to_string(cfg_.seq_len) +
                                    "], got " + std::to_string(max_seq_len));
    }
    ctx_len_ = max_seq_len == 0 ? cfg_.seq_len : max_seq_len;

    if (n_threads < 0) throw std::invalid_argument("n_threads must be >= 0");
    if (n_threads == 0) n_threads = static_cast<int>(env_long("GI_THREADS", 1));
    if (n_threads == 0) n_threads = 1;
    if (n_threads > 256) throw std::invalid_argument("n_threads must be at most 256");
    par_min_weights_ = env_long("GI_PAR_MIN", par_min_weights_);   // tests lower it to force tiny products to split
    if (n_threads > 1) pool_ = std::make_unique<ThreadPool>(n_threads);

    if (max_batch < 0) throw std::invalid_argument("max_batch must be >= 0");
    if (max_batch == 0) max_batch = static_cast<int>(env_long("GI_BATCH", 64));
    if (max_batch == 0) max_batch = 64;
    if (max_batch > 4096) throw std::invalid_argument("max_batch must be at most 4096");
    max_batch_ = max_batch;

    x_.resize(sz(cfg_.dim));
    xb_.resize(sz(cfg_.dim));
    xb2_.resize(sz(cfg_.dim));
    hb_.resize(sz(cfg_.hidden_dim));
    hb2_.resize(sz(cfg_.hidden_dim));
    q_.resize(sz(cfg_.attn_dim()));
    ao_.resize(sz(cfg_.attn_dim()));
    logits_.resize(sz(cfg_.vocab_size));
    bx_.resize(sz(max_batch_) * sz(cfg_.dim));
    bxb_.resize(sz(max_batch_) * sz(cfg_.dim));
    bxb2_.resize(sz(max_batch_) * sz(cfg_.dim));
    bhb_.resize(sz(max_batch_) * sz(cfg_.hidden_dim));
    bhb2_.resize(sz(max_batch_) * sz(cfg_.hidden_dim));
    bq_.resize(sz(max_batch_) * sz(cfg_.attn_dim()));
    bao_.resize(sz(max_batch_) * sz(cfg_.attn_dim()));
    bk_.resize(sz(max_batch_) * sz(cfg_.kv_dim()));
    bv_.resize(sz(max_batch_) * sz(cfg_.kv_dim()));
    row_seq_.resize(sz(max_batch_));
    row_pos_.resize(sz(max_batch_));
    row_table_.resize(sz(max_batch_));
    batt_.resize(sz(pool_ ? pool_->size() : 1) * sz(ctx_len_));

    if (block_size < 0 || kv_blocks < 0) throw std::invalid_argument("block_size and kv_blocks must be >= 0");
    if (block_size == 0) block_size = static_cast<int>(env_long("GI_BLOCK_SIZE", 16));
    if (block_size == 0) block_size = 16;
    if (block_size > ctx_len_) block_size = ctx_len_;   // never bigger than a whole context
    if (kv_blocks == 0) kv_blocks = (ctx_len_ + block_size - 1) / block_size;   // one full-length sequence
    kv_ = std::make_unique<KvPool>(cfg_.n_layers, cfg_.kv_dim(), block_size, kv_blocks);
    default_seq_ = std::make_unique<Sequence>(*kv_);
}

Transformer::~Transformer() = default;

int Transformer::threads() const { return pool_ ? pool_->size() : 1; }

// Runs several independent matrix-vector products with one synchronization of the pool. Each product is cut
// into slices of consecutive rows, one per thread; slice starts are multiples of 4, which is the row blocking of
// the kernels, so every row is computed exactly as in a single-threaded run (bit-identical results). A product
// with fewer than par_min_weights_ weights per thread is not split: thread 0 computes it alone.
void Transformer::run_matvecs(const MatvecJob* jobs, int n) {
    if (!pool_) {
        for (int j = 0; j < n; ++j) matvec(jobs[j].out, jobs[j].W, jobs[j].x, jobs[j].rows, jobs[j].cols);
        return;
    }
    struct Ctx {
        const MatvecJob* jobs;
        int n;
        long par_min;
    } ctx{jobs, n, par_min_weights_};
    pool_->run(
        [](void* p, int t, int nt) {
            const Ctx& c = *static_cast<const Ctx*>(p);
            for (int j = 0; j < c.n; ++j) {
                const MatvecJob& job = c.jobs[j];
                const long weights = static_cast<long>(job.rows) * job.cols;
                if (weights < c.par_min * nt) {   // too small to split
                    if (t == 0) matvec(job.out, job.W, job.x, job.rows, job.cols);
                    continue;
                }
                const int per = (job.rows + nt - 1) / nt;
                const int chunk = (per + 3) / 4 * 4;             // multiple of 4: the kernels' row block
                const int begin = std::min(job.rows, t * chunk);
                const int end = std::min(job.rows, begin + chunk);
                if (begin < end) matvec_rows(job.out, job.W, job.x, begin, end, job.cols);
            }
        },
        &ctx);
}

void Transformer::ensure_rope(int pos) {
    const std::size_t half = sz(cfg_.head_size() / 2);
    while (rope_rows_ <= pos) {
        rope_cos_.resize((sz(rope_rows_) + 1) * half);
        rope_sin_.resize((sz(rope_rows_) + 1) * half);
        kernels::rope_table_row(rope_cos_.data() + sz(rope_rows_) * half, rope_sin_.data() + sz(rope_rows_) * half,
                                rope_rows_, cfg_.head_size(), cfg_.rope_theta);
        ++rope_rows_;
    }
}

namespace {

// Rows of a weight matrix that fit in about 32 KB (the L1 cache); a multiple of 4, the kernels' row block.
int rows_per_tile(const Mat& W, int cols) {
    const std::size_t row_bytes = W.is_q8() ? sz(cols) + 2 * sz(q8_groups(cols)) : sz(cols) * (W.is_bf16() ? 2 : 4);
    long rows = static_cast<long>(32768 / row_bytes) / 4 * 4;
    return rows < 4 ? 4 : static_cast<int>(rows);
}

}  // namespace

namespace {

// Attention for m rows over the KV cache of one layer. Row i is the position positions[i] of the sequence whose
// block table is tables[i]; it attends to positions 0 .. positions[i]. The work items are (row, head) pairs dealt
// out round-robin to the threads (later positions cost more); every item uses its thread's own score row. A
// single-token decode step is the case m == 1. Each item runs the same operations in the same order whatever the
// thread count and whatever the block size, so the output depends on neither.
struct AttCtx {
    const float* q;
    float* ao;
    const KvPool* pool;
    int layer;
    const int* const* tables;
    const int* positions;
    float* scratch;   // [threads, ctx_len]
    int m, n_heads, head_size, kv_mul, attn_dim, kv_dim, ctx_len;
    float inv_scale;
};

void attention_body(void* p, int t, int nt) {
    const AttCtx& c = *static_cast<const AttCtx*>(p);
    float* scores = c.scratch + sz(t) * sz(c.ctx_len);
    const int bs = c.pool->block_size();
    const int items = c.m * c.n_heads;
    for (int item = t; item < items; item += nt) {
        const int i = item / c.n_heads, h = item % c.n_heads;
        const int pos = c.positions[i];
        const int* table = c.tables[i];
        const float* qh = c.q + sz(i) * sz(c.attn_dim) + sz(h) * sz(c.head_size);
        const std::size_t head_off = sz(h / c.kv_mul) * sz(c.head_size);   // GQA: kv_mul query heads share one KV head
        for (int start = 0, b = 0; start <= pos; start += bs, ++b) {
            const float* kb = c.pool->keys(table[b], c.layer) + head_off;
            const int cnt = std::min(bs, pos + 1 - start);
            for (int j = 0; j < cnt; ++j) {
                scores[start + j] = kernels::dot(qh, kb + sz(j) * sz(c.kv_dim), c.head_size) * c.inv_scale;
            }
        }
        kernels::softmax(scores, pos + 1);
        float* out = c.ao + sz(i) * sz(c.attn_dim) + sz(h) * sz(c.head_size);
        std::fill(out, out + c.head_size, 0.0f);
        for (int start = 0, b = 0; start <= pos; start += bs, ++b) {
            const float* vb = c.pool->values(table[b], c.layer) + head_off;
            const int cnt = std::min(bs, pos + 1 - start);
            for (int j = 0; j < cnt; ++j) kernels::axpy(out, scores[start + j], vb + sz(j) * sz(c.kv_dim), c.head_size);
        }
    }
}

}  // namespace

// Several tokens through several matrices. Each thread takes a slice of rows (a multiple of 4 as in run_matvecs)
// and walks over it in tiles of rows that fit in the L1 cache; for each tile all ntok tokens are processed, so
// the tile is read from memory once and then from cache. Every output element is still computed by the
// matvec kernel on whole rows, so the bits do not depend on the tiling, the slicing or the number of threads.
void Transformer::run_matmuls(const MatmulJob* jobs, int n, int ntok) {
    struct Ctx {
        const MatmulJob* jobs;
        int n, ntok;
        long par_min;
    } ctx{jobs, n, ntok, par_min_weights_};
    auto body = [](void* p, int t, int nt) {
        const Ctx& c = *static_cast<const Ctx*>(p);
        for (int j = 0; j < c.n; ++j) {
            const MatmulJob& job = c.jobs[j];
            int begin = 0, end = job.rows;
            const long weights = static_cast<long>(job.rows) * job.cols;
            if (weights * c.ntok < c.par_min * nt) {   // too little work to split: thread 0 does it all
                if (t != 0) continue;
            } else {
                const int per = (job.rows + nt - 1) / nt;
                const int chunk = (per + 3) / 4 * 4;
                begin = std::min(job.rows, t * chunk);
                end = std::min(job.rows, begin + chunk);
            }
            const int tile = rows_per_tile(job.W, job.cols);
            for (int r0 = begin; r0 < end; r0 += tile) {
                const int r1 = std::min(end, r0 + tile);
                for (int tok = 0; tok < c.ntok; ++tok) {
                    matvec_rows(job.out + sz(tok) * job.out_stride, job.W, job.x + sz(tok) * job.x_stride, r0, r1, job.cols);
                }
            }
        }
    };
    if (pool_) pool_->run(body, &ctx);
    else body(&ctx, 0, 1);
}

const float* Transformer::forward_batch(const int* tokens, int n, int pos0, bool compute_logits) {
    return forward_batch(*default_seq_, tokens, n, pos0, compute_logits);
}

const float* Transformer::forward_batch(Sequence& seq, const int* tokens, int n, int pos0, bool compute_logits) {
    if (n < 1) throw std::invalid_argument("forward_batch: no tokens");
    if (pos0 < 0 || pos0 > ctx_len_ - n) {
        throw std::out_of_range("forward_batch: positions " + std::to_string(pos0) + ".." + std::to_string(pos0 + n - 1) +
                                " exceed the context length " + std::to_string(ctx_len_));
    }
    if (&seq.pool() != kv_.get()) throw std::invalid_argument("forward_batch: the sequence belongs to another KV pool");
    if (pos0 > seq.length()) {
        throw std::out_of_range("forward_batch: start position " + std::to_string(pos0) + " leaves a gap, the sequence holds " +
                                std::to_string(seq.length()) + " positions");
    }
    for (int i = 0; i < n; ++i) {
        if (tokens[i] < 0 || tokens[i] >= cfg_.vocab_size) {
            throw std::out_of_range("forward_batch: token id " + std::to_string(tokens[i]) + " out of range");
        }
    }
    if (!prepare_write(seq, pos0, pos0 + n - 1)) {
        seq.truncate(seq.length());
        throw std::runtime_error("forward_batch: the KV pool has no free block left");
    }
    if (trace_) {   // the trace wants every token: one by one
        const float* logits = nullptr;
        for (int i = 0; i < n; ++i) logits = forward(seq, tokens[i], pos0 + i, compute_logits && i == n - 1);
        return compute_logits ? logits : nullptr;
    }
    for (int done = 0; done < n; done += max_batch_) {
        const int m = std::min(max_batch_, n - done);
        batch_chunk(seq, tokens + done, m, pos0 + done, compute_logits && done + m == n);
    }
    return compute_logits ? logits_.data() : nullptr;
}

void Transformer::batch_chunk(Sequence& seq, const int* tokens, int m, int p0, bool want_logits) {
    for (int i = 0; i < m; ++i) {
        row_seq_[sz(i)] = &seq;
        row_pos_[sz(i)] = p0 + i;
    }
    run_rows(tokens, row_seq_.data(), row_pos_.data(), m, want_logits);
    seq.set_length(p0 + m);
}

void Transformer::run_rows(const int* tokens, Sequence* const* seqs, const int* positions, int m, bool want_logits,
                           const unsigned char* row_logits) {
    if (profile_) {
        profile_->last = std::chrono::steady_clock::now();
        profile_->forward_calls += m;
    }
    const int dim = cfg_.dim;
    const int hidden = cfg_.hidden_dim;
    const int attn_dim = cfg_.attn_dim();
    const int kv_dim = cfg_.kv_dim();
    const int kv_mul = cfg_.kv_mul();
    const int head_size = cfg_.head_size();
    const int n_heads = cfg_.n_heads;
    const int bs = kv_->block_size();
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float eps = cfg_.rms_norm_eps;

    int max_pos = 0;
    for (int i = 0; i < m; ++i) {
        row_table_[sz(i)] = seqs[i]->block_table();
        max_pos = std::max(max_pos, positions[i]);
    }
    ensure_rope(max_pos);
    const std::size_t half = sz(head_size / 2);

    float* X = bx_.data();
    float* XB = bxb_.data();
    float* XB2 = bxb2_.data();
    float* HB = bhb_.data();
    float* HB2 = bhb2_.data();
    float* Q = bq_.data();
    float* AO = bao_.data();

    for (int i = 0; i < m; ++i) {
        float* x = X + sz(i) * sz(dim);
        if (w_.token_embedding.is_bf16()) kernels::widen_bf16(x, w_.token_embedding.bf16 + sz(tokens[i]) * sz(dim), dim);
        else std::memcpy(x, w_.token_embedding.f32 + sz(tokens[i]) * sz(dim), sz(dim) * sizeof(float));
    }
    lap(profile_, ForwardProfile::Embed);

    for (int l = 0; l < cfg_.n_layers; ++l) {
        const std::size_t layer = sz(l);
        const LayerWeights& lw = w_.layers[layer];
        float* kbase = bk_.data();   // the new keys and values are computed here and then stored into their blocks
        float* vbase = bv_.data();

        // ---------------- attention block ----------------
        for (int i = 0; i < m; ++i) kernels::rmsnorm(XB + sz(i) * sz(dim), X + sz(i) * sz(dim), lw.rms_att, dim, eps);
        lap(profile_, ForwardProfile::Norm);

        const MatmulJob qkv[3] = {{Q, lw.wq, XB, attn_dim, dim, sz(attn_dim), sz(dim)},
                                  {kbase, lw.wk, XB, kv_dim, dim, sz(kv_dim), sz(dim)},
                                  {vbase, lw.wv, XB, kv_dim, dim, sz(kv_dim), sz(dim)}};
        run_matmuls(qkv, 3, m);
        lap(profile_, ForwardProfile::QkvProj);

        if (cfg_.qk_norm) {   // Qwen3: every q and k head is RMS-normalized (with its own weights) before RoPE
            for (int i = 0; i < m; ++i) {
                float* qi = Q + sz(i) * sz(attn_dim);
                float* ki = kbase + sz(i) * sz(kv_dim);
                for (int h = 0; h < n_heads; ++h) kernels::rmsnorm(qi + sz(h) * sz(head_size), qi + sz(h) * sz(head_size), lw.q_norm, head_size, eps);
                for (int h = 0; h < cfg_.n_kv_heads; ++h) kernels::rmsnorm(ki + sz(h) * sz(head_size), ki + sz(h) * sz(head_size), lw.k_norm, head_size, eps);
            }
            lap(profile_, ForwardProfile::Norm);
        }

        for (int i = 0; i < m; ++i) {
            const float* c = rope_cos_.data() + sz(positions[i]) * half;
            const float* s = rope_sin_.data() + sz(positions[i]) * half;
            kernels::rope_apply(Q + sz(i) * sz(attn_dim), attn_dim, c, s, head_size, cfg_.rope_style);
            kernels::rope_apply(kbase + sz(i) * sz(kv_dim), kv_dim, c, s, head_size, cfg_.rope_style);
        }
        // Store the new keys and values into their blocks (a chunk may cross block boundaries, and rows of
        // different sequences go to different blocks).
        for (int i = 0; i < m; ++i) {
            const int pos = positions[i];
            const int block = row_table_[sz(i)][pos / bs];
            const std::size_t off = sz(pos % bs) * sz(kv_dim);
            std::memcpy(kv_->keys(block, l) + off, kbase + sz(i) * sz(kv_dim), sz(kv_dim) * sizeof(float));
            std::memcpy(kv_->values(block, l) + off, vbase + sz(i) * sz(kv_dim), sz(kv_dim) * sizeof(float));
        }
        lap(profile_, ForwardProfile::Rope);

        // Attention: item = (row, head); a row sees positions 0 .. its own (the rows' keys and values are all in
        // the cache already). Items are dealt out round-robin because later positions cost more.
        const AttCtx ac{Q, AO, kv_.get(), l, row_table_.data(), positions, batt_.data(), m, n_heads, head_size, kv_mul,
                        attn_dim, kv_dim, ctx_len_, inv_scale};
        if (pool_) pool_->run(attention_body, const_cast<AttCtx*>(&ac));
        else attention_body(const_cast<AttCtx*>(&ac), 0, 1);
        lap(profile_, ForwardProfile::Attention);

        const MatmulJob oproj = {XB2, lw.wo, AO, dim, attn_dim, sz(dim), sz(attn_dim)};
        run_matmuls(&oproj, 1, m);
        lap(profile_, ForwardProfile::OutProj);
        kernels::add(X, XB2, m * dim);
        lap(profile_, ForwardProfile::Residual);

        // ---------------- feed-forward block (SwiGLU) ----------------
        for (int i = 0; i < m; ++i) kernels::rmsnorm(XB + sz(i) * sz(dim), X + sz(i) * sz(dim), lw.rms_ffn, dim, eps);
        lap(profile_, ForwardProfile::Norm);
        const MatmulJob gate_up[2] = {{HB, lw.w1, XB, hidden, dim, sz(hidden), sz(dim)},
                                      {HB2, lw.w3, XB, hidden, dim, sz(hidden), sz(dim)}};
        run_matmuls(gate_up, 2, m);
        lap(profile_, ForwardProfile::FfnGateUp);
        kernels::silu(HB, m * hidden);
        kernels::mul(HB, HB2, m * hidden);
        lap(profile_, ForwardProfile::Activation);
        const MatmulJob down = {XB, lw.w2, HB, dim, hidden, sz(dim), sz(hidden)};
        run_matmuls(&down, 1, m);
        lap(profile_, ForwardProfile::FfnDown);
        kernels::add(X, XB, m * dim);
        lap(profile_, ForwardProfile::Residual);
    }

    if (row_logits) {   // logits for any subset of the rows: final norm per row, then one matrix pass for all of them
        int k = 0;
        for (int i = 0; i < m; ++i) {
            if (!row_logits[i]) continue;
            kernels::rmsnorm(XB + sz(k) * sz(dim), X + sz(i) * sz(dim), w_.rms_final, dim, eps);
            ++k;
        }
        lap(profile_, ForwardProfile::Norm);
        if (k == 0) return;
        if (rlogits_.size() < sz(k) * sz(cfg_.vocab_size)) rlogits_.resize(sz(k) * sz(cfg_.vocab_size));
        const MatmulJob head = {rlogits_.data(), w_.wcls, XB, cfg_.vocab_size, dim, sz(cfg_.vocab_size), sz(dim)};
        run_matmuls(&head, 1, k);
        lap(profile_, ForwardProfile::Classifier);
        return;
    }
    if (!want_logits) return;
    float* last = X + sz(m - 1) * sz(dim);
    kernels::rmsnorm(last, last, w_.rms_final, dim, eps);
    lap(profile_, ForwardProfile::Norm);
    const MatvecJob head = {logits_.data(), w_.wcls, last, cfg_.vocab_size, dim};
    run_matvecs(&head, 1);
    lap(profile_, ForwardProfile::Classifier);
}

// Room for positions first..last of seq, and every block they touch belongs to seq alone (copy-on-write). Blocks that are
// still shared with the prefix cache or another sequence are copied first, so a write never changes what others read.
bool Transformer::prepare_write(Sequence& seq, int first, int last) {
    if (!seq.reserve(last + 1)) return false;
    const int bs = kv_->block_size();
    for (int idx = first / bs; idx <= last / bs; ++idx) {
        if (!seq.make_writable(idx)) return false;
    }
    return true;
}

const float* Transformer::forward_rows(const StepRow* rows, int n) {
    if (n < 1 || n > max_batch_) {
        throw std::invalid_argument("forward_rows: " + std::to_string(n) + " rows, expected 1 .. " + std::to_string(max_batch_));
    }
    std::vector<int> tokens(sz(n));
    std::vector<unsigned char> want(sz(n));
    int k = 0;
    for (int i = 0; i < n; ++i) {
        const StepRow& r = rows[i];
        if (r.seq == nullptr || &r.seq->pool() != kv_.get()) throw std::invalid_argument("forward_rows: bad or foreign sequence");
        if (r.token < 0 || r.token >= cfg_.vocab_size) {
            throw std::out_of_range("forward_rows: token id " + std::to_string(r.token) + " out of range");
        }
        int pos = r.seq->length();
        for (int j = 0; j < i; ++j) pos += (rows[j].seq == r.seq) ? 1 : 0;
        if (pos >= ctx_len_) {
            throw std::out_of_range("forward_rows: position " + std::to_string(pos) + " exceeds the context length " +
                                    std::to_string(ctx_len_));
        }
        row_seq_[sz(i)] = r.seq;
        row_pos_[sz(i)] = pos;
        tokens[sz(i)] = r.token;
        want[sz(i)] = r.want_logits ? 1 : 0;
        k += r.want_logits ? 1 : 0;
    }
    // Reserve every new position first; on failure give the blocks back so that nothing has changed.
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) ok = prepare_write(*row_seq_[sz(i)], row_pos_[sz(i)], row_pos_[sz(i)]);
    if (!ok) {
        for (int i = 0; i < n; ++i) row_seq_[sz(i)]->truncate(row_seq_[sz(i)]->length());
        throw std::runtime_error("forward_rows: the KV pool has no free block left");
    }
    run_rows(tokens.data(), row_seq_.data(), row_pos_.data(), n, false, want.data());
    for (int i = 0; i < n; ++i) row_seq_[sz(i)]->set_length(row_pos_[sz(i)] + 1);   // later rows of a sequence overwrite earlier values
    return k > 0 ? rlogits_.data() : nullptr;
}

std::size_t Transformer::kv_cache_bytes() const {
    return kv_->bytes();
}

std::size_t Transformer::scratch_bytes() const {
    return (x_.size() + xb_.size() + xb2_.size() + hb_.size() + hb2_.size() + q_.size() + ao_.size() +
            logits_.size() + bx_.size() + bxb_.size() + bxb2_.size() + bhb_.size() + bhb2_.size() + bq_.size() +
            bao_.size() + bk_.size() + bv_.size() + batt_.size() + rope_cos_.size() + rope_sin_.size() +
            rlogits_.size()) * sizeof(float) +
           row_seq_.size() * (sizeof(Sequence*) + sizeof(int) + sizeof(const int*));
}

const float* Transformer::forward(int token, int pos, bool compute_logits) {
    return forward(*default_seq_, token, pos, compute_logits);
}

const float* Transformer::forward(Sequence& seq, int token, int pos, bool compute_logits) {
    if (token < 0 || token >= cfg_.vocab_size) {
        throw std::out_of_range("forward: token id " + std::to_string(token) + " out of range");
    }
    if (pos < 0 || pos >= ctx_len_) {
        throw std::out_of_range("forward: position " + std::to_string(pos) +
                                " exceeds the context length " + std::to_string(ctx_len_));
    }
    if (&seq.pool() != kv_.get()) throw std::invalid_argument("forward: the sequence belongs to another KV pool");
    if (pos > seq.length()) {
        throw std::out_of_range("forward: position " + std::to_string(pos) + " leaves a gap, the sequence holds " +
                                std::to_string(seq.length()) + " positions");
    }
    if (!prepare_write(seq, pos, pos)) {
        seq.truncate(seq.length());
        throw std::runtime_error("forward: the KV pool has no free block left");
    }

    if (profile_) {
        profile_->last = std::chrono::steady_clock::now();
        ++profile_->forward_calls;
    }
    const int dim = cfg_.dim;
    const int hidden = cfg_.hidden_dim;
    const int attn_dim = cfg_.attn_dim();  // n_heads * head_dim (== dim for llama2.c models)
    const int kv_dim = cfg_.kv_dim();
    const int kv_mul = cfg_.kv_mul();
    const int head_size = cfg_.head_size();
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float eps = cfg_.rms_norm_eps;

    ensure_rope(pos);
    const float* rope_c = rope_cos_.data() + sz(pos) * sz(head_size / 2);
    const float* rope_s = rope_sin_.data() + sz(pos) * sz(head_size / 2);

    float* x = x_.data();
    float* xb = xb_.data();
    float* xb2 = xb2_.data();
    float* hb = hb_.data();
    float* hb2 = hb2_.data();
    float* q = q_.data();
    float* ao = ao_.data();

    // 1. Token embedding: copy one row of the embedding table into the residual stream.
    if (w_.token_embedding.is_bf16()) {
        kernels::widen_bf16(x, w_.token_embedding.bf16 + sz(token) * sz(dim), dim);
    } else {
        std::memcpy(x, w_.token_embedding.f32 + sz(token) * sz(dim), sz(dim) * sizeof(float));
    }
    lap(profile_, ForwardProfile::Embed);
    if (trace_) trace_("embedding", -1, x, dim);

    for (int l = 0; l < cfg_.n_layers; ++l) {
        const std::size_t layer = sz(l);
        const LayerWeights& lw = w_.layers[layer];

        // Where the current position lives in the paged KV cache.
        const int block = seq.block_table()[pos / kv_->block_size()];
        const std::size_t row_off = sz(pos % kv_->block_size()) * sz(kv_dim);

        // ---------------- attention block ----------------
        kernels::rmsnorm(xb, x, lw.rms_att, dim, eps);
        lap(profile_, ForwardProfile::Norm);

        // K and V of the current position are written directly into the cache.
        float* k = kv_->keys(block, l) + row_off;
        float* v = kv_->values(block, l) + row_off;
        const MatvecJob qkv[3] = {{q, lw.wq, xb, attn_dim, dim}, {k, lw.wk, xb, kv_dim, dim}, {v, lw.wv, xb, kv_dim, dim}};
        run_matvecs(qkv, 3);
        lap(profile_, ForwardProfile::QkvProj);

        if (cfg_.qk_norm) {   // Qwen3: every q and k head is RMS-normalized (with its own weights) before RoPE
            for (int h = 0; h < cfg_.n_heads; ++h) kernels::rmsnorm(q + sz(h) * sz(head_size), q + sz(h) * sz(head_size), lw.q_norm, head_size, eps);
            for (int h = 0; h < cfg_.n_kv_heads; ++h) kernels::rmsnorm(k + sz(h) * sz(head_size), k + sz(h) * sz(head_size), lw.k_norm, head_size, eps);
            lap(profile_, ForwardProfile::Norm);
        }

        // Rotary position embedding on q and k (v is not rotated).
        kernels::rope_apply(q, attn_dim, rope_c, rope_s, head_size, cfg_.rope_style);
        kernels::rope_apply(k, kv_dim, rope_c, rope_s, head_size, cfg_.rope_style);
        lap(profile_, ForwardProfile::Rope);

        // Each query head attends over all positions 0..pos of its (shared) KV head; the heads are spread over
        // the threads when there is enough work to pay for waking them.
        {
            const int* table = seq.block_table();
            const AttCtx ac{q, ao, kv_.get(), l, &table, &pos, batt_.data(), 1, cfg_.n_heads, head_size, kv_mul,
                            attn_dim, kv_dim, ctx_len_, inv_scale};
            const int nt = pool_ ? pool_->size() : 1;
            const long work = static_cast<long>(cfg_.n_heads) * (pos + 1) * head_size * 2;   // multiply-adds
            if (pool_ && work >= par_min_weights_ * nt) pool_->run(attention_body, const_cast<AttCtx*>(&ac));
            else attention_body(const_cast<AttCtx*>(&ac), 0, 1);
        }

        lap(profile_, ForwardProfile::Attention);

        // Output projection, then add to the residual stream.
        const MatvecJob oproj = {xb2, lw.wo, ao, dim, attn_dim};
        run_matvecs(&oproj, 1);
        lap(profile_, ForwardProfile::OutProj);
        kernels::add(x, xb2, dim);
        lap(profile_, ForwardProfile::Residual);
        if (trace_) trace_("attn_out", l, x, dim);

        // ---------------- feed-forward block (SwiGLU) ----------------
        kernels::rmsnorm(xb, x, lw.rms_ffn, dim, eps);
        lap(profile_, ForwardProfile::Norm);
        const MatvecJob gate_up[2] = {{hb, lw.w1, xb, hidden, dim}, {hb2, lw.w3, xb, hidden, dim}};   // gate, up
        run_matvecs(gate_up, 2);
        lap(profile_, ForwardProfile::FfnGateUp);
        kernels::silu(hb, hidden);
        kernels::mul(hb, hb2, hidden);
        lap(profile_, ForwardProfile::Activation);
        const MatvecJob down = {xb, lw.w2, hb, dim, hidden};
        run_matvecs(&down, 1);
        lap(profile_, ForwardProfile::FfnDown);
        kernels::add(x, xb, dim);
        lap(profile_, ForwardProfile::Residual);
        if (trace_) trace_("ffn_out", l, x, dim);
    }

    // 3. Final norm and classifier (for shared weights, wcls is the embedding table).
    seq.set_length(pos + 1);
    if (!compute_logits) return nullptr;
    kernels::rmsnorm(x, x, w_.rms_final, dim, eps);
    lap(profile_, ForwardProfile::Norm);
    if (trace_) trace_("final_norm", -1, x, dim);
    const MatvecJob head = {logits_.data(), w_.wcls, x, cfg_.vocab_size, dim};
    run_matvecs(&head, 1);
    lap(profile_, ForwardProfile::Classifier);

    return logits_.data();
}

}  // namespace geneinfer
