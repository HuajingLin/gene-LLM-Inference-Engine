// device_check_main.cpp - checks every operation of a compute device (gpu/device_ops.h) on its own.
//
//   gi_device_check --device cuda|emu [--model DIR]
//
// Each operation runs on generated inputs and is compared with a reference written here in plain double-precision
// loops (nothing shared with the kernels). Output buffers are surrounded by sentinel values that must survive.
// When a GPU result is wrong, the first failing line names the operation and the shape. One line per group:
//   <operation> cases=<n> worst=<error>          error = max|got - want| / max(1, max|want|)
// With --model, the GpuTransformer API is exercised too (position gaps, rewinding, batch limits) and the bytes it
// put on the device are printed (weight_bytes=..., device_bytes=...).
// Exit code 0 if everything is within 1e-5, else 1.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/device_ops.h"
#include "gpu/gpu_transformer.h"

using namespace geneinfer;

namespace {

const gpu::DeviceOps* dev = nullptr;
int failures = 0;
const float kSentinel = -777.25f;

struct Rng {   // xorshift64*: the same numbers everywhere
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed * 2685821657736338717ULL + 1442695040888963407ULL) {}
    std::uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    float uniform() { return static_cast<float>((next() >> 40) / 16777216.0) * 2.0f - 1.0f; }   // [-1, 1)
};

std::vector<float> random_floats(Rng& rng, std::size_t n, float scale = 1.0f) {
    std::vector<float> v(n);
    for (float& x : v) x = rng.uniform() * scale;
    return v;
}

float widen(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

std::uint16_t narrow(float f) {   // truncation is fine: the test only needs some bfloat16 values
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return static_cast<std::uint16_t>(u >> 16);
}

// A device buffer with a host mirror.
struct Buf {
    void* d = nullptr;
    std::size_t bytes = 0;
    Buf() = default;
    Buf(const void* host, std::size_t n) : bytes(n) {
        d = dev->alloc(n);
        dev->upload(d, host, n);
    }
    explicit Buf(const std::vector<float>& v) : Buf(v.data(), v.size() * sizeof(float)) {}
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() {
        if (d) dev->release(d);
    }
    float* f() const { return static_cast<float*>(d); }
    std::vector<float> get() const {
        std::vector<float> v(bytes / sizeof(float));
        dev->download(v.data(), d, bytes);
        return v;
    }
};

struct Group {
    const char* name;
    int cases = 0;
    double worst = 0.0;
    explicit Group(const char* n) : name(n) {}
    // got / want: whole buffers; `mask[i]` false marks a sentinel that must still hold kSentinel.
    void check(const std::string& what, const std::vector<float>& got, const std::vector<double>& want,
               const std::vector<char>& is_output) {
        ++cases;
        double max_want = 1.0, max_diff = 0.0;
        bool sentinel_ok = true, finite = true;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!is_output[i]) {
                if (got[i] != kSentinel) sentinel_ok = false;
                continue;
            }
            if (!std::isfinite(got[i])) finite = false;
            max_want = std::max(max_want, std::fabs(want[i]));
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(got[i]) - want[i]));
        }
        const double e = finite ? max_diff / max_want : INFINITY;
        if (e > worst) worst = e;
        if (!sentinel_ok || !(e < 1e-5)) {
            ++failures;
            std::printf("FAIL %s [%s]: error %.3e%s\n", name, what.c_str(), e,
                        sentinel_ok ? "" : ", memory outside the output was overwritten");
        }
    }
    void bits(const std::string& what, bool same) {
        ++cases;
        if (!same) {
            ++failures;
            std::printf("FAIL %s [%s]: results are not bit-identical\n", name, what.c_str());
        }
    }
    ~Group() { std::printf("%-10s cases=%d worst=%.2e\n", name, cases, worst); }
};

std::string shape(std::initializer_list<int> v) {
    std::string s;
    for (int x : v) s += (s.empty() ? "" : "x") + std::to_string(x);
    return s;
}

void check_embed(Rng& rng) {
    Group g("embed");
    for (int bf16 = 0; bf16 < 2; ++bf16) {
        for (int dim : {1, 5, 64}) {
            const int vocab = 37, m = 9;
            const std::vector<float> table = random_floats(rng, static_cast<std::size_t>(vocab) * dim);
            std::vector<std::uint16_t> table16(table.size());
            for (std::size_t i = 0; i < table.size(); ++i) table16[i] = narrow(table[i]);
            std::vector<int> tokens(m);
            for (int& t : tokens) t = static_cast<int>(rng.next() % vocab);
            tokens[0] = 0;
            tokens[1] = vocab - 1;
            const Buf dt(bf16 ? static_cast<const void*>(table16.data()) : table.data(), table.size() * (bf16 ? 2 : 4));
            const Buf dk(tokens.data(), tokens.size() * sizeof(int));
            const Buf out(std::vector<float>(static_cast<std::size_t>(m + 1) * dim, kSentinel));
            dev->embed(out.f(), dt.d, bf16, static_cast<const int*>(dk.d), m, dim);
            std::vector<double> want(static_cast<std::size_t>(m + 1) * dim, 0.0);
            std::vector<char> is_out(want.size(), 0);
            for (int i = 0; i < m; ++i) {
                for (int d = 0; d < dim; ++d) {
                    const std::size_t src = static_cast<std::size_t>(tokens[i]) * dim + d;
                    want[static_cast<std::size_t>(i) * dim + d] = bf16 ? widen(table16[src]) : table[src];
                    is_out[static_cast<std::size_t>(i) * dim + d] = 1;
                }
            }
            g.check((bf16 ? "bf16 " : "f32 ") + shape({m, dim}), out.get(), want, is_out);
        }
    }
}

void check_rmsnorm(Rng& rng) {
    Group g("rmsnorm");
    for (int in_place = 0; in_place < 2; ++in_place) {
        for (int size : {1, 7, 31, 32, 33, 64, 100, 300}) {
            for (float eps : {1e-5f, 0.05f}) {
                const int nvec = 5;
                const std::size_t n = static_cast<std::size_t>(nvec) * size;
                std::vector<float> x = random_floats(rng, n + size, 3.0f);
                for (std::size_t i = n; i < x.size(); ++i) x[i] = kSentinel;
                const std::vector<float> w = random_floats(rng, size);
                const Buf dx(x), dw(w);
                const Buf dout(std::vector<float>(n + size, kSentinel));
                const Buf scratch(std::vector<float>(static_cast<std::size_t>(gpu::kPartials) * nvec, kSentinel));
                dev->rmsnorm(in_place ? dx.f() : dout.f(), dx.f(), dw.f(), nvec, size, eps, scratch.f());
                std::vector<double> want(n + size, 0.0);
                std::vector<char> is_out(n + size, 0);
                for (int v = 0; v < nvec; ++v) {
                    double ss = 0.0;
                    for (int i = 0; i < size; ++i) ss += static_cast<double>(x[v * size + i]) * x[v * size + i];
                    const double scale = 1.0 / std::sqrt(ss / size + static_cast<double>(eps));
                    for (int i = 0; i < size; ++i) {
                        want[static_cast<std::size_t>(v) * size + i] = w[i] * scale * x[v * size + i];
                        is_out[static_cast<std::size_t>(v) * size + i] = 1;
                    }
                }
                g.check(std::string(in_place ? "in place " : "") + shape({nvec, size}), (in_place ? dx : dout).get(), want, is_out);
            }
        }
    }
}

void check_matmul(Rng& rng) {
    Group g("matmul");
    for (int bf16 = 0; bf16 < 2; ++bf16) {
        for (int cols : {1, 31, 32, 33, 100, 264, 520}) {   // 264, 520: several 16-byte groups per thread of the wide kernel
            for (int rows : {1, 5, 13}) {
                for (int ntok : {1, 7, 8, 9, 17}) {
                    const int out_stride = rows + 3, x_stride = cols + 2;
                    const std::vector<float> W = random_floats(rng, static_cast<std::size_t>(rows) * cols);
                    std::vector<std::uint16_t> W16(W.size());
                    for (std::size_t i = 0; i < W.size(); ++i) W16[i] = narrow(W[i]);
                    const std::vector<float> x = random_floats(rng, static_cast<std::size_t>(ntok) * x_stride, 2.0f);
                    const Buf dw(bf16 ? static_cast<const void*>(W16.data()) : W.data(), W.size() * (bf16 ? 2 : 4));
                    const Buf dx(x);
                    const std::size_t out_n = static_cast<std::size_t>(ntok) * out_stride;
                    const Buf dout(std::vector<float>(out_n, kSentinel));
                    const Buf partial(std::vector<float>(static_cast<std::size_t>(gpu::kPartials) * ntok * rows, kSentinel));
                    dev->matmul(dout.f(), out_stride, dw.d, bf16, dx.f(), x_stride, ntok, rows, cols, partial.f());
                    std::vector<double> want(out_n, 0.0);
                    std::vector<char> is_out(out_n, 0);
                    for (int t = 0; t < ntok; ++t) {
                        for (int r = 0; r < rows; ++r) {
                            double acc = 0.0;
                            for (int c = 0; c < cols; ++c) {
                                const std::size_t wi = static_cast<std::size_t>(r) * cols + c;
                                acc += static_cast<double>(bf16 ? widen(W16[wi]) : W[wi]) * x[static_cast<std::size_t>(t) * x_stride + c];
                            }
                            want[static_cast<std::size_t>(t) * out_stride + r] = acc;
                            is_out[static_cast<std::size_t>(t) * out_stride + r] = 1;
                        }
                    }
                    const std::vector<float> got = dout.get();
                    const std::string what = std::string(bf16 ? "bf16 " : "f32 ") + shape({ntok, rows, cols});
                    g.check(what, got, want, is_out);
                    // Every token alone must give the very same bits as in the batch.
                    bool same = true;
                    for (int t = 0; t < ntok; t += (ntok > 9 ? 5 : 3)) {
                        const Buf one(std::vector<float>(static_cast<std::size_t>(rows), kSentinel));
                        dev->matmul(one.f(), rows, dw.d, bf16, dx.f() + static_cast<std::size_t>(t) * x_stride, x_stride, 1, rows, cols,
                                    partial.f());
                        const std::vector<float> o = one.get();
                        same = same && std::memcmp(o.data(), got.data() + static_cast<std::size_t>(t) * out_stride,
                                                   static_cast<std::size_t>(rows) * sizeof(float)) == 0;
                    }
                    g.bits(what + " token alone", same);
                }
            }
        }
    }
}

void check_rope(Rng& rng) {
    Group g("rope");
    for (int half_split = 0; half_split < 2; ++half_split) {
        for (int head_size : {2, 6, 16}) {
            for (int heads : {1, 3}) {
                const int m = 4, n = heads * head_size, stride = n + 5, half = head_size / 2;
                std::vector<float> v = random_floats(rng, static_cast<std::size_t>(m) * stride);
                const std::vector<float> ang = random_floats(rng, static_cast<std::size_t>(m) * half, 3.0f);
                std::vector<float> c(ang.size()), s(ang.size());
                for (std::size_t i = 0; i < ang.size(); ++i) {
                    c[i] = std::cos(ang[i]);
                    s[i] = std::sin(ang[i]);
                }
                std::vector<double> want(v.size(), 0.0);
                std::vector<char> is_out(v.size(), 0);
                for (int i = 0; i < m; ++i) {
                    for (int j = n; j < stride; ++j) v[static_cast<std::size_t>(i) * stride + j] = kSentinel;
                    for (int h = 0; h < heads; ++h) {
                        for (int k = 0; k < half; ++k) {
                            const std::size_t base = static_cast<std::size_t>(i) * stride + static_cast<std::size_t>(h) * head_size;
                            const std::size_t ia = half_split ? base + k : base + 2 * k;
                            const std::size_t ib = half_split ? base + k + half : base + 2 * k + 1;
                            const double cc = c[static_cast<std::size_t>(i) * half + k], ss = s[static_cast<std::size_t>(i) * half + k];
                            want[ia] = v[ia] * cc - v[ib] * ss;
                            want[ib] = v[ia] * ss + v[ib] * cc;
                            is_out[ia] = is_out[ib] = 1;
                        }
                    }
                }
                const Buf dv(v), dc(c), ds(s);
                dev->rope(dv.f(), stride, m, n, dc.f(), ds.f(), head_size, half_split);
                g.check(std::string(half_split ? "half-split " : "interleaved ") + shape({m, heads, head_size}), dv.get(), want, is_out);
            }
        }
    }
}

// Paged attention and store_kv on a pool with several layers. Every row gets its own block table (blocks scattered
// over the pool in a scrambled order); the rows are stored with store_kv and then attend through their tables.
// The reference keeps each row's history as a plain array. Unused pool memory holds NaN.
void check_attention(Rng& rng) {
    Group g("attention");
    struct Case {
        int m, n_heads, kv_mul, head_size, block_size;
        int positions[6];
        float q_scale;
    };
    const Case cases[] = {{1, 2, 1, 4, 16, {0}, 1.0f},          {1, 4, 2, 6, 4, {9}, 1.0f},           {5, 4, 4, 8, 3, {0, 1, 2, 3, 4}, 1.0f},
                          {6, 6, 3, 2, 1, {3, 0, 8, 5, 5, 1}, 1.0f}, {3, 2, 1, 16, 7, {30, 14, 21}, 1.0f}, {4, 4, 2, 8, 5, {20, 21, 4, 5}, 60.0f},
                          {2, 3, 1, 4, 64, {40, 63}, 300.0f},   {3, 4, 2, 4, 2, {7, 7, 6}, 1.0f}};
    const int n_layers = 3, layer = 1;
    for (const Case& c : cases) {
        const int kv_heads = c.n_heads / c.kv_mul, kv_dim = kv_heads * c.head_size, attn = c.n_heads * c.head_size;
        const int bs = c.block_size;
        int max_pos = 0;
        for (int i = 0; i < c.m; ++i) max_pos = std::max(max_pos, c.positions[i]);
        const int stride = max_pos / bs + 1;
        const long block_floats = static_cast<long>(n_layers) * 2 * bs * kv_dim;
        const int n_blocks = c.m * stride + 3;
        // Scrambled block ids: row i, table index b -> a block of its own.
        std::vector<int> order(static_cast<std::size_t>(n_blocks));
        for (int b = 0; b < n_blocks; ++b) order[static_cast<std::size_t>(b)] = b;
        for (int b = n_blocks - 1; b > 0; --b) std::swap(order[static_cast<std::size_t>(b)], order[rng.next() % static_cast<std::uint64_t>(b + 1)]);
        std::vector<int> tables(static_cast<std::size_t>(c.m) * stride);
        for (std::size_t i = 0; i < tables.size(); ++i) tables[i] = order[i];
        // Histories: row i has positions 0 .. positions[i]; the last one is "new" and goes through store_kv.
        std::vector<float> pool(static_cast<std::size_t>(n_blocks) * static_cast<std::size_t>(block_floats), NAN);
        std::vector<std::vector<float>> hist_k(static_cast<std::size_t>(c.m)), hist_v(static_cast<std::size_t>(c.m));
        std::vector<float> new_k(static_cast<std::size_t>(c.m) * kv_dim), new_v(new_k.size());
        const long layer_off = static_cast<long>(layer) * 2 * bs * kv_dim;
        for (int i = 0; i < c.m; ++i) {
            const int n = c.positions[i] + 1;
            hist_k[static_cast<std::size_t>(i)] = random_floats(rng, static_cast<std::size_t>(n) * kv_dim);
            hist_v[static_cast<std::size_t>(i)] = random_floats(rng, static_cast<std::size_t>(n) * kv_dim);
            for (int p = 0; p < n; ++p) {
                const float* ks = hist_k[static_cast<std::size_t>(i)].data() + static_cast<std::size_t>(p) * kv_dim;
                const float* vs = hist_v[static_cast<std::size_t>(i)].data() + static_cast<std::size_t>(p) * kv_dim;
                if (p == c.positions[i]) {   // the new position: not in the pool yet
                    std::memcpy(new_k.data() + static_cast<std::size_t>(i) * kv_dim, ks, static_cast<std::size_t>(kv_dim) * sizeof(float));
                    std::memcpy(new_v.data() + static_cast<std::size_t>(i) * kv_dim, vs, static_cast<std::size_t>(kv_dim) * sizeof(float));
                    continue;
                }
                const long base = static_cast<long>(tables[static_cast<std::size_t>(i) * stride + p / bs]) * block_floats + layer_off +
                                  static_cast<long>(p % bs) * kv_dim;
                std::memcpy(pool.data() + base, ks, static_cast<std::size_t>(kv_dim) * sizeof(float));
                std::memcpy(pool.data() + base + static_cast<long>(bs) * kv_dim, vs, static_cast<std::size_t>(kv_dim) * sizeof(float));
            }
        }
        const std::vector<float> q = random_floats(rng, static_cast<std::size_t>(c.m) * attn, c.q_scale);
        const Buf dq(q), dpool(pool), dk(new_k), dv(new_v);
        const Buf dtables(tables.data(), tables.size() * sizeof(int));
        const Buf dpos(c.positions, static_cast<std::size_t>(c.m) * sizeof(int));
        const gpu::PagedKv kv{dpool.f(), static_cast<const int*>(dtables.d), static_cast<const int*>(dpos.d), stride, block_floats, layer_off, bs, kv_dim};
        dev->store_kv(kv, dk.f(), dv.f(), c.m);
        // store_kv wrote exactly the new rows: the pool now equals the host copy with those rows filled in.
        std::vector<float> want_pool = pool;
        for (int i = 0; i < c.m; ++i) {
            const int p = c.positions[i];
            const long base = static_cast<long>(tables[static_cast<std::size_t>(i) * stride + p / bs]) * block_floats + layer_off +
                              static_cast<long>(p % bs) * kv_dim;
            std::memcpy(want_pool.data() + base, new_k.data() + static_cast<std::size_t>(i) * kv_dim, static_cast<std::size_t>(kv_dim) * sizeof(float));
            std::memcpy(want_pool.data() + base + static_cast<long>(bs) * kv_dim, new_v.data() + static_cast<std::size_t>(i) * kv_dim,
                        static_cast<std::size_t>(kv_dim) * sizeof(float));
        }
        const std::vector<float> got_pool = dpool.get();
        const std::string what = shape({c.m, c.n_heads, c.kv_mul, c.head_size, bs}) + (c.q_scale > 1.0f ? " large scores" : "");
        g.bits("store_kv " + what, std::memcmp(got_pool.data(), want_pool.data(), want_pool.size() * sizeof(float)) == 0);

        const std::size_t out_n = static_cast<std::size_t>(c.m) * attn;
        const Buf dout(std::vector<float>(out_n + 7, kSentinel));
        const int max_len = max_pos + 1;
        const Buf scores(std::vector<float>(static_cast<std::size_t>(c.m) * c.n_heads * max_len, NAN));
        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(c.head_size));
        dev->attention(dout.f(), dq.f(), kv, c.m, max_len, c.n_heads, c.head_size, c.kv_mul, inv_scale, scores.f());
        std::vector<double> want(out_n + 7, 0.0);
        std::vector<char> is_out(out_n + 7, 0);
        for (int i = 0; i < c.m; ++i) {
            const int n = c.positions[i] + 1;
            const std::vector<float>& keys = hist_k[static_cast<std::size_t>(i)];
            const std::vector<float>& values = hist_v[static_cast<std::size_t>(i)];
            for (int h = 0; h < c.n_heads; ++h) {
                const std::size_t qo = static_cast<std::size_t>(i) * attn + static_cast<std::size_t>(h) * c.head_size;
                const std::size_t ko = static_cast<std::size_t>(h / c.kv_mul) * c.head_size;
                std::vector<double> p(static_cast<std::size_t>(n));
                double mx = -INFINITY;
                for (int j = 0; j < n; ++j) {
                    double acc = 0.0;
                    for (int d = 0; d < c.head_size; ++d) acc += static_cast<double>(q[qo + d]) * keys[static_cast<std::size_t>(j) * kv_dim + ko + d];
                    p[static_cast<std::size_t>(j)] = static_cast<double>(static_cast<float>(acc) * inv_scale);
                    mx = std::max(mx, p[static_cast<std::size_t>(j)]);
                }
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    p[static_cast<std::size_t>(j)] = std::exp(p[static_cast<std::size_t>(j)] - mx);
                    sum += p[static_cast<std::size_t>(j)];
                }
                for (int d = 0; d < c.head_size; ++d) {
                    double acc = 0.0;
                    for (int j = 0; j < n; ++j) acc += p[static_cast<std::size_t>(j)] / sum * values[static_cast<std::size_t>(j) * kv_dim + ko + d];
                    want[qo + d] = acc;
                    is_out[qo + d] = 1;
                }
            }
        }
        g.check(what, dout.get(), want, is_out);
        // The pool is only read by attention.
        const std::vector<float> after = dpool.get();
        g.bits("attention leaves the pool alone " + what, std::memcmp(after.data(), got_pool.data(), after.size() * sizeof(float)) == 0);
    }
    // device -> device copy
    const std::vector<float> src = random_floats(rng, 100);
    const Buf a(src), b(std::vector<float>(100, kSentinel));
    dev->copy(b.f() + 10, a.f() + 20, 30 * sizeof(float));
    std::vector<float> want_b(100, kSentinel);
    std::memcpy(want_b.data() + 10, src.data() + 20, 30 * sizeof(float));
    const std::vector<float> got_b = b.get();
    g.bits("copy", std::memcmp(got_b.data(), want_b.data(), 100 * sizeof(float)) == 0);
}

void check_elementwise(Rng& rng) {
    Group g("add/silu");
    for (int n : {1, 33, 1000}) {
        const std::vector<float> a = random_floats(rng, static_cast<std::size_t>(n) + 4, 8.0f);
        const std::vector<float> b = random_floats(rng, static_cast<std::size_t>(n) + 4, 2.0f);
        std::vector<float> a_in = a;
        for (int i = n; i < n + 4; ++i) a_in[static_cast<std::size_t>(i)] = kSentinel;
        std::vector<char> is_out(a.size(), 0);
        std::vector<double> want_add(a.size(), 0.0), want_silu(a.size(), 0.0);
        for (int i = 0; i < n; ++i) {
            const double x = a[static_cast<std::size_t>(i)], y = b[static_cast<std::size_t>(i)];
            is_out[static_cast<std::size_t>(i)] = 1;
            want_add[static_cast<std::size_t>(i)] = x + y;
            want_silu[static_cast<std::size_t>(i)] = x / (1.0 + std::exp(-x)) * y;
        }
        const Buf db(b);
        {
            const Buf da(a_in);
            dev->add(da.f(), db.f(), n);
            g.check("add " + shape({n}), da.get(), want_add, is_out);
        }
        {
            const Buf da(a_in);
            dev->silu_mul(da.f(), db.f(), n);
            g.check("silu_mul " + shape({n}), da.get(), want_silu, is_out);
        }
    }
}

template <class F>
void expect_throw(const char* what, F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return;
    }
    ++failures;
    std::printf("FAIL api [%s]: no exception\n", what);
}

void check_api(const std::string& model_dir) {
    const Checkpoint ck = Checkpoint::load(model_dir);
    const Config& cfg = ck.config();
    if (cfg.seq_len < 24) throw std::runtime_error("the API check needs a model with a context of at least 24 positions");
    const int ctx = 24;
    GpuTransformer model(ck, dev, ctx, 5);
    std::printf("weight_bytes=%zu device_bytes=%zu\n", model.weight_bytes(), model.device_bytes());
    const std::size_t vocab = static_cast<std::size_t>(cfg.vocab_size);
    std::vector<int> tokens(static_cast<std::size_t>(ctx));
    for (int i = 0; i < ctx; ++i) tokens[static_cast<std::size_t>(i)] = (i * 5 + 1) % cfg.vocab_size;

    expect_throw("forward at position 1 of an empty sequence", [&] { model.forward(1, 1); });
    expect_throw("batch at position 1 of an empty sequence", [&] { model.forward_batch(tokens.data(), 2, 1); });
    expect_throw("batch of no tokens", [&] { model.forward_batch(tokens.data(), 0, 0); });
    expect_throw("position beyond the context", [&] { model.forward(1, ctx); });
    expect_throw("negative position", [&] { model.forward(1, -1); });
    expect_throw("batch beyond the context", [&] { model.forward_batch(tokens.data(), ctx, 1); });

    // Reference run: token by token.
    std::vector<std::vector<float>> ref(static_cast<std::size_t>(ctx));
    for (int pos = 0; pos < ctx; ++pos) {
        const float* l = model.forward(tokens[static_cast<std::size_t>(pos)], pos);
        ref[static_cast<std::size_t>(pos)].assign(l, l + vocab);
    }
    expect_throw("forward one past the end", [&] { model.forward(1, ctx); });
    auto same = [&](const char* what, const float* got, int pos) {
        if (!got || std::memcmp(got, ref[static_cast<std::size_t>(pos)].data(), vocab * sizeof(float)) != 0) {
            ++failures;
            std::printf("FAIL api [%s]: logits of position %d differ from the first run\n", what, pos);
        }
    };
    // Rewind to the middle and feed the same tokens again: same logits.
    const int mid = ctx / 2;
    same("rewind, single", model.forward(tokens[static_cast<std::size_t>(mid)], mid), mid);
    expect_throw("gap after a rewind", [&] { model.forward(1, mid + 2); });
    expect_throw("batch gap after a rewind", [&] { model.forward_batch(tokens.data(), 1, mid + 2); });
    same("append after a rewind", model.forward(tokens[static_cast<std::size_t>(mid) + 1], mid + 1), mid + 1);
    // A batch longer than max_batch (5) from position 2, on top of the kept prefix.
    same("rewind, batch in pieces", model.forward_batch(tokens.data() + 2, ctx - 2, 2), ctx - 1);
    if (model.forward_batch(tokens.data(), 3, 0, false) != nullptr || model.forward(tokens[3], 3, false) != nullptr) {
        ++failures;
        std::printf("FAIL api: logits returned although none were asked for\n");
    }
    same("after calls without logits", model.forward(tokens[4], 4), 4);
    // Other tokens at the start change the logits (the comparison above is not vacuous).
    const float* other = model.forward((tokens[0] + 1) % cfg.vocab_size, 0);
    if (std::memcmp(other, ref[0].data(), vocab * sizeof(float)) == 0) {
        ++failures;
        std::printf("FAIL api: another token gives the same logits\n");
    }
    // Sequences: one of another model's pool is refused; two sequences of this pool do not disturb each other.
    {
        GpuTransformer other(ck, dev, ctx, 5, 4, 0);
        Sequence foreign = other.new_sequence();
        expect_throw("forward on a sequence of another pool", [&] { model.forward(foreign, 1, 0); });
        expect_throw("forward_batch on a sequence of another pool", [&] { model.forward_batch(foreign, tokens.data(), 2, 0); });
        {
            GpuTransformer rows_model(ck, dev, ctx, 5, 4, 12);   // room for several sequences
            Sequence own = rows_model.new_sequence();
            const SequenceModel::StepRow mixed[2] = {{&own, 1, true}, {&foreign, 1, true}};
            expect_throw("forward_rows with a sequence of another pool", [&] { rows_model.forward_rows(mixed, 2); });
            if (own.length() != 0 || own.n_blocks() != 0) {
                ++failures;
                std::printf("FAIL api: a refused forward_rows changed a sequence\n");
            }
            // Twelve rows of three sequences in one pass, every row asking for logits (more than one classifier group).
            Sequence s1 = rows_model.new_sequence(), s2 = rows_model.new_sequence();
            std::vector<SequenceModel::StepRow> rows;
            for (int i = 0; i < 4; ++i) {
                rows.push_back({&own, tokens[static_cast<std::size_t>(i)], true});
                rows.push_back({&s1, tokens[static_cast<std::size_t>(i)], true});
                rows.push_back({&s2, tokens[static_cast<std::size_t>(i)], true});
            }
            // max_batch is 5 here: too many rows are refused, five are fine.
            expect_throw("more rows than max_batch", [&] { rows_model.forward_rows(rows.data(), 12); });
            const float* l = rows_model.forward_rows(rows.data(), 5);   // own: positions 0, 1; s1: 0, 1; s2: 0
            const int row_pos[5] = {0, 0, 0, 1, 1};
            for (int r = 0; r < 5; ++r) same("forward_rows, three sequences", l ? l + static_cast<std::size_t>(r) * vocab : nullptr, row_pos[r]);
        }
        Sequence a = other.new_sequence(), b = other.new_sequence();
        const int na = ctx / 2, nb = ctx / 2 - 4;   // block size 4: together they leave exactly one block free
        for (int pos = 0; pos < na; ++pos) {
            const float* la = other.forward(a, tokens[static_cast<std::size_t>(pos)], pos);
            same("two sequences, first", la, pos);
            if (pos < nb) other.forward(b, (tokens[static_cast<std::size_t>(pos)] + 3) % cfg.vocab_size, pos, false);
        }
        KvPool& pool = other.kv_pool();
        const int free_before = pool.free_blocks(), blocks_before = b.n_blocks();
        // 8 more positions need two blocks, one is free: refused, and the block taken on the way is given back.
        expect_throw("batch the pool cannot hold", [&] { other.forward_batch(b, tokens.data(), 8, nb); });
        if (ctx != 24 || free_before != 1 || b.length() != nb || b.n_blocks() != blocks_before || pool.free_blocks() != free_before) {
            ++failures;
            std::printf("FAIL api: a refused batch changed the sequence or kept a block (free %d -> %d, blocks %d -> %d)\n", free_before,
                        pool.free_blocks(), blocks_before, b.n_blocks());
        }
        expect_throw("host access to device blocks", [&] { pool.keys(0, 0); });
        GpuTransformer big_blocks(ck, dev, ctx, 5, 1000, 0);
        if (model.kv_pool().block_size() != 16 || model.kv_pool().n_blocks() != 2 || big_blocks.kv_pool().block_size() != ctx ||
            big_blocks.kv_pool().n_blocks() != 1 || pool.block_size() != 4 || pool.n_blocks() != 6) {
            ++failures;
            std::printf("FAIL api: default block size / pool size are wrong\n");
        }
    }
    std::printf("api        checked\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string device = "emu", model;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--device" || a == "--model") && i + 1 < argc) {
            (a == "--device" ? device : model) = argv[++i];
        } else {
            std::fprintf(stderr, "usage: %s --device cuda|emu [--model DIR]\n", argv[0]);
            return 2;
        }
    }
    try {
        dev = gpu::device_by_name(device);
        std::printf("device: %s (%s)\n", dev->name, dev->describe().c_str());
        Rng rng(2026);
        check_embed(rng);
        check_rmsnorm(rng);
        check_matmul(rng);
        check_rope(rng);
        check_attention(rng);
        check_elementwise(rng);
        if (!model.empty()) check_api(model);
        dev->sync();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
