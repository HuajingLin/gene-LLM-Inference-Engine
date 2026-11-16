// device_kernels.inc - the device kernels and the operations built from them. No include guard: this file is
// included exactly once by each backend (device_cuda.cu and device_host.cpp), inside its own namespace, after it
// has defined
//   GI_DEV                        qualifiers of a function that runs on the device
//   template <class K> void launch(const typename K::Args& args, long n);
// launch() runs K::body(i, args) once for every index i in [0, n): on a GPU as n threads, in the emulation as a
// loop. A body therefore
//   - writes only elements that belong to its own index, and reads nothing that another index of the same launch
//     writes (bodies of one launch run in any order, or all at once);
//   - uses no shared memory and no synchronization, so the loop of the emulation executes exactly the same code.
// Results do not depend on the order of the indices; the emulation can run them backwards or shuffled to check it.
//
// Only plain C and <math.h> functions are used in the bodies (fmaf, expf, sqrtf): they exist on both sides.

// A bfloat16 value is the upper half of a float32: widening is a 16-bit shift, exact.
static GI_DEV float gi_widen_bf16(unsigned short bits) {
    union {
        unsigned int u;
        float f;
    } v;
    v.u = static_cast<unsigned int>(bits) << 16;
    return v.f;
}

constexpr int kTokChunk = kTokensPerPass;   // tokens that share one pass over a weight row in matmul

// ---------------------------------------------------------------------------------------------------------------
// embed: index = (row, d)
struct EmbedKernel {
    struct Args {
        float* x;
        const void* table;
        const int* tokens;
        int bf16, dim;
    };
    static GI_DEV void body(long i, const Args& a) {
        const long row = i / a.dim;
        const long d = i % a.dim;
        const long src = static_cast<long>(a.tokens[row]) * a.dim + d;
        a.x[i] = a.bf16 ? gi_widen_bf16(static_cast<const unsigned short*>(a.table)[src])
                        : static_cast<const float*>(a.table)[src];
    }
};

// ---------------------------------------------------------------------------------------------------------------
// rmsnorm, step 1: index = (vector, s). Thread s adds the squares of the elements s, s + 32, s + 64, ... (the same
// split into 32 partial sums as matmul: one thread adding a whole vector alone takes far longer than the launch).
struct RmsnormPartialKernel {
    struct Args {
        float* partial;   // [nvec, kPartials]
        const float* x;
        int size;
    };
    static GI_DEV void body(long i, const Args& a) {
        const int s = static_cast<int>(i % kPartials);
        const float* x = a.x + (i / kPartials) * a.size;
        float acc = 0.0f;
        for (int j = s; j < a.size; j += kPartials) acc = fmaf(x[j], x[j], acc);
        a.partial[i] = acc;
    }
};

// rmsnorm, step 2: index = (vector, element). Every thread adds the 32 partial sums of its vector pairwise and
// scales its own element; it reads x[i] before it writes out[i], so out == x is fine.
struct RmsnormScaleKernel {
    struct Args {
        float* out;
        const float* x;
        const float* weight;
        const float* partial;
        int size;
        float eps;
    };
    static GI_DEV void body(long i, const Args& a) {
        const long vec = i / a.size;
        const int d = static_cast<int>(i % a.size);
        const float* p = a.partial + vec * kPartials;
        float v[kPartials];
        for (int q = 0; q < kPartials; ++q) v[q] = p[q];
        for (int width = kPartials / 2; width >= 1; width /= 2) {
            for (int q = 0; q < width; ++q) v[q] = v[q] + v[q + width];
        }
        const float scale = 1.0f / sqrtf(v[0] / static_cast<float>(a.size) + a.eps);
        a.out[i] = a.weight[d] * (scale * a.x[i]);
    }
};

// ---------------------------------------------------------------------------------------------------------------
// matmul, step 1: index = (token chunk, row, s). Thread s of a row sums the columns s, s + 32, s + 64, ... so the
// 32 threads of a GPU warp read 32 neighbouring weights at every step (one coalesced memory access). Up to
// kTokChunk tokens are handled by the same thread: each weight is read once and used for all of them. The sum of
// one (token, row, s) is always the same chain of fused multiply-adds, whatever the number of tokens.
struct MatmulPartialKernel {
    struct Args {
        float* partial;   // [ntok, rows, kPartials]
        const void* W;
        const float* x;
        long x_stride;
        int bf16, ntok, rows, cols;
    };
    static GI_DEV void body(long i, const Args& a) {
        const int s = static_cast<int>(i % kPartials);
        const long rest = i / kPartials;
        const int r = static_cast<int>(rest % a.rows);
        const int t0 = static_cast<int>(rest / a.rows) * kTokChunk;
        const int k = a.ntok - t0 < kTokChunk ? a.ntok - t0 : kTokChunk;
        float acc[kTokChunk];
        for (int t = 0; t < kTokChunk; ++t) acc[t] = 0.0f;
        const float* x0 = a.x + static_cast<long>(t0) * a.x_stride;
        const long row_off = static_cast<long>(r) * a.cols;
        if (a.bf16) {
            const unsigned short* w = static_cast<const unsigned short*>(a.W) + row_off;
            for (int c = s; c < a.cols; c += kPartials) {
                const float wv = gi_widen_bf16(w[c]);
                for (int t = 0; t < k; ++t) acc[t] = fmaf(wv, x0[static_cast<long>(t) * a.x_stride + c], acc[t]);
            }
        } else {
            const float* w = static_cast<const float*>(a.W) + row_off;
            for (int c = s; c < a.cols; c += kPartials) {
                const float wv = w[c];
                for (int t = 0; t < k; ++t) acc[t] = fmaf(wv, x0[static_cast<long>(t) * a.x_stride + c], acc[t]);
            }
        }
        for (int t = 0; t < k; ++t) {
            a.partial[(static_cast<long>(t0 + t) * a.rows + r) * kPartials + s] = acc[t];
        }
    }
};

// The wide variant of step 1. Thread s of a row takes the column groups s, s + 32, s + 64, ...; a group is 16 bytes
// of weights (8 bfloat16 or 4 float32 values) fetched with ONE aligned 16-byte load, so a warp reads 512
// neighbouring bytes per step instead of 64 or 128. Inside a group the columns are added in increasing order.
// Requires: cols a multiple of the group size (then every row starts on a 16-byte boundary of the matrix).
struct alignas(16) GiWord4 {
    unsigned int w[4];
};

struct MatmulWideKernel {
    typedef MatmulPartialKernel::Args Args;
    static GI_DEV void body(long i, const Args& a) {
        const int s = static_cast<int>(i % kPartials);
        const long rest = i / kPartials;
        const int r = static_cast<int>(rest % a.rows);
        const int t0 = static_cast<int>(rest / a.rows) * kTokChunk;
        const int k = a.ntok - t0 < kTokChunk ? a.ntok - t0 : kTokChunk;
        float acc[kTokChunk];
        for (int t = 0; t < kTokChunk; ++t) acc[t] = 0.0f;
        const float* x0 = a.x + static_cast<long>(t0) * a.x_stride;
        const int per_group = a.bf16 ? 8 : 4;
        const int groups = a.cols / per_group;
        const long row_bytes = static_cast<long>(r) * a.cols * (a.bf16 ? 2 : 4);
        const GiWord4* w = reinterpret_cast<const GiWord4*>(static_cast<const char*>(a.W) + row_bytes);
        for (int g = s; g < groups; g += kPartials) {
#ifdef __CUDACC__
            const GiWord4 v = w[g];   // one aligned 16-byte load
#else
            GiWord4 v;                // the emulation copies the same bytes without type punning
            memcpy(&v, w + g, sizeof v);
#endif
            const int c0 = g * per_group;
            if (a.bf16) {
                for (int j = 0; j < 8; ++j) {   // value j: the low or high half of word j / 2 (little-endian)
                    const float wv = gi_widen_bf16(static_cast<unsigned short>((v.w[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu));
                    for (int t = 0; t < k; ++t) acc[t] = fmaf(wv, x0[static_cast<long>(t) * a.x_stride + c0 + j], acc[t]);
                }
            } else {
                for (int j = 0; j < 4; ++j) {
                    union {
                        unsigned int u;
                        float f;
                    } bits;
                    bits.u = v.w[j];
                    for (int t = 0; t < k; ++t) acc[t] = fmaf(bits.f, x0[static_cast<long>(t) * a.x_stride + c0 + j], acc[t]);
                }
            }
        }
        for (int t = 0; t < k; ++t) {
            a.partial[(static_cast<long>(t0 + t) * a.rows + r) * kPartials + s] = acc[t];
        }
    }
};

// matmul, step 2: index = (token, row). Adds the 32 partial sums pairwise (16 + 16, 8 + 8, ...).
struct MatmulReduceKernel {
    struct Args {
        float* out;
        long out_stride;
        const float* partial;
        int rows;
    };
    static GI_DEV void body(long j, const Args& a) {
        const long t = j / a.rows;
        const long r = j % a.rows;
        const float* p = a.partial + j * kPartials;
        float v[kPartials];
        for (int q = 0; q < kPartials; ++q) v[q] = p[q];
        for (int width = kPartials / 2; width >= 1; width /= 2) {
            for (int q = 0; q < width; ++q) v[q] = v[q] + v[q + width];
        }
        a.out[t * a.out_stride + r] = v[0];
    }
};

// ---------------------------------------------------------------------------------------------------------------
// rope: index = (row, pair of values).
struct RopeKernel {
    struct Args {
        float* v;
        long stride;
        const float* cos_rows;
        const float* sin_rows;
        int pairs, head_size, half_split;   // pairs = values per row / 2
    };
    static GI_DEV void body(long i, const Args& a) {
        const long row = i / a.pairs;
        const int p = static_cast<int>(i % a.pairs);
        const int half = a.head_size / 2;
        const int base = (p / half) * a.head_size;
        const int k = p % half;
        const float c = a.cos_rows[row * half + k];
        const float s = a.sin_rows[row * half + k];
        float* v = a.v + row * a.stride + base;
        float* pa = a.half_split ? v + k : v + 2 * k;
        float* pb = a.half_split ? v + k + half : v + 2 * k + 1;
        const float a0 = *pa;
        const float b0 = *pb;
        *pa = a0 * c - b0 * s;
        *pb = a0 * s + b0 * c;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Paged KV cache: where the keys of position `pos` of row `row` start (add block_size * kv_dim for the values).
static GI_DEV const float* gi_kv_row(const PagedKv& kv, long row, int pos) {
    const long block = kv.tables[row * kv.table_stride + pos / kv.block_size];
    return kv.pool + block * kv.block_floats + kv.layer_offset + static_cast<long>(pos % kv.block_size) * kv.kv_dim;
}

// store_kv: index = (keys or values, row, d).
struct StoreKvKernel {
    struct Args {
        PagedKv kv;
        const float* k;
        const float* v;
        int m;
    };
    static GI_DEV void body(long i, const Args& a) {
        const long per = static_cast<long>(a.m) * a.kv.kv_dim;
        const int which = static_cast<int>(i / per);   // 0: keys, 1: values
        const long rem = i % per;
        const long row = rem / a.kv.kv_dim;
        const long d = rem % a.kv.kv_dim;
        const float* dst = gi_kv_row(a.kv, row, a.kv.positions[row]) + (which ? static_cast<long>(a.kv.block_size) * a.kv.kv_dim : 0) + d;
        const_cast<float*>(dst)[0] = (which ? a.v : a.k)[rem];
    }
};

// attention, step 1: index = (row, head, key position j). score = q . k_j / sqrt(head_size) for j <= the row's
// own position; later positions are left untouched (and never read).
struct AttScoreKernel {
    struct Args {
        float* scores;   // [m, n_heads, len]
        const float* q;
        PagedKv kv;
        int len, n_heads, head_size, kv_mul;
        float inv_scale;
    };
    static GI_DEV void body(long i, const Args& a) {
        const int j = static_cast<int>(i % a.len);
        const long rest = i / a.len;
        const int h = static_cast<int>(rest % a.n_heads);
        const long row = rest / a.n_heads;
        if (j > a.kv.positions[row]) return;
        const float* q = a.q + (row * a.n_heads + h) * a.head_size;
        const float* k = gi_kv_row(a.kv, row, j) + (h / a.kv_mul) * a.head_size;
        float acc = 0.0f;
        for (int d = 0; d < a.head_size; ++d) acc = fmaf(q[d], k[d], acc);
        a.scores[i] = acc * a.inv_scale;
    }
};

// attention, step 2: index = (row, head). Softmax over the scores of positions 0 .. pos, in place. The sum is
// accumulated in double like on the CPU: a float sum over thousands of terms loses several digits.
struct AttSoftmaxKernel {
    struct Args {
        float* scores;
        const int* positions;
        int len, n_heads;
    };
    static GI_DEV void body(long i, const Args& a) {
        const int n = a.positions[i / a.n_heads] + 1;
        float* s = a.scores + i * a.len;
        float max_val = s[0];
        for (int j = 1; j < n; ++j) {
            if (s[j] > max_val) max_val = s[j];
        }
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            s[j] = expf(s[j] - max_val);
            sum += static_cast<double>(s[j]);
        }
        const float fsum = static_cast<float>(sum);
        for (int j = 0; j < n; ++j) s[j] /= fsum;
    }
};

// attention, step 3: index = (row, head, d). out = sum_j p_j * v_j[d], block after block in position order (so
// the chain of additions is the same for every block size).
struct AttMixKernel {
    struct Args {
        float* out;
        const float* scores;
        PagedKv kv;
        int len, n_heads, head_size, kv_mul;
    };
    static GI_DEV void body(long i, const Args& a) {
        const int d = static_cast<int>(i % a.head_size);
        const long rh = i / a.head_size;
        const int h = static_cast<int>(rh % a.n_heads);
        const long row = rh / a.n_heads;
        const int pos = a.kv.positions[row];
        const float* p = a.scores + rh * a.len;
        const long value_off = static_cast<long>(a.kv.block_size) * a.kv.kv_dim + (h / a.kv_mul) * a.head_size + d;
        float acc = 0.0f;
        for (int start = 0; start <= pos; start += a.kv.block_size) {
            const float* v = gi_kv_row(a.kv, row, start) + value_off;
            const int count = pos + 1 - start < a.kv.block_size ? pos + 1 - start : a.kv.block_size;
            for (int j = 0; j < count; ++j) acc = fmaf(p[start + j], v[static_cast<long>(j) * a.kv.kv_dim], acc);
        }
        a.out[i] = acc;
    }
};

// ---------------------------------------------------------------------------------------------------------------
struct AddKernel {
    struct Args {
        float* a;
        const float* b;
    };
    static GI_DEV void body(long i, const Args& a) { a.a[i] += a.b[i]; }
};

struct SiluMulKernel {
    struct Args {
        float* a;
        const float* b;
    };
    static GI_DEV void body(long i, const Args& a) {
        const float x = a.a[i];
        a.a[i] = x * (1.0f / (1.0f + expf(-x))) * a.b[i];
    }
};

// ---------------------------------------------------------------------------------------------------------------
// The operations of DeviceOps (host functions: they only launch kernels).

// GI_DEV_MATMUL=narrow selects the one-weight-per-step kernel; wide (the default) the 16-byte kernel where the
// shape allows it. Measured on a Tesla T4 with Llama-3.2-1B: wide decodes about 1.5 times as fast.
static bool gi_matmul_wide() {
    static const bool wide = [] {
        const char* v = getenv("GI_DEV_MATMUL");
        if (!v || !*v || strcmp(v, "wide") == 0) return true;
        if (strcmp(v, "narrow") == 0) return false;
        throw std::runtime_error(std::string("GI_DEV_MATMUL must be narrow or wide, got ") + v);
    }();
    return wide;
}

static void op_embed(float* x, const void* table, int bf16, const int* tokens, int m, int dim) {
    const EmbedKernel::Args a{x, table, tokens, bf16, dim};
    launch<EmbedKernel>(a, static_cast<long>(m) * dim);
}

static void op_rmsnorm(float* out, const float* x, const float* weight, int nvec, int size, float eps, float* partial) {
    const RmsnormPartialKernel::Args p{partial, x, size};
    launch<RmsnormPartialKernel>(p, static_cast<long>(nvec) * kPartials);
    const RmsnormScaleKernel::Args a{out, x, weight, partial, size, eps};
    launch<RmsnormScaleKernel>(a, static_cast<long>(nvec) * size);
}

static void op_matmul(float* out, long out_stride, const void* W, int bf16, const float* x, long x_stride, int ntok,
                      int rows, int cols, float* partial) {
    const MatmulPartialKernel::Args p{partial, W, x, x_stride, bf16, ntok, rows, cols};
    const long chunks = (ntok + kTokChunk - 1) / kTokChunk;
    if (gi_matmul_wide() && cols % (bf16 ? 8 : 4) == 0) launch<MatmulWideKernel>(p, chunks * rows * kPartials);
    else launch<MatmulPartialKernel>(p, chunks * rows * kPartials);
    const MatmulReduceKernel::Args r{out, out_stride, partial, rows};
    launch<MatmulReduceKernel>(r, static_cast<long>(ntok) * rows);
}

static void op_rope(float* v, long stride, int m, int n, const float* cos_rows, const float* sin_rows, int head_size,
                    int half_split) {
    const RopeKernel::Args a{v, stride, cos_rows, sin_rows, n / 2, head_size, half_split};
    launch<RopeKernel>(a, static_cast<long>(m) * (n / 2));
}

static void op_store_kv(const PagedKv& kv, const float* k, const float* v, int m) {
    const StoreKvKernel::Args a{kv, k, v, m};
    launch<StoreKvKernel>(a, 2L * m * kv.kv_dim);
}

static void op_attention(float* out, const float* q, const PagedKv& kv, int m, int max_len, int n_heads, int head_size,
                         int kv_mul, float inv_scale, float* scores) {
    const long row_heads = static_cast<long>(m) * n_heads;
    const AttScoreKernel::Args s{scores, q, kv, max_len, n_heads, head_size, kv_mul, inv_scale};
    launch<AttScoreKernel>(s, row_heads * max_len);
    const AttSoftmaxKernel::Args f{scores, kv.positions, max_len, n_heads};
    launch<AttSoftmaxKernel>(f, row_heads);
    const AttMixKernel::Args x{out, scores, kv, max_len, n_heads, head_size, kv_mul};
    launch<AttMixKernel>(x, row_heads * head_size);
}

static void op_add(float* a, const float* b, long n) {
    const AddKernel::Args k{a, b};
    launch<AddKernel>(k, n);
}

static void op_silu_mul(float* a, const float* b, long n) {
    const SiluMulKernel::Args k{a, b};
    launch<SiluMulKernel>(k, n);
}

static void fill_kernel_ops(DeviceOps& t) {
    t.embed = op_embed;
    t.rmsnorm = op_rmsnorm;
    t.matmul = op_matmul;
    t.rope = op_rope;
    t.store_kv = op_store_kv;
    t.attention = op_attention;
    t.add = op_add;
    t.silu_mul = op_silu_mul;
}
