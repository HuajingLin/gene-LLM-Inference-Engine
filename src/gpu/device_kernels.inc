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

constexpr int kTokChunk = 8;   // tokens that share one pass over a weight row in matmul

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
// rmsnorm: index = vector. One thread reads its whole vector, then writes it (so out == x is fine).
struct RmsnormKernel {
    struct Args {
        float* out;
        const float* x;
        const float* weight;
        int size;
        float eps;
    };
    static GI_DEV void body(long v, const Args& a) {
        const float* x = a.x + v * a.size;
        float* out = a.out + v * a.size;
        float sum_sq = 0.0f;
        for (int i = 0; i < a.size; ++i) sum_sq = fmaf(x[i], x[i], sum_sq);
        const float scale = 1.0f / sqrtf(sum_sq / static_cast<float>(a.size) + a.eps);
        for (int i = 0; i < a.size; ++i) out[i] = a.weight[i] * (scale * x[i]);
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
// attention, step 1: index = (row, head, key position j). score = q . k_j / sqrt(head_size) for j <= the row's
// own position; later positions are left untouched (and never read).
struct AttScoreKernel {
    struct Args {
        float* scores;   // [m, n_heads, len]
        const float* q;
        const float* keys;
        int len, pos0, n_heads, head_size, kv_mul, kv_dim;
        float inv_scale;
    };
    static GI_DEV void body(long i, const Args& a) {
        const int j = static_cast<int>(i % a.len);
        const long rest = i / a.len;
        const int h = static_cast<int>(rest % a.n_heads);
        const long row = rest / a.n_heads;
        if (j > a.pos0 + row) return;
        const float* q = a.q + (row * a.n_heads + h) * a.head_size;
        const float* k = a.keys + static_cast<long>(j) * a.kv_dim + (h / a.kv_mul) * a.head_size;
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
        int len, pos0, n_heads;
    };
    static GI_DEV void body(long i, const Args& a) {
        const long row = i / a.n_heads;
        const int n = a.pos0 + static_cast<int>(row) + 1;
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

// attention, step 3: index = (row, head, d). out = sum_j p_j * v_j[d].
struct AttMixKernel {
    struct Args {
        float* out;
        const float* scores;
        const float* values;
        int len, pos0, n_heads, head_size, kv_mul, kv_dim;
    };
    static GI_DEV void body(long i, const Args& a) {
        const int d = static_cast<int>(i % a.head_size);
        const long rh = i / a.head_size;
        const int h = static_cast<int>(rh % a.n_heads);
        const long row = rh / a.n_heads;
        const int n = a.pos0 + static_cast<int>(row) + 1;
        const float* p = a.scores + rh * a.len;
        const float* v = a.values + (h / a.kv_mul) * a.head_size + d;
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) acc = fmaf(p[j], v[static_cast<long>(j) * a.kv_dim], acc);
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

static void op_embed(float* x, const void* table, int bf16, const int* tokens, int m, int dim) {
    const EmbedKernel::Args a{x, table, tokens, bf16, dim};
    launch<EmbedKernel>(a, static_cast<long>(m) * dim);
}

static void op_rmsnorm(float* out, const float* x, const float* weight, int nvec, int size, float eps) {
    const RmsnormKernel::Args a{out, x, weight, size, eps};
    launch<RmsnormKernel>(a, nvec);
}

static void op_matmul(float* out, long out_stride, const void* W, int bf16, const float* x, long x_stride, int ntok,
                      int rows, int cols, float* partial) {
    const MatmulPartialKernel::Args p{partial, W, x, x_stride, bf16, ntok, rows, cols};
    const long chunks = (ntok + kTokChunk - 1) / kTokChunk;
    launch<MatmulPartialKernel>(p, chunks * rows * kPartials);
    const MatmulReduceKernel::Args r{out, out_stride, partial, rows};
    launch<MatmulReduceKernel>(r, static_cast<long>(ntok) * rows);
}

static void op_rope(float* v, long stride, int m, int n, const float* cos_rows, const float* sin_rows, int head_size,
                    int half_split) {
    const RopeKernel::Args a{v, stride, cos_rows, sin_rows, n / 2, head_size, half_split};
    launch<RopeKernel>(a, static_cast<long>(m) * (n / 2));
}

static void op_attention(float* out, const float* q, const float* keys, const float* values, int m, int pos0,
                         int n_heads, int head_size, int kv_mul, int kv_dim, float inv_scale, float* scores) {
    const int len = pos0 + m;
    const long row_heads = static_cast<long>(m) * n_heads;
    const AttScoreKernel::Args s{scores, q, keys, len, pos0, n_heads, head_size, kv_mul, kv_dim, inv_scale};
    launch<AttScoreKernel>(s, row_heads * len);
    const AttSoftmaxKernel::Args f{scores, len, pos0, n_heads};
    launch<AttSoftmaxKernel>(f, row_heads);
    const AttMixKernel::Args x{out, scores, values, len, pos0, n_heads, head_size, kv_mul, kv_dim};
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
    t.attention = op_attention;
    t.add = op_add;
    t.silu_mul = op_silu_mul;
}
