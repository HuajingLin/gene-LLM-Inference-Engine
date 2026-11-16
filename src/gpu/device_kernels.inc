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

#ifndef __CUDACC__
// The GPU reads 16 bytes at once only from 16-byte boundaries. The emulation copies bytes and would not care, so it
// checks the address instead: a kernel that would do an unaligned load on the GPU fails here.
static inline void gi_emu_check_aligned(const void* p) {
    if (reinterpret_cast<std::uintptr_t>(p) % 16 != 0) throw std::logic_error("emu device: unaligned 16-byte load");
}
#endif

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
        float* partial;   // [ntok, rows, 1 << log2p]
        const void* W;
        const float* x;
        long x_stride;
        int bf16, ntok, rows, cols;
        int log2p;        // the partial sums per output element are 2^log2p (8, 16 or 32)
    };
    static GI_DEV void body(long i, const Args& a) {
        const unsigned u = static_cast<unsigned>(i);   // 32-bit index math: 64-bit division is slow on a GPU
        const int np = 1 << a.log2p;
        const int s = static_cast<int>(u & static_cast<unsigned>(np - 1));
        const unsigned rest = u >> a.log2p;
        const int r = static_cast<int>(rest % static_cast<unsigned>(a.rows));
        const int t0 = static_cast<int>(rest / static_cast<unsigned>(a.rows)) * kTokChunk;
        const int k = a.ntok - t0 < kTokChunk ? a.ntok - t0 : kTokChunk;
        float acc[kTokChunk];
        for (int t = 0; t < kTokChunk; ++t) acc[t] = 0.0f;
        const float* x0 = a.x + static_cast<long>(t0) * a.x_stride;
        const long row_off = static_cast<long>(r) * a.cols;
        if (a.bf16) {
            const unsigned short* w = static_cast<const unsigned short*>(a.W) + row_off;
            for (int c = s; c < a.cols; c += np) {
                const float wv = gi_widen_bf16(w[c]);
                for (int t = 0; t < k; ++t) acc[t] = fmaf(wv, x0[static_cast<long>(t) * a.x_stride + c], acc[t]);
            }
        } else {
            const float* w = static_cast<const float*>(a.W) + row_off;
            for (int c = s; c < a.cols; c += np) {
                const float wv = w[c];
                for (int t = 0; t < k; ++t) acc[t] = fmaf(wv, x0[static_cast<long>(t) * a.x_stride + c], acc[t]);
            }
        }
        for (int t = 0; t < k; ++t) {
            a.partial[(static_cast<long>(t0 + t) * a.rows + r) * np + s] = acc[t];
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
        const unsigned u = static_cast<unsigned>(i);   // 32-bit index math: 64-bit division is slow on a GPU
        const int np = 1 << a.log2p;
        const int s = static_cast<int>(u & static_cast<unsigned>(np - 1));
        const unsigned rest = u >> a.log2p;
        const int r = static_cast<int>(rest % static_cast<unsigned>(a.rows));
        const int t0 = static_cast<int>(rest / static_cast<unsigned>(a.rows)) * kTokChunk;
        const int k = a.ntok - t0 < kTokChunk ? a.ntok - t0 : kTokChunk;
        float acc[kTokChunk];
        for (int t = 0; t < kTokChunk; ++t) acc[t] = 0.0f;
        const float* x0 = a.x + static_cast<long>(t0) * a.x_stride;
        const int per_group = a.bf16 ? 8 : 4;
        const int groups = a.cols / per_group;
        const long row_bytes = static_cast<long>(r) * a.cols * (a.bf16 ? 2 : 4);
        const GiWord4* w = reinterpret_cast<const GiWord4*>(static_cast<const char*>(a.W) + row_bytes);
        for (int g = s; g < groups; g += np) {
#ifdef __CUDACC__
            const GiWord4 v = w[g];   // one aligned 16-byte load
#else
            gi_emu_check_aligned(w + g);
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
            a.partial[(static_cast<long>(t0 + t) * a.rows + r) * np + s] = acc[t];
        }
    }
};

// The tile variant of step 1: index = (token chunk, group of R rows, s). One thread now holds a small tile of
// the product in registers: per 16 bytes of columns it loads the activations of up to 8 tokens ONCE (16-byte loads)
// and uses them for R weight rows, so there are a few loads per hundreds of multiply-adds instead of one
// load per multiply-add. Every (token, row, s) sum is the same chain of operations as in the wide kernel: the two
// give the same bits. Requires, besides the wide kernel's condition, activations that start on a 16-byte boundary
// with a row stride that is a multiple of 4 floats.

static GI_DEV void gi_load4(const float* p, float* out) {
#ifdef __CUDACC__
    struct alignas(16) Float4 {
        float f[4];
    };
    const Float4 v = *reinterpret_cast<const Float4*>(p);   // one aligned 16-byte load
    out[0] = v.f[0];
    out[1] = v.f[1];
    out[2] = v.f[2];
    out[3] = v.f[3];
#else
    gi_emu_check_aligned(p);
    memcpy(out, p, 4 * sizeof(float));
#endif
}

// nr rows (r0 ..), k tokens (t0 ..). Called with constant nr, k and bf16 for the common shapes, so that the loops
// have fixed lengths there.
template <int R>
static GI_DEV void gi_tile_body(const MatmulPartialKernel::Args& a, int s, int r0, int nr, int t0, int k, int bf16) {
    const int np = 1 << a.log2p;
    float acc[R][kTokChunk];
    for (int rr = 0; rr < R; ++rr) {
        for (int t = 0; t < kTokChunk; ++t) acc[rr][t] = 0.0f;
    }
    const int per_group = bf16 ? 8 : 4;
    const int groups = a.cols / per_group;
    const long row_bytes = static_cast<long>(a.cols) * (bf16 ? 2 : 4);
    const char* w0 = static_cast<const char*>(a.W) + static_cast<long>(r0) * row_bytes;
    const float* x0 = a.x + static_cast<long>(t0) * a.x_stride;
    for (int g = s; g < groups; g += np) {
        float xv[kTokChunk][8];
        for (int t = 0; t < k; ++t) {
            const float* xt = x0 + static_cast<long>(t) * a.x_stride + static_cast<long>(g) * per_group;
            gi_load4(xt, xv[t]);
            if (bf16) gi_load4(xt + 4, xv[t] + 4);
        }
        for (int rr = 0; rr < nr; ++rr) {
            const GiWord4* wp = reinterpret_cast<const GiWord4*>(w0 + static_cast<long>(rr) * row_bytes) + g;
#ifdef __CUDACC__
            const GiWord4 v = *wp;
#else
            gi_emu_check_aligned(wp);
            GiWord4 v;
            memcpy(&v, wp, sizeof v);
#endif
            if (bf16) {
                for (int j = 0; j < 8; ++j) {
                    const float wv = gi_widen_bf16(static_cast<unsigned short>((v.w[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu));
                    for (int t = 0; t < k; ++t) acc[rr][t] = fmaf(wv, xv[t][j], acc[rr][t]);
                }
            } else {
                for (int j = 0; j < 4; ++j) {
                    union {
                        unsigned int u;
                        float f;
                    } bits;
                    bits.u = v.w[j];
                    for (int t = 0; t < k; ++t) acc[rr][t] = fmaf(bits.f, xv[t][j], acc[rr][t]);
                }
            }
        }
    }
    for (int rr = 0; rr < nr; ++rr) {
        for (int t = 0; t < k; ++t) {
            a.partial[(static_cast<long>(t0 + t) * a.rows + r0 + rr) * np + s] = acc[rr][t];
        }
    }
}

template <int R>   // R: weight rows per thread (4 or 8)
struct MatmulTileKernel {
    typedef MatmulPartialKernel::Args Args;
    static GI_DEV void body(long i, const Args& a) {
        const unsigned u = static_cast<unsigned>(i);
        const int s = static_cast<int>(u & ((1u << a.log2p) - 1u));
        const unsigned rest = u >> a.log2p;
        const unsigned row_groups = static_cast<unsigned>((a.rows + R - 1) / R);
        const int r0 = static_cast<int>(rest % row_groups) * R;
        const int t0 = static_cast<int>(rest / row_groups) * kTokChunk;
        const int nr = a.rows - r0 < R ? a.rows - r0 : R;
        const int k = a.ntok - t0 < kTokChunk ? a.ntok - t0 : kTokChunk;
        if (a.bf16) {
            if (nr == R && k == kTokChunk) gi_tile_body<R>(a, s, r0, R, t0, kTokChunk, 1);
            else if (nr == R && k == 1) gi_tile_body<R>(a, s, r0, R, t0, 1, 1);
            else gi_tile_body<R>(a, s, r0, nr, t0, k, 1);
        } else {
            if (nr == R && k == kTokChunk) gi_tile_body<R>(a, s, r0, R, t0, kTokChunk, 0);
            else if (nr == R && k == 1) gi_tile_body<R>(a, s, r0, R, t0, 1, 0);
            else gi_tile_body<R>(a, s, r0, nr, t0, k, 0);
        }
    }
};

// matmul, step 2: index = (token, row). Adds the partial sums pairwise (for 32: 16 + 16, 8 + 8, ...).
struct MatmulReduceKernel {
    struct Args {
        float* out;
        long out_stride;
        const float* partial;
        int rows, log2p;
    };
    static GI_DEV void body(long j, const Args& a) {
        const unsigned u = static_cast<unsigned>(j);
        const unsigned t = u / static_cast<unsigned>(a.rows);
        const unsigned r = u % static_cast<unsigned>(a.rows);
        const int np = 1 << a.log2p;
        const float* p = a.partial + (j << a.log2p);
        float v[kPartials];
        for (int q = 0; q < np; ++q) v[q] = p[q];
        for (int width = np / 2; width >= 1; width /= 2) {
            for (int q = 0; q < width; ++q) v[q] = v[q] + v[q + width];
        }
        a.out[static_cast<long>(t) * a.out_stride + r] = v[0];
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

// GI_DEV_MATMUL picks the kernel of the partial sums:
//   narrow  one weight per step
//   wide    16 bytes of weights per step (falls back to narrow when the column count does not allow it)
//   tile    (default) wide, with 4 rows x up to 8 tokens per thread (falls back to wide when the activations are
//           not aligned)
// wide and tile give the same bits; narrow adds in another order. Measured on a Tesla T4 with Llama-3.2-1B
// (bfloat16): single-sequence decode 32 tok/s narrow, 59 wide, 78 tile; 32 concurrent requests 33 tok/s wide, 137 tile.
enum GiMatmulMode { kMatmulNarrow, kMatmulWide, kMatmulTile };

static GiMatmulMode gi_matmul_mode() {
    static const GiMatmulMode mode = [] {
        const char* v = getenv("GI_DEV_MATMUL");
        if (!v || !*v || strcmp(v, "tile") == 0) return kMatmulTile;
        if (strcmp(v, "wide") == 0) return kMatmulWide;
        if (strcmp(v, "narrow") == 0) return kMatmulNarrow;
        throw std::runtime_error(std::string("GI_DEV_MATMUL must be tile, wide or narrow, got ") + v);
    }();
    return mode;
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

// Two tuning knobs of matmul, read once from the environment:
//   GI_DEV_PARTIALS   8, 16 or 32 (default): partial sums, that is threads, per output element. Fewer partial sums
//                     mean fewer threads that each run longer. Changes the order of the additions (last bits).
//   GI_DEV_TILE_ROWS  4 (default) or 8: weight rows per thread of the tile kernel. Does not change any result.
static int gi_env_choice(const char* name, int fallback, int a, int b, int c) {
    const char* v = getenv(name);
    if (!v || !*v) return fallback;
    const int n = atoi(v);
    if (n != a && n != b && n != c) {
        throw std::runtime_error(std::string(name) + " must be " + std::to_string(a) + (b != a ? ", " + std::to_string(b) : std::string()) +
                                 " or " + std::to_string(c) + ", got " + v);
    }
    return n;
}

static int gi_matmul_log2p() {
    static const int log2p = [] {
        const int p = gi_env_choice("GI_DEV_PARTIALS", 32, 8, 16, 32);
        return p == 8 ? 3 : p == 16 ? 4 : 5;
    }();
    return log2p;
}

static int gi_tile_rows() {
    static const int rows = gi_env_choice("GI_DEV_TILE_ROWS", 4, 4, 4, 8);
    return rows;
}

static void op_matmul(float* out, long out_stride, const void* W, int bf16, const float* x, long x_stride, int ntok,
                      int rows, int cols, float* partial) {
    const int log2p = gi_matmul_log2p();
    const long np = 1L << log2p;
    const MatmulPartialKernel::Args p{partial, W, x, x_stride, bf16, ntok, rows, cols, log2p};
    const long chunks = (ntok + kTokChunk - 1) / kTokChunk;
    if (chunks * rows * np >= 2147483647L) throw std::runtime_error("device matmul: too many threads for one launch");
    const GiMatmulMode mode = gi_matmul_mode();
    const bool wide_ok = mode != kMatmulNarrow && cols % (bf16 ? 8 : 4) == 0;
    const bool tile_ok = wide_ok && x_stride % 4 == 0 && reinterpret_cast<std::uintptr_t>(x) % 16 == 0;
    if (tile_ok && mode == kMatmulTile) {
        if (gi_tile_rows() == 8) launch<MatmulTileKernel<8> >(p, chunks * ((rows + 7) / 8) * np);
        else launch<MatmulTileKernel<4> >(p, chunks * ((rows + 3) / 4) * np);
    } else if (wide_ok) {
        launch<MatmulWideKernel>(p, chunks * rows * np);
    } else {
        launch<MatmulPartialKernel>(p, chunks * rows * np);
    }
    const MatmulReduceKernel::Args r{out, out_stride, partial, rows, log2p};
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
