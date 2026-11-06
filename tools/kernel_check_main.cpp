// kernel_check_main.cpp - runs one kernel on binary data from stdin, writes the result to stdout.
// It exists so that tests/verify_kernels.py can compare the C++ kernels with NumPy.
//
//   gi_kernel_check matvec  <rows> <cols>          stdin: W[rows*cols], x[cols]       -> out[rows]
//   gi_kernel_check rmsnorm <n>                    stdin: x[n], weight[n]             -> out[n]
//   gi_kernel_check softmax <n>                    stdin: x[n]                        -> out[n]
//   gi_kernel_check silu    <n>                    stdin: x[n]                        -> out[n]
//   gi_kernel_check dot     <n>                    stdin: a[n], b[n]                  -> out[1]
//   gi_kernel_check axpy    <n> <alpha>            stdin: y[n], x[n]                  -> y + alpha*x
//   gi_kernel_check mul     <n>                    stdin: a[n], b[n]                  -> a*b
//   gi_kernel_check add     <n>                    stdin: a[n], b[n]                  -> a+b
//   gi_kernel_check isa                            -> text: "active=<isa> supported=<list>"
//   gi_kernel_check rope    <n> <pos> <head_size> [theta] [interleaved|half_split]
//                                                  stdin: v[n]                        -> out[n]
//
// All data is raw little-endian float32. The instruction set follows the environment variable GI_ISA
// (scalar, avx2, avx512; default: the best the CPU has). GI_OFFSET=k (0..15) shifts every input and output
// array by k floats inside its allocation, so that SIMD loads and stores are tested at unaligned addresses.
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <fcntl.h>
#  include <io.h>
#endif

#include "kernels/kernel_table.h"
#include "kernels/kernels.h"

namespace k = geneinfer::kernels;

namespace {

// A float array that starts `offset` floats after the beginning of its allocation (see GI_OFFSET).
class Buf {
public:
    Buf(std::size_t count, std::size_t offset) : store_(count + offset + 16, -777.0f), off_(offset), n_(count) {}
    float* data() { return store_.data() + off_; }
    const float* data() const { return store_.data() + off_; }
    std::vector<float> values() const { return std::vector<float>(data(), data() + n_); }
    // The guard values around the array must be untouched after a kernel ran.
    bool guards_intact() const {
        for (std::size_t i = 0; i < off_; ++i) if (store_[i] != -777.0f) return false;
        for (std::size_t i = off_ + n_; i < store_.size(); ++i) if (store_[i] != -777.0f) return false;
        return true;
    }

private:
    std::vector<float> store_;
    std::size_t off_, n_;
};

std::size_t g_offset = 0;

Buf read_buf(std::size_t count) {
    Buf b(count, g_offset);
    const std::size_t got = std::fread(b.data(), sizeof(float), count, stdin);
    if (got != count) {
        throw std::runtime_error("expected " + std::to_string(count) + " floats on stdin, got " + std::to_string(got));
    }
    return b;
}

void check_guards(const Buf& b, const char* what) {
    if (!b.guards_intact()) throw std::runtime_error(std::string("kernel wrote outside its array: ") + what);
}

std::vector<float> read_floats(std::size_t count) {
    std::vector<float> data(count);
    const std::size_t got = std::fread(data.data(), sizeof(float), count, stdin);
    if (got != count) {
        throw std::runtime_error("expected " + std::to_string(count) + " floats on stdin, got " +
                                 std::to_string(got));
    }
    return data;
}

void write_floats(const std::vector<float>& data) {
    std::fwrite(data.data(), sizeof(float), data.size(), stdout);
}

int arg_int(char** argv, int i, const char* name) {
    const int v = std::atoi(argv[i]);
    if (v <= 0 && std::string(argv[i]) != "0") {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + argv[i]);
    }
    return v;
}

int usage() {
    std::fprintf(stderr,
                 "usage: gi_kernel_check <op> <args...>\n"
                 "  matvec <rows> <cols> | rmsnorm <n> | softmax <n> | silu <n>\n"
                 "  dot <n> | axpy <n> <alpha> | mul <n> | add <n> |\n"
                 "  rope <n> <pos> <head_size> [theta] [interleaved|half_split]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "isa") {
        std::string list;
        for (const k::Isa i : {k::Isa::Scalar, k::Isa::Avx2, k::Isa::Avx512}) {
            if (k::isa_supported(i)) list += (list.empty() ? "" : ",") + std::string(k::isa_name(i));
        }
        // which operations really run other code than the scalar reference
        const k::KernelTable cur = k::current_table(), ref = k::scalar_table();
        std::string simd;
        auto note = [&](bool differs, const char* name) { if (differs) simd += (simd.empty() ? "" : ",") + std::string(name); };
        note(cur.matvec != ref.matvec, "matvec");
        note(cur.rmsnorm != ref.rmsnorm, "rmsnorm");
        note(cur.dot != ref.dot, "dot");
        note(cur.axpy != ref.axpy, "axpy");
        note(cur.mul != ref.mul, "mul");
        note(cur.add != ref.add, "add");
        std::printf("active=%s supported=%s simd_ops=%s\n", k::isa_name(k::active_isa()), list.c_str(), simd.c_str());
        return 0;
    }
    if (argc < 3) return usage();
    if (const char* off = std::getenv("GI_OFFSET")) {
        g_offset = static_cast<std::size_t>(std::atoi(off));
        if (g_offset > 15) { std::fprintf(stderr, "error: GI_OFFSET must be 0..15\n"); return 2; }
    }

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    try {
        const std::string op = argv[1];

        if (op == "matvec" && argc == 4) {
            const int rows = arg_int(argv, 2, "rows"), cols = arg_int(argv, 3, "cols");
            if (rows <= 0 || cols <= 0) throw std::runtime_error("rows and cols must be positive");
            const Buf W = read_buf(static_cast<std::size_t>(rows) * cols);
            const Buf x = read_buf(static_cast<std::size_t>(cols));
            Buf out(static_cast<std::size_t>(rows), g_offset);
            k::matvec(out.data(), W.data(), x.data(), rows, cols);
            check_guards(out, "matvec");
            write_floats(out.values());
        } else if (op == "rmsnorm" && argc == 3) {
            const int n = arg_int(argv, 2, "n");
            if (n <= 0) throw std::runtime_error("n must be positive");
            const Buf x = read_buf(static_cast<std::size_t>(n));
            const Buf w = read_buf(static_cast<std::size_t>(n));
            Buf out(static_cast<std::size_t>(n), g_offset);
            k::rmsnorm(out.data(), x.data(), w.data(), n);
            check_guards(out, "rmsnorm");
            write_floats(out.values());
        } else if ((op == "softmax" || op == "silu") && argc == 3) {
            const int n = arg_int(argv, 2, "n");
            if (n <= 0) throw std::runtime_error("n must be positive");
            auto x = read_floats(static_cast<std::size_t>(n));
            if (op == "softmax") k::softmax(x.data(), n);
            else k::silu(x.data(), n);
            write_floats(x);
        } else if (op == "dot" && argc == 3) {
            const int n = arg_int(argv, 2, "n");
            if (n <= 0) throw std::runtime_error("n must be positive");
            const Buf a = read_buf(static_cast<std::size_t>(n));
            const Buf b = read_buf(static_cast<std::size_t>(n));
            write_floats({k::dot(a.data(), b.data(), n)});
        } else if (op == "axpy" && argc == 4) {
            const int n = arg_int(argv, 2, "n");
            if (n <= 0) throw std::runtime_error("n must be positive");
            char* end = nullptr;
            const float alpha = std::strtof(argv[3], &end);
            if (end == argv[3] || *end != '\0') throw std::runtime_error("invalid alpha");
            Buf y = read_buf(static_cast<std::size_t>(n));
            const Buf x = read_buf(static_cast<std::size_t>(n));
            k::axpy(y.data(), alpha, x.data(), n);
            check_guards(y, "axpy");
            write_floats(y.values());
        } else if ((op == "mul" || op == "add") && argc == 3) {
            const int n = arg_int(argv, 2, "n");
            if (n <= 0) throw std::runtime_error("n must be positive");
            Buf a = read_buf(static_cast<std::size_t>(n));
            const Buf b = read_buf(static_cast<std::size_t>(n));
            if (op == "mul") k::mul(a.data(), b.data(), n);
            else k::add(a.data(), b.data(), n);
            check_guards(a, "mul/add");
            write_floats(a.values());
        } else if (op == "rope" && argc >= 5 && argc <= 7) {
            const int n = arg_int(argv, 2, "n"), pos = arg_int(argv, 3, "pos");
            const int head_size = arg_int(argv, 4, "head_size");
            float theta = 10000.0f;
            geneinfer::RopeStyle style = geneinfer::RopeStyle::Interleaved;
            if (argc >= 6) {
                char* end = nullptr;
                theta = std::strtof(argv[5], &end);
                if (end == argv[5] || *end != '\0' || !(theta > 1.0f)) throw std::runtime_error("invalid theta");
            }
            if (argc == 7) {
                const std::string s = argv[6];
                if (s == "half_split") style = geneinfer::RopeStyle::HalfSplit;
                else if (s != "interleaved") throw std::runtime_error("style must be interleaved or half_split");
            }
            if (n <= 0 || pos < 0 || head_size <= 0 || head_size % 2 != 0 || n % head_size != 0) {
                throw std::runtime_error("rope needs n > 0, pos >= 0, even head_size dividing n");
            }
            auto v = read_floats(static_cast<std::size_t>(n));
            k::rope(v.data(), n, pos, head_size, theta, style);
            write_floats(v);
        } else {
            return usage();
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
