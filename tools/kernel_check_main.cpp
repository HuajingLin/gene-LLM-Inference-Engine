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
//   gi_kernel_check rope    <n> <pos> <head_size>  stdin: v[n]                        -> out[n]
//
// All data is raw little-endian float32.
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

#include "kernels/kernels.h"

namespace k = geneinfer::kernels;

namespace {

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
                 "  dot <n> | axpy <n> <alpha> | mul <n> | add <n> | rope <n> <pos> <head_size>\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    try {
        const std::string op = argv[1];

        if (op == "matvec" && argc == 4) {
            const int rows = arg_int(argv, 2, "rows"), cols = arg_int(argv, 3, "cols");
            if (rows <= 0 || cols <= 0) throw std::runtime_error("rows and cols must be positive");
            const auto W = read_floats(static_cast<std::size_t>(rows) * cols);
            const auto x = read_floats(static_cast<std::size_t>(cols));
            std::vector<float> out(static_cast<std::size_t>(rows));
            k::matvec(out.data(), W.data(), x.data(), rows, cols);
            write_floats(out);
        } else if (op == "rmsnorm" && argc == 3) {
            const int n = arg_int(argv, 2, "n");
            if (n <= 0) throw std::runtime_error("n must be positive");
            const auto x = read_floats(static_cast<std::size_t>(n));
            const auto w = read_floats(static_cast<std::size_t>(n));
            std::vector<float> out(static_cast<std::size_t>(n));
            k::rmsnorm(out.data(), x.data(), w.data(), n);
            write_floats(out);
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
            const auto a = read_floats(static_cast<std::size_t>(n));
            const auto b = read_floats(static_cast<std::size_t>(n));
            write_floats({k::dot(a.data(), b.data(), n)});
        } else if (op == "axpy" && argc == 4) {
            const int n = arg_int(argv, 2, "n");
            if (n <= 0) throw std::runtime_error("n must be positive");
            char* end = nullptr;
            const float alpha = std::strtof(argv[3], &end);
            if (end == argv[3] || *end != '\0') throw std::runtime_error("invalid alpha");
            auto y = read_floats(static_cast<std::size_t>(n));
            const auto x = read_floats(static_cast<std::size_t>(n));
            k::axpy(y.data(), alpha, x.data(), n);
            write_floats(y);
        } else if ((op == "mul" || op == "add") && argc == 3) {
            const int n = arg_int(argv, 2, "n");
            if (n <= 0) throw std::runtime_error("n must be positive");
            auto a = read_floats(static_cast<std::size_t>(n));
            const auto b = read_floats(static_cast<std::size_t>(n));
            if (op == "mul") k::mul(a.data(), b.data(), n);
            else k::add(a.data(), b.data(), n);
            write_floats(a);
        } else if (op == "rope" && argc == 5) {
            const int n = arg_int(argv, 2, "n"), pos = arg_int(argv, 3, "pos");
            const int head_size = arg_int(argv, 4, "head_size");
            if (n <= 0 || pos < 0 || head_size <= 0 || head_size % 2 != 0 || n % head_size != 0) {
                throw std::runtime_error("rope needs n > 0, pos >= 0, even head_size dividing n");
            }
            auto v = read_floats(static_cast<std::size_t>(n));
            k::rope(v.data(), n, pos, head_size);
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
