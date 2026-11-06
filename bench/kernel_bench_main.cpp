// kernel_bench_main.cpp - micro-benchmark of the matrix-vector product, per instruction set.
//
//   gi_kernel_bench [--seconds S] [--shapes ROWSxCOLS,ROWSxCOLS,...]
//
// For every supported instruction set and every shape it times matvec on a weight matrix and prints the best
// of several repetitions as GFLOP/s (2 flops per weight) and GB/s of weights streamed. Shapes whose matrix is
// much larger than the CPU caches measure memory bandwidth (decoding is limited by it); small shapes measure
// the compute limit. Default shapes are those of SmolLM2-135M: q/o projections, MLP, and the output
// head (113 MB of weights).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "kernels/kernels.h"

namespace k = geneinfer::kernels;

namespace {

struct Shape {
    int rows, cols;
};

std::vector<Shape> parse_shapes(const std::string& text) {
    std::vector<Shape> out;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t end = text.find(',', pos);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(pos, end - pos);
        const std::size_t x = item.find('x');
        if (x == std::string::npos) return {};
        const int r = std::atoi(item.substr(0, x).c_str()), c = std::atoi(item.substr(x + 1).c_str());
        if (r <= 0 || c <= 0) return {};
        out.push_back({r, c});
        pos = end + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    double seconds = 0.4;
    std::string shapes_arg = "576x576,1536x576,576x1536,49152x576";
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (flag == "--shapes" && i + 1 < argc) shapes_arg = argv[++i];
        else { std::fprintf(stderr, "usage: gi_kernel_bench [--seconds S] [--shapes RxC,RxC,...]\n"); return 2; }
    }
    const std::vector<Shape> shapes = parse_shapes(shapes_arg);
    if (shapes.empty() || !(seconds > 0.0)) { std::fprintf(stderr, "error: bad arguments\n"); return 2; }

    std::printf("%-8s %-12s %10s %10s %12s %10s\n", "isa", "shape", "MiB", "GFLOP/s", "GB/s", "us/call");
    for (const Shape& s : shapes) {
        std::vector<float> W(static_cast<std::size_t>(s.rows) * s.cols), x(static_cast<std::size_t>(s.cols)),
            out(static_cast<std::size_t>(s.rows));
        unsigned long long state = 88172645463325252ull;
        auto rnd = [&]() {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            return static_cast<float>(static_cast<double>(state >> 40) / 16777216.0 - 0.5);
        };
        for (float& v : W) v = rnd();
        for (float& v : x) v = rnd();
        const double bytes = static_cast<double>(W.size()) * sizeof(float);
        for (const k::Isa isa : {k::Isa::Scalar, k::Isa::Avx2, k::Isa::Avx512}) {
            if (!k::set_isa(isa)) continue;
            k::matvec(out.data(), W.data(), x.data(), s.rows, s.cols);   // warm up (page faults, caches)
            double best = 1e30;
            const auto begin = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() < seconds) {
                const auto t0 = std::chrono::steady_clock::now();
                k::matvec(out.data(), W.data(), x.data(), s.rows, s.cols);
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            volatile float sink = out[0];
            (void)sink;
            std::printf("%-8s %-12s %10.2f %10.2f %12.2f %10.1f\n", k::isa_name(isa),
                        (std::to_string(s.rows) + "x" + std::to_string(s.cols)).c_str(), bytes / 1048576.0,
                        2.0 * static_cast<double>(W.size()) / best / 1e9, bytes / best / 1e9, best * 1e6);
        }
    }
    return 0;
}
