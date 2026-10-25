// sampler_check_main.cpp - exposes the sampler for tests/verify_sampler.py (binary stdin/stdout).
//
//   gi_sampler_check uniform <vocab> <temperature> <top_k> <top_p> <count>
//       stdin: logits[vocab] (float32), then u[count] (float32 in [0,1))
//       stdout: count token ids (int32)      one deterministic draw per u
//   gi_sampler_check rng <vocab> <temperature> <top_k> <top_p> <seed> <count>
//       stdin: logits[vocab]                 stdout: count token ids (int32) drawn with the
//                                            sampler's internal random generator
//   gi_sampler_check rngdump <seed> <count>
//       stdout: count uniform floats from the internal generator
#include <cstdint>
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

#include "sampler/sampler.h"

using namespace geneinfer;

namespace {

std::vector<float> read_floats(std::size_t count) {
    std::vector<float> data(count);
    if (std::fread(data.data(), sizeof(float), count, stdin) != count) {
        throw std::runtime_error("not enough data on stdin");
    }
    return data;
}

double num(const char* s) {
    char* end = nullptr;
    const double d = std::strtod(s, &end);
    if (end == s || *end != '\0') throw std::runtime_error(std::string("invalid number: ") + s);
    return d;
}

// Seeds are 64-bit integers: parse them exactly (a double would lose bits above 2^53).
std::uint64_t seed_arg(const char* s) {
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') throw std::runtime_error(std::string("invalid seed: ") + s);
    return static_cast<std::uint64_t>(v);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "uniform" && argc == 7) {
            const int vocab = static_cast<int>(num(argv[2]));
            SamplerConfig cfg;
            cfg.temperature = static_cast<float>(num(argv[3]));
            cfg.top_k = static_cast<int>(num(argv[4]));
            cfg.top_p = static_cast<float>(num(argv[5]));
            const std::size_t count = static_cast<std::size_t>(num(argv[6]));
            Sampler sampler(cfg, vocab);
            const auto logits = read_floats(static_cast<std::size_t>(vocab));
            const auto us = read_floats(count);
            std::vector<std::int32_t> out(count);
            for (std::size_t i = 0; i < count; ++i) out[i] = sampler.sample(logits.data(), us[i]);
            std::fwrite(out.data(), sizeof(std::int32_t), out.size(), stdout);
        } else if (mode == "rng" && argc == 8) {
            const int vocab = static_cast<int>(num(argv[2]));
            SamplerConfig cfg;
            cfg.temperature = static_cast<float>(num(argv[3]));
            cfg.top_k = static_cast<int>(num(argv[4]));
            cfg.top_p = static_cast<float>(num(argv[5]));
            cfg.seed = seed_arg(argv[6]);
            const std::size_t count = static_cast<std::size_t>(num(argv[7]));
            Sampler sampler(cfg, vocab);
            const auto logits = read_floats(static_cast<std::size_t>(vocab));
            std::vector<std::int32_t> out(count);
            for (std::size_t i = 0; i < count; ++i) out[i] = sampler.sample(logits.data());
            std::fwrite(out.data(), sizeof(std::int32_t), out.size(), stdout);
        } else if (mode == "rngdump" && argc == 4) {
            SamplerConfig cfg;
            cfg.seed = seed_arg(argv[2]);
            const std::size_t count = static_cast<std::size_t>(num(argv[3]));
            Sampler sampler(cfg, 1);
            std::vector<float> out(count);
            for (std::size_t i = 0; i < count; ++i) out[i] = sampler.next_uniform();
            std::fwrite(out.data(), sizeof(float), out.size(), stdout);
        } else {
            std::fprintf(stderr, "usage: gi_sampler_check uniform|rng|rngdump ...\n");
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
