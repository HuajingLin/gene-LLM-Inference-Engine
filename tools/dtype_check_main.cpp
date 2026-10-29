// dtype_check_main.cpp - converts all 65536 16-bit patterns to float32 (for tests/verify_dtype.py).
//   gi_dtype_check bf16|f16     stdin: 65536 x uint16     stdout: 65536 x float32
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "loader/dtype.h"

#ifdef _WIN32
#  include <fcntl.h>
#  include <io.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc != 2) {
        std::fprintf(stderr, "usage: gi_dtype_check bf16|f16\n");
        return 2;
    }
    const std::string kind = argv[1];
    geneinfer::DType dtype;
    if (kind == "bf16") dtype = geneinfer::DType::BF16;
    else if (kind == "f16") dtype = geneinfer::DType::F16;
    else if (kind == "f32") dtype = geneinfer::DType::F32;
    else { std::fprintf(stderr, "unknown type %s\n", kind.c_str()); return 2; }

    const std::size_t elem = dtype == geneinfer::DType::F32 ? 4 : 2;
    std::vector<unsigned char> in;
    unsigned char buf[4096];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, stdin)) > 0) in.insert(in.end(), buf, buf + n);
    if (in.size() % elem != 0) {
        std::fprintf(stderr, "input size is not a multiple of %zu bytes\n", elem);
        return 1;
    }
    std::vector<float> out(in.size() / elem);
    geneinfer::convert_to_f32(dtype, in.data(), out.data(), out.size());
    std::fwrite(out.data(), sizeof(float), out.size(), stdout);
    return 0;
}
