// unicode_check_main.cpp - exposes the Unicode helpers for tests/verify_unicode.py.
//   gi_unicode_check classes   stdout: 0x110000 bytes, one per code point: bit0 = letter, bit1 = number, bit2 = space
//   gi_unicode_check decode    stdin: bytes             stdout: the decoded code points as little-endian uint32
//   gi_unicode_check nfc       stdin: UTF-8 texts separated by NUL bytes  stdout: their NFC forms, separated the same way
//   gi_unicode_check encode    stdin: uint32 code points  stdout: their UTF-8 encoding
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <fcntl.h>
#  include <io.h>
#endif

#include "tokenizer/unicode.h"

using namespace geneinfer;

static std::string read_stdin() {
    std::string data;
    char buf[65536];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, stdin)) > 0) data.append(buf, n);
    return data;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc != 2) {
        std::fprintf(stderr, "usage: gi_unicode_check classes|decode|encode\n");
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "classes") {
        std::vector<unsigned char> out(0x110000);
        for (char32_t cp = 0; cp < 0x110000; ++cp) {
            out[cp] = static_cast<unsigned char>((unicode::is_letter(cp) ? 1 : 0) | (unicode::is_number(cp) ? 2 : 0) |
                                                 (unicode::is_space(cp) ? 4 : 0));
        }
        std::fwrite(out.data(), 1, out.size(), stdout);
    } else if (mode == "decode") {
        const std::string in = read_stdin();
        std::vector<std::uint32_t> out;
        std::size_t pos = 0;
        while (pos < in.size()) out.push_back(static_cast<std::uint32_t>(unicode::next_code_point(in, pos)));
        std::fwrite(out.data(), sizeof(std::uint32_t), out.size(), stdout);
    } else if (mode == "nfc") {
        // stdin: lines of text separated by a NUL byte; stdout: the NFC form of each, separated the same way
        const std::string in = read_stdin();
        std::string out;
        std::size_t start = 0;
        while (start <= in.size()) {
            std::size_t end = in.find('\0', start);
            if (end == std::string::npos) end = in.size();
            out += unicode::nfc(in.substr(start, end - start));
            if (end < in.size()) out += '\0';
            start = end + 1;
        }
        std::fwrite(out.data(), 1, out.size(), stdout);
    } else if (mode == "encode") {
        const std::string in = read_stdin();
        std::string out;
        for (std::size_t i = 0; i + 4 <= in.size(); i += 4) {
            std::uint32_t cp = static_cast<unsigned char>(in[i]) | (static_cast<unsigned char>(in[i + 1]) << 8) |
                               (static_cast<unsigned char>(in[i + 2]) << 16) |
                               (static_cast<std::uint32_t>(static_cast<unsigned char>(in[i + 3])) << 24);
            unicode::append_utf8(out, cp);
        }
        std::fwrite(out.data(), 1, out.size(), stdout);
    } else {
        std::fprintf(stderr, "unknown mode\n");
        return 2;
    }
    return 0;
}
