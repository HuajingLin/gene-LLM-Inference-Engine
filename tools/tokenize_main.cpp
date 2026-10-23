// tokenize_main.cpp - command-line front end for the tokenizer.
//
//   gi_tokenize <tokenizer.bin> <vocab_size> encode [--no-bos] [--eos]
//       reads text from stdin (raw bytes), prints token ids separated by spaces
//   gi_tokenize <tokenizer.bin> <vocab_size> decode
//       reads whitespace-separated token ids from stdin, prints the text
//
// Examples:
//   printf 'Once upon a time' | gi_tokenize models/tokenizer.bin 32000 encode
//   echo '1 9038 2501 263 931' | gi_tokenize models/tokenizer.bin 32000 decode
//
// Note: use printf (or echo -n) for encode; a trailing newline is part of the text.
#include <cstdio>
#include <exception>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <fcntl.h>
#  include <io.h>
#endif

#include "tokenizer/tokenizer.h"

using namespace geneinfer;

static int usage(const char* prog) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  %s <tokenizer.bin> <vocab_size> encode [--no-bos] [--eos]\n"    //BOS = Beginning Of Sequence
                 "  %s <tokenizer.bin> <vocab_size> decode\n",                      //EOS = End Of Sequence
                 prog, prog);
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 4) return usage(argv[0]);

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    try {
        const std::string path = argv[1];
        const int vocab_size = std::stoi(argv[2]);
        const std::string mode = argv[3];
        const Tokenizer tok = Tokenizer::load(path, vocab_size);

        const std::string input((std::istreambuf_iterator<char>(std::cin)),
                                std::istreambuf_iterator<char>());

        if (mode == "encode") {
            bool bos = true, eos = false;
            for (int i = 4; i < argc; ++i) {
                const std::string flag = argv[i];
                if (flag == "--no-bos") bos = false;
                else if (flag == "--eos") eos = true;
                else return usage(argv[0]);
            }
            const std::vector<int> ids = tok.encode(input, bos, eos);
            for (std::size_t i = 0; i < ids.size(); ++i) {
                std::printf(i ? " %d" : "%d", ids[i]);
            }
            std::printf("\n");
            return 0;
        }

        if (mode == "decode") {
            std::istringstream iss(input);
            std::vector<int> ids;
            int v;
            while (iss >> v) ids.push_back(v);
            if (!iss.eof()) {
                std::fprintf(stderr, "error: stdin must contain only integer token ids\n");
                return 1;
            }
            const std::string text = tok.decode(ids);
            std::fwrite(text.data(), 1, text.size(), stdout);
            return 0;
        }

        return usage(argv[0]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
