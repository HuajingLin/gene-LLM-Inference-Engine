// hf_tokenize_main.cpp - command line front end for the Hugging Face byte-level BPE tokenizer.
//
//   gi_hf_tokenize <model_dir> encode [--bos]            stdin: text (raw bytes)  stdout: token ids
//   gi_hf_tokenize <model_dir> decode [--skip-special]   stdin: token ids         stdout: text (raw bytes)
//   gi_hf_tokenize <model_dir> pretokenize               stdin: text   stdout: the pre-tokenizer pieces, one per
//                                                        line, as JSON strings (for debugging and tests)
//   gi_hf_tokenize <model_dir> info                      a summary of what was loaded
//
// Batch modes (one process for many inputs; JSON on stdin and stdout, used by the tests):
//   encode-batch [--bos]        stdin: ["text", ...]                 stdout: [[ids...], ...]
//   pretokenize-batch           stdin: ["text", ...]                 stdout: [["piece", ...], ...]
//   decode-batch [--skip-special]  stdin: [[ids...], ...]            stdout: ["hex of the bytes", ...]
//
// Use printf (or echo -n) to feed text: a trailing newline is part of the text.
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

#include "tokenizer/hf_tokenizer.h"
#include "util/json.h"

using namespace geneinfer;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: gi_hf_tokenize <model_dir> encode [--bos] | decode [--skip-special] | pretokenize | info\n");
        return 2;
    }
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    try {
        const HfTokenizer tok = HfTokenizer::load(argv[1]);
        const std::string mode = argv[2];
        bool flag_bos = false, flag_skip = false;
        for (int i = 3; i < argc; ++i) {
            const std::string f = argv[i];
            if (f == "--bos") flag_bos = true;
            else if (f == "--skip-special") flag_skip = true;
            else { std::fprintf(stderr, "unknown option %s\n", f.c_str()); return 2; }
        }

        if (mode == "info") {
            std::printf("%s\n", tok.describe().c_str());
            return 0;
        }
        const std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());

        if (mode == "encode-batch" || mode == "pretokenize-batch" || mode == "decode-batch") {
            const JsonValue in = parse_json(input);
            JsonValue out = JsonValue::make_array();
            for (const JsonValue& item : in.items()) {
                JsonValue row = JsonValue::make_array();
                if (mode == "encode-batch") {
                    for (const int id : tok.encode(item.as_string(), flag_bos)) row.push_back(JsonValue::make_number(id));
                    out.push_back(std::move(row));
                } else if (mode == "pretokenize-batch") {
                    for (const std::string& piece : tok.pre_tokenize(item.as_string())) row.push_back(JsonValue::make_string(piece));
                    out.push_back(std::move(row));
                } else {
                    std::vector<int> ids;
                    for (const JsonValue& v : item.items()) ids.push_back(static_cast<int>(v.as_int()));
                    const std::string bytes = tok.decode(ids, flag_skip);
                    static const char* hex = "0123456789abcdef";
                    std::string h;
                    for (const char c : bytes) { h += hex[(static_cast<unsigned char>(c) >> 4) & 15]; h += hex[static_cast<unsigned char>(c) & 15]; }
                    out.push_back(JsonValue::make_string(h));
                }
            }
            const std::string text = dump_json(out);
            std::fwrite(text.data(), 1, text.size(), stdout);
            return 0;
        }

        if (mode == "encode") {
            const std::vector<int> ids = tok.encode(input, flag_bos);
            for (std::size_t i = 0; i < ids.size(); ++i) std::printf(i ? " %d" : "%d", ids[i]);
            std::printf("\n");
        } else if (mode == "decode") {
            std::istringstream iss(input);
            std::vector<int> ids;
            int v;
            while (iss >> v) ids.push_back(v);
            if (!iss.eof()) { std::fprintf(stderr, "error: stdin must contain only integer token ids\n"); return 1; }
            const std::string text = tok.decode(ids, flag_skip);
            std::fwrite(text.data(), 1, text.size(), stdout);
        } else if (mode == "pretokenize") {
            for (const std::string& piece : tok.pre_tokenize(input)) {
                std::printf("%s\n", dump_json(JsonValue::make_string(piece)).c_str());
            }
        } else {
            std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
