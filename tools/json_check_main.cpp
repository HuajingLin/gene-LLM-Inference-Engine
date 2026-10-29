// json_check_main.cpp - parses JSON from stdin and prints it back in compact form.
// Exit code 1 (message on stderr) if the input is not valid JSON.
// Used by tests/verify_json.py to compare the parser with Python's json module.
#include <cstdio>
#include <exception>
#include <iostream>
#include <iterator>
#include <string>

#include "util/json.h"

int main() {
    try {
        const std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        const std::string out = geneinfer::dump_json(geneinfer::parse_json(text));
        std::fwrite(out.data(), 1, out.size(), stdout);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
