#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace geneinfer {

namespace {

constexpr int kMinVocabSize = 259;             // <unk>, <s>, </s> and 256 byte tokens
constexpr std::int32_t kMaxPieceBytes = 1 << 20;  // sanity bound against corrupt files

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("tokenizer: " + msg); }

template <typename T>
T read_pod(std::ifstream& in, const char* what) {
    T value;
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(T))) {
        fail(std::string("file truncated while reading ") + what);
    }
    return value;
}

bool is_continuation_byte(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

}  // namespace

Tokenizer Tokenizer::load(const std::string& path, int vocab_size) {
    if (vocab_size < kMinVocabSize) {
        fail("vocab_size must be at least 259, got " + std::to_string(vocab_size));
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open file: " + path);

    Tokenizer tk;
    tk.max_token_length_ = read_pod<std::int32_t>(in, "max_token_length");
    tk.vocab_.reserve(static_cast<std::size_t>(vocab_size));
    tk.scores_.reserve(static_cast<std::size_t>(vocab_size));
    tk.lookup_.reserve(static_cast<std::size_t>(vocab_size) * 2);

    for (int i = 0; i < vocab_size; ++i) {
        const float score = read_pod<float>(in, "token score");
        const std::int32_t len = read_pod<std::int32_t>(in, "token length");
        if (len < 0 || len > kMaxPieceBytes) {
            fail("invalid token length " + std::to_string(len) + " at id " + std::to_string(i));
        }

        std::string piece(static_cast<std::size_t>(len), '\0');
        in.read(&piece[0], len);
        if (in.gcount() != static_cast<std::streamsize>(len)) {
            fail("file truncated while reading token text at id " + std::to_string(i) +
                 " (is vocab_size too large?)");
        }

        tk.lookup_.emplace(piece, i);  // emplace keeps the first id if a piece repeats
        tk.vocab_.push_back(std::move(piece));
        tk.scores_.push_back(score);
    }

    if (in.peek() != std::ifstream::traits_type::eof()) {
        fail("file has more data after " + std::to_string(vocab_size) +
             " tokens (is vocab_size too small?)");
    }
    return tk;
}

int Tokenizer::find(const std::string& piece) const {
    const auto it = lookup_.find(piece);
    return it == lookup_.end() ? -1 : it->second;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) const {
    std::vector<int> tokens;
    tokens.reserve(text.size() + 3);

    if (bos) tokens.push_back(kBos);

    // SentencePiece adds a "dummy prefix" space in front of any non-empty text.
    if (!text.empty()) {
        const int space = find(" ");
        if (space < 0) fail("vocabulary has no ' ' token (needed for the dummy prefix)");
        tokens.push_back(space);
    }

    // Step 1: split the text into UTF-8 code points and map each one to a token.
    // A code point is a lead byte followed by up to 3 continuation bytes. If the
    // vocabulary does not contain it, fall back to one token per byte (byte + 3).
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i + 1;
        while (j < n && j - i < 4 && is_continuation_byte(text[j])) ++j;

        const int id = find(text.substr(i, j - i));
        if (id >= 0) {
            tokens.push_back(id);
        } else {
            for (std::size_t k = i; k < j; ++k) {
                tokens.push_back(static_cast<int>(static_cast<unsigned char>(text[k])) +
                                 kByteFallbackOffset);
            }
        }
        i = j;
    }

    // Step 2: BPE. Repeatedly merge the adjacent pair whose concatenation is in the
    // vocabulary with the highest score. Ties go to the leftmost pair.
    std::string merged;
    for (;;) {
        float best_score = -1e10f;
        int best_id = -1;
        std::size_t best_idx = 0;

        for (std::size_t k = 0; k + 1 < tokens.size(); ++k) {
            merged.assign(vocab_[static_cast<std::size_t>(tokens[k])]);
            merged.append(vocab_[static_cast<std::size_t>(tokens[k + 1])]);
            const int id = find(merged);
            if (id >= 0 && scores_[static_cast<std::size_t>(id)] > best_score) {
                best_score = scores_[static_cast<std::size_t>(id)];
                best_id = id;
                best_idx = k;
            }
        }

        if (best_id < 0) break;  // nothing left to merge
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
    }

    if (eos) tokens.push_back(kEos);
    return tokens;
}

std::string Tokenizer::decode_piece(int prev_token, int token) const {
    if (token < 0 || token >= vocab_size()) {
        throw std::out_of_range("tokenizer: token id out of range: " + std::to_string(token));
    }
    const std::string& piece = vocab_[static_cast<std::size_t>(token)];

    // After BOS, the dummy-prefix space at the start of the first piece is dropped.
    std::size_t start = 0;
    if (prev_token == kBos && !piece.empty() && piece[0] == ' ') start = 1;

    // Byte-fallback token "<0xXX>" -> the single byte 0xXX.
    if (piece.size() - start == 6 && piece.compare(start, 3, "<0x") == 0 &&
        piece[start + 5] == '>') {
        const int hi = hex_value(piece[start + 3]);
        const int lo = hex_value(piece[start + 4]);
        if (hi >= 0 && lo >= 0) return std::string(1, static_cast<char>(hi * 16 + lo));
    }
    return piece.substr(start);
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string out;
    int prev = kNoToken;
    for (const int t : tokens) {
        if (t != kBos && t != kEos) out += decode_piece(prev, t);
        prev = t;
    }
    return out;
}

}  // namespace geneinfer
