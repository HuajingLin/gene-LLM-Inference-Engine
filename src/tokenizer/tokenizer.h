// tokenizer.h - BPE tokenizer compatible with llama2.c's tokenizer.bin.
//
// File layout (little-endian):
//   int32  max_token_length
//   repeated vocab_size times:
//     float32 score, int32 length, <length> bytes of token text
//
// Conventions (SentencePiece / Llama 2):
//   id 0 = <unk>, id 1 = <s> (BOS), id 2 = </s> (EOS)
//   ids 3..258 are byte-fallback tokens "<0x00>".."<0xFF>" (id = byte + 3)
//   a leading space (the "dummy prefix") is added to every non-empty input
//
// The behavior intentionally mirrors run.c from llama2.c so that results can be
// compared one-to-one with tools/llama2c_io.py.
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace geneinfer {

class Tokenizer {
public:
    static constexpr int kUnk = 0;
    static constexpr int kBos = 1;
    static constexpr int kEos = 2;
    static constexpr int kNoToken = -1;            // "no previous token" marker for decode_piece
    static constexpr int kByteFallbackOffset = 3;  // byte value + 3 = token id

    // Throws std::runtime_error on I/O errors, truncated files, or if the file does not
    // contain exactly `vocab_size` entries (use Config::vocab_size from the checkpoint).
    static Tokenizer load(const std::string& path, int vocab_size);

    int vocab_size() const { return static_cast<int>(vocab_.size()); }
    int max_token_length() const { return max_token_length_; }

    // Text (UTF-8 bytes) -> token ids.
    std::vector<int> encode(const std::string& text, bool bos = true, bool eos = false) const;

    // Raw bytes of one token. `prev_token` is the token that came right before it
    // (kNoToken if none): after BOS, a leading space is dropped. Byte-fallback tokens
    // such as "<0x0A>" are converted to the single byte they stand for.
    // Throws std::out_of_range for an invalid token id.
    std::string decode_piece(int prev_token, int token) const;

    // Concatenation of decode_piece over a whole sequence. BOS/EOS produce no text.
    // decode(encode(s)) == s for typical text (the dummy-prefix space is removed again).
    std::string decode(const std::vector<int>& tokens) const;

private:
    Tokenizer() = default;

    int find(const std::string& piece) const;  // token id of an exact piece, or -1

    int max_token_length_ = 0;
    std::vector<std::string> vocab_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int> lookup_;  // piece -> id (first occurrence wins)
};

}  // namespace geneinfer
