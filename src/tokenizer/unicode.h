// unicode.h - the little Unicode support that byte-level BPE tokenizers need:
// UTF-8 decoding/encoding and the character classes \p{L}, \p{N} and \s.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace geneinfer {
namespace unicode {

constexpr char32_t kReplacement = 0xFFFD;

bool is_letter(char32_t cp);       // general category L*  (regex \p{L})
bool is_number(char32_t cp);       // general category N*  (regex \p{N})
bool is_space(char32_t cp);        // Unicode White_Space  (regex \s)

// Unicode normalization form C (canonical decomposition, canonical ordering, canonical composition) of UTF-8 text,
// like the "NFC" normalizer of Hugging Face tokenizers. Bytes that are not valid UTF-8 are copied through unchanged
// and separate the text around them (they neither combine nor reorder). Pure ASCII is returned as it is.
std::string nfc(const std::string& utf8);

// Decodes the code point that starts at s[pos] and advances pos past it. Malformed input (stray
// continuation bytes, truncated or overlong sequences, surrogates, values above U+10FFFF) yields
// U+FFFD and consumes one byte, so decoding always makes progress and never reads out of bounds.
char32_t next_code_point(const std::string& s, std::size_t& pos);

// Appends the UTF-8 encoding of `cp` to `out` (invalid values are written as U+FFFD).
void append_utf8(std::string& out, char32_t cp);

}  // namespace unicode
}  // namespace geneinfer
