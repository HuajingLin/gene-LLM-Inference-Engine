#include "tokenizer/unicode.h"

#include <algorithm>
#include <iterator>

namespace geneinfer {
namespace unicode {

namespace {

struct CodePointRange {
    std::uint32_t lo;
    std::uint32_t hi;
};

#include "tokenizer/unicode_tables.inc"

template <std::size_t N>
bool in_ranges(const CodePointRange (&table)[N], char32_t cp) {
    // Binary search for the last range whose lo <= cp, then check hi.
    const CodePointRange* end = table + N;
    const CodePointRange* it = std::upper_bound(
        table, end, static_cast<std::uint32_t>(cp),
        [](std::uint32_t value, const CodePointRange& r) { return value < r.lo; });
    if (it == table) return false;
    --it;
    return cp <= it->hi;
}

}  // namespace

bool is_letter(char32_t cp) { return in_ranges(kLetterRanges, cp); }
bool is_number(char32_t cp) { return in_ranges(kNumberRanges, cp); }
bool is_space(char32_t cp) { return in_ranges(kSpaceRanges, cp); }

char32_t next_code_point(const std::string& s, std::size_t& pos) {
    const std::size_t n = s.size();
    const unsigned char b0 = static_cast<unsigned char>(s[pos]);
    if (b0 < 0x80) {
        ++pos;
        return b0;
    }

    int extra = 0;
    char32_t cp = 0;
    char32_t min_value = 0;
    if ((b0 & 0xE0) == 0xC0) { extra = 1; cp = b0 & 0x1F; min_value = 0x80; }
    else if ((b0 & 0xF0) == 0xE0) { extra = 2; cp = b0 & 0x0F; min_value = 0x800; }
    else if ((b0 & 0xF8) == 0xF0) { extra = 3; cp = b0 & 0x07; min_value = 0x10000; }
    else { ++pos; return kReplacement; }  // stray continuation byte or invalid lead byte

    if (pos + static_cast<std::size_t>(extra) >= n) {  // fewer bytes left than the lead byte announces
        ++pos;
        return kReplacement;
    }
    for (int i = 1; i <= extra; ++i) {
        const unsigned char b = static_cast<unsigned char>(s[pos + static_cast<std::size_t>(i)]);
        if ((b & 0xC0) != 0x80) { ++pos; return kReplacement; }
        cp = (cp << 6) | (b & 0x3F);
    }
    if (cp < min_value || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) { ++pos; return kReplacement; }
    pos += static_cast<std::size_t>(extra) + 1;
    return cp;
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = kReplacement;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

}  // namespace unicode
}  // namespace geneinfer
