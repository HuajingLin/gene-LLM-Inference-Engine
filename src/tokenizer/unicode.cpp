#include "tokenizer/unicode.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

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

#include "tokenizer/unicode_norm_tables.inc"

int ccc_of(char32_t cp) {
    const CccRange* end = kCccRanges + sizeof kCccRanges / sizeof kCccRanges[0];
    const CccRange* it = std::upper_bound(kCccRanges, end, static_cast<std::uint32_t>(cp),
                                          [](std::uint32_t value, const CccRange& r) { return value < r.lo; });
    if (it == kCccRanges) return 0;
    --it;
    return cp <= it->hi ? it->ccc : 0;
}

constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr int kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

// Full canonical decomposition of one code point, appended to `out`.
void decompose(char32_t cp, std::vector<char32_t>& out) {
    if (cp >= kSBase && cp < kSBase + kSCount) {   // Hangul syllable: arithmetic
        const int s = static_cast<int>(cp - kSBase);
        out.push_back(kLBase + static_cast<char32_t>(s / kNCount));
        out.push_back(kVBase + static_cast<char32_t>((s % kNCount) / kTCount));
        if (s % kTCount) out.push_back(kTBase + static_cast<char32_t>(s % kTCount));
        return;
    }
    const DecompEntry* end = kDecomp + sizeof kDecomp / sizeof kDecomp[0];
    const DecompEntry* it = std::lower_bound(kDecomp, end, static_cast<std::uint32_t>(cp),
                                             [](const DecompEntry& e, std::uint32_t value) { return e.cp < value; });
    if (it == end || it->cp != cp) {
        out.push_back(cp);
        return;
    }
    decompose(it->first, out);
    if (it->second) decompose(it->second, out);
}

// The primary composite of the pair, or 0.
char32_t compose_pair(char32_t a, char32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
        return kSBase + (static_cast<char32_t>(a - kLBase) * kVCount + static_cast<char32_t>(b - kVBase)) * kTCount;
    }
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount) {
        return a + (b - kTBase);
    }
    const ComposeEntry* end = kCompose + sizeof kCompose / sizeof kCompose[0];
    const ComposeEntry* it = std::lower_bound(kCompose, end, std::make_pair(static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b)),
                                              [](const ComposeEntry& e, const std::pair<std::uint32_t, std::uint32_t>& key) {
                                                  return e.first != key.first ? e.first < key.first : e.second < key.second;
                                              });
    return (it != end && it->first == a && it->second == b) ? static_cast<char32_t>(it->composite) : 0;
}

void nfc_segment(const std::vector<char32_t>& in, std::string& out) {
    std::vector<char32_t> d;
    d.reserve(in.size() + 8);
    for (const char32_t cp : in) decompose(cp, d);

    // canonical ordering: inside every run of non-starters, stable sort by combining class
    for (std::size_t i = 1; i < d.size(); ++i) {
        const int c = ccc_of(d[i]);
        if (c == 0) continue;
        std::size_t j = i;
        while (j > 0 && ccc_of(d[j - 1]) > c) {
            std::swap(d[j - 1], d[j]);
            --j;
        }
    }

    // canonical composition
    std::vector<char32_t> res;
    res.reserve(d.size());
    std::size_t starter = 0;
    bool have_starter = false;
    int last_ccc = 0;
    for (const char32_t ch : d) {
        const int c = ccc_of(ch);
        if (have_starter) {
            const bool adjacent = res.size() - 1 == starter;
            const bool blocked = !adjacent && last_ccc >= c;
            if (!blocked) {
                const char32_t composite = compose_pair(res[starter], ch);
                if (composite != 0) {
                    res[starter] = composite;
                    continue;
                }
            }
        }
        if (c == 0) {
            starter = res.size();
            have_starter = true;
        }
        res.push_back(ch);
        last_ccc = c;
    }
    for (const char32_t cp : res) append_utf8(out, cp);
}

}  // namespace

std::string nfc(const std::string& utf8) {
    const bool ascii = std::all_of(utf8.begin(), utf8.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; });
    if (ascii) return utf8;
    std::string out;
    out.reserve(utf8.size());
    std::vector<char32_t> segment;
    for (std::size_t pos = 0; pos < utf8.size();) {
        const std::size_t start = pos;
        const char32_t cp = next_code_point(utf8, pos);
        if (cp == kReplacement && pos - start == 1) {   // an invalid byte (a real U+FFFD takes three): copy it through
            nfc_segment(segment, out);
            segment.clear();
            out += utf8[start];
            continue;
        }
        segment.push_back(cp);
    }
    nfc_segment(segment, out);
    return out;
}

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
