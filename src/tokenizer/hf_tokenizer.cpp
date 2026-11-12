#include "tokenizer/hf_tokenizer.h"

#include <algorithm>
#include <climits>
#include <filesystem>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>

#include "tokenizer/unicode.h"
#include "util/json.h"

namespace geneinfer {

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("hf tokenizer: " + msg); }

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

JsonValue read_json(const fs::path& path) {
    try {
        return parse_json(read_file(path));
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.rfind("hf tokenizer:", 0) == 0) throw;
        fail(path.filename().string() + ": " + msg);
    }
}

int to_id(const JsonValue& v, const char* what) {
    const std::int64_t n = v.as_int();
    if (n < 0 || n > 100000000) fail(std::string("invalid ") + what + " " + std::to_string(n));
    return static_cast<int>(n);
}

// ---- the byte-level alphabet of GPT-2: every byte gets a printable code point -----------------
// Printable bytes (33..126, 161..172, 174..255) map to themselves; the other 68 bytes map, in order,
// to code points 256, 257, ... so that no token contains a space or a control character.
struct ByteLevelTables {
    std::array<char32_t, 256> byte_to_cp{};
    std::array<int, 512> cp_to_byte{};   // -1 where a code point is not part of the alphabet

    ByteLevelTables() {
        cp_to_byte.fill(-1);
        int next = 0;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const char32_t cp = printable ? static_cast<char32_t>(b) : static_cast<char32_t>(256 + next++);
            byte_to_cp[static_cast<std::size_t>(b)] = cp;
            cp_to_byte[cp] = b;
        }
    }
};

const ByteLevelTables& tables() {
    static const ByteLevelTables t;
    return t;
}

// ---- pre-tokenizer pieces -----------------------------------------------------------------------
// Digits: every numeric character (category N) becomes a piece of its own (individual) or runs of
// them are grouped; everything between them forms the other pieces.
void split_digits(const std::string& s, bool individual, std::vector<std::string>& out) {
    std::string run;
    bool run_is_digits = false;
    std::size_t pos = 0;
    while (pos < s.size()) {
        const std::size_t start = pos;
        const char32_t cp = unicode::next_code_point(s, pos);
        const bool digit = unicode::is_number(cp);
        const std::string ch = s.substr(start, pos - start);
        if (digit && individual) {
            if (!run.empty()) { out.push_back(run); run.clear(); }
            out.push_back(ch);
            continue;
        }
        if (!run.empty() && digit != run_is_digits) { out.push_back(run); run.clear(); }
        run += ch;
        run_is_digits = digit;
    }
    if (!run.empty()) out.push_back(run);
}

// The GPT-2 regular expression, hand-written because std::regex has no \p{L} / \p{N}:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Alternatives are tried in this order at every position; the first one that matches wins.
void gpt2_split(const std::string& s, std::vector<std::string>& out) {
    std::vector<char32_t> cp;
    std::vector<std::size_t> off;   // byte offset of every code point, plus the end of the string
    for (std::size_t pos = 0; pos < s.size();) {
        off.push_back(pos);
        cp.push_back(unicode::next_code_point(s, pos));
    }
    off.push_back(s.size());
    const std::size_t n = cp.size();

    auto is_l = [&](std::size_t i) { return unicode::is_letter(cp[i]); };
    auto is_n = [&](std::size_t i) { return unicode::is_number(cp[i]); };
    auto is_s = [&](std::size_t i) { return unicode::is_space(cp[i]); };
    auto is_other = [&](std::size_t i) { return !is_s(i) && !is_l(i) && !is_n(i); };

    static const char* const kContractions[] = {"s", "t", "re", "ve", "m", "ll", "d"};

    std::size_t i = 0;
    while (i < n) {
        std::size_t end = i;

        // 's|'t|'re|'ve|'m|'ll|'d
        if (cp[i] == U'\'') {
            for (const char* suffix : kContractions) {
                std::size_t len = 0;
                while (suffix[len]) ++len;
                bool ok = i + len < n;
                for (std::size_t k = 0; ok && k < len; ++k) ok = cp[i + 1 + k] == static_cast<char32_t>(suffix[k]);
                if (ok) { end = i + 1 + len; break; }
            }
        }

        // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`  (the optional space is a plain U+0020)
        if (end == i) {
            const std::size_t j = (cp[i] == U' ') ? i + 1 : i;
            if (j < n) {
                std::size_t k = j;
                if (is_l(j)) { while (k < n && is_l(k)) ++k; }
                else if (is_n(j)) { while (k < n && is_n(k)) ++k; }
                else if (is_other(j)) { while (k < n && is_other(k)) ++k; }
                if (k > j) end = k;
            }
        }

        // `\s+(?!\S)` and `\s+`
        if (end == i) {
            std::size_t k = i;
            while (k < n && is_s(k)) ++k;                 // the whitespace run [i, k)
            if (k == n || k - i == 1) end = k;            // up to the end of the text, or a single character
            else end = k - 1;                             // leave the last one to join the next token
        }

        out.push_back(s.substr(off[i], off[end] - off[i]));
        i = end;
    }
}

// The Qwen2 / Qwen3 regular expression, hand-written like the GPT-2 one:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Alternatives are tried in order at every position; inside one, the greedy parts give back characters when the
// rest cannot match (that backtracking is spelled out in the comments below).
const char* const kQwen2Pattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

void qwen2_split(const std::string& s, std::vector<std::string>& out) {
    std::vector<char32_t> cp;
    std::vector<std::size_t> off;
    for (std::size_t pos = 0; pos < s.size();) {
        off.push_back(pos);
        cp.push_back(unicode::next_code_point(s, pos));
    }
    off.push_back(s.size());
    const std::size_t n = cp.size();

    auto is_l = [&](std::size_t i) { return unicode::is_letter(cp[i]); };
    auto is_n = [&](std::size_t i) { return unicode::is_number(cp[i]); };
    auto is_s = [&](std::size_t i) { return unicode::is_space(cp[i]); };
    auto is_nl = [&](std::size_t i) { return cp[i] == U'\r' || cp[i] == U'\n'; };
    auto is_other = [&](std::size_t i) { return !is_s(i) && !is_l(i) && !is_n(i); };   // [^\s\p{L}\p{N}]
    // case-insensitive ASCII letter; U+017F (long s) folds to 's'
    auto is_ci = [&](std::size_t i, char ch) {
        const char32_t c = cp[i];
        if (c == static_cast<char32_t>(ch) || c == static_cast<char32_t>(ch - 'a' + 'A')) return true;
        return ch == 's' && c == 0x17F;
    };

    std::size_t i = 0;
    while (i < n) {
        std::size_t end = i;

        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp[i] == U'\'' && i + 1 < n) {
            if (is_ci(i + 1, 's') || is_ci(i + 1, 't') || is_ci(i + 1, 'm') || is_ci(i + 1, 'd')) end = i + 2;
            else if (i + 2 < n && ((is_ci(i + 1, 'r') && is_ci(i + 2, 'e')) || (is_ci(i + 1, 'v') && is_ci(i + 2, 'e')) ||
                                   (is_ci(i + 1, 'l') && is_ci(i + 2, 'l')))) end = i + 3;
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+  (the optional character is anything but CR, LF, a letter or a number; if no
        //    letter follows it, giving it back cannot help because it is not a letter itself)
        if (end == i) {
            std::size_t j = i;
            if (!is_nl(i) && !is_l(i) && !is_n(i) && i + 1 < n && is_l(i + 1)) j = i + 1;
            if (j < n && is_l(j)) {
                std::size_t k = j;
                while (k < n && is_l(k)) ++k;
                end = k;
            }
        }

        // 3. \p{N}  (one digit at a time)
        if (end == i && is_n(i)) end = i + 1;

        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (end == i) {
            const std::size_t j = (cp[i] == U' ') ? i + 1 : i;
            if (j < n && is_other(j)) {
                std::size_t k = j;
                while (k < n && is_other(k)) ++k;
                while (k < n && is_nl(k)) ++k;
                end = k;
            }
        }

        // 5. \s*[\r\n]+  : the longest whitespace run, given back until it ends with a CR or LF
        // 6. \s+(?!\S)    : a whitespace run up to the end of the text, or all but its last character
        // 7. \s+          : a single whitespace character in front of something else
        if (end == i && is_s(i)) {
            std::size_t k = i;
            while (k < n && is_s(k)) ++k;   // the whitespace run [i, k)
            std::size_t last_nl = k;
            for (std::size_t m = k; m > i; --m) {
                if (is_nl(m - 1)) { last_nl = m - 1; break; }
            }
            if (last_nl != k) end = last_nl + 1;                       // alternative 5
            else if (k == n || k - i == 1) end = k;                    // alternative 6 (to the end) / 7 (one character)
            else end = k - 1;                                          // alternative 6
        }

        if (end == i) end = i + 1;   // cannot happen (every character is a letter, number, other or space); keeps progress certain
        out.push_back(s.substr(off[i], off[end] - off[i]));
        i = end;
    }
}

}  // namespace

// ==================================================================================================
//  loading
// ==================================================================================================
HfTokenizer HfTokenizer::load(const std::string& directory) {
    const fs::path dir(directory);
    if (!fs::exists(dir / "tokenizer.json")) fail("no tokenizer.json in " + directory);
    const JsonValue root = read_json(dir / "tokenizer.json");
    if (!root.is_object()) fail("tokenizer.json is not a JSON object");

    HfTokenizer tk;

    // ---- things this implementation does not do: refuse instead of guessing --------------------
    if (const JsonValue* n = root.find("normalizer")) {
        if (!n->is_null()) {
            if (n->at("type").as_string() != "NFC") fail("normalizer '" + n->at("type").as_string() + "' is not supported (only NFC)");
            tk.nfc_ = true;
        }
    }
    if (const JsonValue* p = root.find("post_processor")) {
        if (!p->is_null() && p->at("type").as_string() != "ByteLevel") {
            fail("post_processor '" + p->at("type").as_string() + "' is not supported (it would add tokens)");
        }
    }
    {
        const JsonValue* d = root.find("decoder");
        if (!d || d->is_null() || d->at("type").as_string() != "ByteLevel") fail("only the ByteLevel decoder is supported");
    }

    // ---- pre-tokenizer ----------------------------------------------------------------------------
    {
        const JsonValue* pt = root.find("pre_tokenizer");
        if (!pt || pt->is_null()) fail("a pre_tokenizer is required");
        std::vector<const JsonValue*> stages;
        if (pt->at("type").as_string() == "Sequence") {
            for (const JsonValue& s : pt->at("pretokenizers").items()) stages.push_back(&s);
        } else {
            stages.push_back(pt);
        }
        bool seen_byte_level = false;
        for (std::size_t i = 0; i < stages.size(); ++i) {
            const std::string type = stages[i]->at("type").as_string();
            if (type == "Split" && i == 0 && stages.size() == 2) {
                const JsonValue* pattern = stages[i]->find("pattern");
                const JsonValue* regex = pattern ? pattern->find("Regex") : nullptr;
                if (!regex || !regex->is_string() || regex->as_string() != kQwen2Pattern) {
                    fail("a Split pre_tokenizer is supported only with the Qwen2 regular expression");
                }
                const JsonValue* behavior = stages[i]->find("behavior");
                const JsonValue* invert = stages[i]->find("invert");
                if (!behavior || !behavior->is_string() || behavior->as_string() != "Isolated" ||
                    (invert && invert->is_bool() && invert->as_bool())) {
                    fail("the Split pre_tokenizer must use behavior Isolated without invert");
                }
                tk.qwen2_split_ = true;
            } else if (type == "Digits" && i == 0 && stages.size() == 2) {
                tk.split_digits_ = true;
                const JsonValue* ind = stages[i]->find("individual_digits");
                tk.individual_digits_ = ind && ind->is_bool() && ind->as_bool();
            } else if (type == "ByteLevel" && i + 1 == stages.size()) {
                const JsonValue* prefix = stages[i]->find("add_prefix_space");
                if (prefix && prefix->is_bool() && prefix->as_bool()) fail("ByteLevel add_prefix_space=true is not supported");
                const JsonValue* regex = stages[i]->find("use_regex");
                tk.use_regex_ = !(regex && regex->is_bool() && !regex->as_bool());
                seen_byte_level = true;
            } else {
                fail("unsupported pre_tokenizer layout (stage " + std::to_string(i) + " is '" + type +
                     "'); supported: ByteLevel, or Digits followed by ByteLevel");
            }
        }
        if (!seen_byte_level) fail("the pre_tokenizer must end with ByteLevel");
        if (tk.qwen2_split_ && tk.use_regex_) fail("a Split pre_tokenizer needs ByteLevel with use_regex=false after it");
    }

    // ---- model ----------------------------------------------------------------------------------------
    const JsonValue& model = root.at("model");
    if (model.at("type").as_string() != "BPE") fail("model type '" + model.at("type").as_string() + "' is not supported");
    for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
        const JsonValue* v = model.find(key);
        if (v && v->is_string() && !v->as_string().empty()) fail(std::string("model.") + key + " is not supported");
    }
    if (const JsonValue* v = model.find("byte_fallback")) {
        if (v->is_bool() && v->as_bool()) fail("byte_fallback is not supported (byte-level BPE does not need it)");
    }
    if (const JsonValue* v = model.find("ignore_merges")) tk.ignore_merges_ = v->is_bool() && v->as_bool();

    // vocabulary
    const auto& vocab = model.at("vocab").members();
    int max_id = -1;
    for (const auto& kv : vocab) max_id = std::max(max_id, to_id(kv.second, "token id"));
    tk.id_to_token_.assign(static_cast<std::size_t>(max_id + 1), std::string());
    tk.has_token_.assign(tk.id_to_token_.size(), 0);
    tk.token_to_id_.reserve(vocab.size() * 2);
    for (const auto& kv : vocab) {
        const int id = to_id(kv.second, "token id");
        if (tk.has_token_[static_cast<std::size_t>(id)]) fail("two tokens share the id " + std::to_string(id));
        tk.id_to_token_[static_cast<std::size_t>(id)] = kv.first;
        tk.has_token_[static_cast<std::size_t>(id)] = 1;
        tk.token_to_id_.emplace(kv.first, id);
    }

    // added tokens
    if (const JsonValue* added = root.find("added_tokens")) {
        for (const JsonValue& a : added->items()) {
            for (const char* flag : {"lstrip", "rstrip", "single_word"}) {
                const JsonValue* f = a.find(flag);
                if (f && f->is_bool() && f->as_bool()) fail(std::string("added token flag '") + flag + "' is not supported");
            }
            const int id = to_id(a.at("id"), "added token id");
            const std::string content = a.at("content").as_string();
            if (content.empty()) fail("an added token has empty content");
            if (static_cast<std::size_t>(id) >= tk.id_to_token_.size()) {
                tk.id_to_token_.resize(static_cast<std::size_t>(id) + 1);
                tk.has_token_.resize(static_cast<std::size_t>(id) + 1, 0);
            }
            if (!tk.has_token_[static_cast<std::size_t>(id)]) {
                tk.id_to_token_[static_cast<std::size_t>(id)] = content;
                tk.has_token_[static_cast<std::size_t>(id)] = 1;
                tk.token_to_id_.emplace(content, id);
            }
            tk.added_.push_back({id, content});
        }
    }
    tk.special_.assign(tk.id_to_token_.size(), 0);
    if (const JsonValue* added = root.find("added_tokens")) {
        for (const JsonValue& a : added->items()) {
            const JsonValue* sp = a.find("special");
            if (sp && sp->is_bool() && sp->as_bool()) tk.special_[static_cast<std::size_t>(to_id(a.at("id"), "id"))] = 1;
        }
    }
    for (std::size_t i = 0; i < tk.added_.size(); ++i) {
        tk.added_by_first_byte_[static_cast<unsigned char>(tk.added_[i].content[0])].push_back(static_cast<int>(i));
    }
    for (auto& list : tk.added_by_first_byte_) {   // longest first: leftmost-longest matching
        std::stable_sort(list.begin(), list.end(), [&](int a, int b) {
            return tk.added_[static_cast<std::size_t>(a)].content.size() > tk.added_[static_cast<std::size_t>(b)].content.size();
        });
    }

    // merges: "left right" strings or [left, right] pairs, in priority order
    const auto& merges = model.at("merges").items();
    tk.merges_.reserve(merges.size() * 2);
    for (std::size_t r = 0; r < merges.size(); ++r) {
        std::string left, right;
        if (merges[r].is_string()) {
            const std::string& m = merges[r].as_string();
            const std::size_t sp = m.find(' ');
            if (sp == std::string::npos || sp == 0 || m.find(' ', sp + 1) != std::string::npos) {
                fail("merge #" + std::to_string(r) + " is not of the form 'left right'");
            }
            left = m.substr(0, sp);
            right = m.substr(sp + 1);
        } else {
            const auto& pair = merges[r].items();
            if (pair.size() != 2) fail("merge #" + std::to_string(r) + " is not a pair");
            left = pair[0].as_string();
            right = pair[1].as_string();
        }
        const int l = tk.token_to_id(left), rt = tk.token_to_id(right), merged = tk.token_to_id(left + right);
        if (l < 0 || rt < 0 || merged < 0) fail("merge #" + std::to_string(r) + " refers to a token that is not in the vocabulary");
        const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(l)) << 32) | static_cast<std::uint32_t>(rt);
        tk.merges_.emplace(key, MergeInfo{static_cast<int>(r), merged});   // emplace keeps the first (best) rank
    }
    tk.n_merges_ = merges.size();

    if (const JsonValue* unk = model.find("unk_token")) {
        if (unk->is_string()) {
            tk.unk_id_ = tk.token_to_id(unk->as_string());
            if (tk.unk_id_ < 0) fail("unk_token is not in the vocabulary");
        }
    }

    // the token of every byte
    tk.byte_token_.fill(-1);
    for (int b = 0; b < 256; ++b) {
        std::string s;
        unicode::append_utf8(s, tables().byte_to_cp[static_cast<std::size_t>(b)]);
        tk.byte_token_[static_cast<std::size_t>(b)] = tk.token_to_id(s);
    }

    // beginning / end of sequence ids from config.json (optional)
    if (fs::exists(dir / "config.json")) {
        const JsonValue cfg = read_json(dir / "config.json");
        if (const JsonValue* b = cfg.find("bos_token_id")) {
            if (b->is_number()) tk.bos_id_ = to_id(*b, "bos_token_id");
        }
        if (const JsonValue* e = cfg.find("eos_token_id")) {
            if (e->is_number()) tk.eos_ids_.push_back(to_id(*e, "eos_token_id"));
            else if (e->is_array()) for (const JsonValue& x : e->items()) tk.eos_ids_.push_back(to_id(x, "eos_token_id"));
        }
    }
    // generation_config.json may list more end-of-sequence ids (Qwen3: <|im_end|> and <|endoftext|>); add the new ones
    if (fs::exists(dir / "generation_config.json")) {
        const JsonValue gen = read_json(dir / "generation_config.json");
        if (const JsonValue* e = gen.find("eos_token_id")) {
            std::vector<int> more;
            if (e->is_number()) more.push_back(to_id(*e, "eos_token_id"));
            else if (e->is_array()) for (const JsonValue& x : e->items()) more.push_back(to_id(x, "eos_token_id"));
            for (const int id : more) {
                if (std::find(tk.eos_ids_.begin(), tk.eos_ids_.end(), id) == tk.eos_ids_.end()) tk.eos_ids_.push_back(id);
            }
        }
    }
    return tk;
}

int HfTokenizer::token_to_id(const std::string& token) const {
    const auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

bool HfTokenizer::is_special(int id) const {
    return id >= 0 && static_cast<std::size_t>(id) < special_.size() && special_[static_cast<std::size_t>(id)] != 0;
}

// ==================================================================================================
//  encoding
// ==================================================================================================
std::string HfTokenizer::normalize(const std::string& text) const { return nfc_ ? unicode::nfc(text) : text; }

std::vector<std::string> HfTokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> stage;
    if (qwen2_split_) {
        if (!text.empty()) qwen2_split(text, stage);
        return stage;
    }
    if (split_digits_) split_digits(text, individual_digits_, stage);
    else if (!text.empty()) stage.push_back(text);
    if (!use_regex_) return stage;
    std::vector<std::string> out;
    for (const std::string& piece : stage) gpt2_split(piece, out);
    return out;
}

// Merge priority queue entry: lowest rank first, then the leftmost position.
namespace {
struct Candidate {
    int rank;
    int pos;
    int left_id;
    int right_id;
    int merged;
    bool operator>(const Candidate& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
};
}  // namespace

void HfTokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
    if (piece.empty()) return;

    if (ignore_merges_) {
        std::string mapped;
        for (const char c : piece) unicode::append_utf8(mapped, tables().byte_to_cp[static_cast<unsigned char>(c)]);
        const auto it = token_to_id_.find(mapped);
        if (it != token_to_id_.end()) { out.push_back(it->second); return; }
    }

    // One symbol per byte (bytes whose token is missing are dropped, or become the unknown token).
    struct Symbol { int id; int prev; int next; };
    std::vector<Symbol> sym;
    sym.reserve(piece.size());
    for (const char c : piece) {
        int id = byte_token_[static_cast<unsigned char>(c)];
        if (id < 0) id = unk_id_;
        if (id >= 0) sym.push_back({id, 0, 0});
    }
    const int n = static_cast<int>(sym.size());
    for (int i = 0; i < n; ++i) {
        sym[static_cast<std::size_t>(i)].prev = i - 1;
        sym[static_cast<std::size_t>(i)].next = (i + 1 < n) ? i + 1 : -1;
    }

    // Symbols form a linked list; a min-heap holds the mergeable neighbouring pairs. Entries that
    // went stale because a neighbour changed are recognised when popped and skipped.
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> heap;
    auto push_pair = [&](int pos) {
        if (pos < 0) return;                                     // no left neighbour
        const int next = sym[static_cast<std::size_t>(pos)].next;
        if (next < 0) return;                                    // no right neighbour
        const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sym[static_cast<std::size_t>(pos)].id)) << 32) |
                                  static_cast<std::uint32_t>(sym[static_cast<std::size_t>(next)].id);
        const auto it = merges_.find(key);
        if (it != merges_.end()) {
            heap.push({it->second.rank, pos, sym[static_cast<std::size_t>(pos)].id, sym[static_cast<std::size_t>(next)].id, it->second.merged});
        }
    };
    for (int i = 0; i + 1 < n; ++i) push_pair(i);

    std::vector<char> dead(static_cast<std::size_t>(n), 0);
    while (!heap.empty()) {
        const Candidate c = heap.top();
        heap.pop();
        const std::size_t p = static_cast<std::size_t>(c.pos);
        if (dead[p] || sym[p].id != c.left_id) continue;
        const int next = sym[p].next;
        if (next < 0 || sym[static_cast<std::size_t>(next)].id != c.right_id) continue;

        sym[p].id = c.merged;                                   // merge the pair into the left symbol
        const int after = sym[static_cast<std::size_t>(next)].next;
        sym[p].next = after;
        if (after >= 0) sym[static_cast<std::size_t>(after)].prev = c.pos;
        dead[static_cast<std::size_t>(next)] = 1;
        push_pair(sym[p].prev);                                  // the pair to the left changed
        push_pair(c.pos);                                        // and the pair to the right
    }
    for (int i = 0; i >= 0 && i < n; i = sym[static_cast<std::size_t>(i)].next) {
        if (!dead[static_cast<std::size_t>(i)]) out.push_back(sym[static_cast<std::size_t>(i)].id);
    }
}

void HfTokenizer::encode_plain(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;
    for (const std::string& piece : pre_tokenize(normalize(text))) bpe(piece, out);
}

int HfTokenizer::match_added(const std::string& text, std::size_t pos) const {
    for (const int idx : added_by_first_byte_[static_cast<unsigned char>(text[pos])]) {
        const std::string& content = added_[static_cast<std::size_t>(idx)].content;
        if (text.compare(pos, content.size(), content) == 0) return idx;   // longest candidates come first
    }
    return -1;
}

std::vector<int> HfTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> ids;
    if (add_bos) {
        if (bos_id_ < 0) fail("add_bos requested but the model defines no bos_token_id");
        ids.push_back(bos_id_);
    }
    // Added tokens are found in the raw text first; the stretches between them are tokenized normally.
    std::size_t segment_start = 0, i = 0;
    while (i < text.size()) {
        const int m = added_.empty() ? -1 : match_added(text, i);
        if (m < 0) { ++i; continue; }
        encode_plain(text.substr(segment_start, i - segment_start), ids);
        ids.push_back(added_[static_cast<std::size_t>(m)].id);
        i += added_[static_cast<std::size_t>(m)].content.size();
        segment_start = i;
    }
    encode_plain(text.substr(segment_start), ids);
    return ids;
}

// ==================================================================================================
//  decoding
// ==================================================================================================
std::string HfTokenizer::decode_piece(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size() || !has_token_[static_cast<std::size_t>(id)]) {
        throw std::out_of_range("hf tokenizer: token id out of range: " + std::to_string(id));
    }
    const std::string& token = id_to_token_[static_cast<std::size_t>(id)];
    // Map every character back to its byte. A token with a character outside the byte-level alphabet
    // (an added token such as "<|im_end|>" written with other symbols) is used as plain text.
    std::string bytes;
    for (std::size_t pos = 0; pos < token.size();) {
        const char32_t cp = unicode::next_code_point(token, pos);
        if (cp >= 512 || tables().cp_to_byte[cp] < 0) return token;
        bytes += static_cast<char>(tables().cp_to_byte[cp]);
    }
    return bytes;
}

std::string HfTokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::string out;
    for (const int id : ids) {
        if (skip_special && is_special(id)) continue;
        out += decode_piece(id);
    }
    return out;
}

std::string HfTokenizer::describe() const {
    std::ostringstream os;
    os << "byte-level BPE: " << id_to_token_.size() << " token ids, " << n_merges_ << " merges, " << added_.size()
       << " added tokens\nnormalizer: " << (nfc_ ? "NFC" : "none")
       << "\npre-tokenizer: " << (split_digits_ ? (individual_digits_ ? "digits (one per digit), " : "digits (grouped), ") : "")
       << (qwen2_split_ ? "Qwen2 regex" : use_regex_ ? "GPT-2 regex" : "no regex") << "\nbos id: " << bos_id_ << ", eos ids:";
    for (const int e : eos_ids_) os << ' ' << e;
    if (eos_ids_.empty()) os << " none";
    return os.str();
}

}  // namespace geneinfer
