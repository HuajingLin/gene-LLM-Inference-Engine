#include "util/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace geneinfer {

namespace {

constexpr int kMaxDepth = 200;

[[noreturn]] void type_error(const char* wanted) {
    throw std::runtime_error(std::string("json: value is not ") + wanted);
}

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    JsonValue parse_document() {
        if (s_.compare(0, 3, "\xEF\xBB\xBF") == 0) pos_ = 3;  // UTF-8 byte-order mark
        skip_ws();
        JsonValue v = parse_value(0);
        skip_ws();
        if (pos_ != s_.size()) fail("unexpected data after the JSON value");
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("json: " + msg + " at offset " + std::to_string(pos_));
    }

    bool at_end() const { return pos_ >= s_.size(); }
    char peek() const { return at_end() ? '\0' : s_[pos_]; }

    void skip_ws() {
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    void expect_literal(const char* lit) {
        for (const char* p = lit; *p; ++p) {
            if (at_end() || s_[pos_] != *p) fail(std::string("invalid literal, expected '") + lit + "'");
            ++pos_;
        }
    }

    JsonValue parse_value(int depth) {
        if (depth > kMaxDepth) fail("nesting is too deep");
        if (at_end()) fail("unexpected end of input");
        const char c = s_[pos_];
        if (c == '{') return parse_object(depth);
        if (c == '[') return parse_array(depth);
        if (c == '"') return JsonValue::make_string(parse_string());
        if (c == 't') { expect_literal("true"); return JsonValue::make_bool(true); }
        if (c == 'f') { expect_literal("false"); return JsonValue::make_bool(false); }
        if (c == 'n') { expect_literal("null"); return JsonValue(); }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail(std::string("unexpected character '") + c + "'");
    }

    JsonValue parse_object(int depth) {
        ++pos_;  // '{'
        JsonValue obj = JsonValue::make_object();
        skip_ws();
        if (peek() == '}') { ++pos_; return obj; }
        for (;;) {
            skip_ws();
            if (peek() != '"') fail("expected a string key");
            std::string key = parse_string();
            skip_ws();
            if (peek() != ':') fail("expected ':' after the key");
            ++pos_;
            skip_ws();
            obj.set(key, parse_value(depth + 1));
            skip_ws();
            if (peek() == ',') { ++pos_; continue; }
            if (peek() == '}') { ++pos_; return obj; }
            fail("expected ',' or '}'");
        }
    }

    JsonValue parse_array(int depth) {
        ++pos_;  // '['
        JsonValue arr = JsonValue::make_array();
        skip_ws();
        if (peek() == ']') { ++pos_; return arr; }
        for (;;) {
            skip_ws();
            arr.push_back(parse_value(depth + 1));
            skip_ws();
            if (peek() == ',') { ++pos_; continue; }
            if (peek() == ']') { ++pos_; return arr; }
            fail("expected ',' or ']'");
        }
    }

    JsonValue parse_number() {
        const std::size_t start = pos_;
        if (peek() == '-') ++pos_;
        if (peek() == '0') {
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') ++pos_;
        } else {
            fail("invalid number");
        }
        if (peek() == '.') {
            ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) fail("digits expected after the decimal point");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) fail("digits expected in the exponent");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        const std::string token = s_.substr(start, pos_ - start);
        const double d = std::strtod(token.c_str(), nullptr);
        if (!std::isfinite(d)) { pos_ = start; fail("number out of range"); }
        return JsonValue::make_number(d);
    }

    unsigned parse_hex4() {
        if (pos_ + 4 > s_.size()) fail("incomplete \\u escape");
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s_[pos_ + static_cast<std::size_t>(i)];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
            else fail("invalid hex digit in \\u escape");
        }
        pos_ += 4;
        return v;
    }

    static void append_utf8(std::string& out, unsigned cp) {
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

    std::string parse_string() {
        ++pos_;  // opening quote
        std::string out;
        for (;;) {
            if (at_end()) fail("unterminated string");
            const char c = s_[pos_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) { --pos_; fail("control character in string"); }
            if (c != '\\') { out += c; continue; }

            if (at_end()) fail("unterminated escape");
            const char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = parse_hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate: must be followed by a low one
                        if (pos_ + 2 > s_.size() || s_[pos_] != '\\' || s_[pos_ + 1] != 'u') fail("lone surrogate");
                        pos_ += 2;
                        const unsigned lo = parse_hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("invalid surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("lone surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    --pos_;
                    fail("invalid escape sequence");
            }
        }
    }

    const std::string& s_;
    std::size_t pos_ = 0;
};

void dump_string(const std::string& s, std::string& out) {
    out += '"';
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += ch;  // UTF-8 bytes pass through unchanged
                }
        }
    }
    out += '"';
}

void dump_value(const JsonValue& v, std::string& out) {
    switch (v.type()) {
        case JsonValue::Type::Null: out += "null"; break;
        case JsonValue::Type::Bool: out += v.as_bool() ? "true" : "false"; break;
        case JsonValue::Type::Number: {
            const double d = v.as_number();
            char buf[40];
            if (d == std::floor(d) && std::fabs(d) < 1e15) std::snprintf(buf, sizeof buf, "%.0f", d);
            else std::snprintf(buf, sizeof buf, "%.17g", d);
            out += buf;
            break;
        }
        case JsonValue::Type::String: dump_string(v.as_string(), out); break;
        case JsonValue::Type::Array: {
            out += '[';
            bool first = true;
            for (const JsonValue& item : v.items()) {
                if (!first) out += ',';
                first = false;
                dump_value(item, out);
            }
            out += ']';
            break;
        }
        case JsonValue::Type::Object: {
            out += '{';
            bool first = true;
            for (const auto& kv : v.members()) {
                if (!first) out += ',';
                first = false;
                dump_string(kv.first, out);
                out += ':';
                dump_value(kv.second, out);
            }
            out += '}';
            break;
        }
    }
}

}  // namespace

// ---- JsonValue ---------------------------------------------------------------------------
JsonValue JsonValue::make_bool(bool b) { JsonValue v; v.type_ = Type::Bool; v.bool_ = b; return v; }
JsonValue JsonValue::make_number(double d) { JsonValue v; v.type_ = Type::Number; v.number_ = d; return v; }
JsonValue JsonValue::make_string(std::string s) { JsonValue v; v.type_ = Type::String; v.string_ = std::move(s); return v; }
JsonValue JsonValue::make_array() { JsonValue v; v.type_ = Type::Array; return v; }
JsonValue JsonValue::make_object() { JsonValue v; v.type_ = Type::Object; return v; }

bool JsonValue::as_bool() const {
    if (type_ != Type::Bool) type_error("a boolean");
    return bool_;
}

double JsonValue::as_number() const {
    if (type_ != Type::Number) type_error("a number");
    return number_;
}

std::int64_t JsonValue::as_int() const {
    if (type_ != Type::Number) type_error("a number");
    if (number_ != std::floor(number_) || std::fabs(number_) > 9007199254740992.0) {
        throw std::runtime_error("json: number is not an integer in the exactly representable range");
    }
    return static_cast<std::int64_t>(number_);
}

const std::string& JsonValue::as_string() const {
    if (type_ != Type::String) type_error("a string");
    return string_;
}

const std::vector<JsonValue>& JsonValue::items() const {
    if (type_ != Type::Array) type_error("an array");
    return items_;
}

const std::vector<std::pair<std::string, JsonValue>>& JsonValue::members() const {
    if (type_ != Type::Object) type_error("an object");
    return members_;
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type_ != Type::Object) return nullptr;
    for (const auto& kv : members_) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

const JsonValue& JsonValue::at(const std::string& key) const {
    const JsonValue* v = find(key);
    if (!v) throw std::runtime_error("json: missing key '" + key + "'");
    return *v;
}

void JsonValue::push_back(JsonValue v) {
    if (type_ != Type::Array) type_error("an array");
    items_.push_back(std::move(v));
}

void JsonValue::set(const std::string& key, JsonValue v) {
    if (type_ != Type::Object) type_error("an object");
    for (auto& kv : members_) {
        if (kv.first == key) { kv.second = std::move(v); return; }
    }
    members_.emplace_back(key, std::move(v));
}

JsonValue parse_json(const std::string& text) { return Parser(text).parse_document(); }

std::string dump_json(const JsonValue& value) {
    std::string out;
    dump_value(value, out);
    return out;
}

}  // namespace geneinfer
