// json.h - a small, strict JSON parser (RFC 8259), enough for config.json and safetensors headers.
//
//   JsonValue v = parse_json(text);
//   int layers = v.at("num_hidden_layers").as_int();
//   if (const JsonValue* x = v.find("rope_theta")) theta = x->as_number();
//
// Strictness: no comments, no trailing commas, no NaN/Infinity, no leading zeros, no raw control
// characters in strings, no lone surrogates in \u escapes. A leading UTF-8 byte-order mark is
// skipped. Nesting is limited to 200 levels. Duplicate object keys: the last value wins.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace geneinfer {

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    JsonValue() = default;  // null
    static JsonValue make_bool(bool b);
    static JsonValue make_number(double d);
    static JsonValue make_string(std::string s);
    static JsonValue make_array();
    static JsonValue make_object();

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    // The accessors below throw std::runtime_error if the value has a different type.
    bool as_bool() const;
    double as_number() const;
    std::int64_t as_int() const;  // the number must be integral (and fit in int64)
    const std::string& as_string() const;
    const std::vector<JsonValue>& items() const;                             // arrays
    const std::vector<std::pair<std::string, JsonValue>>& members() const;   // objects, in file order

    // Objects: pointer to the value of `key`, or nullptr if absent (or if this is not an object).
    const JsonValue* find(const std::string& key) const;
    // Like find(), but throws std::runtime_error naming the missing key.
    const JsonValue& at(const std::string& key) const;

    // Building (used by the parser).
    void push_back(JsonValue v);                 // arrays
    void set(const std::string& key, JsonValue v);  // objects; replaces an existing key

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<JsonValue> items_;
    std::vector<std::pair<std::string, JsonValue>> members_;
};

// Parses a complete JSON document. Throws std::runtime_error("json: ... at offset N").
JsonValue parse_json(const std::string& text);

// Compact serialization (numbers use enough digits to round-trip exactly).
std::string dump_json(const JsonValue& value);

}  // namespace geneinfer
