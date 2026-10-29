#include "loader/safetensors.h"

#include <stdexcept>

#include "util/json.h"

namespace geneinfer {

namespace {

constexpr std::uint64_t kMaxHeaderBytes = 100ull * 1024 * 1024;  // sanity bound (spec limit)

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
    throw std::runtime_error("safetensors (" + path + "): " + msg);
}

}  // namespace

std::size_t SafetensorsEntry::numel() const {
    std::size_t n = 1;
    for (const std::int64_t d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

SafetensorsFile SafetensorsFile::open(const std::string& path) {
    SafetensorsFile st;
    st.file_ = MappedFile(path);
    const unsigned char* base = st.file_.data();
    const std::size_t size = st.file_.size();

    if (size < 8) fail(path, "file is too small to contain a header");
    std::uint64_t header_size = 0;
    for (int i = 7; i >= 0; --i) header_size = (header_size << 8) | base[i];  // little-endian
    if (header_size > kMaxHeaderBytes || header_size > size - 8) {
        fail(path, "header size " + std::to_string(header_size) + " does not fit in the file");
    }

    const std::size_t data_start = 8 + static_cast<std::size_t>(header_size);
    const std::size_t data_len = size - data_start;

    JsonValue header;
    try {
        header = parse_json(std::string(reinterpret_cast<const char*>(base) + 8,
                                        static_cast<std::size_t>(header_size)));
    } catch (const std::exception& e) {
        fail(path, std::string("invalid header JSON: ") + e.what());
    }
    if (!header.is_object()) fail(path, "the header is not a JSON object");

    for (const auto& kv : header.members()) {
        if (kv.first == "__metadata__") continue;
        const JsonValue& info = kv.second;
        if (!info.is_object()) fail(path, "entry '" + kv.first + "' is not an object");

        SafetensorsEntry e;
        e.name = kv.first;
        try {
            e.dtype_name = info.at("dtype").as_string();
            e.dtype = dtype_from_name(e.dtype_name);
            for (const JsonValue& d : info.at("shape").items()) {
                const std::int64_t dim = d.as_int();
                if (dim < 0) fail(path, "negative dimension in '" + kv.first + "'");
                e.shape.push_back(dim);
            }
            const auto& offsets = info.at("data_offsets").items();
            if (offsets.size() != 2) fail(path, "data_offsets of '" + kv.first + "' must have 2 numbers");
            const std::int64_t b = offsets[0].as_int(), en = offsets[1].as_int();
            if (b < 0 || en < b || static_cast<std::uint64_t>(en) > data_len) {
                fail(path, "data_offsets of '" + kv.first + "' lie outside the data region");
            }
            e.begin = data_start + static_cast<std::size_t>(b);
            e.end = data_start + static_cast<std::size_t>(en);
        } catch (const std::runtime_error& err) {
            const std::string msg = err.what();
            if (msg.rfind("safetensors", 0) == 0) throw;  // already one of ours
            fail(path, "malformed entry '" + kv.first + "': " + msg);
        }

        const std::size_t elem = dtype_size(e.dtype_name);
        if (elem != 0 && e.numel() * elem != e.byte_size()) {
            fail(path, "tensor '" + kv.first + "': " + std::to_string(e.byte_size()) +
                       " bytes of data do not match shape x " + e.dtype_name);
        }
        st.index_[e.name] = st.entries_.size();
        st.entries_.push_back(std::move(e));
    }
    return st;
}

const SafetensorsEntry* SafetensorsFile::find(const std::string& name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : &entries_[it->second];
}

}  // namespace geneinfer
