// safetensors.h - reader for the safetensors file format.
//
// File layout:
//   8 bytes   little-endian unsigned integer N = size of the JSON header
//   N bytes   JSON object: { "<tensor name>": {"dtype": "BF16", "shape": [..],
//                                              "data_offsets": [begin, end]}, ...,
//                            "__metadata__": {...} }
//   rest      the tensor data; data_offsets are relative to the start of this region
//
// The file is memory-mapped; tensors are looked up by name and handed out as raw bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "loader/dtype.h"
#include "loader/mapped_file.h"

namespace geneinfer {

struct SafetensorsEntry {
    std::string name;
    std::string dtype_name;            // as written in the file, e.g. "BF16"
    DType dtype = DType::Other;
    std::vector<std::int64_t> shape;
    std::size_t begin = 0;             // absolute byte offset of the data inside the file
    std::size_t end = 0;               // absolute offset one past the data

    std::size_t numel() const;
    std::size_t byte_size() const { return end - begin; }
};

class SafetensorsFile {
public:
    // Throws std::runtime_error if the file is missing, truncated or its header is inconsistent
    // (offsets outside the data region, byte size not matching shape x dtype, ...).
    static SafetensorsFile open(const std::string& path);

    // Entry for a tensor name, or nullptr.
    const SafetensorsEntry* find(const std::string& name) const;

    // All entries in the order they appear in the header.
    const std::vector<SafetensorsEntry>& entries() const { return entries_; }

    // Pointer to the first byte of a tensor's data inside the memory mapping.
    const unsigned char* data(const SafetensorsEntry& e) const { return file_.data() + e.begin; }

    std::size_t file_size() const { return file_.size(); }

private:
    SafetensorsFile() = default;

    MappedFile file_;
    std::vector<SafetensorsEntry> entries_;
    std::unordered_map<std::string, std::size_t> index_;
};

}  // namespace geneinfer
