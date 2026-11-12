// mapped_file.h - read-only memory-mapped file (cross-platform: POSIX mmap / Windows MapViewOfFile)
//
// Why mmap instead of reading the file into a vector:
//   1. weight pointers can point straight into the file (zero copy);
//   2. the OS pages data in on demand, so even multi-GB (1B-class) models start quickly;
//   3. several processes can share the same physical pages.
#pragma once

#include <cstddef>
#include <string>

namespace geneinfer {

class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path);  // throws std::runtime_error on failure
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    const unsigned char* data() const { return data_; }
    std::size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }

private:
    void close() noexcept;

    const unsigned char* data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;     // HANDLE
    void* mapping_ = nullptr;  // HANDLE
#endif
};

}  // namespace geneinfer
