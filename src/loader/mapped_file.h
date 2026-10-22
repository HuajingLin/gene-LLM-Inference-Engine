// mapped_file.h — Read-only memory-mapped file (cross-platform: POSIX mmap / Windows MapViewOfFile)
//
// Why use mmap instead of reading data into a std::vector?
//   1. Weight pointers can point directly into the file—zero-copy.
//   2. The operating system loads pages into memory on demand, ensuring fast startup even for 1B-parameter-scale models (several GBs).
//   3. Multiple processes can share the same physical memory pages.
#pragma once

#include <cstddef>
#include <string>

namespace geneinfer {

class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path);  // Failure Throw std::runtime_error
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
