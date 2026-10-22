#include "mapped_file.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace geneinfer {

#ifdef _WIN32

MappedFile::MappedFile(const std::string& path) {
    const std::filesystem::path p(path);

    HANDLE file = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("cannot open file: " + path);
    }

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file, &sz) || sz.QuadPart <= 0) {
        CloseHandle(file);
        throw std::runtime_error("file is empty or size unknown: " + path);
    }

    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        CloseHandle(file);
        throw std::runtime_error("CreateFileMapping failed: " + path);
    }

    void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(mapping);
        CloseHandle(file);
        throw std::runtime_error("MapViewOfFile failed: " + path);
    }

    file_ = file;
    mapping_ = mapping;
    data_ = static_cast<const unsigned char*>(view);
    size_ = static_cast<std::size_t>(sz.QuadPart);
}

void MappedFile::close() noexcept {
    if (data_) UnmapViewOfFile(const_cast<unsigned char*>(data_));
    if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
    if (file_) CloseHandle(static_cast<HANDLE>(file_));
    data_ = nullptr;
    size_ = 0;
    file_ = nullptr;
    mapping_ = nullptr;
}

#else  // POSIX

MappedFile::MappedFile(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("cannot open file: " + path + " (" + std::strerror(errno) + ")");
    }

    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("file is empty or size unknown: " + path);
    }

    const std::size_t size = static_cast<std::size_t>(st.st_size);
    void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // Once the mapping has been established, the fd can be closed; the mapping will remain valid.
    if (addr == MAP_FAILED) {
        throw std::runtime_error("mmap failed: " + path + " (" + std::strerror(errno) + ")");
    }

    data_ = static_cast<const unsigned char*>(addr);
    size_ = size;
}

void MappedFile::close() noexcept {
    if (data_) ::munmap(const_cast<unsigned char*>(data_), size_);
    data_ = nullptr;
    size_ = 0;
}

#endif

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept {
    *this = std::move(other);
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
#ifdef _WIN32
        file_ = other.file_;
        mapping_ = other.mapping_;
        other.file_ = nullptr;
        other.mapping_ = nullptr;
#endif
    }
    return *this;
}

}  // namespace geneinfer
