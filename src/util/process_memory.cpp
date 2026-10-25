#include "util/process_memory.h"

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <psapi.h>
#elif defined(__linux__)
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#else
#  include <sys/resource.h>
#endif

namespace geneinfer {

#if defined(_WIN32)

static bool query(PROCESS_MEMORY_COUNTERS& pmc) {
    pmc.cb = sizeof(pmc);
    return GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)) != 0;
}

std::size_t peak_rss_bytes() {
    PROCESS_MEMORY_COUNTERS pmc;
    return query(pmc) ? static_cast<std::size_t>(pmc.PeakWorkingSetSize) : 0;
}

std::size_t current_rss_bytes() {
    PROCESS_MEMORY_COUNTERS pmc;
    return query(pmc) ? static_cast<std::size_t>(pmc.WorkingSetSize) : 0;
}

#elif defined(__linux__)

// Reads a "Key:   <number> kB" line from /proc/self/status and returns bytes.
static std::size_t read_status_kb(const char* key) {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    std::size_t result = 0;
    const std::size_t key_len = std::strlen(key);
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, key, key_len) == 0 && line[key_len] == ':') {
            result = static_cast<std::size_t>(std::strtoull(line + key_len + 1, nullptr, 10)) * 1024u;
            break;
        }
    }
    std::fclose(f);
    return result;
}

std::size_t peak_rss_bytes() { return read_status_kb("VmHWM"); }
std::size_t current_rss_bytes() { return read_status_kb("VmRSS"); }

#else  // other POSIX systems (e.g. macOS): only the peak is available through getrusage

std::size_t peak_rss_bytes() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#  if defined(__APPLE__)
    return static_cast<std::size_t>(ru.ru_maxrss);          // bytes on macOS
#  else
    return static_cast<std::size_t>(ru.ru_maxrss) * 1024u;  // kilobytes elsewhere
#  endif
}

std::size_t current_rss_bytes() { return peak_rss_bytes(); }

#endif

}  // namespace geneinfer
