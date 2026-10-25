// process_memory.h - resident memory of the current process (for benchmarks).
//
// "Resident" means pages that are actually in RAM. Because model weights are memory-mapped,
// they only count once they have been touched, so the numbers grow as the model runs.
#pragma once

#include <cstddef>

namespace geneinfer {

// Highest resident set size reached so far, in bytes (0 if the platform gives no answer).
std::size_t peak_rss_bytes();

// Resident set size right now, in bytes (0 if the platform gives no answer).
std::size_t current_rss_bytes();

}  // namespace geneinfer
