// device_host.cpp - the "emu" device: the kernels of device_kernels.inc run on the CPU, one index after another.
//
// It exists so that the kernel code can be tested on a machine without a GPU: the bodies and the operations are
// the very same source the CUDA backend compiles; only launch() and the memory functions differ. Two things make
// the emulation stricter than a quiet loop:
//   - "device memory" is filled with NaN patterns when it is allocated, so a kernel that reads memory nothing
//     has written shows up in the results;
//   - GI_EMU_ORDER=reverse or GI_EMU_ORDER=shuffle runs the indices of every launch backwards or in a scrambled
//     order. On a GPU they run all at once, so a correct kernel gives the same bits in any order.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#include "gpu/device_ops.h"

namespace geneinfer {
namespace gpu {

namespace {

enum class Order { Forward, Reverse, Shuffle };

Order emu_order() {
    static const Order order = [] {
        const char* v = std::getenv("GI_EMU_ORDER");
        if (!v || !*v || std::strcmp(v, "forward") == 0) return Order::Forward;
        if (std::strcmp(v, "reverse") == 0) return Order::Reverse;
        if (std::strcmp(v, "shuffle") == 0) return Order::Shuffle;
        throw std::runtime_error(std::string("GI_EMU_ORDER must be forward, reverse or shuffle, got ") + v);
    }();
    return order;
}

#define GI_DEV inline

template <class K>
void launch(const typename K::Args& args, long n) {
    switch (emu_order()) {
    case Order::Forward:
        for (long i = 0; i < n; ++i) K::body(i, args);
        break;
    case Order::Reverse:
        for (long i = n - 1; i >= 0; --i) K::body(i, args);
        break;
    case Order::Shuffle: {
        // i -> (i * step + offset) mod n visits every index once when step and n share no factor.
        static const long primes[] = {1000003L, 999983L, 7919L, 104729L};
        long step = 1;
        for (const long p : primes) {
            if (n % p != 0) {
                step = p;
                break;
            }
        }
        const long offset = n / 3;
        for (long i = 0; i < n; ++i) {
            const long j = static_cast<long>((static_cast<unsigned long long>(i) * static_cast<unsigned long long>(step) +
                                              static_cast<unsigned long long>(offset)) % static_cast<unsigned long long>(n));
            K::body(j, args);
        }
        break;
    }
    }
}

#include "gpu/device_kernels.inc"

void* emu_alloc(std::size_t bytes) {
    void* p = std::malloc(bytes ? bytes : 1);
    if (!p) throw std::runtime_error("emu device: out of memory allocating " + std::to_string(bytes) + " bytes");
    std::memset(p, 0xFF, bytes);   // float NaN, int -1: reads of unwritten memory are visible
    return p;
}

void emu_release(void* p) { std::free(p); }
void emu_copy(void* dst, const void* src, std::size_t bytes) { std::memcpy(dst, src, bytes); }
void emu_sync() {}
std::string emu_describe() { return "host emulation of the device kernels (for tests; slow)"; }

DeviceOps make_emu_ops() {
    DeviceOps t{};
    t.name = "emu";
    t.alloc = emu_alloc;
    t.release = emu_release;
    t.upload = emu_copy;
    t.download = emu_copy;
    t.copy = emu_copy;
    t.sync = emu_sync;
    t.describe = emu_describe;
    fill_kernel_ops(t);
    return t;
}

}  // namespace

const DeviceOps* emu_ops() {
    static const DeviceOps table = make_emu_ops();
    (void)emu_order();   // report a bad GI_EMU_ORDER when the device is chosen, not in the middle of a pass
    return &table;
}

#ifndef GI_HAVE_CUDA
const DeviceOps* cuda_ops() { return nullptr; }
#endif

const DeviceOps* device_by_name(const std::string& name) {
    if (name == "emu") return emu_ops();
    if (name == "cuda") {
        const DeviceOps* ops = cuda_ops();
        if (!ops) throw std::runtime_error("this build has no CUDA support (configure with -DGI_CUDA=ON on a machine with the CUDA toolkit)");
        return ops;
    }
    throw std::runtime_error("unknown device '" + name + "' (expected cpu, cuda or emu)");
}

}  // namespace gpu
}  // namespace geneinfer
