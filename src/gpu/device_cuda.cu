// device_cuda.cu - the CUDA device: every index of a launch is one GPU thread.
//
// Compiled only with -DGI_CUDA=ON (needs the CUDA toolkit; any GPU of compute capability 3.5 or newer works, a
// Tesla T4 included). The kernels themselves are in device_kernels.inc, shared with the host emulation.
#include <cuda_runtime.h>

#include <math.h>

#include <stdlib.h>
#include <string.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "gpu/device_ops.h"

namespace geneinfer {
namespace gpu {

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
    }
}

#define GI_DEV __device__ __forceinline__

constexpr int kBlockThreads = 256;   // a multiple of 32: the 32 partial sums of a matmul row are one warp

template <class K>
__global__ void run_kernel(typename K::Args args, long n) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) K::body(i, args);
}

template <class K>
void launch(const typename K::Args& args, long n) {
    if (n <= 0) return;
    const long blocks = (n + kBlockThreads - 1) / kBlockThreads;
    if (blocks > 2147483647L) throw std::runtime_error("CUDA launch: too many threads for one kernel");
    run_kernel<K><<<static_cast<unsigned int>(blocks), kBlockThreads>>>(args, n);
    check(cudaGetLastError(), "kernel launch");
}

#include "gpu/device_kernels.inc"

void* cuda_alloc(std::size_t bytes) {
    void* p = nullptr;
    const cudaError_t err = cudaMalloc(&p, bytes ? bytes : 1);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA: cannot allocate " + std::to_string(bytes >> 20) + " MiB of device memory (" +
                                 cudaGetErrorString(err) + ")");
    }
    return p;
}

void cuda_release(void* p) { cudaFree(p); }

void cuda_upload(void* dst, const void* src, std::size_t bytes) {
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "upload");
}

void cuda_download(void* dst, const void* src, std::size_t bytes) {
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "download");
}

void cuda_copy(void* dst, const void* src, std::size_t bytes) {
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "device copy");
}

void cuda_sync() { check(cudaDeviceSynchronize(), "synchronize"); }

std::string cuda_describe() {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    return std::string(prop.name) + ", " + std::to_string(total_bytes >> 20) + " MiB (" + std::to_string(free_bytes >> 20) +
           " MiB free), compute capability " + std::to_string(prop.major) + "." + std::to_string(prop.minor);
}

DeviceOps make_cuda_ops() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count < 1) {
        throw std::runtime_error(std::string("CUDA: no usable GPU found (") + cudaGetErrorString(err) + ")");
    }
    DeviceOps t{};
    t.name = "cuda";
    t.alloc = cuda_alloc;
    t.release = cuda_release;
    t.upload = cuda_upload;
    t.download = cuda_download;
    t.copy = cuda_copy;
    t.sync = cuda_sync;
    t.describe = cuda_describe;
    fill_kernel_ops(t);
    return t;
}

}  // namespace

const DeviceOps* cuda_ops() {
    static const DeviceOps table = make_cuda_ops();
    return &table;
}

}  // namespace gpu
}  // namespace geneinfer
