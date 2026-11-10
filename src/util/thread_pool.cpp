#include "util/thread_pool.h"

#include <stdexcept>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define GI_CPU_RELAX() _mm_pause()
#else
#define GI_CPU_RELAX() ((void)0)
#endif

namespace geneinfer {

namespace {
constexpr int kSpinIterations = 20000;   // ~tens of microseconds of busy-waiting before a worker goes to sleep
}

ThreadPool::ThreadPool(int n_threads) : n_(n_threads) {
    if (n_threads < 1) throw std::invalid_argument("ThreadPool: n_threads must be >= 1");
    workers_.reserve(static_cast<std::size_t>(n_threads - 1));
    for (int i = 1; i < n_threads; ++i) workers_.emplace_back([this, i] { worker_main(i); });
}

ThreadPool::~ThreadPool() {
    stop_.store(true);
    {
        std::lock_guard<std::mutex> lock(mutex_);   // a worker between its check and its wait cannot miss the notify
    }
    wake_.notify_all();
    for (std::thread& t : workers_) t.join();
}

void ThreadPool::worker_main(int index) {
    unsigned long long seen = 0;
    for (;;) {
        // spin first: the next run() usually comes right away
        int spins = 0;
        while (generation_.load(std::memory_order_acquire) == seen && !stop_.load(std::memory_order_acquire) &&
               spins < kSpinIterations) {
            GI_CPU_RELAX();
            ++spins;
        }
        if (generation_.load() == seen && !stop_.load()) {
            std::unique_lock<std::mutex> lock(mutex_);
            sleepers_.fetch_add(1);
            wake_.wait(lock, [&] { return generation_.load() != seen || stop_.load(); });
            sleepers_.fetch_sub(1);
        }
        if (stop_.load()) return;
        seen = generation_.load(std::memory_order_acquire);
        fn_(ctx_, index, n_);
        remaining_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void ThreadPool::run(Fn fn, void* ctx) {
    if (n_ == 1) {
        fn(ctx, 0, 1);
        return;
    }
    fn_ = fn;
    ctx_ = ctx;
    remaining_.store(n_ - 1, std::memory_order_relaxed);
    generation_.fetch_add(1);                 // seq_cst: publishes fn_, ctx_ and remaining_ to the workers
    if (sleepers_.load() > 0) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
        }
        wake_.notify_all();
    }
    fn(ctx, 0, n_);                           // the calling thread takes part
    while (remaining_.load(std::memory_order_acquire) != 0) GI_CPU_RELAX();
}

}  // namespace geneinfer
