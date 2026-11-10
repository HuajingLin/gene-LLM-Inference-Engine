// thread_pool.h - a minimal fork/join thread pool for data-parallel loops over the rows of a matrix.
//
// run(fn, ctx) calls fn(ctx, t, n) for t = 0..n-1 at the same time (t = 0 on the calling thread, the others on
// persistent worker threads) and returns when all calls have finished. There is no task queue: one broadcast
// per call and one counter to wait on, which keeps the synchronization cost to about a microsecond when the
// workers are awake. A forward pass makes ~120 such calls per token, so this cost matters.
//
// Idle workers spin for a short time after a call (the next call usually follows within microseconds), then
// sleep on a condition variable so that an idle program does not burn CPU.
//
// Not reentrant: run() must be called from one thread at a time and fn must not call run().
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace geneinfer {

class ThreadPool {
public:
    using Fn = void (*)(void* ctx, int thread_index, int thread_count);

    // n_threads >= 1 is the total number of threads that take part, including the calling thread.
    // Throws std::invalid_argument for n_threads < 1.
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return n_; }
    void run(Fn fn, void* ctx);

private:
    void worker_main(int index);

    int n_;
    std::vector<std::thread> workers_;
    Fn fn_ = nullptr;
    void* ctx_ = nullptr;
    std::atomic<unsigned long long> generation_{0};   // incremented for every run()
    std::atomic<int> remaining_{0};                   // workers that have not finished the current run
    std::atomic<int> sleepers_{0};
    std::atomic<bool> stop_{false};
    std::mutex mutex_;
    std::condition_variable wake_;
};

}  // namespace geneinfer
