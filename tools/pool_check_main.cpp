// pool_check_main.cpp - stress test of ThreadPool (tests/verify_threads.py runs it).
//
//   gi_pool_check <threads> <runs> [pause_every pause_us]
//
// Every run() must call fn exactly once for every thread index, all calls must be finished when run() returns,
// and writes made by the workers must be visible to the caller afterwards. Optional pauses between runs make
// the workers fall asleep, so the wake-up path is exercised too. Prints "ok <calls>" or exits 1.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "util/thread_pool.h"

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        std::fprintf(stderr, "usage: gi_pool_check <threads> <runs> [pause_every pause_us]\n");
        return 2;
    }
    const int threads = std::atoi(argv[1]);
    const long runs = std::atol(argv[2]);
    const long pause_every = argc == 5 ? std::atol(argv[3]) : 0;
    const long pause_us = argc == 5 ? std::atol(argv[4]) : 0;
    if (threads < 1 || runs < 1) { std::fprintf(stderr, "error: bad arguments\n"); return 2; }

    try {
        geneinfer::ThreadPool pool(threads);
        struct Ctx {
            std::vector<long> slot;          // plain (non-atomic) memory: visibility must come from the pool
            std::atomic<int> calls{0};
            long round = 0;
            std::atomic<int> bad{0};
        } ctx;
        ctx.slot.assign(static_cast<std::size_t>(threads), -1);

        long total_calls = 0;
        for (long r = 1; r <= runs; ++r) {
            ctx.calls.store(0);
            ctx.round = r;
            pool.run(
                [](void* p, int t, int n) {
                    Ctx& c = *static_cast<Ctx*>(p);
                    if (n < 1) c.bad.fetch_add(1);
                    c.slot[static_cast<std::size_t>(t)] = c.round;   // each index writes its own slot
                    c.calls.fetch_add(1);
                },
                &ctx);
            // here every call must be complete and its write visible
            if (ctx.calls.load() != threads) {
                std::fprintf(stderr, "run %ld: %d calls, expected %d\n", r, ctx.calls.load(), threads);
                return 1;
            }
            for (int t = 0; t < threads; ++t) {
                if (ctx.slot[static_cast<std::size_t>(t)] != r) {
                    std::fprintf(stderr, "run %ld: thread index %d wrote %ld\n", r, t, ctx.slot[static_cast<std::size_t>(t)]);
                    return 1;
                }
            }
            total_calls += ctx.calls.load();
            if (pause_every > 0 && r % pause_every == 0) std::this_thread::sleep_for(std::chrono::microseconds(pause_us));
        }
        if (ctx.bad.load() != 0) { std::fprintf(stderr, "a call got thread_count < 1\n"); return 1; }
        std::printf("ok %ld\n", total_calls);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
