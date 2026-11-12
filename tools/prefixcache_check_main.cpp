// prefixcache_check_main.cpp - script driver for PrefixCache, Sequence::adopt and copy-on-write (tests/verify_prefixcache.py).
//
// Reads commands from standard input, one per line, and prints one answer line per command:
//   pool BS NB                 a pool with block size BS and NB blocks (1 layer, kv_dim 2) and an empty prefix cache
//   seq S                      (re)creates the empty sequence S
//   fill S t1 t2 ...          appends tokens: reserves room, writes the value of each token into its position, sets the length
//                              -> ok | full (nothing appended)
//   insert S                   registers the full blocks of S in the cache -> number of new entries
//   adopt S t1 t2 ...         S must be empty: match(t1 ...) and adopt the blocks -> "<n> <block ids>"
//   truncate S N               rewind S to N positions -> ok | error
//   destroy S                  -> ok
//   evict N                    -> number of freed blocks
//   evictable                  -> number
//   cow S INDEX                make_writable(INDEX) on S -> ok | full -> then the block id now at INDEX is printed: "ok B"
//   read S POS                 value stored at position POS of S (the token written by fill), "error" if POS >= length
//   state                      free blocks, cache entries, then "ref(block)" of every block and the block table of every sequence
// "error" is printed for commands that throw.
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "model/kv_cache.h"
#include "model/prefix_cache.h"

using namespace geneinfer;

int main() {
    std::unique_ptr<KvPool> pool;
    std::unique_ptr<PrefixCache> cache;
    std::map<int, std::unique_ptr<Sequence>> seqs;
    std::map<int, std::vector<int>> tokens;   // tokens written into each sequence
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        try {
            if (cmd == "pool") {
                int bs, nb;
                in >> bs >> nb;
                seqs.clear();
                cache.reset();
                pool = std::make_unique<KvPool>(1, 2, bs, nb);
                cache = std::make_unique<PrefixCache>(*pool);
                std::puts("ok");
            } else if (cmd == "seq") {
                int s;
                in >> s;
                seqs[s] = std::make_unique<Sequence>(*pool);
                tokens[s].clear();
                std::puts("ok");
            } else if (cmd == "fill") {
                int s, t;
                in >> s;
                std::vector<int> add;
                while (in >> t) add.push_back(t);
                Sequence& q = *seqs.at(s);
                const int len = q.length();
                if (!q.reserve(len + static_cast<int>(add.size()))) {
                    q.truncate(len);
                    std::puts("full");
                    continue;
                }
                const int bs = pool->block_size();
                bool writable = true;
                for (int idx = len / bs; writable && idx <= (len + static_cast<int>(add.size()) - 1) / bs && !add.empty(); ++idx) {
                    writable = q.make_writable(idx);
                }
                if (!writable) {
                    q.truncate(len);
                    std::puts("full");
                    continue;
                }
                for (std::size_t i = 0; i < add.size(); ++i) {
                    const int pos = len + static_cast<int>(i);
                    pool->keys(q.block_table()[pos / bs], 0)[(pos % bs) * 2] = static_cast<float>(add[i]);
                }
                q.set_length(len + static_cast<int>(add.size()));
                tokens[s].insert(tokens[s].end(), add.begin(), add.end());
                std::puts("ok");
            } else if (cmd == "insert") {
                int s;
                in >> s;
                Sequence& q = *seqs.at(s);
                std::printf("%d\n", cache->insert(tokens[s].data(), q.length(), q.block_table(), q.n_blocks()));
            } else if (cmd == "adopt") {
                int s, t;
                in >> s;
                std::vector<int> toks;
                while (in >> t) toks.push_back(t);
                std::vector<int> blocks;
                const int n = cache->match(toks.data(), static_cast<int>(toks.size()), blocks);
                try {
                    seqs.at(s)->adopt(blocks);
                } catch (...) {
                    for (int b : blocks) pool->release(b);   // not empty: give the references back
                    throw;
                }
                tokens[s].assign(toks.begin(), toks.begin() + n * pool->block_size());
                std::printf("%d", n);
                for (int b : blocks) std::printf(" %d", b);
                std::printf("\n");
            } else if (cmd == "truncate") {
                int s, n;
                in >> s >> n;
                seqs.at(s)->truncate(n);
                tokens[s].resize(static_cast<std::size_t>(n));
                std::puts("ok");
            } else if (cmd == "destroy") {
                int s;
                in >> s;
                seqs.erase(s);
                tokens.erase(s);
                std::puts("ok");
            } else if (cmd == "evict") {
                int n;
                in >> n;
                std::printf("%d\n", cache->evict(n));
            } else if (cmd == "evictable") {
                std::printf("%d\n", cache->evictable());
            } else if (cmd == "cow") {
                int s, idx;
                in >> s >> idx;
                Sequence& q = *seqs.at(s);
                if (!q.make_writable(idx)) {
                    std::puts("full");
                } else {
                    std::printf("ok %d\n", q.block_table()[idx]);
                }
            } else if (cmd == "read") {
                int s, pos;
                in >> s >> pos;
                Sequence& q = *seqs.at(s);
                if (pos < 0 || pos >= q.length()) {
                    std::puts("error");
                } else {
                    const int bs = pool->block_size();
                    std::printf("%g\n", static_cast<double>(pool->keys(q.block_table()[pos / bs], 0)[(pos % bs) * 2]));
                }
            } else if (cmd == "overwrite") {   // writes a value straight into a position WITHOUT copy-on-write (to test that blocks are separate)
                int s, pos;
                float v;
                in >> s >> pos >> v;
                Sequence& q = *seqs.at(s);
                const int bs = pool->block_size();
                pool->keys(q.block_table()[pos / bs], 0)[(pos % bs) * 2] = v;
                std::puts("ok");
            } else if (cmd == "state") {
                std::printf("%d %d |", pool->free_blocks(), cache->entries());
                for (int b = 0; b < pool->n_blocks(); ++b) std::printf(" %d", pool->ref_count(b));
                for (const auto& kv : seqs) {
                    std::printf(" | %d: %d", kv.first, kv.second->length());
                    for (int i = 0; i < kv.second->n_blocks(); ++i) std::printf(" %d", kv.second->block_table()[i]);
                }
                std::printf("\n");
            } else {
                std::puts("error");
            }
        } catch (const std::exception&) {
            std::puts("error");
        }
        std::fflush(stdout);
    }
    return 0;
}
