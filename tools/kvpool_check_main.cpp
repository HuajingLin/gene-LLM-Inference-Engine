// gi_kvpool_check - drives KvPool / Sequence from a script on stdin, for tests/verify_kvpool.py.
//
// One command per line, one answer line per command:
//   pool L D BS NB          create the pool (layers, kv_dim, block size, blocks)              -> ok | error
//   alloc                   KvPool::allocate()                                                -> block id | full
//   retain B / release B    reference counting                                                -> ok | error
//   refs B                  reference count                                                   -> n | error
//   new S                   a new Sequence named S (S = 0..15)                                -> ok
//   reserve S N             Sequence::reserve                                                 -> ok | full
//   setlen S N              Sequence::set_length                                              -> ok | error
//   truncate S N            Sequence::truncate                                                -> ok | error
//   destroy S               destroys the Sequence                                             -> ok
//   move S T                T = std::move(S) (move construction)                              -> ok
//   assign S T              T = std::move(S) into an existing Sequence (move assignment)      -> ok
//   write S POS LAYER V     fills the key row with V and the value row with V + 0.5           -> ok
//   read S POS LAYER        -> first and last float of the key and value rows of that position
//   state                   -> free blocks, then per-sequence "length capacity block-table"
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "model/kv_cache.h"

using geneinfer::KvPool;
using geneinfer::Sequence;

int main() {
    std::unique_ptr<KvPool> pool;
    std::map<int, std::unique_ptr<Sequence>> seqs;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        try {
            if (cmd == "pool") {
                int l, d, bs, nb;
                in >> l >> d >> bs >> nb;
                seqs.clear();
                pool = std::make_unique<KvPool>(l, d, bs, nb);
                std::puts("ok");
            } else if (cmd == "alloc") {
                const int b = pool->allocate();
                if (b < 0) std::puts("full");
                else std::printf("%d\n", b);
            } else if (cmd == "retain" || cmd == "release" || cmd == "refs") {
                int b;
                in >> b;
                if (cmd == "retain") { pool->retain(b); std::puts("ok"); }
                else if (cmd == "release") { pool->release(b); std::puts("ok"); }
                else std::printf("%d\n", pool->ref_count(b));
            } else if (cmd == "new") {
                int s;
                in >> s;
                seqs[s] = std::make_unique<Sequence>(*pool);
                std::puts("ok");
            } else if (cmd == "reserve") {
                int s, n;
                in >> s >> n;
                std::puts(seqs.at(s)->reserve(n) ? "ok" : "full");
            } else if (cmd == "setlen") {
                int s, n;
                in >> s >> n;
                seqs.at(s)->set_length(n);
                std::puts("ok");
            } else if (cmd == "truncate") {
                int s, n;
                in >> s >> n;
                seqs.at(s)->truncate(n);
                std::puts("ok");
            } else if (cmd == "destroy") {
                int s;
                in >> s;
                seqs.erase(s);
                std::puts("ok");
            } else if (cmd == "move") {
                int s, t;
                in >> s >> t;
                auto moved = std::make_unique<Sequence>(std::move(*seqs.at(s)));
                seqs[t] = std::move(moved);   // replaces (and destroys) an old T; S stays, empty
                std::puts("ok");
            } else if (cmd == "assign") {
                int s, t;
                in >> s >> t;
                *seqs.at(t) = std::move(*seqs.at(s));
                std::puts("ok");
            } else if (cmd == "write" || cmd == "read") {
                int s, pos, layer;
                in >> s >> pos >> layer;
                Sequence& q = *seqs.at(s);
                const int bs = pool->block_size();
                if (pos < 0 || pos >= q.capacity()) throw std::out_of_range("position outside the reserved blocks");
                const int block = q.block_table()[pos / bs];
                const std::size_t off = static_cast<std::size_t>(pos % bs) * static_cast<std::size_t>(pool->kv_dim());
                float* k = pool->keys(block, layer) + off;
                float* v = pool->values(block, layer) + off;
                const std::size_t last = static_cast<std::size_t>(pool->kv_dim()) - 1;
                if (cmd == "write") {
                    float val;
                    in >> val;
                    for (int i = 0; i < pool->kv_dim(); ++i) { k[i] = val; v[i] = val + 0.5f; }
                    std::puts("ok");
                } else {
                    std::printf("%g %g %g %g\n", k[0], k[last], v[0], v[last]);
                }
            } else if (cmd == "state") {
                std::printf("%d", pool->free_blocks());
                for (const auto& [id, q] : seqs) {
                    std::printf(" | %d: %d %d", id, q->length(), q->capacity());
                    for (int i = 0; i < q->n_blocks(); ++i) std::printf(" %d", q->block_table()[i]);
                }
                std::puts("");
            } else {
                std::puts("error unknown command");
            }
        } catch (const std::exception&) {
            std::puts("error");
        }
        std::fflush(stdout);
    }
    return 0;
}
