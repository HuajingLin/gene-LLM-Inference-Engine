// prefix_cache.h - reuse of the keys and values of prompt prefixes that were computed before (prefix caching).
//
// The keys and values of a position depend only on the tokens up to that position. So when a new request starts with
// tokens that an earlier request has already processed, the blocks holding those positions can be shared instead of
// computed again. The cache indexes FULL blocks: an entry says "the block that holds the tokens T[0..bs) at the
// positions after the chain of entries ending in block P is block B". Looking up a prompt walks the chain block by
// block, so only a true prefix matches (the key contains the whole chain through the parent block id, and the
// tokens are compared exactly, no hashes). Partially filled blocks are never shared.
//
// The cache owns one reference to every block it indexes. Sequences that use a block hold their own references.
// Blocks that only the cache references can be evicted, least recently used leaf first, when the pool needs memory.
#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "model/kv_cache.h"

namespace geneinfer {

class PrefixCache {
public:
    explicit PrefixCache(KvPool& pool);
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;
    ~PrefixCache();   // gives back the cache's references

    // The longest cached chain of full blocks that is a prefix of tokens[0..n). Every returned block is retained
    // once for the caller (hand them to Sequence::adopt, or release them). Returns the number of blocks.
    int match(const int* tokens, int n, std::vector<int>& blocks);

    // Registers the full blocks of a sequence whose first n_tokens positions hold `tokens`: block i of block_table
    // (i < n_tokens / block_size, i < n_blocks) becomes the cached copy of the i-th block of tokens unless the cache
    // already knows that chain link. Returns how many blocks were newly registered.
    int insert(const int* tokens, int n_tokens, const int* block_table, int n_blocks);

    // Frees up to `count` blocks nobody else uses, least recently used leaves first; returns how many were freed.
    // To reach a block that only the cache holds but that sits above a block a sequence is using, the cache may also
    // forget the entries below it (the sequence keeps its block).
    int evict(int count);

    // How many blocks evict() could free right now: the entries that only the cache references.
    int evictable() const;

    int entries() const { return static_cast<int>(index_.size()); }
    void clear();   // evicts everything the cache holds (blocks still used by sequences stay with them)

private:
    struct Entry {
        bool used = false;
        int parent = -1;                 // block id of the previous link, -1 for the first block of a prompt
        std::vector<int> tokens;         // block_size tokens
        long tick = 0;                   // last use
        int children = 0;
    };
    using Key = std::pair<int, std::vector<int>>;

    void remove(int block);

    KvPool& pool_;
    std::map<Key, int> index_;           // (parent block, tokens) -> block
    std::vector<Entry> by_block_;        // by block id
    long clock_ = 0;
};

}  // namespace geneinfer
