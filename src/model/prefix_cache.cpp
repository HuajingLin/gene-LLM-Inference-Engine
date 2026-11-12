#include "model/prefix_cache.h"

#include <algorithm>

namespace geneinfer {

PrefixCache::PrefixCache(KvPool& pool) : pool_(pool), by_block_(static_cast<std::size_t>(pool.n_blocks())) {}

PrefixCache::~PrefixCache() { clear(); }

int PrefixCache::match(const int* tokens, int n, std::vector<int>& blocks) {
    const int bs = pool_.block_size();
    int parent = -1;
    int found = 0;
    blocks.clear();
    for (int i = 0; (i + 1) * bs <= n; ++i) {
        Key key{parent, std::vector<int>(tokens + i * bs, tokens + (i + 1) * bs)};
        const auto it = index_.find(key);
        if (it == index_.end()) break;
        const int b = it->second;
        pool_.retain(b);
        by_block_[static_cast<std::size_t>(b)].tick = ++clock_;
        blocks.push_back(b);
        parent = b;
        ++found;
    }
    return found;
}

int PrefixCache::insert(const int* tokens, int n_tokens, const int* block_table, int n_blocks) {
    const int bs = pool_.block_size();
    const int full = std::min(n_tokens / bs, n_blocks);
    int parent = -1;
    int added = 0;
    for (int i = 0; i < full; ++i) {
        Key key{parent, std::vector<int>(tokens + i * bs, tokens + (i + 1) * bs)};
        const auto it = index_.find(key);
        if (it != index_.end()) {   // this link is known (maybe under another block id with identical contents): follow it
            by_block_[static_cast<std::size_t>(it->second)].tick = ++clock_;
            parent = it->second;
            continue;
        }
        const int b = block_table[i];
        if (by_block_[static_cast<std::size_t>(b)].used) break;   // cannot happen for consistent callers; never index a block twice
        Entry& e = by_block_[static_cast<std::size_t>(b)];
        e.used = true;
        e.parent = parent;
        e.tokens = key.second;
        e.tick = ++clock_;
        e.children = 0;
        if (parent >= 0) ++by_block_[static_cast<std::size_t>(parent)].children;
        pool_.retain(b);
        index_.emplace(std::move(key), b);
        parent = b;
        ++added;
    }
    return added;
}

void PrefixCache::remove(int block) {
    Entry& e = by_block_[static_cast<std::size_t>(block)];
    index_.erase(Key{e.parent, e.tokens});
    if (e.parent >= 0) --by_block_[static_cast<std::size_t>(e.parent)].children;
    e.used = false;
    e.tokens.clear();
    pool_.release(block);
}

int PrefixCache::evict(int count) {
    int freed = 0;
    while (freed < count) {
        // Leaves only (an entry's children are keyed by its block id). Prefer a leaf only the cache references: dropping
        // it frees a block. If every such block sits above a leaf that a sequence is using, drop one of those leaves
        // instead (the sequence keeps its block, the cache just forgets it); that frees nothing yet but makes the
        // block above evictable.
        int victim = -1;
        bool victim_frees = false;
        for (int b = 0; b < pool_.n_blocks(); ++b) {
            const Entry& e = by_block_[static_cast<std::size_t>(b)];
            if (!e.used || e.children != 0) continue;
            const bool frees = pool_.ref_count(b) == 1;
            if (!frees) {   // useful only if some entry above it is evictable once it is gone
                bool above = false;
                for (int a = e.parent; a >= 0; a = by_block_[static_cast<std::size_t>(a)].parent) above = above || pool_.ref_count(a) == 1;
                if (!above) continue;
            }
            if (victim < 0 || (frees && !victim_frees) || (frees == victim_frees && e.tick < by_block_[static_cast<std::size_t>(victim)].tick)) {
                victim = b;
                victim_frees = frees;
            }
        }
        if (victim < 0) break;
        remove(victim);
        if (victim_frees) ++freed;
    }
    return freed;
}

int PrefixCache::evictable() const {
    int n = 0;
    for (int b = 0; b < pool_.n_blocks(); ++b) {
        if (by_block_[static_cast<std::size_t>(b)].used && pool_.ref_count(b) == 1) ++n;
    }
    return n;
}

void PrefixCache::clear() {
    for (int b = 0; b < pool_.n_blocks(); ++b) {
        if (by_block_[static_cast<std::size_t>(b)].used) {
            // Children first would keep the child counts tidy, but everything goes, so only the maps matter.
            Entry& e = by_block_[static_cast<std::size_t>(b)];
            e.used = false;
            pool_.release(b);
        }
    }
    index_.clear();
    for (Entry& e : by_block_) {
        e.tokens.clear();
        e.children = 0;
        e.parent = -1;
    }
}

}  // namespace geneinfer
