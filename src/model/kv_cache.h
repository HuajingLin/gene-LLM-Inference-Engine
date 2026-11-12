// kv_cache.h - a paged KV cache: a pool of fixed-size blocks, and per-sequence block tables (the idea of vLLM's
// PagedAttention).
//
// The keys and values of all sequences live in one KvPool: n_blocks blocks, each holding block_size consecutive
// positions of EVERY layer ([layer][keys, values][position in block][kv_dim]). A Sequence owns a list of block
// ids (its block table): position p of the sequence is in block table[p / block_size] at row p % block_size.
// Blocks are taken from the pool when a sequence grows and given back when it is cleared, truncated or destroyed,
// so the memory of finished sequences is reused at once and no sequence has to reserve a whole context up front.
// Blocks are reference counted: retain() lets a second owner share a block (prefix caching), release() drops one
// reference and frees the block when none is left.
//
// The pool memory is not zero-filled: a position is always written before it is read, and untouched pages cost
// nothing, so a big pool only occupies what is really used.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace geneinfer {

class KvPool {
public:
    // block_size >= 1 positions per block; n_blocks >= 1. Throws std::invalid_argument for bad sizes.
    KvPool(int n_layers, int kv_dim, int block_size, int n_blocks);
    KvPool(const KvPool&) = delete;
    KvPool& operator=(const KvPool&) = delete;

    int n_layers() const { return n_layers_; }
    int kv_dim() const { return kv_dim_; }
    int block_size() const { return block_size_; }
    int n_blocks() const { return n_blocks_; }
    int free_blocks() const { return static_cast<int>(free_.size()); }
    std::size_t block_bytes() const { return block_floats_ * sizeof(float); }
    std::size_t bytes() const { return block_bytes() * static_cast<std::size_t>(n_blocks_); }

    // A free block with reference count 1, or -1 when the pool is exhausted. Blocks come back in the order they
    // were freed (most recent first), which keeps recently used memory in use.
    int allocate();
    void retain(int block);    // one more owner of an allocated block
    void release(int block);   // one owner less; the block is free again when the count reaches 0
    int ref_count(int block) const;

    // The keys / values of `block` for `layer`: block_size rows of kv_dim floats.
    float* keys(int block, int layer) { return base(block, layer); }
    float* values(int block, int layer) { return base(block, layer) + row_floats(); }
    const float* keys(int block, int layer) const { return base(block, layer); }
    const float* values(int block, int layer) const { return base(block, layer) + row_floats(); }

private:
    std::size_t row_floats() const { return static_cast<std::size_t>(block_size_) * static_cast<std::size_t>(kv_dim_); }
    float* base(int block, int layer) const;

    int n_layers_, kv_dim_, block_size_, n_blocks_;
    std::size_t block_floats_;
    std::unique_ptr<float[]> mem_;
    std::vector<int> refs_;
    std::vector<int> free_;   // stack of free block ids
};

// The cache of one sequence: its block table and its length (the number of positions whose keys and values are
// stored). The blocks are returned to the pool when the Sequence is destroyed.
class Sequence {
public:
    explicit Sequence(KvPool& pool) : pool_(&pool) {}
    Sequence(Sequence&& other) noexcept;
    Sequence& operator=(Sequence&& other) noexcept;
    Sequence(const Sequence&) = delete;
    Sequence& operator=(const Sequence&) = delete;
    ~Sequence() { clear(); }

    int length() const { return length_; }
    int capacity() const { return static_cast<int>(blocks_.size()) * pool_->block_size(); }
    int n_blocks() const { return static_cast<int>(blocks_.size()); }
    const int* block_table() const { return blocks_.data(); }
    KvPool& pool() const { return *pool_; }

    // Makes room for n positions in total. Returns false, keeping everything it had (plus any blocks it did get),
    // when the pool has no block left. Never shrinks.
    bool reserve(int n);

    // Forgets all positions >= n (n <= length()) and gives back the blocks that are no longer needed.
    void truncate(int n);
    void clear() { truncate(0); }

    // Set by the forward pass after it has stored positions up to n - 1 (n <= capacity()).
    void set_length(int n);

private:
    KvPool* pool_;
    std::vector<int> blocks_;
    int length_ = 0;
};

}  // namespace geneinfer
