#include "model/kv_cache.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace geneinfer {

KvPool::KvPool(int n_layers, int kv_dim, int block_size, int n_blocks)
    : n_layers_(n_layers), kv_dim_(kv_dim), block_size_(block_size), n_blocks_(n_blocks) {
    if (n_layers < 1 || kv_dim < 1 || block_size < 1 || n_blocks < 1) {
        throw std::invalid_argument("KvPool: layers, kv_dim, block_size and n_blocks must all be >= 1");
    }
    block_floats_ = static_cast<std::size_t>(n_layers) * 2 * static_cast<std::size_t>(block_size) * static_cast<std::size_t>(kv_dim);
    mem_.reset(new float[block_floats_ * static_cast<std::size_t>(n_blocks)]);   // not zero-filled, see the header
    refs_.assign(static_cast<std::size_t>(n_blocks), 0);
    free_.reserve(static_cast<std::size_t>(n_blocks));
    for (int b = n_blocks - 1; b >= 0; --b) free_.push_back(b);   // block 0 is handed out first
}

float* KvPool::base(int block, int layer) const {
    return mem_.get() + static_cast<std::size_t>(block) * block_floats_ +
           static_cast<std::size_t>(layer) * 2 * row_floats();
}

int KvPool::allocate() {
    if (free_.empty()) return -1;
    const int b = free_.back();
    free_.pop_back();
    refs_[static_cast<std::size_t>(b)] = 1;
    return b;
}

void KvPool::retain(int block) {
    if (block < 0 || block >= n_blocks_ || refs_[static_cast<std::size_t>(block)] < 1) {
        throw std::logic_error("KvPool::retain: block " + std::to_string(block) + " is not allocated");
    }
    ++refs_[static_cast<std::size_t>(block)];
}

void KvPool::release(int block) {
    if (block < 0 || block >= n_blocks_ || refs_[static_cast<std::size_t>(block)] < 1) {
        throw std::logic_error("KvPool::release: block " + std::to_string(block) + " is not allocated");
    }
    if (--refs_[static_cast<std::size_t>(block)] == 0) free_.push_back(block);
}

int KvPool::ref_count(int block) const {
    if (block < 0 || block >= n_blocks_) throw std::out_of_range("KvPool::ref_count: bad block id");
    return refs_[static_cast<std::size_t>(block)];
}

Sequence::Sequence(Sequence&& other) noexcept
    : pool_(other.pool_), blocks_(std::move(other.blocks_)), length_(other.length_) {
    other.blocks_.clear();
    other.length_ = 0;
}

Sequence& Sequence::operator=(Sequence&& other) noexcept {
    if (this != &other) {
        clear();
        pool_ = other.pool_;
        blocks_ = std::move(other.blocks_);
        length_ = other.length_;
        other.blocks_.clear();
        other.length_ = 0;
    }
    return *this;
}

bool Sequence::reserve(int n) {
    const int bs = pool_->block_size();
    const std::size_t needed = (static_cast<std::size_t>(n < 0 ? 0 : n) + static_cast<std::size_t>(bs) - 1) / static_cast<std::size_t>(bs);
    while (blocks_.size() < needed) {
        const int b = pool_->allocate();
        if (b < 0) return false;
        blocks_.push_back(b);
    }
    return true;
}

void Sequence::truncate(int n) {
    if (n < 0 || n > length_) throw std::out_of_range("Sequence::truncate: new length must be in [0, length]");
    const int bs = pool_->block_size();
    const std::size_t keep = (static_cast<std::size_t>(n) + static_cast<std::size_t>(bs) - 1) / static_cast<std::size_t>(bs);
    while (blocks_.size() > keep) {
        pool_->release(blocks_.back());
        blocks_.pop_back();
    }
    length_ = n;
}

void Sequence::set_length(int n) {
    if (n < 0 || n > capacity()) throw std::out_of_range("Sequence::set_length: beyond the reserved capacity");
    length_ = n;
}

}  // namespace geneinfer
