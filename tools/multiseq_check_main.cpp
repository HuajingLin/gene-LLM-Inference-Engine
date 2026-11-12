// multiseq_check_main.cpp - several sequences sharing one paged KV pool.
//
//   gi_multiseq <model> --max-seq-len N [--block-size B] [--kv-blocks K] [--seed S] [--mode interleave|rows|cow|exhaust|rows_exhaust]
//
// interleave: three sequences (random token streams) are run alone to get reference logits, then run again in a
//   random interleaving of single-token steps and small batches on one shared pool, with random rewinds
//   (re-running earlier positions). Every logits row must be bit-identical to the reference.
// rows: forward_rows() with rows of up to four sequences in one pass (single decode rows and prefill chunks mixed,
//   a random subset of the rows asking for logits); each requested row must be bit-identical to the reference.
//   A pass the pool cannot hold must throw and change nothing.
// cow: two sequences share the first blocks of one history (KvPool::retain + Sequence::adopt); the second is rewound
//   into the middle of a shared block and continues with other tokens, through forward(), forward_batch() or
//   forward_rows(). Copy-on-write must keep the first sequence unchanged: both must match their references.
// exhaust: a pool too small for two sequences. The first sequence fills it; the next token of either sequence must be
//   refused with an exception that leaves the sequence length unchanged; after the first sequence is destroyed the
//   second one must work and give the reference logits.
// Prints "ok <what>" lines and finally "PASS", or a "FAIL ..." line (exit code 1).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <random>
#include <utility>
#include <string>
#include <vector>

#include "model/transformer.h"

using namespace geneinfer;

namespace {

struct Failure {
    std::string what;
};

void expect(bool cond, const std::string& what) {
    if (!cond) throw Failure{what};
}

std::vector<int> random_tokens(std::mt19937& rng, int n, int vocab) {
    std::vector<int> t(static_cast<std::size_t>(n));
    for (int& x : t) x = static_cast<int>(rng() % static_cast<unsigned>(vocab));
    return t;
}

// Reference: logits after every position, one token at a time, on a fresh sequence.
std::vector<std::vector<float>> run_alone(Transformer& m, const std::vector<int>& toks, int vocab) {
    Sequence seq = m.new_sequence();
    std::vector<std::vector<float>> out;
    for (std::size_t i = 0; i < toks.size(); ++i) {
        const float* l = m.forward(seq, toks[i], static_cast<int>(i));
        out.emplace_back(l, l + vocab);
    }
    return out;
}

bool same(const float* a, const std::vector<float>& b) {
    return std::memcmp(a, b.data(), b.size() * sizeof(float)) == 0;
}

void interleave(Transformer& m, std::mt19937& rng, int vocab, int ctx) {
    const int nseq = 3;
    std::vector<std::vector<int>> toks;
    std::vector<std::vector<std::vector<float>>> ref;
    for (int s = 0; s < nseq; ++s) {
        toks.push_back(random_tokens(rng, ctx / 2 + static_cast<int>(rng() % static_cast<unsigned>(ctx / 2)), vocab));
        ref.push_back(run_alone(m, toks.back(), vocab));
    }
    std::vector<Sequence> seqs;
    for (int s = 0; s < nseq; ++s) seqs.push_back(m.new_sequence());
    std::vector<int> done(nseq, 0);
    int steps = 0, batches = 0, rewinds = 0;
    for (;;) {
        std::vector<int> open;
        for (int s = 0; s < nseq; ++s)
            if (done[s] < static_cast<int>(toks[s].size())) open.push_back(s);
        if (open.empty()) break;
        const int s = open[rng() % open.size()];
        const int total = static_cast<int>(toks[s].size());
        const int kind = static_cast<int>(rng() % 6);
        if (kind == 0 && done[s] > 0) {   // rewind and redo
            const int back = static_cast<int>(rng() % static_cast<unsigned>(done[s] + 1));
            seqs[s].truncate(back);
            done[s] = back;
            ++rewinds;
            continue;
        }
        if (kind <= 2) {   // a batch
            const int n = 1 + static_cast<int>(rng() % static_cast<unsigned>(std::min(5, total - done[s])));
            const float* l = m.forward_batch(seqs[s], toks[s].data() + done[s], n, done[s], true);
            expect(same(l, ref[s][static_cast<std::size_t>(done[s] + n - 1)]),
                   "batch logits differ from the single-sequence reference (sequence " + std::to_string(s) + ")");
            done[s] += n;
            ++batches;
        } else {
            const float* l = m.forward(seqs[s], toks[s][static_cast<std::size_t>(done[s])], done[s]);
            expect(same(l, ref[s][static_cast<std::size_t>(done[s])]),
                   "logits differ from the single-sequence reference (sequence " + std::to_string(s) + ")");
            ++done[s];
            ++steps;
        }
        expect(seqs[s].length() == done[s], "sequence length is not the number of processed tokens");
    }
    std::printf("ok interleave: %d single steps, %d batches, %d rewinds\n", steps, batches, rewinds);
}

void rows_mode(Transformer& m, std::mt19937& rng, int vocab, int ctx) {
    const int nseq = 4;
    std::vector<std::vector<int>> toks;
    std::vector<std::vector<std::vector<float>>> ref;
    for (int s = 0; s < nseq; ++s) {
        toks.push_back(random_tokens(rng, 1 + static_cast<int>(rng() % static_cast<unsigned>(ctx)), vocab));
        ref.push_back(run_alone(m, toks.back(), vocab));
    }
    std::vector<Sequence> seqs;
    for (int s = 0; s < nseq; ++s) seqs.push_back(m.new_sequence());
    std::vector<int> done(nseq, 0);
    int passes = 0, rows_total = 0, checked = 0;
    for (;;) {
        std::vector<Transformer::StepRow> rows;
        std::vector<std::pair<int, int>> origin;   // sequence, position of each row
        std::vector<int> take(nseq, 0);
        // visit the sequences in random order; each contributes 0 .. 6 consecutive rows
        std::vector<int> order(nseq);
        for (int s = 0; s < nseq; ++s) order[static_cast<std::size_t>(s)] = s;
        std::shuffle(order.begin(), order.end(), rng);
        for (int s : order) {
            const int left = static_cast<int>(toks[static_cast<std::size_t>(s)].size()) - done[static_cast<std::size_t>(s)];
            const int want_rows = std::min(left, static_cast<int>(rng() % 7));
            for (int j = 0; j < want_rows && static_cast<int>(rows.size()) < m.max_batch(); ++j) {
                const int pos = done[static_cast<std::size_t>(s)] + j;
                rows.push_back({&seqs[static_cast<std::size_t>(s)], toks[static_cast<std::size_t>(s)][static_cast<std::size_t>(pos)],
                                (rng() & 1u) != 0});
                origin.push_back({s, pos});
                ++take[static_cast<std::size_t>(s)];
            }
        }
        if (rows.empty()) {
            bool any_left = false;
            for (int s = 0; s < nseq; ++s) any_left |= done[static_cast<std::size_t>(s)] < static_cast<int>(toks[static_cast<std::size_t>(s)].size());
            if (!any_left) break;
            continue;
        }
        const float* l = m.forward_rows(rows.data(), static_cast<int>(rows.size()));
        int k = 0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (!rows[i].want_logits) continue;
            expect(same(l + static_cast<std::size_t>(k) * static_cast<std::size_t>(vocab),
                        ref[static_cast<std::size_t>(origin[i].first)][static_cast<std::size_t>(origin[i].second)]),
                   "forward_rows logits differ from the single-sequence reference (sequence " + std::to_string(origin[i].first) + ")");
            ++k;
            ++checked;
        }
        expect((k == 0) == (l == nullptr), "forward_rows returned a wrong null/non-null logits pointer");
        for (int s = 0; s < nseq; ++s) {
            done[static_cast<std::size_t>(s)] += take[static_cast<std::size_t>(s)];
            expect(seqs[static_cast<std::size_t>(s)].length() == done[static_cast<std::size_t>(s)], "sequence length wrong after forward_rows");
        }
        ++passes;
        rows_total += static_cast<int>(rows.size());
    }
    // Bounds: a pass whose rows would go past the context must be refused and must not change anything.
    {
        Sequence full = m.new_sequence();
        const std::vector<int> t = random_tokens(rng, ctx, vocab);
        m.forward_batch(full, t.data(), ctx - 1, 0, false);
        Sequence other = m.new_sequence();
        const Transformer::StepRow two[2] = {{&full, t[0], true}, {&full, t[1], true}};
        bool threw = false;
        try {
            m.forward_rows(two, 2);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        expect(threw && full.length() == ctx - 1, "a pass running past the context was not refused cleanly");
        const Transformer::StepRow last[2] = {{&full, t[ctx - 1], true}, {&other, t[0], true}};
        const float* l = m.forward_rows(last, 2);   // exactly the last position: allowed
        expect(l != nullptr && full.length() == ctx && other.length() == 1, "the last position of the context was refused");
        expect(same(l, run_alone(m, t, vocab)[static_cast<std::size_t>(ctx - 1)]), "logits at the last position differ");
        threw = false;
        try {
            const Transformer::StepRow one = {&full, t[0], true};
            m.forward_rows(&one, 1);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        expect(threw, "a row beyond the last position was accepted");
        const Transformer::StepRow bad = {&other, vocab, true};
        threw = false;
        try {
            m.forward_rows(&bad, 1);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        expect(threw && other.length() == 1, "a token id outside the vocabulary was accepted");
        {
            std::vector<Transformer::StepRow> many(static_cast<std::size_t>(m.max_batch()) + 1, Transformer::StepRow{&other, t[0], false});
            threw = false;
            try {
                m.forward_rows(many.data(), static_cast<int>(many.size()));
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            expect(threw && other.length() == 1, "more rows than max_batch were accepted");
        }
        threw = false;
        try {
            m.forward_rows(&bad, 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "zero rows were accepted");
    }
    std::printf("ok rows: %d passes, %d rows, %d logits rows checked\n", passes, rows_total, checked);
}

void rows_exhaust(Transformer& m, std::mt19937& rng, int vocab) {
    KvPool& pool = m.kv_pool();
    const int bs = pool.block_size();
    Sequence a = m.new_sequence();
    Sequence b = m.new_sequence();
    const std::vector<int> t = random_tokens(rng, 3 * pool.n_blocks() * bs, vocab);
    // fill most of the pool with a: leave exactly one free block
    const int fill = (pool.n_blocks() - 1) * bs;
    if (fill + bs + 1 > m.max_seq_len()) {   // the context is too short to build this situation
        std::printf("ok rows_exhaust: skipped (context too short for this pool)\n");
        return;
    }
    for (int i = 0; i < fill; ++i) m.forward(a, t[static_cast<std::size_t>(i)], i, false);
    expect(pool.free_blocks() == 1, "setup: expected one free block");
    // a pass that needs two new blocks (one for a, one for b, but a also crosses into a second): a gets bs rows, b gets 1
    std::vector<Transformer::StepRow> rows;
    for (int i = 0; i < bs + 1; ++i) rows.push_back({&a, t[static_cast<std::size_t>(fill + i)], false});
    rows.push_back({&b, t[0], true});
    if (static_cast<int>(rows.size()) > m.max_batch()) return;   // not testable with this batch size
    const int free_before = pool.free_blocks();
    bool threw = false;
    try {
        m.forward_rows(rows.data(), static_cast<int>(rows.size()));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "forward_rows did not refuse a pass the pool cannot hold");
    expect(a.length() == fill && b.length() == 0, "a refused forward_rows changed a length");
    expect(pool.free_blocks() == free_before && a.n_blocks() == pool.n_blocks() - 1 && b.n_blocks() == 0,
           "a refused forward_rows kept blocks reserved");
    // a smaller pass still works afterwards and matches the reference
    const auto refb = run_alone(m, {t[0]}, vocab);
    const Transformer::StepRow one = {&b, t[0], true};
    const float* l = m.forward_rows(&one, 1);
    expect(l != nullptr && same(l, refb[0]), "forward_rows after a refused pass gives wrong logits");
    std::printf("ok rows_exhaust: refused cleanly, state unchanged, still usable\n");
}

void cow_mode(Transformer& m, std::mt19937& rng, int vocab, int ctx) {
    KvPool& pool = m.kv_pool();
    const int bs = pool.block_size();
    int rounds = 0;
    for (int variant = 0; variant < 3; ++variant) {
        const int n = std::max(bs + 2, ctx / 2 + static_cast<int>(rng() % static_cast<unsigned>(ctx / 2)));
        if (n + 4 > ctx) continue;
        const std::vector<int> a = random_tokens(rng, n + 3, vocab);    // sequence A: n tokens, then 3 more after B diverged
        const int shared_blocks = n / bs;
        if (shared_blocks < 1) continue;
        const int m_rewind = 1 + static_cast<int>(rng() % static_cast<unsigned>(shared_blocks * bs - 1 > 0 ? shared_blocks * bs - 1 : 1));   // inside the shared part
        std::vector<int> b(a.begin(), a.begin() + m_rewind);
        const std::vector<int> extra = random_tokens(rng, 5, vocab);
        b.insert(b.end(), extra.begin(), extra.end());
        if (static_cast<int>(b.size()) > ctx) continue;
        const auto ref_a = run_alone(m, a, vocab);
        const auto ref_b = run_alone(m, b, vocab);

        Sequence sa = m.new_sequence();
        for (int i = 0; i < n; ++i) m.forward(sa, a[static_cast<std::size_t>(i)], i, false);
        Sequence sb = m.new_sequence();
        std::vector<int> shared(sa.block_table(), sa.block_table() + shared_blocks);
        for (int blk : shared) pool.retain(blk);
        sb.adopt(shared);
        expect(sb.length() == shared_blocks * bs, "adopt did not set the length");
        sb.truncate(m_rewind);   // into the middle of the shared region
        // B continues with its own tokens
        if (variant == 0) {
            for (std::size_t i = static_cast<std::size_t>(m_rewind); i < b.size(); ++i) {
                const float* l = m.forward(sb, b[i], static_cast<int>(i));
                expect(same(l, ref_b[i]), "B (forward) differs from its reference after copy-on-write");
            }
        } else if (variant == 1) {
            const float* l = m.forward_batch(sb, b.data() + m_rewind, static_cast<int>(b.size()) - m_rewind, m_rewind, true);
            expect(same(l, ref_b.back()), "B (forward_batch) differs from its reference after copy-on-write");
        } else {
            std::vector<Transformer::StepRow> rows;
            for (std::size_t i = static_cast<std::size_t>(m_rewind); i < b.size(); ++i) rows.push_back({&sb, b[i], i + 1 == b.size()});
            const float* l = m.forward_rows(rows.data(), static_cast<int>(rows.size()));
            expect(l != nullptr && same(l, ref_b.back()), "B (forward_rows) differs from its reference after copy-on-write");
        }
        expect(sb.block_table()[(m_rewind) / bs] != sa.block_table()[m_rewind / bs], "B still shares the block it wrote into");
        // A goes on and must not have seen B's writes
        for (int i = n; i < n + 3; ++i) {
            const float* l = m.forward(sa, a[static_cast<std::size_t>(i)], i);
            expect(same(l, ref_a[static_cast<std::size_t>(i)]), "A differs from its reference after B wrote into a shared block");
        }
        // and A re-reading its whole history in one batch gives the reference too
        {
            Sequence sc = m.new_sequence();
            const float* l = m.forward_batch(sc, a.data(), n + 3, 0, true);
            expect(same(l, ref_a.back()), "reference batch differs");
        }
        ++rounds;
    }
    expect(pool.free_blocks() <= pool.n_blocks(), "pool accounting broken");
    std::printf("ok cow: %d rounds, diverged sequences match their references\n", rounds);
}

void exhaust(Transformer& m, std::mt19937& rng, int vocab) {
    KvPool& pool = m.kv_pool();
    const int capacity = pool.n_blocks() * pool.block_size();
    const std::vector<int> a = random_tokens(rng, capacity + 1, vocab);
    const std::vector<int> b = random_tokens(rng, std::min(4, capacity), vocab);
    const auto ref_b = run_alone(m, b, vocab);
    expect(pool.free_blocks() == pool.n_blocks(), "run_alone leaked blocks");
    {
        Sequence sa = m.new_sequence();
        Sequence sb = m.new_sequence();
        for (int i = 0; i < capacity; ++i) m.forward(sa, a[static_cast<std::size_t>(i)], i, false);
        expect(sa.length() == capacity && pool.free_blocks() == 0, "sequence A did not fill the pool");
        bool threw = false;
        try {
            m.forward(sa, a[static_cast<std::size_t>(capacity)], capacity);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expect(threw, "a full pool did not refuse the next position of A");
        expect(sa.length() == capacity, "a refused forward changed the length of A");
        threw = false;
        try {
            m.forward(sb, b[0], 0);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expect(threw, "a full pool did not refuse sequence B");
        expect(sb.length() == 0 && sb.n_blocks() == 0, "a refused forward changed B");
        threw = false;
        try {
            m.forward_batch(sb, b.data(), static_cast<int>(b.size()), 0);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expect(threw && sb.length() == 0, "a full pool did not refuse a batch for B cleanly");
        sa.truncate(capacity - pool.block_size() * (pool.n_blocks() > 1 ? 1 : 0));   // give one block back
        if (pool.n_blocks() > 1) {
            expect(pool.free_blocks() == 1, "truncate did not give a block back");
        }
    }   // A and B destroyed here
    expect(pool.free_blocks() == pool.n_blocks(), "destroying the sequences did not return every block");
    Sequence sb = m.new_sequence();
    for (std::size_t i = 0; i < b.size(); ++i) {
        const float* l = m.forward(sb, b[i], static_cast<int>(i));
        expect(same(l, ref_b[i]), "B after the exhaustion differs from its reference");
    }
    std::printf("ok exhaust: pool of %d blocks refused cleanly and recovered\n", pool.n_blocks());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model> --max-seq-len N [--block-size B] [--kv-blocks K] [--seed S] [--mode interleave|rows|cow|exhaust|rows_exhaust]\n", argv[0]);
        return 2;
    }
    try {
        int ctx = 0, bs = 0, nblocks = 0;
        unsigned seed = 1;
        std::string mode = "interleave";
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            const std::string v = argv[++i];
            if (a == "--max-seq-len") ctx = std::stoi(v);
            else if (a == "--block-size") bs = std::stoi(v);
            else if (a == "--kv-blocks") nblocks = std::stoi(v);
            else if (a == "--seed") seed = static_cast<unsigned>(std::stoul(v));
            else if (a == "--mode") mode = v;
            else throw std::runtime_error("unknown option " + a);
        }
        const Checkpoint ck = Checkpoint::load(argv[1]);
        Transformer m(ck, ctx, 0, 0, bs, nblocks);
        std::mt19937 rng(seed);
        const int vocab = m.config().vocab_size;
        if (mode == "interleave") interleave(m, rng, vocab, m.max_seq_len());
        else if (mode == "rows") rows_mode(m, rng, vocab, m.max_seq_len());
        else if (mode == "rows_exhaust") rows_exhaust(m, rng, vocab);
        else if (mode == "cow") cow_mode(m, rng, vocab, m.max_seq_len());
        else if (mode == "exhaust") exhaust(m, rng, vocab);
        else throw std::runtime_error("unknown mode " + mode);
        std::printf("PASS\n");
        return 0;
    } catch (const Failure& f) {
        std::printf("FAIL %s\n", f.what.c_str());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
