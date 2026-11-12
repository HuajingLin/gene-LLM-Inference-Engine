// multiseq_check_main.cpp - several sequences sharing one paged KV pool.
//
//   gi_multiseq <model> --max-seq-len N [--block-size B] [--kv-blocks K] [--seed S] [--mode interleave|exhaust]
//
// interleave: three sequences (random token streams) are run alone to get reference logits, then run again in a
//   random interleaving of single-token steps and small batches on one shared pool, with random rewinds
//   (re-running earlier positions). Every logits row must be bit-identical to the reference.
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
        std::fprintf(stderr, "usage: %s <model> --max-seq-len N [--block-size B] [--kv-blocks K] [--seed S] [--mode interleave|exhaust]\n", argv[0]);
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
