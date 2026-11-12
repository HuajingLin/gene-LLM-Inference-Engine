#include "runtime/engine.h"

#include <algorithm>
#include <stdexcept>

namespace geneinfer {

struct Engine::Job {
    int id = 0;
    EngineRequest request;
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<Sequence> seq;   // exists while the job is running
    std::vector<int> tokens;         // prompt followed by the generated tokens
    int fed = 0;                     // tokens[0 .. fed) are in the KV cache
    int cached_upto = 0;             // tokens[0 .. cached_upto) are registered in the prefix cache
    int steps = 0;                   // position limit (prompt and generated tokens)
    EngineResult result;
};

Engine::Engine(Transformer& model, const TextTokenizer& tokenizer, const EngineConfig& config)
    : model_(model), tokenizer_(tokenizer), cfg_(config) {
    if (cfg_.max_running < 1) throw std::invalid_argument("Engine: max_running must be >= 1");
    if (cfg_.step_rows < 0 || cfg_.prefill_chunk < 0) throw std::invalid_argument("Engine: step_rows and prefill_chunk must be >= 0");
    if (cfg_.step_rows == 0 || cfg_.step_rows > model_.max_batch()) cfg_.step_rows = model_.max_batch();
    if (cfg_.prefill_chunk == 0) cfg_.prefill_chunk = cfg_.step_rows;
    if (cfg_.prefix_cache) cache_ = std::make_unique<PrefixCache>(model_.kv_pool());
}

Engine::~Engine() = default;

int Engine::submit(EngineRequest request) {
    const int context = model_.max_seq_len();
    if (request.prompt.empty()) throw std::invalid_argument("Engine::submit: prompt has no tokens");
    if (static_cast<int>(request.prompt.size()) > context) {
        throw std::invalid_argument("Engine::submit: prompt of " + std::to_string(request.prompt.size()) +
                                    " tokens does not fit in the context length " + std::to_string(context));
    }
    for (int t : request.prompt) {
        if (t < 0 || t >= model_.config().vocab_size) throw std::invalid_argument("Engine::submit: token id out of range");
    }
    auto job = std::make_unique<Job>();
    job->id = static_cast<int>(jobs_.size());
    job->sampler = std::make_unique<Sampler>(request.sampler, model_.config().vocab_size);   // validates the config
    job->steps = (request.max_positions <= 0) ? context : std::min(request.max_positions, context);
    job->tokens = request.prompt;
    job->result.id = job->id;
    job->result.prompt_tokens = request.prompt;
    job->request = std::move(request);
    Job* raw = job.get();
    jobs_.push_back(std::move(job));
    waiting_.push_back(raw);
    return raw->id;
}

bool Engine::idle() const { return waiting_.empty() && running_.empty(); }
int Engine::running() const { return static_cast<int>(running_.size()); }
int Engine::waiting() const { return static_cast<int>(waiting_.size()); }

const EngineResult& Engine::result(int id) const {
    if (id < 0 || id >= static_cast<int>(jobs_.size())) throw std::out_of_range("Engine::result: unknown request id");
    return jobs_[static_cast<std::size_t>(id)]->result;
}

void Engine::admit() {
    while (!waiting_.empty() && static_cast<int>(running_.size()) < cfg_.max_running) {
        Job* j = waiting_.front();
        waiting_.pop_front();
        j->seq = std::make_unique<Sequence>(model_.new_sequence());
        j->fed = 0;
        j->cached_upto = 0;
        if (cache_) {
            // Reuse the blocks of an earlier request with the same first tokens. At least one token must still be fed (it
            // produces the logits), and the position limit must leave a row to process.
            const int usable = std::min(static_cast<int>(j->tokens.size()) - 1, j->steps - 1);
            std::vector<int> blocks;
            const int n = usable > 0 ? cache_->match(j->tokens.data(), usable, blocks) : 0;
            if (n > 0) {
                j->seq->adopt(blocks);
                j->fed = n * model_.kv_pool().block_size();
                j->cached_upto = j->fed;
                j->result.cached_tokens += j->fed;
                stats_.prefix_tokens += j->fed;
                ++stats_.prefix_hits;
            }
        }
        running_.push_back(j);
    }
    stats_.max_running_seen = std::max(stats_.max_running_seen, static_cast<int>(running_.size()));
}

// Registers the blocks that became full since the last call (the tokens of positions 0 .. fed).
void Engine::cache_new_blocks(Job& job) {
    if (!cache_) return;
    const int bs = model_.kv_pool().block_size();
    if (job.fed / bs <= job.cached_upto / bs) return;
    cache_->insert(job.tokens.data(), job.fed, job.seq->block_table(), job.seq->n_blocks());
    job.cached_upto = job.fed;
}

void Engine::finish(Job& job, const std::string& error) {
    job.result.finished = true;
    job.result.error = error;
    job.result.last_pass = stats_.passes;
    job.seq.reset();   // gives the blocks back
    running_.erase(std::find(running_.begin(), running_.end(), &job));
}

void Engine::preempt_youngest() {
    Job* j = running_.back();
    running_.pop_back();
    j->seq.reset();
    j->fed = 0;
    ++j->result.preemptions;
    ++stats_.preemptions;
    waiting_.push_front(j);
}

bool Engine::step() {
    admit();
    if (running_.empty()) return false;

    const int bs = model_.kv_pool().block_size();
    for (;;) {
        // ---- build the pass: generating requests first (one row each), then prompt chunks, oldest first ----
        std::vector<Transformer::StepRow> rows;
        std::vector<int> job_rows(running_.size(), 0);
        std::vector<int> job_logit(running_.size(), -1);   // index of the job's row among the rows that return logits
        int logit_rows = 0;
        int free_blocks = model_.kv_pool().free_blocks() + (cache_ ? cache_->evictable() : 0);   // cached blocks nobody uses can be evicted
        int needed_blocks = 0;
        int budget = cfg_.step_rows;
        auto blocks_for = [&](const Job& j, int extra) {
            const int need = (j.seq->length() + extra + bs - 1) / bs;
            return std::max(0, need - j.seq->n_blocks());
        };
        for (int phase = 0; phase < 2; ++phase) {
            for (std::size_t k = 0; k < running_.size(); ++k) {
                Job& j = *running_[k];
                const int pending = static_cast<int>(j.tokens.size()) - j.fed;
                const bool decoding = pending == 1;
                if ((phase == 0) != decoding) continue;
                int take = std::min({pending, j.steps - j.fed, budget, phase == 0 ? 1 : cfg_.prefill_chunk});
                while (take > 0 && blocks_for(j, take) > free_blocks) --take;   // as many rows as the free blocks allow
                if (take <= 0) continue;
                free_blocks -= blocks_for(j, take);
                needed_blocks += blocks_for(j, take);
                budget -= take;
                job_rows[k] = take;
                for (int r = 0; r < take; ++r) {
                    const int idx = j.fed + r;
                    const bool wants = idx == static_cast<int>(j.tokens.size()) - 1;
                    rows.push_back({j.seq.get(), j.tokens[static_cast<std::size_t>(idx)], wants});
                    if (wants) job_logit[k] = logit_rows++;
                }
            }
        }
        if (rows.empty()) {
            // Nothing fits in the free blocks. Free memory by recomputing the youngest request later; a lone request
            // that does not fit even by itself can never finish.
            if (running_.size() == 1) {
                finish(*running_.front(), "the KV pool is too small for this request");
                return true;
            }
            preempt_youngest();
            continue;
        }

        // ---- run it ----
        if (cache_ && needed_blocks > model_.kv_pool().free_blocks()) {
            stats_.evicted_blocks += cache_->evict(needed_blocks - model_.kv_pool().free_blocks());
        }
        const float* logits = model_.forward_rows(rows.data(), static_cast<int>(rows.size()));
        ++stats_.passes;
        stats_.rows += static_cast<long>(rows.size());
        stats_.max_rows_in_pass = std::max(stats_.max_rows_in_pass, static_cast<int>(rows.size()));
        for (int n : job_rows) stats_.max_chunk = std::max(stats_.max_chunk, n);
        const int vocab = model_.config().vocab_size;

        // ---- sample, advance, retire ----
        std::vector<Job*> done;
        for (std::size_t k = 0; k < running_.size(); ++k) {
            Job& j = *running_[k];
            if (job_rows[k] == 0) continue;
            if (j.result.first_pass < 0) j.result.first_pass = stats_.passes;
            j.fed += job_rows[k];
            cache_new_blocks(j);
            if (j.fed == static_cast<int>(j.tokens.size())) {   // its last row asked for logits: sample
                const int prev = j.tokens.back();
                const int next = j.sampler->sample(logits + static_cast<std::size_t>(job_logit[k]) * static_cast<std::size_t>(vocab));
                ++stats_.logits_rows;
                j.result.generated_tokens.push_back(next);
                j.tokens.push_back(next);
                if (j.result.first_token_pass < 0) j.result.first_token_pass = stats_.passes;
                if (tokenizer_.is_end_token(next)) {
                    j.result.stopped_by_end_token = true;
                    done.push_back(&j);
                    continue;
                }
                std::string piece = tokenizer_.decode_piece(prev, next);
                j.result.text += piece;
                if (j.request.on_piece) j.request.on_piece(piece);
            }
            if (j.fed >= j.steps) done.push_back(&j);   // position limit reached
        }
        for (Job* j : done) finish(*j, "");
        return true;
    }
}

void Engine::run_until_idle() {
    while (step()) {
    }
}

}  // namespace geneinfer
