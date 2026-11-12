// engine.h - serving several requests at once on one model: request queue, scheduler, continuous batching.
//
// Every step() builds ONE forward pass (Transformer::forward_rows) out of rows of the running requests: a request that
// is generating contributes one row (its last token), a request that is still reading its prompt contributes a chunk
// of prompt tokens. Requests join and leave between steps, so the batch never waits for its slowest member. All
// weights are read once per pass for all rows, which is what makes decoding several requests cheaper per token.
//
// Each request behaves exactly as generate() would run it alone with the same sampler seed: same tokens, same stop
// rules (end token, position limit). Logits are bit-identical to the single-sequence path, so the outputs are equal.
//
// Memory: every request owns a Sequence on the model's KV pool. When the pool runs dry the scheduler first schedules
// fewer rows; if no row fits at all it preempts the most recently admitted request (its blocks are freed and it is
// recomputed later: prompt and tokens generated so far are fed again, which gives the same result). A request that
// cannot make progress even when running alone is finished with an error text.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "model/transformer.h"
#include "sampler/sampler.h"
#include "tokenizer/text_tokenizer.h"

namespace geneinfer {

struct EngineConfig {
    int max_running = 8;       // requests that hold KV memory and take part in a pass at the same time
    int step_rows = 0;         // rows per pass; 0 = the model's max_batch
    int prefill_chunk = 0;     // most prompt tokens one request adds to a pass; 0 = no limit besides step_rows
};

struct EngineRequest {
    std::vector<int> prompt;           // encoded prompt, not empty
    int max_positions = 0;             // prompt + generated tokens at most; 0 or too large = the context length
    SamplerConfig sampler;
    std::function<void(const std::string&)> on_piece;   // streamed text of every generated token (optional)
};

struct EngineResult {
    int id = -1;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;   // an end token, if one was sampled, is the last entry (not part of text)
    std::string text;
    bool finished = false;
    bool stopped_by_end_token = false;
    std::string error;                   // not empty if the request failed (e.g. the KV pool is too small for it)
    int preemptions = 0;
    long first_pass = -1;                // pass in which the request first took part / produced its first token
    long first_token_pass = -1;
    long last_pass = -1;
};

struct EngineStats {
    long passes = 0;
    long rows = 0;            // rows over all passes
    long logits_rows = 0;
    long preemptions = 0;
    int max_rows_in_pass = 0;
    int max_chunk = 0;        // most rows one request contributed to a single pass (prompt chunks)
    int max_running_seen = 0;
};

class Engine {
public:
    Engine(Transformer& model, const TextTokenizer& tokenizer, const EngineConfig& config = {});
    ~Engine();

    // Queues a request and returns its id (0, 1, 2 ...). Throws std::invalid_argument for an empty prompt, a prompt
    // that does not fit in the context, or a token id outside the vocabulary.
    int submit(EngineRequest request);

    // Runs one forward pass. Returns false (and does nothing) if no request is waiting or running.
    bool step();

    void run_until_idle();   // step() until everything has finished
    bool idle() const;
    int running() const;
    int waiting() const;

    const EngineResult& result(int id) const;
    const EngineStats& stats() const { return stats_; }

private:
    struct Job;
    void admit();
    void preempt_youngest();
    void finish(Job& job, const std::string& error);

    Transformer& model_;
    const TextTokenizer& tokenizer_;
    EngineConfig cfg_;
    std::vector<std::unique_ptr<Job>> jobs_;   // by id
    std::deque<Job*> waiting_;
    std::vector<Job*> running_;                // in admission order
    EngineStats stats_;
};

}  // namespace geneinfer
