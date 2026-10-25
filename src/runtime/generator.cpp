#include "runtime/generator.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

namespace geneinfer {

GenerateResult generate(Transformer& model, const Tokenizer& tokenizer, Sampler& sampler,
                        const std::vector<int>& prompt_tokens, int max_positions,
                        const PieceCallback& on_piece) {
    using Clock = std::chrono::steady_clock;

    if (prompt_tokens.empty()) throw std::invalid_argument("generate: prompt has no tokens");
    const int n_prompt = static_cast<int>(prompt_tokens.size());
    const int context = model.max_seq_len();
    if (n_prompt > context) {
        throw std::invalid_argument("generate: prompt of " + std::to_string(n_prompt) +
                                    " tokens does not fit in the context length " +
                                    std::to_string(context));
    }
    const int steps = (max_positions <= 0) ? context : std::min(max_positions, context);

    GenerateResult result;
    result.prompt_tokens = prompt_tokens;

    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        const auto start = Clock::now();

        // Only the last prompt position and later positions need logits.
        const bool needs_logits = pos >= n_prompt - 1;
        const float* logits = model.forward(token, pos, needs_logits);

        int next;
        if (!needs_logits) {
            next = prompt_tokens[static_cast<std::size_t>(pos) + 1];  // still feeding the prompt
        } else {
            next = sampler.sample(logits);
            result.generated_tokens.push_back(next);
        }

        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (pos < n_prompt) {
            result.prefill_ms += ms;
        } else {
            result.decode_ms += ms;
            ++result.decode_steps;
        }
        ++pos;

        if (next == Tokenizer::kBos || next == Tokenizer::kEos) {
            result.stopped_by_terminator = true;
            break;
        }
        if (pos >= n_prompt) {  // `next` is a generated token
            std::string piece = tokenizer.decode_piece(token, next);
            result.text += piece;
            if (on_piece) on_piece(piece);
        }
        token = next;
    }
    return result;
}

}  // namespace geneinfer
