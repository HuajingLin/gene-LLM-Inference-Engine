// generator.h - the generation loop: prefill the prompt, then sample token after token.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "model/transformer.h"
#include "sampler/sampler.h"
#include "tokenizer/tokenizer.h"

namespace geneinfer {

struct GenerateResult {
    std::vector<int> prompt_tokens;
    // Sampled tokens. If generation stopped at BOS/EOS, that terminator is the last entry
    // (it is not part of `text`).
    std::vector<int> generated_tokens;
    std::string text;  // decoded generated tokens (raw bytes, may end inside a UTF-8 sequence)

    bool stopped_by_terminator = false;  // BOS/EOS sampled (as opposed to hitting the limit)
    double prefill_ms = 0.0;             // forward passes of the prompt, incl. the first sample
    double decode_ms = 0.0;              // all later forward passes + sampling
    int decode_steps = 0;                // number of iterations counted in decode_ms
};

// Called with the decoded bytes of every newly generated token (for streaming output).
using PieceCallback = std::function<void(const std::string& piece)>;

// Feeds `prompt_tokens` (already encoded, normally starting with BOS) through the model and
// then generates until a BOS/EOS is sampled or `max_positions` positions are used in total
// (prompt included; 0 or too large means the model's context length).
//
// With temperature 0 this loop produces exactly the tokens of tools/ref.py.
// Throws std::invalid_argument if the prompt is empty or does not fit in the context.
GenerateResult generate(Transformer& model, const Tokenizer& tokenizer, Sampler& sampler,
                        const std::vector<int>& prompt_tokens, int max_positions,
                        const PieceCallback& on_piece = nullptr);

}  // namespace geneinfer
