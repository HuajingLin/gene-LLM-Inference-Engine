// text_tokenizer.h - one interface over the two tokenizer families the engine can run with.
//
//   llama2.c   tokenizer.bin (SentencePiece-style, score merges, BOS is always the first token)
//   huggingface tokenizer.json in a model directory (byte-level BPE)
//
// The generation loop and the command line tool only need: turn a prompt into token ids, turn tokens
// back into bytes, and recognize the tokens that end a generation. Everything that differs between
// the families (BOS handling, the leading-space rule, which tokens end a sequence) lives behind this.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace geneinfer {

class TextTokenizer {
public:
    virtual ~TextTokenizer() = default;

    virtual std::string kind() const = 0;   // "llama2.c" or "huggingface"
    virtual int vocab_size() const = 0;

    // Prompt text -> the token ids the model is fed first.
    // Throws std::runtime_error if a prompt cannot be encoded (e.g. empty prompt and no BOS token).
    virtual std::vector<int> encode_prompt(const std::string& text) const = 0;

    // Raw bytes of one generated token; `prev` is the token before it (-1 if none).
    virtual std::string decode_piece(int prev, int token) const = 0;
    virtual std::string decode(const std::vector<int>& tokens) const = 0;

    // True for tokens that end a generation when the model samples them.
    virtual bool is_end_token(int token) const = 0;
};

// llama2.c tokenizer.bin; every prompt starts with BOS; BOS and EOS end a generation.
std::unique_ptr<TextTokenizer> load_llama2c_tokenizer(const std::string& path, int model_vocab_size);

// Hugging Face directory (tokenizer.json + optional config.json). No token is added to the prompt unless
// `add_bos` is set (like transformers); an empty prompt falls back to BOS because the model needs at
// least one input token. The tokens listed as eos_token_id in config.json end a generation.
std::unique_ptr<TextTokenizer> load_hf_tokenizer(const std::string& directory, bool add_bos);

}  // namespace geneinfer