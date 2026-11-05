#include "tokenizer/text_tokenizer.h"

#include <algorithm>
#include <stdexcept>

#include "tokenizer/hf_tokenizer.h"
#include "tokenizer/tokenizer.h"

namespace geneinfer {
namespace {

class Llama2cText final : public TextTokenizer {
public:
    explicit Llama2cText(Tokenizer tok) : tok_(std::move(tok)) {}

    std::string kind() const override { return "llama2.c"; }
    int vocab_size() const override { return tok_.vocab_size(); }
    std::vector<int> encode_prompt(const std::string& text) const override {
        return tok_.encode(text, /*bos=*/true, /*eos=*/false);
    }
    std::string decode_piece(int prev, int token) const override { return tok_.decode_piece(prev, token); }
    std::string decode(const std::vector<int>& tokens) const override { return tok_.decode(tokens); }
    bool is_end_token(int token) const override { return token == Tokenizer::kBos || token == Tokenizer::kEos; }

private:
    Tokenizer tok_;
};

class HfText final : public TextTokenizer {
public:
    HfText(HfTokenizer tok, bool add_bos) : tok_(std::move(tok)), add_bos_(add_bos) {}

    std::string kind() const override { return "huggingface"; }
    int vocab_size() const override { return tok_.vocab_size(); }
    std::vector<int> encode_prompt(const std::string& text) const override {
        std::vector<int> ids = tok_.encode(text, add_bos_);
        if (ids.empty()) {  // the model cannot run on nothing: start from BOS
            if (tok_.bos_id() < 0) {
                throw std::runtime_error("the prompt is empty and the model defines no BOS token to start from");
            }
            ids.push_back(tok_.bos_id());
        }
        return ids;
    }
    // Byte-level BPE has no leading-space rule: a token is exactly its bytes, whatever came before.
    // Model vocabularies are often padded beyond the tokenizer's (to a multiple of 64, say); the model can
    // sample such an id, and like Hugging Face's decode it stands for no text.
    std::string decode_piece(int, int token) const override { return piece_or_empty(token); }
    std::string decode(const std::vector<int>& tokens) const override {
        std::string out;
        for (const int t : tokens) out += piece_or_empty(t);
        return out;
    }
    bool is_end_token(int token) const override {
        const auto& e = tok_.eos_ids();
        return std::find(e.begin(), e.end(), token) != e.end();
    }

private:
    std::string piece_or_empty(int token) const {
        try {
            return tok_.decode_piece(token);
        } catch (const std::out_of_range&) {
            return std::string();
        }
    }

    HfTokenizer tok_;
    bool add_bos_;
};

}  // namespace

std::unique_ptr<TextTokenizer> load_llama2c_tokenizer(const std::string& path, int model_vocab_size) {
    return std::make_unique<Llama2cText>(Tokenizer::load(path, model_vocab_size));
}

std::unique_ptr<TextTokenizer> load_hf_tokenizer(const std::string& directory, bool add_bos) {
    return std::make_unique<HfText>(HfTokenizer::load(directory), add_bos);
}

}  // namespace geneinfer