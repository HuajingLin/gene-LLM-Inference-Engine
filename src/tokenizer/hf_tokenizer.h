// hf_tokenizer.h - byte-level BPE tokenizer in the Hugging Face "tokenizer.json" format
// (GPT-2 family: SmolLM2, GPT-2, many Llama-architecture models).
//
// Supported configuration (anything else is rejected with a clear error, never guessed):
//   model          BPE, no continuing-subword prefix / end-of-word suffix, no byte fallback
//   normalizer     none, or NFC (Unicode normalization form C)
//   pre_tokenizer  ByteLevel(add_prefix_space=false)                      or
//                  Sequence[Digits(individual_digits), ByteLevel(add_prefix_space=false)]   or
//                  Sequence[Split(<the Qwen2 regular expression>, Isolated), ByteLevel(use_regex=false)]
//   post_processor none or ByteLevel (adds no tokens)
//   decoder        ByteLevel
//   added_tokens   matched literally in the input text, without lstrip / rstrip / single_word
//
// Pipeline of encode(): split the text at added tokens (e.g. "<|im_start|>") -> normalize each part (NFC, if the
// tokenizer says so) -> pre-tokenize it (digits, then the GPT-2 regex; or the Qwen2 regex) -> map bytes to the printable "byte-level" alphabet -> merge pairs
// by priority (BPE). decode() reverses the last step and returns raw UTF-8 bytes.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace geneinfer {

class HfTokenizer {
public:
    // `directory` holds tokenizer.json (required) and config.json (optional: bos/eos token ids).
    // Throws std::runtime_error on missing files, invalid JSON or unsupported settings.
    static HfTokenizer load(const std::string& directory);

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }   // largest id + 1
    int token_to_id(const std::string& token) const;                           // -1 if unknown
    bool is_special(int id) const;                                             // added token marked "special"
    // True when tokenizer.json's post_processor prepends the BOS token (Hugging Face then adds it by default, as for
    // Llama 3); encode() does so only on request (add_bos).
    bool template_adds_bos() const { return template_adds_bos_; }
    int bos_id() const { return bos_id_; }                                     // -1 if the model has none
    const std::vector<int>& eos_ids() const { return eos_ids_; }               // end-of-sequence token ids

    // Text (UTF-8) -> token ids. Like Hugging Face's default, no special token is added unless
    // `add_bos` is set (which needs bos_id() >= 0).
    std::vector<int> encode(const std::string& text, bool add_bos = false) const;

    // The raw bytes one token stands for (added tokens: their text). Concatenating the pieces of a
    // sequence gives its text; a multi-byte character may be split between two pieces.
    std::string decode_piece(int id) const;
    std::string decode(const std::vector<int>& ids, bool skip_special = false) const;

    // The normalizer on its own (identity when the tokenizer has none); exposed for tests.
    std::string normalize(const std::string& text) const;

    // The pieces the pre-tokenizer cuts an (already normalized) plain text into, before BPE; exposed for tests.
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    std::string describe() const;   // one-paragraph summary of what was loaded

private:
    HfTokenizer() = default;

    struct AddedToken {
        int id;
        std::string content;
    };
    struct MergeInfo {
        int rank;     // position in the merges list: lower merges first
        int merged;   // id of the merged token
    };

    void encode_plain(const std::string& text, std::vector<int>& out) const;
    void bpe(const std::string& piece, std::vector<int>& out) const;
    int match_added(const std::string& text, std::size_t pos) const;   // index into added_, or -1

    std::vector<std::string> id_to_token_;
    std::vector<char> has_token_;
    std::vector<char> special_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::uint64_t, MergeInfo> merges_;     // key: left id << 32 | right id
    std::vector<AddedToken> added_;
    std::array<std::vector<int>, 256> added_by_first_byte_;   // indices into added_, longest first
    std::array<int, 256> byte_token_{};                       // token id of each byte, -1 if absent

    int unk_id_ = -1;
    bool ignore_merges_ = false;
    bool split_digits_ = false;
    bool individual_digits_ = false;
    bool use_regex_ = true;
    bool qwen2_split_ = false;      // the Split pre-tokenizer with the Qwen2 or the Llama 3 regular expression
    int split_max_digits_ = 1;      // longest run of digits that forms one piece: 1 (Qwen2) or 3 (Llama 3)
    bool nfc_ = false;              // normalizer: NFC
    int bos_id_ = -1;
    bool template_adds_bos_ = false;
    int template_bos_id_ = -1;
    std::vector<int> eos_ids_;
    std::size_t n_merges_ = 0;
};

}  // namespace geneinfer
