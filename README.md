# Gene LLM Engine
Lightweight LLM Inference Engine
Core Data Flow:
```
Text Input
→ Tokenizer (text → token IDs)
→ Prefill (processes the entire prompt at once, populates the KV Cache)
→ Decode Loop (generates one token at a time)
  Embedding → N × Transformer Blocks → RMSNorm → LM Head → logits
→ Sampler (selects the next token) → appended to the KV Cache
→ Detokenizer (token IDs → text stream output)
```
[Project Wiki](https://github.com/HuajingLin/gene-LLM-Inference-Engine/wiki/M1.-Preparing-for-comparison-and-verification)

# run
```
cmake -S . -B build
cmake --build build --config Release

$ python3 tests/verify_forward.py --exe build/gi_forward \
    --model models/stories15M/stories15M.bin --golden ref_out/ref.json
```

## Project Structure

```
ppm-project/
├── CMakeLists.txt
├── .gitignore
├── src/
│   ├── loader/
│   │   ├── mapped_file.h / .cpp    # Cross-platform mmap (POSIX / Windows)
│   │   └── checkpoint.h / .cpp     # Config、Weights、Checkpoint::load
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.h / .cpp      #tokenizer
│   │   └── kernel_check_main.cpp
│   │
│   ├── kernels/
│   │   └── kernels.h / .cpp
│   │
│   ├── model/
│   │   └── transformer.h / .cpp
│   │
│   ├── sampler/
│   │   └── sampler.h / .cpp
│   │
│   └── runtime/
│       └── generator.h / c.pp
│   
├── models/
│   └── stories15M/         # TinyStories
│   
├── tools/
│   ├── inspect_main.cpp    # gi_inspect: Print configuration and the fingerprint of each tensor
│   ├── llama2c_io.py       # Loading checkpoints/tokenizers using NumPy
│   ├── ref.py              # M0: Reference forward + greedy generation; outputs the standar answer
│   ├── make_dummy.py       # Generate a small, random model for offline testing purposes
│   ├── tokenize_main.cpp   # gi_tokenize
│   ├── kernel_check_main.cpp
│   ├── forward_main.cpp
│   ├── generate_main.cpp
│   └── sampler_check_main.cpp
│   
└── tests/
    ├── verify_loader.py    # M1 Verification: C++ Output vs. NumPy Element-wise Comparison
    ├── verify_tokenizer.py # Comparison with Python reference implementation
    ├── verify_kernels.py
    ├── verify_forward.py
    ├── verify_sampler.py
    └── verify_generate.py
```