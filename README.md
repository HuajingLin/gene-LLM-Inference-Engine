# Gene LLM Engine
Lightweight LLM Inference Engine

```diff
- Stage 1: stories15M
+ Stage 2: SmolLM2-135M
```
[Project Wiki](https://github.com/HuajingLin/gene-LLM-Inference-Engine/wiki/M1.-Preparing-for-comparison-and-verification)

## run
```
cmake -S . -B build
cmake --build build --config Release

$ python3 tests/verify_forward.py --exe build/gi_forward \
    --model models/stories15M/stories15M.bin --golden ref_out/ref.json
```

## Project Structure
Stage 1: M1 ~ M6
```
geneInfer/
├── CMakeLists.txt
├── .gitignore
├── src/
│   ├── loader/                     # M (Milestone)
│   │   ├── mapped_file.h / .cpp    # M1: Cross-platform mmap (POSIX / Windows)
│   │   └── checkpoint.h / .cpp     # M1: Config、Weights、Checkpoint::load
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.h / .cpp      # M2: tokenizer
│   │   └── kernel_check_main.cpp
│   │
│   ├── kernels/
│   │   └── kernels.h / .cpp        # M3: operators
│   │
│   ├── model/
│   │   └── transformer.h / .cpp    # M4
│   │
│   ├── sampler/
│   │   └── sampler.h / .cpp        # M5
│   │
│   ├── runtime/
│   │   └── generator.h / c.pp      # M5
│   │
│   └── util/
│       └── process_memory.h / .cpp # M6
│   
├── models/
│   ├── stories15M/             # TinyStories
│   └── SmolLM2-135M/           # SmolLM2
│   
├── tools/
│   ├── ref.py                  # M1: Reference forward + greedy generation; outputs the standar answer
│   ├── llama2c_io.py           # M1: Loading checkpoints/tokenizers using NumPy
│   ├── make_dummy.py           # M1: Generate a small, random model for offline testing purposes
│   ├── inspect_main.cpp        # M1: gi_inspect: Print configuration and the fingerprint of each tensor
│   ├── tokenize_main.cpp       # M2: gi_tokenize
│   ├── kernel_check_main.cpp   # M3: gi_kernel_check
│   ├── forward_main.cpp        # M4: gi_forward
│   ├── generate_main.cpp       # M5: gi_generate
│   └── sampler_check_main.cpp  # M5: gi_sampler_check
│   
├── tests/
│    ├── verify_loader.py       # M1: Verification: C++ Output vs. NumPy Element-wise Comparison
│    ├── verify_tokenizer.py    # M2: Comparison with Python reference implementation
│    ├── verify_kernels.py      # M3
│    ├── verify_forward.py      # M4
│    ├── verify_sampler.py      # M5
│    ├── verify_generate.py     # M5
│    └── verify_bench.py        # M6
│
└── bench/
     └── bench_main.cpp         # M6: gi_bench
```