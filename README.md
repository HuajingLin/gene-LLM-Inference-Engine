# Gene LLM Engine
Lightweight LLM Inference Engine

```diff
# Models used
+ Stage 1: stories15M
+ Stage 2: SmolLM2-135M
+ Stage 3a: SmolLM2-135M
+ Stage 3b: SmolLM2-135M
- Stage 3b: SmolLM2-135M
```
[Project Wiki](https://github.com/HuajingLin/gene-LLM-Inference-Engine/wiki/Stage-3c)

## run
```diff
cmake -S . -B build
cmake --build build --config Release

# End-to-end performance testing (one run for each of the three instruction sets) model Transformer, Sampler
$ for isa in scalar avx2 avx512; do ./build/gi_bench --model models/SmolLM2-135M --isa $isa --label $isa; done
```

## Main Stages
### Stage 1: M1 ~ M6   
### Stage 2: M7 ~ M10   
### Stage 3a: Performance Optimization: SIMD matvec   
### Stage 3b: Weights are stored in bf16 format  
### Stage 3c: Multithreading 
```diff
                        scalar      avx2        avx512
decode                  7.6 tok/s   23.4 tok/s  23.0 tok/s
prefil                  9.9 tok/s   30.7 tok/s  29.8 tok/s
effective bandwidth     4.2 GB/s    12.7 GB/s   12.5 GB/s
```
- SIMD makes computation about three times faster, approaching the upper limit of memory bandwidth.
- It’s almost identical to AVX2. This indicates that the process has become memory-bound; further increasing computational speed is pointless, as time is simply spent waiting for data to be fetched from memory.

### Stage 3b: Weights are stored in bf16 format.
```diff
            f32 weights     bf16 native
decode      26.5 tok/s      42.1 tok/s
prefill     34.2 tok/s      56.0 tok/s
peak RSS    866 MiB         352 MiB
```
All the checks pass, and the speedup is real: decode went from 37.7 to 23.1 ms per token (26.5 to 42.1 tok/s), with logits bit-identical to the float32-expanded weights and 24/24 greedy tokens matching PyTorch.

### Stage 3c: Multithreading 
```diff
# threads: 4
| label | prefill tok/s | decode tok/s | median ms/token | GFLOP/s | peak RSS MiB |
|-------|---------------|--------------|-----------------|---------|--------------|
| —     | 77.6          | 59.3         | 16.634          | 16.17   | 352.6        |
```

## Project Structure
```
geneInfer/
├── CMakeLists.txt
├── .gitignore
├── src/
│   ├── loader/                     # M (Milestone)
│   │   ├── mapped_file.h / .cpp    # M1: Cross-platform mmap (POSIX / Windows)
│   │   ├── checkpoint.h / .cpp     # M1: Config、Weights、Checkpoint::load
│   │   ├── dtype.h / .cpp          # M7: Conversion from f16 to float32
│   │   ├── safetensors.h / .cpp    # M7: tensors reader
│   │   └── hf_checkpoint.cpp       # M7: Hugging Face Model directory loading
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.h / .cpp      # M2: tokenizer
│   │   ├── hf_tokenizer.h / .cpp   # M9
│   │   ├── unicode.h / .cpp        # M9
│   │   └── text_tokenizer.h / .cpp # M10
│   │
│   ├── kernels/
│   │   ├── kernels.h / .cpp        # M3: operators
│   │   ├── kernel_table.h / .cpp   # Stage 3a
│   │   ├── kernels_avx2.cpp        # Stage 3a
│   │   └── kernels_avx512.cpp      # Stage 3a
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
│   ├── util/
│   │   ├── process_memory.h / .cpp # M6
│   │   ├── json.h / .cpp           # M7 JSON Parser
│   │   └── jthread_pool.h / .cpp   # Stage 3c
│   │
│   └── common/
│       └── rope_style.h            # M8
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
│   ├── sampler_check_main.cpp  # M5: gi_sampler_check
│   ├── hf_io.py                # M7: Read and write HF tensors
│   ├── make_dummy_hf.py        # M7: generating random numbers for model
│   ├── json_check_main.cpp     # M7: gi_json_check
│   ├── dtype_check_main.cpp    # M7: gi_dtype_check
│   ├── hf_golden.py            # M8: takes the prompt ids from the tokenizers library
│   ├── env_check.py            # M8
│   ├── hf_tokenize_main.cpp    # M9: gi_hf_tokenize
│   ├── unicode_check_main.cpp  # M9: gi_unicode_check
│   ├── hf_tok_ref.py           # M9 standard answer
│   ├── tok_corpus.py           # M9 Test corpus
│   ├── make_dummy_hf_tokenizer.py  # M9 
│   ├── hf_tokenizer_golden.py      # M9 HF Golden Data
│   ├── gen_unicode_tables.py       # M9
│   ├── hf_tokenizer_probe.p        # M9
│   └── pool_check_main.cpp         # stage 3c
│
├── tests/
│    ├── verify_loader.py       # M1: Verification: C++ Output vs. NumPy Element-wise Comparison
│    ├── verify_tokenizer.py    # M2: Comparison with Python reference implementation
│    ├── verify_kernels.py      # M3
│    ├── verify_forward.py      # M4
│    ├── verify_sampler.py      # M5
│    ├── verify_generate.py     # M5
│    ├── verify_bench.py        # M6
│    ├── verify_json.py         # M7
│    ├── verify_dtype.py        # M7
│    ├── verify_hf_loader.py    # M7
│    ├── verify_hf_tokenizer.py # M9
│    ├── verify_unicode.py      # M9
│    ├── verify_generate_hf.py  # M10
│    └── verify_threads.py      # stage 3c
│
└── bench/
     ├── bench_main.cpp         # M6: gi_bench
     └── kernel_bench_main.cpp  # Stage 3a: gi_kernel_bench
```