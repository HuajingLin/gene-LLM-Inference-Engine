# Gene LLM Engine
Lightweight LLM Inference Engine

```diff
# Models used
+ Stage 1: stories15M
+ Stage 2: SmolLM2-135M
+ Stage 3a ~ 3f: SmolLM2-135M
+ Stage 3 M11: Qwen3.06B
- Stage 4: Qwen3.06B
```

## run
```diff
cmake -S . -B build
cmake --build build --config Release
```

## Main Stages
### Stage 1: M1 ~ M6   
### Stage 2: M7 ~ M10

### Stage 3a: SIMD matvec
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
# Threading works, but it only buys about 1.3x.
```
### Stage 3d: int8 quantization
Speed: int8 gives 9.4–9.7 ms/token with 2–3 threads, about 105 tok/s. That is 1.75x faster than the best bf16 run (16.4 ms). The kernels and the thread bit-identity check passed.   
Accuracy is worse than I’d like. Mean KL is 1.2e-2 and the logit error is 3.4%.    
[Project Wiki](https://github.com/HuajingLin/gene-LLM-Inference-Engine/wiki/Stage-3d)

### Stage 3e: GI_Q8_SKIP
per token  median 8.772 ms, so the sensitive-layer fix costs nothing in speed.

### Stage 3f: Batched prefill 
```diff
Threads	Token-by-token	     Batched	     Speedup
1	     94.5 tok/s	     155.3 tok/s	1.64x
2	     116.4 tok/s	270.8 tok/s	2.3x
3	     140.5 tok/s	312.1 tok/s	2.2x
4	     120.7 tok/s	333.9 tok/s	2.8x
```
### Stage 3.M11: Qwen3-0.6B
| label | prefill tok/s | decode tok/s | median ms/token | GFLOP/s | peak RSS MiB |
|-------|---------------|--------------|-----------------|---------|--------------|
| -     | 116.8         | 27.7         | 35.585          | 33.50   | 1761.4       |

### Stage 4.1: the paged KV cache
| label | prefill tok/s | decode tok/s | median ms/token | GFLOP/s | peak RSS MiB |
|-------|---------------|--------------|-----------------|---------|--------------|
| -     | 111.5         | 27.9         | 35.781          | 33.32   | 1761.6       |

### Stage 4.2 batching
```diff
concurrent  wall (s)   tokens/s     tok/s/request  rows/pass    speedup    outputs
1           2.539      25.2         25.2           1.23         1.00       identical to running alone
2           3.269      39.2         19.6           2.47         1.55       identical to running alone
4           5.343      47.9         12.0           4.94         1.90       identical to running alone
8           8.623      59.4         7.4            9.58         2.36       identical to running alone
(tokens/s counts generated tokens only; wall time includes the prompts.)
```
### Stage 4.3 prefix caching
the 1-request run takes 0.83 s against 1.81 s without, and the 4-request run takes 1.79 s against 7.0 s, about 2.2x and 3.9x faster.

### Stage 4.4 multi-row kernel 
```diff
			     Multi-token	Per-token
Prefill, run 1		158.8 tok/s	110.4 tok/s
Prefill, run 2		155.0 tok/s	111.1 tok/s
Decode         	28.2 tok/s	28.2 tok/s
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
│   │   ├── hf_checkpoint.cpp       # M7: Hugging Face Model directory loading
│   │   └── quantize.h / .cpp       # stage 3d
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.h / .cpp      # M2: tokenizer
│   │   ├── hf_tokenizer.h / .cpp   # M9
│   │   ├── unicode.h / .cpp        # M9
│   │   ├── text_tokenizer.h / .cpp # M10
│   │   └── unicode_norm_tables.inc # M11
│   │
│   ├── kernels/
│   │   ├── kernels.h / .cpp        # M3: operators
│   │   ├── kernel_table.h / .cpp   # Stage 3a
│   │   ├── kernels_avx2.cpp        # Stage 3a
│   │   └── kernels_avx512.cpp      # Stage 3a
│   │
│   ├── model/
│   │   ├── transformer.h / .cpp    # M4
│   │   ├── kv_cache.h / .cpp       # Stage 4.1
│   │   └── prefix_cache.h / .cpp   # Stage 4.3
│   │
│   ├── sampler/
│   │   └── sampler.h / .cpp        # M5
│   │
│   ├── runtime/
│   │   ├── generator.h / c.pp      # M5
│   │   └── engine.h / c.pp         # stage 4.2
│   │
│   ├── util/
│   │   ├── process_memory.h / .cpp # M6
│   │   ├── json.h / .cpp           # M7 JSON Parser
│   │   └── jthread_pool.h / .cpp   # Stage 3c
│   │
│   └── common/
│       ├── rope_style.h        # M8
│       └── q8.h                # stage 3d
│   
├── models/
│   ├── stories15M/             # TinyStories
│   ├── SmolLM2-135M/
│   └── Qwen3-0.6B/
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
│   ├── pool_check_main.cpp         # stage 3c
│   ├── quant_dump_main.cpp         # stage 3d
│   ├── quant_stats.py              # stage 3e
│   ├── gen_norm_tables.py      # M11
│   ├── multiseq_check_main.cpp # stage 4.1
│   ├── kvpool_check_main.cpp   # stage 4.1
│   └── prefixcache_check_main.cpp  # stage 4.3
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
│    ├── verify_threads.py      # stage 3c
│    ├── verify_quantize.py     # stage 3d
│    ├── quant_accuracy.py      # stage 3d
│    ├── verify_batch.py        # stage 3f
│    ├── verify_nfc.py          # M11
│    ├── verify_kvpool.py       # stage 4.1
│    ├── verify_multiseq.py     # stage 4.1
│    └── verify_prefixcache.py  # stage 4.3
│
└── bench/
     ├── bench_main.cpp         # M6: gi_bench
     └── kernel_bench_main.cpp  # Stage 3a: gi_kernel_bench
```