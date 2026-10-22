# Gene LLM Engine

Learning LLM Engine

# run
```
cmake -S . -B build
cmake --build build --config Release
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
│   └── tokenizer/
│       └── tokenizer.cpp   #
│   
├── models/
│   └── stories15M/         # TinyStories
├── tools/
│   ├── inspect_main.cpp    # gi_inspect: Print configuration and the fingerprint of each tensor
│   ├── llama2c_io.py       # Loading checkpoints/tokenizers using NumPy
│   ├── ref.py              # M0: Reference forward + greedy generation; outputs the standar answer
│   └── make_dummy.py       # Generate a small, random model for offline testing purposes
└── tests/
    └── verify_loader.py    # M1 Verification: C++ Output vs. NumPy Element-wise Comparison
```