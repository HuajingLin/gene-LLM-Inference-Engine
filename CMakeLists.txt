cmake_minimum_required(VERSION 3.16)
project(GeneInfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release for single-config generators (Makefile/Ninja).
# For MSVC multi-config builds use: cmake --build build --config Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(geneinfer STATIC
  src/loader/mapped_file.cpp
  src/loader/checkpoint.cpp
  src/loader/hf_checkpoint.cpp
  src/loader/dtype.cpp
  src/loader/quantize.cpp
  src/loader/safetensors.cpp
  src/util/json.cpp
  src/util/thread_pool.cpp
  src/tokenizer/tokenizer.cpp
  src/tokenizer/unicode.cpp
  src/tokenizer/hf_tokenizer.cpp
  src/tokenizer/text_tokenizer.cpp
  src/kernels/kernels.cpp
  src/model/transformer.cpp
  src/sampler/sampler.cpp
  src/runtime/generator.cpp
  src/util/process_memory.cpp
)
# SIMD kernels: x86-64 with GCC/Clang only (other targets run the scalar kernels). Each SIMD file is compiled
# with its own instruction-set flags; the CPU is checked at run time before one of them is used.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_sources(geneinfer PRIVATE src/kernels/kernels_avx2.cpp src/kernels/kernels_avx512.cpp)
  set_source_files_properties(src/kernels/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  set_source_files_properties(src/kernels/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512vl;-mavx2;-mfma")
  target_compile_definitions(geneinfer PUBLIC GI_HAVE_X86_SIMD=1)
endif()
find_package(Threads REQUIRED)
target_link_libraries(geneinfer PUBLIC Threads::Threads)
target_include_directories(geneinfer PUBLIC src)
if(WIN32)
  target_link_libraries(geneinfer PUBLIC psapi)
endif()

add_executable(gi_inspect tools/inspect_main.cpp)
target_link_libraries(gi_inspect PRIVATE geneinfer)

add_executable(gi_tokenize tools/tokenize_main.cpp)
target_link_libraries(gi_tokenize PRIVATE geneinfer)

add_executable(gi_kernel_check tools/kernel_check_main.cpp)
target_link_libraries(gi_kernel_check PRIVATE geneinfer)

add_executable(gi_forward tools/forward_main.cpp)
target_link_libraries(gi_forward PRIVATE geneinfer)

add_executable(gi_generate tools/generate_main.cpp)
target_link_libraries(gi_generate PRIVATE geneinfer)

add_executable(gi_sampler_check tools/sampler_check_main.cpp)
target_link_libraries(gi_sampler_check PRIVATE geneinfer)

add_executable(gi_bench bench/bench_main.cpp)
target_link_libraries(gi_bench PRIVATE geneinfer)

add_executable(gi_kernel_bench bench/kernel_bench_main.cpp)
target_link_libraries(gi_kernel_bench PRIVATE geneinfer)

add_executable(gi_pool_check tools/pool_check_main.cpp)
target_link_libraries(gi_pool_check PRIVATE geneinfer)
add_executable(gi_quant_dump tools/quant_dump_main.cpp)
target_link_libraries(gi_quant_dump PRIVATE geneinfer)

add_executable(gi_json_check tools/json_check_main.cpp)
target_link_libraries(gi_json_check PRIVATE geneinfer)

add_executable(gi_dtype_check tools/dtype_check_main.cpp)
target_link_libraries(gi_dtype_check PRIVATE geneinfer)

add_executable(gi_hf_tokenize tools/hf_tokenize_main.cpp)
target_link_libraries(gi_hf_tokenize PRIVATE geneinfer)

add_executable(gi_unicode_check tools/unicode_check_main.cpp)
target_link_libraries(gi_unicode_check PRIVATE geneinfer)

foreach(tgt geneinfer gi_inspect gi_tokenize gi_kernel_check gi_forward gi_generate gi_sampler_check gi_bench gi_kernel_bench gi_pool_check gi_quant_dump gi_json_check gi_dtype_check gi_hf_tokenize gi_unicode_check)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W4 /utf-8)
    target_compile_definitions(${tgt} PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()
