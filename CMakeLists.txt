cmake_minimum_required(VERSION 3.16)
project(GeneInfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For single-configuration generators (Makefile/Ninja), 
# the default is Release; for MSVC multi-configuration, use `--config Release`.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(geneinfer STATIC
  src/loader/mapped_file.cpp
  src/loader/checkpoint.cpp
  src/loader/hf_checkpoint.cpp
  src/loader/dtype.cpp
  src/loader/safetensors.cpp
  src/util/json.cpp
  src/tokenizer/tokenizer.cpp
  src/kernels/kernels.cpp
  src/model/transformer.cpp
  src/sampler/sampler.cpp
  src/runtime/generator.cpp
  src/util/process_memory.cpp
)
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

add_executable(gi_json_check tools/json_check_main.cpp)
target_link_libraries(gi_json_check PRIVATE geneinfer)

add_executable(gi_dtype_check tools/dtype_check_main.cpp)
target_link_libraries(gi_dtype_check PRIVATE geneinfer)

foreach(tgt geneinfer gi_inspect gi_tokenize gi_kernel_check gi_forward gi_generate gi_sampler_check gi_bench gi_json_check gi_dtype_check)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W4 /utf-8)
    target_compile_definitions(${tgt} PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()