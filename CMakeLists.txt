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
)
target_include_directories(geneinfer PUBLIC src)

add_executable(gi_inspect tools/inspect_main.cpp)
target_link_libraries(gi_inspect PRIVATE geneinfer)

foreach(tgt geneinfer gi_inspect)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W4 /utf-8)
    target_compile_definitions(${tgt} PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()
