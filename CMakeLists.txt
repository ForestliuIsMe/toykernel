cmake_minimum_required(VERSION 3.18)
project(ToyKernel CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(include)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Print configuration
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
