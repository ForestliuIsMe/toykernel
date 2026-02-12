cmake_minimum_required(VERSION 3.18)
project(ToyKernel CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# All source files
file(GLOB SOURCES "src/*.cu")

# Build executable
cuda_add_executable(toykernel ${SOURCES})

# Include headers
target_include_directories(toykernel PRIVATE include)
