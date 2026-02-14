#!/bin/bash
# Build script for ToyKernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Building ToyKernel ==="
echo "Build directory: ${BUILD_DIR}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON

# Build
make -j$(nproc)

echo "=== Build Complete ==="
echo "Executables: ${BUILD_DIR}/tests/"
