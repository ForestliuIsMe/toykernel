#!/bin/bash
# Benchmark script for ToyKernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Running ToyKernel Benchmarks ==="

if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: build directory not found. Run scripts/build.sh first."
    exit 1
fi

cd "${BUILD_DIR}"

# Run all benchmarks
if [ "$1" == "gemm" ]; then
    echo "Running GEMM benchmarks..."
    ./benchmarks/level2/bench_gemm
elif [ "$1" == "attention" ]; then
    echo "Running Attention benchmarks..."
    ./benchmarks/level3/bench_flash_attention
elif [ "$1" == "quant" ]; then
    echo "Running Quantized GEMM benchmarks..."
    ./benchmarks/level4/bench_quantized_gemm
else
    echo "Running all benchmarks..."
    ./benchmarks/benchmark_all
fi

echo "=== Benchmarks Complete ==="
