#!/bin/bash
# Test script for ToyKernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Running ToyKernel Tests ==="

if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: build directory not found. Run scripts/build.sh first."
    exit 1
fi

cd "${BUILD_DIR}"

# Run all tests
if [ "$1" == "level1" ]; then
    echo "Running Level 1 tests..."
    ./tests/level1_test
elif [ "$1" == "level2" ]; then
    echo "Running Level 2 tests..."
    ./tests/level2_test
elif [ "$1" == "level3" ]; then
    echo "Running Level 3 tests..."
    ./tests/level3_test
elif [ "$1" == "level4" ]; then
    echo "Running Level 4 tests..."
    ./tests/level4_test
else
    echo "Running all tests..."
    ./tests/unittest
fi

echo "=== Tests Complete ==="
