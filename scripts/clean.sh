#!/bin/bash
# Clean script for ToyKernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Cleaning ToyKernel ==="

# Remove build directory
if [ -d "${SCRIPT_DIR}/build" ]; then
    rm -rf "${SCRIPT_DIR}/build"
    echo "Removed: build/"
else
    echo "No build/ directory found."
fi

# Remove generated files
rm -f "${SCRIPT_DIR}/CMakeCache.txt"
rm -rf "${SCRIPT_DIR}/CMakeFiles"
rm -f "${SCRIPT_DIR}/cmake_install.cmake"
rm -f "${SCRIPT_DIR}/Makefile"

echo "=== Clean Complete ==="
