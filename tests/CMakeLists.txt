# ToyKernel Tests CMakeLists.txt

# Catch2 framework (header-only, download or use system)
# If using system: find_package(Catch2 REQUIRED)
# For now, we use simple test framework

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
add_subdirectory(level4)

# Main test executable
cuda_add_executable(unittest unittest.cu)

target_link_libraries(unittest
    level1_vector_ops
    level1_gemv
    level1_softmax
    level1_norm
    level1_activation
    level2_gemm
    level3_attention
    level3_position
    level3_memory
    level3_decoding
    level4_weight_quant
    level4_quantized_ops
    level4_activation_quant
)
