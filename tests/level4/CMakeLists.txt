# Level 4 Tests

cuda_add_executable(level4_test
    test_quantization.cu
    test_quantized_gemm.cu
)

target_link_libraries(level4_test
    level4_weight_quant
    level4_quantized_ops
    level4_activation_quant
)
