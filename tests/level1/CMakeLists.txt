# Level 1 Tests

cuda_add_executable(level1_test
    test_vector_ops.cu
    test_gemv.cu
    test_softmax.cu
    test_norm.cu
    test_activation.cu
)

target_link_libraries(level1_test
    level1_vector_ops
    level1_gemv
    level1_softmax
    level1_norm
    level1_activation
)
