# Level 2 Tests

cuda_add_executable(level2_test
    test_gemm_naive.cu
    test_gemm_sliced_k.cu
    test_gemm_split_k.cu
    test_gemm_persistent.cu
)

target_link_libraries(level2_test
    level2_gemm
    gemm_naive
    gemm_sliced_k
    gemm_split_k
    gemm_persistent
)
