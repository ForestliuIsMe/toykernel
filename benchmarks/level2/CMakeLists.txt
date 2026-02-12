# Level 2 Benchmarks

cuda_add_executable(bench_gemm
    bench_gemm.cu
)

target_link_libraries(bench_gemm
    level2_gemm
    gemm_naive
    gemm_sliced_k
    gemm_split_k
    gemm_persistent
)
