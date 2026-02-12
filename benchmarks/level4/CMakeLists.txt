# Level 4 Benchmarks

cuda_add_executable(bench_quantized_gemm
    bench_quantized_gemm.cu
)

target_link_libraries(bench_quantized_gemm
    level4_quantized_ops
)
