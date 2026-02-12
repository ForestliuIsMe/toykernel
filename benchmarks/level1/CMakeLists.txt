# Level 1 Benchmarks

cuda_add_executable(bench_level1
    bench_vector_ops.cu
    bench_softmax.cu
)

target_link_libraries(bench_level1
    level1_vector_ops
    level1_softmax
)
