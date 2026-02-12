# Level 3 Benchmarks

cuda_add_executable(bench_flash_attention
    bench_flash_attention.cu
)

target_link_libraries(bench_flash_attention
    level3_attention
)
