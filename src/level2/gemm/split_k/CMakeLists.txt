# Split-K GEMM

cuda_add_library(gemm_split_k
    basic.cu
    reduce.cu
    async.cu
)

target_include_directories(gemm_split_k PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
