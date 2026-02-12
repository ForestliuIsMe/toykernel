# Sliced-K GEMM

cuda_add_library(gemm_sliced_k
    basic.cu
    warp.cu
    tensor_core.cu
)

target_include_directories(gemm_sliced_k PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
