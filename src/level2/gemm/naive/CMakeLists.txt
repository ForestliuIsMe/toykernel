# Naive GEMM

cuda_add_library(gemm_naive
    row_major.cu
    col_major.cu
)

target_include_directories(gemm_naive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
