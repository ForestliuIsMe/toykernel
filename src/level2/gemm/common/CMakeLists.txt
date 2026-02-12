# GEMM Common Utilities

cuda_add_library(gemm_common
    gemm_common.cuh
    tile_config.cuh
)

target_include_directories(gemm_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
