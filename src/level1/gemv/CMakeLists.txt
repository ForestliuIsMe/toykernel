# GEMV Library

cuda_add_library(level1_gemv
    gemv.cu
)

target_include_directories(level1_gemv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
