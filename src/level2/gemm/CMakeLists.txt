# GEMM Library

add_subdirectory(naive)
add_subdirectory(sliced_k)
add_subdirectory(split_k)
add_subdirectory(persistent)
add_subdirectory(common)

# Main GEMM library combining all implementations
cuda_add_library(level2_gemm
    gemm.cu
)

target_include_directories(level2_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
