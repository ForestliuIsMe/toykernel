# Persistent GEMM

cuda_add_library(gemm_persistent
    basic.cu
    stream.cu
    tensor_core.cu
)

target_include_directories(gemm_persistent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
