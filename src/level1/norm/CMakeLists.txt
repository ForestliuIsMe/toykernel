# Normalization Library

cuda_add_library(level1_norm
    layernorm.cu
    rmsnorm.cu
)

target_include_directories(level1_norm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
