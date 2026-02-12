# Quantization Common Utilities

cuda_add_library(level4_quant_common
    quant_common.cuh
    scales.cuh
)

target_include_directories(level4_quant_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
