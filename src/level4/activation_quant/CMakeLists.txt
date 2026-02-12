# Activation Quantization Library

cuda_add_library(level4_activation_quant
    smooth_quant.cu
    awq_quant.cu
    static_quant.cu
    dynamic_quant.cu
)

target_include_directories(level4_activation_quant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
