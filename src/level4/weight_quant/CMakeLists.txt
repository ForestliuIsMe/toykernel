# Weight Quantization Library

cuda_add_library(level4_weight_quant
    w8a16_quant.cu
    w4a16_quant.cu
    w4a4_quant.cu
    gptq.cu
)

target_include_directories(level4_weight_quant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
