# Quantized Operations Library

cuda_add_library(level4_quantized_ops
    w8a16_gemm.cu
    w4a16_gemm.cu
    w4a4_gemm.cu
    dequantize.cu
    quantize.cu
)

target_include_directories(level4_quantized_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
