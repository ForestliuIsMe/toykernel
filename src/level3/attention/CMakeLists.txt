# Attention Library

cuda_add_library(level3_attention
    flash_attention_fwd.cu
    flash_attention_bwd.cu
    flash_decoding.cu
    flash_decoding_bwd.cu
)

target_include_directories(level3_attention PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
