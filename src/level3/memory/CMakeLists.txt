# Memory Library

cuda_add_library(level3_memory
    paged_attention.cu
    kv_quant.cu
    h2o_eviction.cu
)

target_include_directories(level3_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
