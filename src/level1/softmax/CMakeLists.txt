# Softmax Library

cuda_add_library(level1_softmax
    softmax.cu
    softmax_warp.cu
)

target_include_directories(level1_softmax PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
