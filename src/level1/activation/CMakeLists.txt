# Activation Functions Library

cuda_add_library(level1_activation
    gelu.cu
    swish.cu
)

target_include_directories(level1_activation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
