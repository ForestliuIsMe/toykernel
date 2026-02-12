# Vector Operations Library

cuda_add_library(level1_vector_ops
    vector_add.cu
    vector_mul.cu
    vector_scale.cu
)

target_include_directories(level1_vector_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
