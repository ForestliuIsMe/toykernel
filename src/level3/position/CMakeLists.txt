# Position Encoding Library

cuda_add_library(level3_position
    rope.cu
    rope_indexed.cu
    alibi.cu
)

target_include_directories(level3_position PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
