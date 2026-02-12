# Decoding Library

cuda_add_library(level3_decoding
    medusa.cu
    speculative_draft.cu
)

target_include_directories(level3_decoding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
