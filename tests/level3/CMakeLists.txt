# Level 3 Tests

cuda_add_executable(level3_test
    test_flash_attention.cu
    test_rope.cu
    test_paged_attention.cu
)

target_link_libraries(level3_test
    level3_attention
    level3_position
    level3_memory
)
