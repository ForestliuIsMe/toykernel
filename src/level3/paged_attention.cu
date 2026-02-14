/**
 * @file paged_attention.cu
 * @brief PagedAttention implementation (vLLM-style)
 *
 * PagedAttention manages KV cache in non-contiguous pages:
 * - Reduces memory fragmentation from speculative decoding
 * - Enables sharing of KV cache across sequences
 * - Uses block tables for flexible memory management
 *
 * Based on vLLM's PagedAttention:
 * https://github.com/vllm-project/vllm
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

#define PAGE_SIZE 16
#define HEAD_DIM 64
#define BLOCK_SIZE 16

/**
 * @brief PagedAttention kernel
 * @param query Query tensor [num_seqs, num_heads, head_dim]
 * @param key_cache KV cache keys [num_blocks, page_size, num_heads, head_dim]
 * @param value_cache KV cache values [num_blocks, page_size, num_heads, head_dim]
 * @param block_tables Block table mapping [num_seqs, max_num_blocks]
 * @param context_lengths Length of context for each sequence
 * @param output Attention output [num_seqs, num_heads, head_dim]
 * @param num_seqs Number of sequences
 * @param num_heads Number of heads
 * @param max_context_len Maximum context length
 */
__global__ void paged_attention_kernel(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_tables,
    const int* context_lengths,
    float* output,
    int num_seqs, int num_heads, int max_context_len
) {
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;

    if (seq_id >= num_seqs) return;

    int context_len = context_lengths[seq_id];
    int num_blocks = (context_len + PAGE_SIZE - 1) / PAGE_SIZE;

    // Get block table for this sequence
    const int* seq_block_table = block_tables + seq_id * 64;  // max 64 blocks

    // Softmax scale
    float softmax_scale = 1.0f / sqrtf((float)HEAD_DIM);

    // Online softmax variables
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float row_output[HEAD_DIM];

    for (int d = 0; d < HEAD_DIM; ++d) {
        row_output[d] = 0.0f;
    }

    // Iterate over pages
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int physical_block = seq_block_table[block_idx];
        int block_offset = (block_idx == num_blocks - 1) ?
                           (context_len % PAGE_SIZE) : PAGE_SIZE;

        // Load key for this block
        float k[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; d += BLOCK_SIZE) {
            int local_d = tid + d;
            if (local_d < HEAD_DIM) {
                int k_idx = physical_block * PAGE_SIZE * num_heads * HEAD_DIM +
                           block_offset * num_heads * HEAD_DIM +
                           head_id * HEAD_DIM + local_d;
                k[local_d] = key_cache[k_idx];
            }
        }

        // Compute QK
        float qk = 0.0f;
        int q_offset = seq_id * num_heads * HEAD_DIM + head_id * HEAD_DIM + tid;
        for (int d = 0; d < HEAD_DIM; ++d) {
            qk += query[q_offset + d] * k[d];
        }
        qk *= softmax_scale;

        // Online softmax update
        float new_max = fmaxf(row_max, qk);
        float exp_qk = expf(qk - new_max);
        float rescale = expf(row_max - new_max);

        // Update output accumulator
        if (row_max != -INFINITY) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                row_output[d] *= rescale;
            }
            row_sum *= rescale;
        }

        // Load value and accumulate
        for (int v_idx = 0; v_idx < block_offset; ++v_idx) {
            float v[HEAD_DIM];
            int v_base = physical_block * PAGE_SIZE * num_heads * HEAD_DIM +
                        v_idx * num_heads * HEAD_DIM +
                        head_id * HEAD_DIM;

            for (int d = 0; d < HEAD_DIM; ++d) {
                v[d] = value_cache[v_base + d];
            }

            // Compute attention weight (simplified)
            // In practice, would compute full QK^T matrix
            float attn_weight = exp_qk / (float)block_offset;

            for (int d = 0; d < HEAD_DIM; ++d) {
                row_output[d] += attn_weight * v[d];
            }
        }

        row_max = new_max;
        row_sum += exp_qk;
    }

    // Normalize
    float row_sum_inv = 1.0f / (row_sum + 1e-6f);
    for (int d = 0; d < HEAD_DIM; ++d) {
        row_output[d] *= row_sum_inv;
    }

    // Write output
    int out_offset = seq_id * num_heads * HEAD_DIM + head_id * HEAD_DIM + tid;
    output[out_offset] = row_output[tid];
}

// Simplified wrapper
void paged_attention(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_tables,
    const int* context_lengths,
    float* output,
    int num_seqs, int num_heads, int max_context_len
) {
    dim3 block(HEAD_DIM);
    dim3 grid(num_seqs, num_heads);
    paged_attention_kernel<<<grid, block>>>(
        query, key_cache, value_cache, block_tables,
        context_lengths, output, num_seqs, num_heads, max_context_len
    );
    CUDA_KERNEL_CHECK();
}
