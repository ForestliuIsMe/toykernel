/**
 * @file flash_decoding.cu
 * @brief FlashDecoding implementation for LLM inference
 *
 * FlashDecoding optimizes the decoding phase of autoregressive LLM inference:
 * - Split KV cache into smaller chunks for parallel processing
 * - Uses max-scoring token to identify relevant chunks
 * - Achieves near-linear speedup with longer sequences
 *
 * O = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

#define DECODING_BLOCK_SIZE 8
#define HEAD_DIM 64

/**
 * @brief FlashDecoding: Compute partial QK^T for all KV chunks
 * Each thread block processes one query token against multiple KV chunks
 * @param Q Query [num_heads, 1, head_dim] (single token)
 * @param K Key cache [num_heads, num_kv_chunks, chunk_size, head_dim]
 * @param softmax_scale Scaling factor
 * @param partial_max Partial max scores [num_heads, num_kv_chunks]
 * @param partial_sum Partial sum of exp [num_heads, num_kv_chunks]
 * @param num_heads Number of heads
 * @param num_kv_chunks Number of KV cache chunks
 * @param chunk_size Size of each chunk
 */
__global__ void flash_decoding_partial_kernel(
    const float* Q, const float* K, float softmax_scale,
    float* partial_max, float* partial_sum,
    int num_heads, int num_kv_chunks, int chunk_size
) {
    int head_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;

    if (head_id >= num_heads || chunk_id >= num_kv_chunks) return;

    // Each thread computes partial QK for a subset of head_dim
    float partial_qk = 0.0f;
    for (int d = tid; d < HEAD_DIM; d += DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE) {
        partial_qk += Q[head_id * HEAD_DIM + d] * K[chunk_id * chunk_size * HEAD_DIM + d];
    }

    // Warp-level reduction for QK value
    for (int offset = DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        partial_qk += __shfl_down_sync(WARP_MASK, partial_qk, offset);
    }

    // Thread 0 writes the QK value
    if (tid == 0) {
        float qk = partial_qk * softmax_scale;
        float exp_qk = expf(qk);

        // Store partial results (will be reduced in next phase)
        int idx = head_id * num_kv_chunks + chunk_id;
        partial_max[idx] = qk;
        partial_sum[idx] = exp_qk;
    }
}

/**
 * @brief FlashDecoding: Reduce partial results and compute final attention
 * @param partial_max Partial max scores
 * @param partial_sum Partial sum of exp
 * @param V Value cache
 * @param O Output
 * @param num_heads, num_kv_chunks, chunk_size, head_dim
 */
__global__ void flash_decoding_reduce_kernel(
    const float* partial_max, const float* partial_sum,
    const float* V, float* O,
    int num_heads, int num_kv_chunks, int chunk_size
) {
    int head_id = blockIdx.x;
    int tid = threadIdx.x;

    if (head_id >= num_heads) return;

    // Compute global max across all chunks
    float global_max = -INFINITY;
    for (int c = 0; c < num_kv_chunks; ++c) {
        int idx = head_id * num_kv_chunks + c;
        global_max = fmaxf(global_max, partial_max[idx]);
    }

    // Compute global sum
    float global_sum = 0.0f;
    for (int c = 0; c < num_kv_chunks; ++c) {
        int idx = head_id * num_kv_chunks + c;
        global_sum += partial_sum[idx] * expf(partial_max[idx] - global_max);
    }

    // Compute weighted sum of V
    float output_val = 0.0f;
    for (int c = 0; c < num_kv_chunks; ++c) {
        int idx = head_id * num_kv_chunks + c;
        float weight = partial_sum[idx] * expf(partial_max[idx] - global_max) / global_sum;

        // Load V for this chunk
        for (int i = tid; i < chunk_size; i += DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE) {
            int v_idx = c * chunk_size * HEAD_DIM + i * HEAD_DIM + (tid % HEAD_DIM);
            output_val += weight * V[v_idx];
        }
    }

    // Warp reduction
    for (int offset = DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        output_val += __shfl_down_sync(WARP_MASK, output_val, offset);
    }

    if (tid == 0) {
        O[head_id * HEAD_DIM] = output_val;
    }
}

/**
 * @brief FlashDecoding wrapper
 */
void flash_decoding(const float* Q, const float* K, const float* V, float* O,
                    float softmax_scale, int num_heads, int num_kv_chunks, int chunk_size) {
    // Allocate partial results
    float *partial_max, *partial_sum;
    size_t partial_size = num_heads * num_kv_chunks * sizeof(float);
    CUDA_CHECK(cudaMalloc(&partial_max, partial_size));
    CUDA_CHECK(cudaMalloc(&partial_sum, partial_size));

    // Phase 1: Compute partial QK^T
    dim3 block1(DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE);
    dim3 grid1(num_heads, num_kv_chunks);
    flash_decoding_partial_kernel<<<grid1, block1>>>(
        Q, K, softmax_scale, partial_max, partial_sum, num_heads, num_kv_chunks, chunk_size
    );
    CUDA_KERNEL_CHECK();

    // Phase 2: Reduce and compute final attention
    dim3 block2(DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE);
    dim3 grid2(num_heads);
    flash_decoding_reduce_kernel<<<grid2, block2>>>(
        partial_max, partial_sum, V, O, num_heads, num_kv_chunks, chunk_size
    );
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaFree(partial_max));
    CUDA_CHECK(cudaFree(partial_sum));
}
