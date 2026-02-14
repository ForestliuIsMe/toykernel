/**
 * @file rope.cu
 * @brief Rotary Position Embedding (RoPE) implementation
 *
 * RoPE encodes position information into query and key vectors:
 * - Applies rotation matrix based on position index
 * - Uses complex number representation implicitly via sin/cos
 * - Enables relative position modeling without explicit attention bias
 *
 * RoPE(x, pos) = x * cos(theta) + rotate(x) * sin(theta)
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

#define WARP_SIZE 32

/**
 * @brief Apply RoPE to query tensor
 * @param q Query tensor [batch, num_heads, seq_len, head_dim]
 * @param output Output tensor
 * @param position_ids Position indices [seq_len]
 * @param inv_freq Inverse frequencies for rotary embedding
 * @param seq_len Sequence length
 * @param num_heads Number of heads
 * @param head_dim Head dimension (must be even)
 */
__global__ void rope_kernel(
    const float* q, float* output,
    const int* position_ids,
    const float* inv_freq,
    int seq_len, int num_heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * num_heads * head_dim;
    if (idx >= total_elements) return;

    // Decode indices
    int tmp = idx;
    int head_dim_idx = tmp % head_dim; tmp /= head_dim;
    int head_id = tmp % num_heads; tmp /= num_heads;
    int seq_id = tmp;

    int pos = position_ids[seq_id];
    float freq = inv_freq[head_dim_idx / 2];
    float theta = pos * freq;

    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    float x1 = q[idx];  // First half of head
    float x2 = q[idx + head_dim / 2];  // Second half of head

    // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    float rot_x1 = x1 * cos_theta - x2 * sin_theta;
    float rot_x2 = x1 * sin_theta + x2 * cos_theta;

    // Write output
    output[idx] = rot_x1;
    output[idx + head_dim / 2] = rot_x2;
}

/**
 * @brief Apply RoPE using warp shuffle for efficiency
 * @param q Query tensor
 * @param output Output tensor
 * @param position_ids Position indices
 * @param inv_freq Inverse frequencies
 */
__global__ void rope_warp_kernel(
    const float* q, float* output,
    const int* position_ids,
    const float* inv_freq,
    int seq_len, int num_heads, int head_dim
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int head_dim = blockDim.x;

    // Each block handles one (head, position) pair
    int head_id = blockIdx.y;
    int seq_id = blockIdx.z;

    if (head_id >= num_heads || seq_id >= seq_len) return;

    int pos = position_ids[seq_id];
    int offset = (head_id * seq_len + seq_id) * head_dim;

    // Load query vector
    float x = q[offset + tid];

    // Compute rotation angle
    int dim_pair = tid / 2;
    float freq = inv_freq[dim_pair];
    float theta = pos * freq;

    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    // Get paired element using shuffle
    bool is_even = (tid % 2 == 0);
    float x_pair = __shfl_sync(WARP_MASK, x, tid ^ 1);

    // Apply rotation
    float rot_x;
    if (is_even) {
        rot_x = x * cos_theta - x_pair * sin_theta;
    } else {
        rot_x = x_pair * cos_theta + x * sin_theta;
    }

    // Write output
    output[offset + tid] = rot_x;
}

/**
 * @brief Compute inv_freq for RoPE
 * @param inv_freq Output array
 * @param head_dim Head dimension
 * @param base Base frequency (typically 10000)
 */
void compute_rope_freqs(float* inv_freq, int head_dim, float base = 10000.0f) {
    for (int i = 0; i < head_dim / 2; ++i) {
        inv_freq[i] = 1.0f / powf(base, (2.0f * i) / head_dim);
    }
}

// Wrapper functions
void rope(const float* q, float* output, const int* position_ids,
          const float* inv_freq, int seq_len, int num_heads, int head_dim) {
    int total_elements = seq_len * num_heads * head_dim;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    rope_kernel<<<grid_size, block_size>>>(
        q, output, position_ids, inv_freq, seq_len, num_heads, head_dim
    );
    CUDA_KERNEL_CHECK();
}

void rope_warp(const float* q, float* output, const int* position_ids,
               const float* inv_freq, int seq_len, int num_heads, int head_dim) {
    dim3 block(head_dim);
    dim3 grid(1, num_heads, seq_len);
    rope_warp_kernel<<<grid, block>>>(
        q, output, position_ids, inv_freq, seq_len, num_heads, head_dim
    );
    CUDA_KERNEL_CHECK();
}
