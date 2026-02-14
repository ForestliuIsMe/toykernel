/**
 * @file softmax.cu
 * @brief Softmax activation function implementations
 *
 * This file implements Softmax and Warp-level Softmax:
 * - Naive Softmax: Standard row-wise softmax
 * - Warp-level Softmax: Optimized using warp shuffle for efficiency
 *
 * softmax(x_i) = exp(x_i) / sum(exp(x_j))
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

/**
 * @brief Naive row-wise softmax
 * @param input Input array
 * @param output Output array (softmax result)
 * @param n Number of rows
 * @param m Number of columns (features per row)
 */
__global__ void softmax_kernel(const float* input, float* output, int n, int m) {
    int row = blockIdx.x;
    if (row >= n) return;

    const float* x = input + row * m;
    float* y = output + row * m;

    // Find max value for numerical stability
    float max_val = x[0];
    for (int i = 1; i < m; ++i) {
        max_val = fmaxf(max_val, x[i]);
    }

    // Compute sum of exp(x_i - max)
    float sum = 0.0f;
    for (int i = 0; i < m; ++i) {
        sum += expf(x[i] - max_val);
    }

    // Compute softmax
    for (int i = 0; i < m; ++i) {
        y[i] = expf(x[i] - max_val) / sum;
    }
}

/**
 * @brief Warp-level softmax using shuffle instructions
 * @param input Input array
 * @param output Output array
 * @param n Number of rows
 * @param m Number of columns (must be multiple of 32)
 */
__global__ void softmax_warp_kernel(const float* input, float* output, int n, int m) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (row >= n) return;

    const float* x = input + row * m;
    float* y = output + row * m;

    // Process 32 elements per warp
    for (int base = 0; base < m; base += WARP_SIZE) {
        int col = base + lane_id;
        float val = (col < m) ? x[col] : -INFINITY;

        // Warp-level max reduction
        float max_val = val;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(WARP_MASK, max_val, offset));
        }
        max_val = __shfl_sync(WARP_MASK, max_val, 0);

        // Warp-level sum of exp(x - max)
        float exp_val = (col < m) ? expf(val - max_val) : 0.0f;
        float sum = exp_val;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(WARP_MASK, sum, offset);
        }
        sum = __shfl_sync(WARP_MASK, sum, 0);

        // Write result
        if (col < m) {
            y[col] = exp_val / sum;
        }
    }
}

// Wrapper functions
void softmax(const float* input, float* output, int n, int m) {
    dim3 block(256);
    dim3 grid(n);
    softmax_kernel<<<grid, block>>>(input, output, n, m);
    CUDA_KERNEL_CHECK();
}

void softmax_warp(const float* input, float* output, int n, int m) {
    dim3 block(256);
    dim3 grid(n);
    softmax_warp_kernel<<<grid, block>>>(input, output, n, m);
    CUDA_KERNEL_CHECK();
}
