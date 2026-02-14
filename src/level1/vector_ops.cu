/**
 * @file vector_ops.cu
 * @brief Basic vector operations: Add, Mul, Scale
 *
 * This file implements fundamental vector operations on GPU:
 * - Vector Add: C = A + B (element-wise)
 * - Vector Mul: C = A * B (element-wise)
 * - Vector Scale: B = alpha * A
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

/**
 * @brief Vector addition: C = A + B
 * @param n Number of elements
 * @param A Input vector A
 * @param B Input vector B
 * @param C Output vector C
 */
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * @brief Vector multiplication: C = A * B
 * @param n Number of elements
 * @param A Input vector A
 * @param B Input vector B
 * @param C Output vector C
 */
__global__ void vector_mul_kernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

/**
 * @brief Vector scale: B = alpha * A
 * @param n Number of elements
 * @param alpha Scale factor
 * @param A Input vector A
 * @param B Output vector B
 */
__global__ void vector_scale_kernel(const float* A, float* B, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = alpha * A[idx];
    }
}

// Wrapper functions for host code
void vector_add(const float* A, const float* B, float* C, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(A, B, C, n);
    CUDA_KERNEL_CHECK();
}

void vector_mul(const float* A, const float* B, float* C, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_mul_kernel<<<grid_size, block_size>>>(A, B, C, n);
    CUDA_KERNEL_CHECK();
}

void vector_scale(const float* A, float* B, float alpha, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_scale_kernel<<<grid_size, block_size>>>(A, B, alpha, n);
    CUDA_KERNEL_CHECK();
}
