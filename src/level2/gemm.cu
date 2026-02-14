/**
 * @file gemm.cu
 * @brief Naive General Matrix Multiplication (GEMM)
 *
 * Implements C = alpha * A * B + beta * C where:
 * - A is an M x K matrix
 * - B is a K x N matrix
 * - C is an M x N matrix
 *
 * This is the foundational implementation for understanding
 * more optimized GEMM algorithms.
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

/**
 * @brief Naive GEMM kernel
 * Each thread computes one element of C
 * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A and rows of B
 * @param alpha Scalar multiplier
 * @param beta Scalar for existing C
 */
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

/**
 * @brief Shared memory tiled GEMM
 * Uses shared memory for block-level caching
 * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A and rows of B
 * @param alpha Scalar multiplier
 * @param beta Scalar for existing C
 */
__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K, float alpha, float beta) {
    const int BLOCK_SIZE = 16;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Loop over K tiles
    for (int m = 0; m < K; m += BLOCK_SIZE) {
        // Load A into shared memory
        if (row < M && (m + tx) < K) {
            As[ty][tx] = A[row * K + m + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B into shared memory
        if (col < N && (m + ty) < K) {
            Bs[ty][tx] = B[(m + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Wrapper functions
void gemm_naive(const float* A, const float* B, float* C,
                int M, int N, int K, float alpha = 1.0f, float beta = 0.0f) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_KERNEL_CHECK();
}

void gemm_tiled(const float* A, const float* B, float* C,
                int M, int N, int K, float alpha = 1.0f, float beta = 0.0f) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_KERNEL_CHECK();
}
