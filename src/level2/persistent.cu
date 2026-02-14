/**
 * @file persistent.cu
 * @brief Persistent Kernel GEMM
 *
 * Implements persistent thread block GEMM where thread blocks
 * stay alive to process multiple tiles, maximizing occupancy
 * and reducing kernel launch overhead.
 *
 * Key optimizations:
 * - Thread blocks persist across K iterations
 * - Use register blocking for better performance
 * - Cooperative thread arrays for memory coalescing
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

// Persistent block GEMM configuration
#define PERSISTENT_BLOCK_M 16
#define PERSISTENT_BLOCK_N 16
#define PERSISTENT_BLOCK_K 16
#define NUM_PERSISTENT_K_ITERATIONS 4

/**
 * @brief Persistent GEMM kernel
 * Thread blocks process multiple K tiles without exiting
 * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M, N, K Matrix dimensions
 */
__global__ void persistent_gemm_kernel(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    // Shared memory for tile caching
    __shared__ float sA[PERSISTENT_BLOCK_M][PERSISTENT_BLOCK_K];
    __shared__ float sB[PERSISTENT_BLOCK_K][PERSISTENT_BLOCK_N];

    // Register-based accumulator
    float regC[PERSISTENT_BLOCK_M][PERSISTENT_BLOCK_N];
    for (int i = 0; i < PERSISTENT_BLOCK_M; ++i) {
        for (int j = 0; j < PERSISTENT_BLOCK_N; ++j) {
            regC[i][j] = 0.0f;
        }
    }

    // Block index
    int block_idx = blockIdx.x;
    int block_id_y = block_idx / gridDim.y;
    int block_id_x = block_idx % gridDim.y;

    // Compute which tile this block is responsible for
    int tile_row = block_id_y * blockDim.y + threadIdx.y;
    int tile_col = block_id_x * blockDim.x + threadIdx.x;

    // Persistent K-loop
    for (int k_tile = 0; k_tile < K; k_tile += PERSISTENT_BLOCK_K * NUM_PERSISTENT_K_ITERATIONS) {
        // Process multiple K tiles while staying alive
        for (int iter = 0; iter < NUM_PERSISTENT_K_ITERATIONS; ++iter) {
            int current_k = k_tile + iter * PERSISTENT_BLOCK_K;

            // Load A tile into shared memory
            int a_row = tile_row + iter * blockDim.y * gridDim.y;
            int a_col = current_k + threadIdx.x;
            if (a_row < M && a_col < K) {
                sA[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
            } else {
                sA[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // Load B tile into shared memory
            int b_row = current_k + threadIdx.y;
            int b_col = tile_col + iter * blockDim.x * gridDim.y;
            if (b_row < K && b_col < N) {
                sB[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
            } else {
                sB[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            // Compute using register blocking
            for (int k = 0; k < PERSISTENT_BLOCK_K; ++k) {
                float a_val = sA[threadIdx.y][k];
                for (int j = 0; j < PERSISTENT_BLOCK_N; ++j) {
                    regC[threadIdx.y][j] += a_val * sB[k][j];
                }
            }

            __syncthreads();
        }

        // Commit results after completing iterations
        int final_row = tile_row + iter * blockDim.y * gridDim.y;
        int final_col = tile_col;
    }

    // Write final result
    int c_row = block_id_y * PERSISTENT_BLOCK_M + threadIdx.y;
    int c_col = block_id_x * PERSISTENT_BLOCK_N + threadIdx.x;

    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = regC[threadIdx.y][threadIdx.x];
    }
}

/**
 * @brief Simplified persistent GEMM with single K iteration
 * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M, N, K Matrix dimensions
 */
__global__ void persistent_gemm_simple_kernel(const float* A, const float* B, float* C,
                                                int M, int N, int K) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float sum = 0.0f;

    // Persistent K loop - stay alive for all K tiles
    for (int k = 0; k < K; k += 16) {
        // Load A
        if (row < M && (k + tx) < K) {
            sA[ty][tx] = A[row * K + k + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B
        if (col < N && (k + ty) < K) {
            sB[ty][tx] = B[(k + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute
        for (int i = 0; i < 16; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Wrapper function
void persistent_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    persistent_gemm_simple_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_KERNEL_CHECK();
}
