/**
 * @file sliced_k.cu
 * @brief Sliced-K GEMM optimization
 *
 * Implements Sliced-K GEMM to improve occupancy and reduce
 * synchronization overhead in blocked GEMM.
 *
 * Key idea: Split K dimension across multiple thread blocks
 * that cooperate via shared memory and atomics.
 *
 * C = alpha * A * B + beta * C
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

#define SLICE_K 4  // Number of K slices per block

/**
 * @brief Sliced-K GEMM kernel
 * Splits K dimension across multiple warps/blocks for better parallelism
 * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M, N, K Matrix dimensions
 */
__global__ void sliced_k_gemm_kernel(const float* A, const float* B, float* C,
                                      int M, int N, int K) {
    const int BLOCK_SIZE = 128;
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each block handles a tile of C
    int M_tile = BLOCK_SIZE;
    int N_tile = BLOCK_SIZE;

    int row = by * M_tile + ty;
    int col = bx * N_tile + tx;

    // Accumulator for this slice
    float acc = 0.0f;

    // Number of K tiles
    int num_k_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Sliced-K: process SLICE_K segments of K
    for (int slice = 0; slice < SLICE_K; ++slice) {
        // Each slice processes a portion of K dimension
        int k_start = (slice * K / SLICE_K);
        int k_end = ((slice + 1) * K / SLICE_K);
        int k_tile_size = k_end - k_start;

        // Load and compute for this slice
        for (int k_tile = 0; k_tile < k_tile_size; k_tile += BLOCK_SIZE) {
            int global_k = k_start + k_tile;

            // Cooperative loading of A
            int a_row = row;
            int a_col = global_k + tx;
            if (a_row < M && a_col < K) {
                As[ty][tx] = A[a_row * K + a_col];
            } else {
                As[ty][tx] = 0.0f;
            }

            // Cooperative loading of B
            int b_row = global_k + ty;
            int b_col = col;
            if (b_row < K && b_col < N) {
                Bs[ty][tx] = B[b_row * N + b_col];
            } else {
                Bs[ty][tx] = 0.0f;
            }

            __syncthreads();

            // Compute partial result
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                acc += As[ty][k] * Bs[k][tx];
            }

            __syncthreads();
        }
    }

    // Write result using atomic add for slices
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], acc);
    }
}

/**
 * @brief Simplified sliced-K with reduction at block level
 * Uses multiple thread blocks per output tile
 */
__global__ void sliced_k_simple_kernel(const float* A, const float* B, float* C,
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

    // K dimension split into slices
    int num_slices = 4;
    int slice_size = (K + num_slices - 1) / num_slices;

    for (int s = 0; s < num_slices; ++s) {
        int k_start = s * slice_size;
        int k_end = min(k_start + slice_size, K);

        // Each thread loads one element from A and B
        if (row < M && (k_start + ty) < K) {
            sA[ty][tx] = A[row * K + k_start + ty];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if (col < N && (k_start + tx) < K) {
            sB[ty][tx] = B[(k_start + tx) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < 16; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Wrapper function
void sliced_k_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    sliced_k_simple_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_KERNEL_CHECK();
}
