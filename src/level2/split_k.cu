/**
 * @file split_k.cu
 * @brief Split-K GEMM optimization
 *
 * Implements Split-K GEMM where K dimension is split across
 * multiple thread blocks that compute partial results and
 * reduce via a final kernel.
 *
 * This approach is useful when K is large and provides:
 * - Better parallelization across K dimension
 * - Reduced shared memory pressure per block
 * - Better utilization of GPU resources
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

/**
 * @brief Split-K Phase 1: Compute partial results
 * Each block computes a partial GEMM for a slice of K
 * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param partial_C Partial results (num_slices * M x N)
 * @param M, N, K Matrix dimensions
 * @param num_slices Number of K splits
 */
__global__ void split_k_phase1_kernel(const float* A, const float* B, float* partial_C,
                                       int M, int N, int K, int num_slices) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int slice_id = bx / ((N + 15) / 16);  // Which slice this block handles
    int col_block = bx % ((N + 15) / 16);

    int k_start = (slice_id * K) / num_slices;
    int k_end = ((slice_id + 1) * K) / num_slices;
    int k_size = k_end - k_start;

    int row = by * 16 + ty;
    int col = col_block * 16 + tx;

    float sum = 0.0f;

    // Process this slice of K
    for (int k = 0; k < k_size; k += 16) {
        // Load A tile
        if (row < M && (k_start + k + tx) < K) {
            sA[ty][tx] = A[row * K + k_start + k + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B tile
        if (col < N && (k_start + k + ty) < K) {
            sB[ty][tx] = B[(k_start + k + ty) * N + col];
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

    // Write partial result
    if (row < M && col < N) {
        int partial_row = slice_id * M + row;
        partial_C[partial_row * N + col] = sum;
    }
}

/**
 * @brief Split-K Phase 2: Reduce partial results
 * Reduces partial results from all slices
 * @param partial_C Partial results from Phase 1
 * @param C Final output matrix C (M x N)
 * @param M, N Matrix dimensions
 * @param num_slices Number of slices to reduce
 */
__global__ void split_k_phase2_kernel(const float* partial_C, float* C,
                                       int M, int N, int num_slices) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int s = 0; s < num_slices; ++s) {
        sum += partial_C[s * M * N + row * N + col];
    }

    C[row * N + col] = sum;
}

/**
 * @brief Combined Split-K GEMM wrapper
 * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M, N, K Matrix dimensions
 */
void split_k_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int num_slices = 4;  // Split K into 4 parts

    // Allocate memory for partial results
    float* partial_C;
    size_t partial_size = M * N * num_slices * sizeof(float);
    CUDA_CHECK(cudaMalloc(&partial_C, partial_size));

    // Phase 1: Compute partial results
    dim3 block1(16, 16);
    int n_col_blocks = (N + 15) / 16;
    int n_row_blocks = (M + 15) / 16;
    dim3 grid1(n_col_blocks * num_slices, n_row_blocks);

    split_k_phase1_kernel<<<grid1, block1>>>(A, B, partial_C, M, N, K, num_slices);
    CUDA_KERNEL_CHECK();
    cudaDeviceSynchronize();

    // Phase 2: Reduce partial results
    dim3 block2(16, 16);
    dim3 grid2((N + 15) / 16, (M + 15) / 16);

    split_k_phase2_kernel<<<grid2, block2>>>(partial_C, C, M, N, num_slices);
    CUDA_KERNEL_CHECK();
    cudaDeviceSynchronize();

    // Cleanup
    CUDA_CHECK(cudaFree(partial_C));
}
