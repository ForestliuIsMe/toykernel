/**
 * @file w8a16_gemm.cu
 * @brief INT8 Weight × FP16 Activation GEMM (W8A16)
 *
 * Implements weight-only quantization for inference:
 * - Weights are quantized to INT8 (8-bit)
 * - Activations remain in FP16
 * - Dequantize during computation
 *
 * Formula: Y = (W_quant * scale) @ X
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../../include/utils.cuh"

#define WARP_SIZE 32
#define TILE_M 16
#define TILE_N 16

/**
 * @brief W8A16 GEMM kernel
 * @param W_quant Quantized weight matrix (INT8) [M x K]
 * @param W_scale Weight scale factor [M]
 * @param X Input activation [K x N]
 * @param Y Output [M x N]
 * @param M, N, K Matrix dimensions
 */
__global__ void w8a16_gemm_kernel(
    const int8_t* W_quant,
    const float* W_scale,
    const half* X,
    half* Y,
    int M, int N, int K
) {
    __shared__ float sX[TILE_N][TILE_K];
    __shared__ int8_t sW[TILE_M][TILE_K];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // Accumulator
    float acc[TILE_N];
    for (int i = 0; i < TILE_N; ++i) {
        acc[i] = 0.0f;
    }

    // Load scale for this row
    float scale = (row < M) ? W_scale[row] : 1.0f;

    // Iterate over K
    for (int k = 0; k < K; k += TILE_K) {
        // Load X tile to shared memory
        int x_row = k + ty;
        int x_col = col;
        if (x_row < K && x_col < N) {
            sX[ty][tx] = __half2float(X[x_row * N + x_col]);
        } else {
            sX[ty][tx] = 0.0f;
        }

        // Load W tile to shared memory
        int w_row = row;
        int w_col = k + tx;
        if (w_row < M && w_col < K) {
            sW[ty][tx] = W_quant[w_row * K + w_col];
        } else {
            sW[ty][tx] = 0;
        }

        __syncthreads();

        // Compute partial result
        for (int kk = 0; kk < TILE_K; ++kk) {
            float w_val = (float)sW[ty][kk] * scale;
            for (int n = 0; n < TILE_N; ++n) {
                acc[n] += w_val * sX[kk][n];
            }
        }

        __syncthreads();
    }

    // Write output
    if (row < M && col < N) {
        Y[row * N + col] = __float2half(acc[tx]);
    }
}

/**
 * @brief Per-channel W8A16 GEMM (scale per output channel)
 */
__global__ void w8a16_gemm_per_channel_kernel(
    const int8_t* W_quant,
    const float* W_scale,  // [M] - scale per output channel
    const half* X,
    half* Y,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Compute dot product
    for (int k = 0; k < K; ++k) {
        sum += (float)W_quant[row * K + k] * __half2float(X[k * N + col]);
    }

    // Apply per-channel scale
    sum *= W_scale[row];

    Y[row * N + col] = __float2half(sum);
}

// Wrapper function
void w8a16_gemm(
    const int8_t* W_quant,
    const float* W_scale,
    const half* X,
    half* Y,
    int M, int N, int K
) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    w8a16_gemm_kernel<<<grid, block>>>(W_quant, W_scale, X, Y, M, N, K);
    CUDA_KERNEL_CHECK();
}
