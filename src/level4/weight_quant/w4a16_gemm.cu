/**
 * @file w4a16_gemm.cu
 * @brief INT4 Weight × FP16 Activation GEMM (W4A16)
 *
 * Implements 4-bit weight quantization for higher compression:
 * - Weights are quantized to INT4 (packed in bytes)
 * - Activations remain in FP16
 * - Higher compression ratio than W8A16
 *
 * Uses group-wise quantization for better accuracy.
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../../include/utils.cuh"

#define TILE_M 16
#define TILE_N 16
#define TILE_K 64
#define GROUP_SIZE 128

/**
 * @brief W4A16 GEMM kernel
 * @param W_quant Quantized weight matrix (INT4 packed) [M x (K/2)]
 * @param W_scale Weight scale factor [M x (K/GROUP_SIZE)]
 * @param W_zp Zero point [M x (K/GROUP_SIZE)]
 * @param X Input activation [K x N]
 * @param Y Output [M x N]
 * @param M, N, K Matrix dimensions
 */
__global__ void w4a16_gemm_kernel(
    const uint8_t* W_quant,  // Packed INT4
    const float* W_scale,
    const float* W_zp,
    const half* X,
    half* Y,
    int M, int N, int K
) {
    __shared__ float sX[TILE_N][TILE_K];
    __shared__ uint8_t sW[TILE_M][TILE_K / 2];  // Half because 4-bit packed

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

    // Iterate over K
    for (int k = 0; k < K; k += TILE_K) {
        // Load X tile
        for (int i = 0; i < 2; ++i) {  // Each thread loads 2 elements
            int x_row = k + ty * 2 + i;
            int x_col = col;
            if (x_row < K && x_col < N) {
                sX[ty * 2 + i][tx] = __half2float(X[x_row * N + x_col]);
            } else {
                sX[ty * 2 + i][tx] = 0.0f;
            }
        }

        // Load W tile (packed INT4)
        int w_row = row;
        int w_col = (k + tx) / 2;
        if (w_row < M && w_col < K / 2) {
            sW[ty][tx] = W_quant[w_row * (K / 2) + w_col];
        } else {
            sW[ty][tx] = 0;
        }

        __syncthreads();

        // Unpack and compute
        for (int kk = 0; kk < TILE_K; kk += 2) {
            // Unpack two INT4 values from one byte
            uint8_t packed = sW[ty][kk / 2];
            int4 w0 = (packed & 0x0F);       // Lower 4 bits
            int4 w1 = (packed >> 4);          // Upper 4 bits

            // Get scale and zero point for this group
            int group0 = (k + kk) / GROUP_SIZE;
            int group1 = (k + kk + 1) / GROUP_SIZE;
            float scale0 = W_scale[row * (K / GROUP_SIZE) + group0];
            float scale1 = W_scale[row * (K / GROUP_SIZE) + group1];
            float zp0 = W_zp[row * (K / GROUP_SIZE) + group0];
            float zp1 = W_zp[row * (K / GROUP_SIZE) + group1];

            // Dequantize and accumulate
            float w0_f = ((float)w0 - zp0) * scale0;
            float w1_f = ((float)w1 - zp1) * scale1;

            for (int n = 0; n < TILE_N; ++n) {
                acc[n] += w0_f * sX[kk][n];
                acc[n] += w1_f * sX[kk + 1][n];
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
 * @brief Simple W4A16 GEMM (without zero point)
 */
__global__ void w4a16_gemm_simple_kernel(
    const uint8_t* W_quant,
    const float* W_scale,
    const half* X,
    half* Y,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Each thread computes one output element
    for (int k = 0; k < K; k += 2) {
        // Unpack INT4
        uint8_t packed = W_quant[row * (K / 2) + k / 2];
        int4 w0 = (packed & 0x0F);
        int4 w1 = (packed >> 4);

        // Get scale
        int group = k / GROUP_SIZE;
        float scale = W_scale[row * (K / GROUP_SIZE) + group];

        sum += (float)w0 * scale * __half2float(X[k * N + col]);
        if (k + 1 < K) {
            sum += (float)w1 * scale * __half2float(X[(k + 1) * N + col]);
        }
    }

    Y[row * N + col] = __float2half(sum);
}

// Wrapper function
void w4a16_gemm(
    const uint8_t* W_quant,
    const float* W_scale,
    const float* W_zp,
    const half* X,
    half* Y,
    int M, int N, int K
) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    w4a16_gemm_kernel<<<grid, block>>>(W_quant, W_scale, W_zp, X, Y, M, N, K);
    CUDA_KERNEL_CHECK();
}
