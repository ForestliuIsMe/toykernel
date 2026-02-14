/**
 * @file quantized_gemm.cu
 * @brief Quantized GEMM operations
 *
 * Implements GEMM with fully quantized computation:
 * - INT8 × INT8 → INT32 accumulation
 * - Requires separate kernel for int8 computation
 * - Output is dequantized to FP16/FP32
 *
 * Used in scenarios where both weights and activations are quantized.
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../../include/utils.cuh"

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

/**
 * @brief INT8 GEMM kernel
 * Both A and B are INT8, output accumulated in INT32 then dequantized
 * @param A Input A (INT8) [M x K]
 * @param B Input B (INT8) [K x N]
 * @param C Scale for A
 * @param D Scale for B
 * @param Y Output (FP16) [M x N]
 * @param M, N, K Matrix dimensions
 */
__global__ void quantized_gemm_int8_kernel(
    const int8_t* A,
    const int8_t* B,
    float scale_a,
    float scale_b,
    half* Y,
    int M, int N, int K
) {
    __shared__ int8_t sA[TILE_M][TILE_K];
    __shared__ int8_t sB[TILE_K][TILE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // INT32 accumulator
    int acc[TILE_N];
    for (int i = 0; i < TILE_N; ++i) {
        acc[i] = 0;
    }

    // Iterate over K
    for (int k = 0; k < K; k += TILE_K) {
        // Load A tile
        if (row < M && (k + tx) < K) {
            sA[ty][tx] = A[row * K + k + tx];
        } else {
            sA[ty][tx] = 0;
        }

        // Load B tile
        if (col < N && (k + ty) < K) {
            sB[ty][tx] = B[(k + ty) * N + col];
        } else {
            sB[ty][tx] = 0;
        }

        __syncthreads();

        // Compute INT8 × INT8 → INT32
        for (int kk = 0; kk < TILE_K; ++kk) {
            int8_t a_val = sA[ty][kk];
            for (int n = 0; n < TILE_N; ++n) {
                acc[n] += (int)a_val * (int)sB[kk][n];
            }
        }

        __syncthreads();
    }

    // Dequantize and write output
    float dequant_scale = scale_a * scale_b;
    if (row < M && col < N) {
        float result = (float)acc[tx] * dequant_scale;
        Y[row * N + col] = __float2half(result);
    }
}

/**
 * @brief Per-tensor quantized GEMM
 */
__global__ void quantized_gemm_per_tensor_kernel(
    const int8_t* A,
    const int8_t* B,
    float scale_a,
    float scale_b,
    half* Y,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int sum = 0;

    // Each thread computes one output element
    for (int k = 0; k < K; ++k) {
        sum += (int)A[row * K + k] * (int)B[k * N + col];
    }

    float dequant = (float)sum * scale_a * scale_b;
    Y[row * N + col] = __float2half(dequant);
}

/**
 * @brief INT8 × INT8 with per-channel dequantization for weights
 */
__global__ void quantized_gemm_mixed_kernel(
    const int8_t* A,
    const int8_t* B,          // [K x N], per-channel scale
    const float* scale_w,     // [N] per-channel scale for weights
    float scale_a,
    half* Y,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += (int)A[row * K + k] * (int)B[k * N + col];
    }

    // Per-channel dequantization
    float dequant = (float)sum * scale_a * scale_w[col];
    Y[row * N + col] = __float2half(dequant);
}

// Wrapper functions
void quantized_gemm_int8(
    const int8_t* A,
    const int8_t* B,
    float scale_a,
    float scale_b,
    half* Y,
    int M, int N, int K
) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    quantized_gemm_int8_kernel<<<grid, block>>>(A, B, scale_a, scale_b, Y, M, N, K);
    CUDA_KERNEL_CHECK();
}

void quantized_gemm(
    const int8_t* A,
    const int8_t* B,
    float scale_a,
    float scale_b,
    half* Y,
    int M, int N, int K
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    quantized_gemm_per_tensor_kernel<<<grid, block>>>(A, B, scale_a, scale_b, Y, M, N, K);
    CUDA_KERNEL_CHECK();
}

void quantized_gemm_mixed(
    const int8_t* A,
    const int8_t* B,
    const float* scale_w,
    float scale_a,
    half* Y,
    int M, int N, int K
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    quantized_gemm_mixed_kernel<<<grid, block>>>(A, B, scale_w, scale_a, Y, M, N, K);
    CUDA_KERNEL_CHECK();
}
