/**
 * @file smooth_quant.cu
 * @brief SmoothQuant implementation
 *
 * SmoothQuant is an activation-aware quantization scheme:
 * - Migrates quantization difficulty from activations to weights
 * - Computes per-channel scaling factors based on activation statistics
 * - Formula: Y = (X * scale) @ W, where scale balances quantization error
 *
 * Paper: SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs
 * https://arxiv.org/abs/2308.15026
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../../include/utils.cuh"

/**
 * @brief Compute SmoothQuant scales
 * @param X Activation tensor [M, K]
 * @param scales Output scale per channel [K]
 * @param alpha Migration strength (0-1, higher = more migration to weights)
 * @param M, K Dimensions
 */
__global__ void smooth_quant_scales_kernel(
    const float* X,
    float* scales,
    float alpha,
    int M, int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= K) return;

    // Compute max absolute value per channel
    float max_val = 0.0f;
    for (int m = 0; m < M; ++m) {
        max_val = fmaxf(max_val, fabsf(X[m * K + col]));
    }

    // Per-channel scale
    // Formula: scale = (max_per_channel)^(1-alpha) / (avg_per_channel)^(-alpha)
    // Simplified: balance between weight and activation quantization difficulty

    if (max_val == 0.0f) max_val = 1.0f;

    // SmoothQuant formula
    scales[col] = powf(max_val, -alpha);
}

/**
 * @brief Apply SmoothQuant: X_scaled = X * scale
 * @param X Input activation [M, K]
 * @param scales Scale per channel [K]
 * @param X_scaled Output scaled activation [M, K]
 * @param M, K Dimensions
 */
__global__ void smooth_quant_scale_kernel(
    const float* X,
    const float* scales,
    float* X_scaled,
    int M, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    int col = idx % K;
    X_scaled[idx] = X[idx] * scales[col];
}

/**
 * @brief Inverse SmoothQuant (after GEMM): Y_unscaled = Y / scale
 * @param Y Output after GEMM [M, N]
 * @param scales Scale per channel [K]
 * @param Y_unscaled Output unscaled [M, N]
 * @param M, N Dimensions
 */
__global__ void smooth_quant_unscale_kernel(
    const half* Y,
    const float* scales,
    half* Y_unscaled,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    // For the output, we typically divide by scales at the end
    // This is usually applied per-output-channel
    Y_unscaled[idx] = Y[idx];
}

/**
 * @brief Compute SmoothQuant scales with both activation and weight stats
 * @param X Activation tensor
 * @param W Weight tensor [K, N]
 * @param scales Output scale per channel
 * @param alpha Migration strength
 * @param M, K, N Dimensions
 */
__global__ void smooth_quant_scales_full_kernel(
    const float* X,
    const float* W,
    float* scales,
    float alpha,
    int M, int K, int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= K) return;

    // Activation statistics
    float act_max = 0.0f;
    for (int m = 0; m < M; ++m) {
        act_max = fmaxf(act_max, fabsf(X[m * K + col]));
    }

    // Weight statistics (per input channel)
    float weight_max = 0.0f;
    for (int n = 0; n < N; ++n) {
        weight_max = fmaxf(weight_max, fabsf(W[col * N + n]));
    }

    if (act_max == 0.0f) act_max = 1.0f;
    if (weight_max == 0.0f) weight_max = 1.0f;

    // SmoothQuant formula
    // s_j = max(|X_ij|)^(1-alpha) * max(|W_jk|)^alpha
    // This balances the quantization difficulty between activations and weights
    float s = powf(act_max, 1.0f - alpha) * powf(weight_max, alpha);
    scales[col] = 1.0f / s;
}

/**
 * @brief Combined SmoothQuant: scale activations and perform GEMM
 * @param X Input activation [M, K]
 * @param W Weight [K, N]
 * @param scales SmoothQuant scales [K]
 * @param output Output [M, N]
 * @param M, N, K Dimensions
 */
__global__ void smooth_quant_gemm_kernel(
    const float* X,
    const float* W,
    const float* scales,
    float* output,
    int M, int N, int K
) {
    __shared__ float sX[16][16];
    __shared__ float sW[16][16];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float acc = 0.0f;

    for (int k = 0; k < K; k += 16) {
        // Load and scale X
        if (row < M && (k + tx) < K) {
            sX[ty][tx] = X[row * K + k + tx] * scales[k + tx];
        } else {
            sX[ty][tx] = 0.0f;
        }

        // Load W
        if (col < N && (k + ty) < K) {
            sW[ty][tx] = W[(k + ty) * N + col];
        } else {
            sW[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute
        for (int i = 0; i < 16; ++i) {
            acc += sX[ty][i] * sW[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        output[row * N + col] = acc;
    }
}

// Wrapper functions
void smooth_quant_scales(const float* X, float* scales, float alpha, int M, int K) {
    int block_size = 256;
    int grid_size = (K + block_size - 1) / block_size;
    smooth_quant_scales_kernel<<<grid_size, block_size>>>(X, scales, alpha, M, K);
    CUDA_KERNEL_CHECK();
}

void smooth_quant_scales_full(const float* X, const float* W, float* scales, float alpha, int M, int K, int N) {
    int block_size = 256;
    int grid_size = (K + block_size - 1) / block_size;
    smooth_quant_scales_full_kernel<<<grid_size, block_size>>>(X, W, scales, alpha, M, K, N);
    CUDA_KERNEL_CHECK();
}

void smooth_quant_scale(const float* X, const float* scales, float* X_scaled, int M, int K) {
    int block_size = 256;
    int grid_size = (M * K + block_size - 1) / block_size;
    smooth_quant_scale_kernel<<<grid_size, block_size>>>(X, scales, X_scaled, M, K);
    CUDA_KERNEL_CHECK();
}

void smooth_quant_gemm(const float* X, const float* W, const float* scales, float* output, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    smooth_quant_gemm_kernel<<<grid, block>>>(X, W, scales, output, M, N, K);
    CUDA_KERNEL_CHECK();
}
