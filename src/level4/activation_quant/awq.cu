/**
 * @file awq.cu
 * @brief AWQ (Activation-aware Weight Quantization) implementation
 *
 * AWQ is a weight-only quantization method that:
 * - Protects salient weights based on activation statistics
 * - Uses per-channel scaling derived from activation magnitudes
 * - Achieves better accuracy than standard weight quantization
 *
 * Paper: AWQ: Activation-aware Weight Quantization for LLM Compression
 * https://arxiv.org/abs/2306.00978
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../../include/utils.cuh"

#define AWQ_GROUP_SIZE 128
#define TILE_M 16
#define TILE_N 16

/**
 * @brief Compute AWQ scales based on activation statistics
 * @param X Activation tensor [M, K]
 * @param scales Output per-channel scales [K]
 * @param M, K Dimensions
 *
 * AWQ scale formula: s = 1 / max(|X[:, k]|)
 * This gives more resolution to weights connected to high-activation inputs
 */
__global__ void awq_scales_kernel(
    const float* X,
    float* scales,
    int M, int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= K) return;

    // Find max absolute activation for this channel
    float max_val = 0.0f;
    for (int m = 0; m < M; ++m) {
        max_val = fmaxf(max_val, fabsf(X[m * K + col]));
    }

    if (max_val == 0.0f) max_val = 1.0f;

    // AWQ: higher activation channel gets higher scale
    // This effectively dequantizes weights with high activation influence
    scales[col] = 1.0f / max_val;
}

/**
 * @brief AWQ: Scale weights before quantization
 * @param W Weight tensor [K, N]
 * @param scales Per-channel scales [K]
 * @param W_scaled Output scaled weight [K, N]
 * @param K, N Dimensions
 */
__global__ void awq_scale_weights_kernel(
    const float* W,
    const float* scales,
    float* W_scaled,
    int K, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;

    int col = idx % N;
    int row = idx / N;

    // Scale each weight by its input channel scale
    W_scaled[idx] = W[idx] * scales[row];
}

/**
 * @brief AWQ: Quantize with scale protection
 * @param W_scaled Scaled weight tensor
 * @param W_quant Quantized output (INT8)
 * @param scales Per-channel scales for dequantization
 * @param K, N Dimensions
 */
__global__ void awq_quantize_kernel(
    const float* W_scaled,
    int8_t* W_quant,
    float* scales,
    int K, int N
) {
    int row = blockIdx.x;
    if (row >= K) return;

    int tid = threadIdx.x;

    // Find max in this row
    extern __shared__ float sdata[];
    sdata[tid] = 0.0f;

    for (int n = tid; n < N; n += blockDim.x) {
        sdata[tid] = fmaxf(sdata[tid], fabsf(W_scaled[row * N + n]));
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    float max_val = sdata[0];
    if (max_val == 0.0f) max_val = 1.0f;

    float quant_scale = 127.0f / max_val;
    float dequant_scale = 1.0f / quant_scale;

    if (tid == 0) {
        scales[row] = dequant_scale;
    }
    __syncthreads();

    // Quantize
    for (int n = tid; n < N; n += blockDim.x) {
        float val = W_scaled[row * N + n];
        float quantized = roundf(val * quant_scale);
        quantized = fmaxf(-128.0f, fminf(127.0f, quantized));
        W_quant[row * N + n] = (int8_t)quantized;
    }
}

/**
 * @brief AWQ GEMM with in-place dequantization
 * @param W_quant Quantized weights [K, N]
 * @param scales AWQ scales [K]
 * @param X Input activations [M, K]
 * @param Y Output [M, N]
 * @param M, N, K Dimensions
 */
__global__ void awq_gemm_kernel(
    const int8_t* W_quant,
    const float* scales,
    const half* X,
    half* Y,
    int M, int N, int K
) {
    __shared__ float sX[TILE_M][TILE_N];
    __shared__ int8_t sW[TILE_N][TILE_K];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float acc[TILE_N];
    for (int i = 0; i < TILE_N; ++i) {
        acc[i] = 0.0f;
    }

    for (int k = 0; k < K; k += TILE_K) {
        // Load X tile
        for (int i = 0; i < 2; ++i) {
            int x_row = row;
            int x_col = col + i * TILE_N / 2;
            if (x_row < M && x_col < N) {
                sX[ty][tx + i * TILE_N / 2] = __half2float(X[x_row * N + x_col]);
            } else {
                sX[ty][tx + i * TILE_N / 2] = 0.0f;
            }
        }

        // Load and scale W tile
        for (int i = 0; i < 2; ++i) {
            int w_row = k / 2 + ty;
            int w_col = col;
            if (w_row < K && w_col < N) {
                int8_t w = W_quant[w_row * N + w_col];
                float scale = scales[w_row];
                sW[ty + i * TILE_K / 2][tx] = (float)w * scale;
            } else {
                sW[ty + i * TILE_K / 2][tx] = 0.0f;
            }
        }

        __syncthreads();

        // Compute
        for (int kk = 0; kk < TILE_K; ++kk) {
            float x_val = sX[ty][kk];
            for (int n = 0; n < TILE_N; ++n) {
                acc[n] += x_val * sW[kk][n];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Y[row * N + col] = __float2half(acc[tx]);
    }
}

/**
 * @brief Simplified AWQ GEMM
 */
__global__ void awq_gemm_simple_kernel(
    const int8_t* W_quant,
    const float* scales,
    const half* X,
    half* Y,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        float w_dequant = (float)W_quant[k * N + col] * scales[k];
        sum += w_dequant * __half2float(X[row * K + k]);
    }

    Y[row * N + col] = __float2half(sum);
}

// Wrapper functions
void awq_scales(const float* X, float* scales, int M, int K) {
    int block_size = 256;
    int grid_size = (K + block_size - 1) / block_size;
    awq_scales_kernel<<<grid_size, block_size>>>(X, scales, M, K);
    CUDA_KERNEL_CHECK();
}

void awq_scale_weights(const float* W, const float* scales, float* W_scaled, int K, int N) {
    int block_size = 256;
    int grid_size = (K * N + block_size - 1) / block_size;
    awq_scale_weights_kernel<<<grid_size, block_size>>>(W, scales, W_scaled, K, N);
    CUDA_KERNEL_CHECK();
}

void awq_quantize(const float* W_scaled, int8_t* W_quant, float* scales, int K, int N) {
    dim3 block(256);
    dim3 grid(K);
    int shared_size = 256 * sizeof(float);
    awq_quantize_kernel<<<grid, block, shared_size>>>(W_scaled, W_quant, scales, K, N);
    CUDA_KERNEL_CHECK();
}

void awq_gemm(const int8_t* W_quant, const float* scales, const half* X, half* Y, int M, int N, int K) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    awq_gemm_kernel<<<grid, block>>>(W_quant, scales, X, Y, M, N, K);
    CUDA_KERNEL_CHECK();
}
