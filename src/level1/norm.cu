/**
 * @file norm.cu
 * @brief Layer normalization and RMS normalization implementations
 *
 * This file implements normalization layers commonly used in LLMs:
 * - LayerNorm: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 * - RMSNorm: y = x * gamma / sqrt(mean(x^2) + eps)
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

/**
 * @brief Layer normalization kernel
 * @param input Input tensor [N, H]
 * @param output Output tensor
 * @param gamma Learnable scale parameter
 * @param beta Learnable shift parameter
 * @param n Number of rows
 * @param h Hidden dimension size
 * @param eps Epsilon for numerical stability
 */
__global__ void layernorm_kernel(const float* input, float* output,
                                  const float* gamma, const float* beta,
                                  int n, int h, float eps) {
    int row = blockIdx.x;
    if (row >= n) return;

    const float* x = input + row * h;
    float* y = output + row * h;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < h; ++i) {
        mean += x[i];
    }
    mean /= h;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < h; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= h;

    float inv_std = rsqrtf(var + eps);

    // Normalize and apply gamma/beta
    for (int i = 0; i < h; ++i) {
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

/**
 * @brief RMS normalization kernel
 * @param input Input tensor [N, H]
 * @param output Output tensor
 * @param gamma Learnable scale parameter
 * @param n Number of rows
 * @param h Hidden dimension size
 * @param eps Epsilon for numerical stability
 */
__global__ void rmsnorm_kernel(const float* input, float* output,
                                const float* gamma, int n, int h, float eps) {
    int row = blockIdx.x;
    if (row >= n) return;

    const float* x = input + row * h;
    float* y = output + row * h;

    // Compute mean of squares
    float ms = 0.0f;
    for (int i = 0; i < h; ++i) {
        ms += x[i] * x[i];
    }
    ms /= h;

    float inv_norm = rsqrtf(ms + eps);

    // Normalize and scale
    for (int i = 0; i < h; ++i) {
        y[i] = x[i] * inv_norm * gamma[i];
    }
}

// Wrapper functions
void layernorm(const float* input, float* output,
               const float* gamma, const float* beta,
               int n, int h, float eps = 1e-5f) {
    dim3 block(1);
    dim3 grid(n);
    layernorm_kernel<<<grid, block>>>(input, output, gamma, beta, n, h, eps);
    CUDA_KERNEL_CHECK();
}

void rmsnorm(const float* input, float* output,
             const float* gamma, int n, int h, float eps = 1e-5f) {
    dim3 block(1);
    dim3 grid(n);
    rmsnorm_kernel<<<grid, block>>>(input, output, gamma, n, h, eps);
    CUDA_KERNEL_CHECK();
}
