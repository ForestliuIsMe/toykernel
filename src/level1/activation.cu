/**
 * @file activation.cu
 * @brief Activation functions: GeLU and Swish
 *
 * This file implements activation functions commonly used in LLMs:
 * - GeLU: Gaussian Error Linear Unit
 *   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * - Swish: x * sigmoid(x)
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

/**
 * @brief GeLU activation function
 * @param input Input tensor
 * @param output Output tensor
 * @param n Number of elements
 */
__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];

    // GeLU approximation
    float sqrt_2_over_pi = 0.7978845608028674f;
    float c = 0.044715f;

    float inner = sqrt_2_over_pi * (x + c * x * x * x);
    float t = tanhf(inner);
    float y = 0.5f * x * (1.0f + t);

    output[idx] = y;
}

/**
 * @brief Swish activation function
 * @param input Input tensor
 * @param output Output tensor
 * @param n Number of elements
 */
__global__ void swish_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];

    // Sigmoid: 1 / (1 + exp(-x))
    float sigmoid = 1.0f / (1.0f + expf(-x));

    output[idx] = x * sigmoid;
}

/**
 * @brief GeLU derivative (for training)
 * @param input Input tensor
 * @param output Output tensor (gradient)
 * @param n Number of elements
 */
__global__ void gelu_grad_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];

    float sqrt_2_over_pi = 0.7978845608028674f;
    float c = 0.044715f;

    float inner = sqrt_2_over_pi * (x + c * x * x * x);
    float t = tanhf(inner);
    float d = expf(-inner * inner);

    // GeLU derivative
    float y = 0.5f * (1.0f + t) + 0.5f * x * d * sqrt_2_over_pi * (1.0f + 3.0f * c * x * x);

    output[idx] = y;
}

// Wrapper functions
void gelu(const float* input, float* output, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    gelu_kernel<<<grid_size, block_size>>>(input, output, n);
    CUDA_KERNEL_CHECK();
}

void swish(const float* input, float* output, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    swish_kernel<<<grid_size, block_size>>>(input, output, n);
    CUDA_KERNEL_CHECK();
}

void gelu_grad(const float* input, float* output, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    gelu_grad_kernel<<<grid_size, block_size>>>(input, output, n);
    CUDA_KERNEL_CHECK();
}
