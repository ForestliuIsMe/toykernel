/**
 * @file quantize.cu
 * @brief Quantization kernel implementations
 *
 * Implements various quantization schemes:
 * - INT8 symmetric quantization
 * - INT8 asymmetric quantization (with zero point)
 * - Per-tensor and per-channel scaling
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../../include/utils.cuh"

/**
 * @brief Quantize float to INT8 (symmetric, per-tensor)
 * @param input Input float tensor
 * @param output Output INT8 tensor
 * @param scale Output scale factor
 * @param n Number of elements
 */
__global__ void quantize_int8_symmetric_kernel(
    const float* input, int8_t* output, float* scale, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Find max absolute value across all threads
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    sdata[tid] = fabsf(input[idx]);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    float max_val = sdata[0];
    if (max_val == 0.0f) max_val = 1.0f;

    float scale_val = 127.0f / max_val;

    if (tid == 0) {
        *scale = 1.0f / scale_val;
    }
    __syncthreads();

    // Quantize
    float quantized = roundf(input[idx] * scale_val);
    quantized = fmaxf(-128.0f, fminf(127.0f, quantized));
    output[idx] = (int8_t)quantized;
}

/**
 * @brief Quantize float to INT8 (asymmetric, per-tensor)
 * @param input Input float tensor
 * @param output Output INT8 tensor
 * @param scale Output scale factor
 * @param zero_point Zero point
 * @param n Number of elements
 */
__global__ void quantize_int8_asymmetric_kernel(
    const float* input, int8_t* output,
    float* scale, int8_t* zero_point, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Find min and max
    sdata[tid] = input[idx];
    __syncthreads();

    // Parallel reduction for min
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float min_val = sdata[0];

    __syncthreads();

    // Parallel reduction for max
    sdata[tid] = input[idx];
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    if (tid == 0) {
        float range = max_val - min_val;
        if (range == 0.0f) range = 1.0f;
        *scale = range / 255.0f;
        *zero_point = (int8_t)roundf(-min_val / *scale);
    }
    __syncthreads();

    // Quantize
    float quantized = roundf(input[idx] / *scale + (float)*zero_point);
    quantized = fmaxf(-128.0f, fminf(127.0f, quantized));
    output[idx] = (int8_t)quantized;
}

/**
 * @brief Dequantize INT8 to float
 * @param input Input INT8 tensor
 * @param scale Scale factor
 * @param zero_point Zero point
 * @param output Output float tensor
 * @param n Number of elements
 */
__global__ void dequantize_int8_kernel(
    const int8_t* input, float scale, int8_t zero_point, float* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = ((float)input[idx] - (float)zero_point) * scale;
}

/**
 * @brief Per-channel quantization
 * @param input Input tensor [M, K]
 * @param output Quantized output [M, K]
 * @param scales Scale per channel [M]
 * @param M, K Dimensions
 */
__global__ void quantize_per_channel_kernel(
    const float* input, int8_t* output, float* scales, int M, int K
) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;

    // Find max in this row
    extern __shared__ float sdata[];
    sdata[tid] = 0.0f;

    for (int k = tid; k < K; k += blockDim.x) {
        sdata[tid] = fmaxf(sdata[tid], fabsf(input[row * K + k]));
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
    float scale = 127.0f / max_val;

    if (tid == 0) {
        scales[row] = 1.0f / scale;
    }
    __syncthreads();

    // Quantize
    for (int k = tid; k < K; k += blockDim.x) {
        float quantized = roundf(input[row * K + k] * scale);
        quantized = fmaxf(-128.0f, fminf(127.0f, quantized));
        output[row * K + k] = (int8_t)quantized;
    }
}

// Wrapper functions
void quantize_int8_symmetric(const float* input, int8_t* output, float* scale, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    int shared_size = block_size * sizeof(float);
    quantize_int8_symmetric_kernel<<<grid_size, block_size, shared_size>>>(input, output, scale, n);
    CUDA_KERNEL_CHECK();
}

void quantize_int8_asymmetric(const float* input, int8_t* output, float* scale, int8_t* zero_point, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    int shared_size = block_size * sizeof(float);
    quantize_int8_asymmetric_kernel<<<grid_size, block_size, shared_size>>>(input, output, scale, zero_point, n);
    CUDA_KERNEL_CHECK();
}

void dequantize_int8(const int8_t* input, float scale, int8_t zero_point, float* output, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    dequantize_int8_kernel<<<grid_size, block_size>>>(input, scale, zero_point, output, n);
    CUDA_KERNEL_CHECK();
}

void quantize_per_channel(const float* input, int8_t* output, float* scales, int M, int K) {
    dim3 block(256);
    dim3 grid(M);
    int shared_size = 256 * sizeof(float);
    quantize_per_channel_kernel<<<grid, block, shared_size>>>(input, output, scales, M, K);
    CUDA_KERNEL_CHECK();
}
