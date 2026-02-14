/**
 * @file gemv.cu
 * @brief General Matrix-Vector Multiplication (GEMV)
 *
 * Implements y = A * x where:
 * - A is an M x N matrix
 * - x is an N-dimensional vector
 * - y is an M-dimensional vector
 *
 * Template parameters for block tiling:
 * - BLOCK_M: Number of rows processed per block
 * - BLOCK_N: Number of columns (vector elements) processed per iteration
 * - THREADS_PER_BLOCK: Total threads per block (should be >= BLOCK_M * BLOCK_N)
 *
 * This is a fundamental operation in linear layers and attention.
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

template<typename T>
__device__ T warp_reduce_add(volatile T d){
    // TODO:
}

// iterative-k
// 每一个warp处理一行数据
// 仅针对 K >= 128的情况
template<int THREADS>
__global__ void gemv_iterative_k_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* y,
    int M,
    int K
){
    int tid = threadIdx.x;
    int lid = LANE_ID();
    int wid = WARP_ID();
    int bid = blockIdx.x;

    const int BLOCK_SIZE_M = THREADS/WARP_SIZE;
    
    float sum = 0.0f;
    for(int k = 0; k < K; k += WARP_SIZE * 4){
        if(lid * 4 + k > K){
            break;
        }
        float4 a_local = FLOAT4(A[OFFSET2D(wid + BLOCK_SIZE_M * bid,lid * 4 + k, K)]);
        float4 x_local = FLOAT4(x[lid * 4 + k]);
        sum += a_local.x * x_local.x;
        sum += a_local.y * x_local.y;
        sum += a_local.z * x_local.z;
        sum += a_local.w * x_local.w;
    }

    sum = warp_reduce_add<float>(sum);

    if(lid == 0){
        y[wid + BLOCK_SIZE_M * bid] = sum;
    }
}


// pitch size must be integer multiply of 128
template<int THREADS, int PITCH_SIZE>
__global__ void gemv_split_k_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* y,
    int M,
    int K
){
    int tid = threadIdx.x;
    int lid = LANE_ID();
    int wid = WARP_ID();
    int bid = blockIdx.x + blockIdx.y * blockDim.x;

    int by = blockIdx.y;
    int bx = blockIdx.x;

    const int BLOCK_SIZE_M = THREADS/WARP_SIZE;
    
    float sum = 0.0f;
    for(int k = 0; k < PITCH_SIZE; k += WARP_SIZE * 4){
        float4 a_local = FLOAT4(A[OFFSET2D(wid + BLOCK_SIZE_M * by,lid * 4 + k + bx * PITCH_SIZE, K)]);
        float4 x_local = FLOAT4(x[lid * 4 + k + bx * PITCH_SIZE]);
        sum += a_local.x * x_local.x;
        sum += a_local.y * x_local.y;
        sum += a_local.z * x_local.z;
        sum += a_local.w * x_local.w;
    }

    sum = warp_reduce_add<float>(sum);

    if(lid == 0){
        atomicAdd(&y[wid + BLOCK_SIZE_M * by], sum);
    }
}




