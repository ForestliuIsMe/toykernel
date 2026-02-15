/**
 * @file gemv.cu
 * @brief General Matrix-Vector Multiplication (GEMV)
 *
 * Implements y = A * x where:
 * - A is an M x K matrix (in row-major order)
 * - x is a K-dimensional vector
 * - y is an M-dimensional vector
 *
 * This file provides two implementations:
 * - gemv_iterative_k: Each warp processes one row, iterative over K
 * - gemv_split_k: Split K dimension across blocks for better parallelism
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

/**
 * @brief Warp-level reduction for addition
 * @tparam T Type of value to reduce
 * @param val Input value
 * @return Reduced sum
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_add(volatile T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(WARP_MASK, val, offset);
    }
    return val;
}

/**
 * @brief GEMV with iterative K dimension
 *
 * Each warp processes one row of the matrix. Threads within a warp
 * cooperate to compute the dot product iteratively over K.
 *
 * Requirements:
 * - THREADS must be multiple of WARP_SIZE (32)
 * - K should be >= 128 for good efficiency
 *
 * @tparam THREADS Total threads per block (default 128)
 * @param A Matrix A (M x K), row-major
 * @param x Vector x (K)
 * @param y Output vector y (M)
 * @param M Number of rows
 * @param K Number of columns
 */
template<int THREADS = 128>
__global__ void gemv_iterative_k_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    constexpr int BLOCK_SIZE_M = THREADS / WARP_SIZE;

    int tid = threadIdx.x;
    int lid = LANE_ID();        // Lane ID within warp [0, 31]
    int wid = WARP_ID();        // Warp ID within block
    int bid = blockIdx.x;       // Block index

    // Global row index: warp ID within block + warps per block * block ID
    int row = wid + BLOCK_SIZE_M * bid;

    // Bounds check: ensure this warp has a valid row to process
    if (row >= M) return;

    float sum = 0.0f;

    // Iterate over K in steps of WARP_SIZE * 4 (128 elements)
    // Each thread processes 4 elements per iteration using float4
    for (int k = 0; k < K; k += WARP_SIZE * 4) {
        __syncthreads();
        // Check bounds for float4 loading
        int k_start = k + lid * 4;

        // Bounds check: ensure we don't read beyond K
        if (k_start + 3 < K) {
            // All 4 elements are valid, use float4 for vectorized load
            float4 a_local = FLOAT4(&A[OFFSET2D(row, k_start, K)]);
            float4 x_local = FLOAT4(&x[k_start]);

            sum += a_local.x * x_local.x;
            sum += a_local.y * x_local.y;
            sum += a_local.z * x_local.z;
            sum += a_local.w * x_local.w;
        } else {
            // Partial load for boundary case
            for (int i = 0; i < 4; ++i) {
                int col = k_start + i;
                if (col < K) {
                    sum += A[OFFSET2D(row, col, K)] * x[col];
                }
            }
        }
    }

    // Warp-level reduction to sum across 32 threads
    __syncthreads();
    sum = warp_reduce_add(sum);

    // Thread 0 in each warp writes the result
    if (lid == 0) {
        y[row] = sum;
    }
}

/**
 * @brief GEMV with split-K parallelism
 *
 * Splits the K dimension across multiple blocks in the x direction.
 * Each block processes a tile of K, and results are accumulated via atomicAdd.
 *
 * Requirements:
 * - PITCH_SIZE must be integer multiple of 128
 * - K should be divisible by PITCH_SIZE for best results
 * - y must be initialized to 0 before calling this kernel
 *
 * @tparam THREADS     Total threads per block
 * @param PITCH_SIZE   K dimension tile size per block in x direction
 * @param A Matrix A (M x K), row-major
 * @param x Vector x (K)
 * @param y Output vector y (M), must be pre-initialized to 0
 * @param M Number of rows
 * @param K Number of columns
 */
template<int THREADS = 128, int PITCH_SIZE = 128>
__global__ void gemv_split_k_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    constexpr int BLOCK_SIZE_M = THREADS / WARP_SIZE;

    int lid = LANE_ID();        // Lane ID [0, 31]
    int wid = WARP_ID();        // Warp ID within block
    int bid_x = blockIdx.x;     // Block index in x direction (K split)
    int bid_y = blockIdx.y;     // Block index in y direction (row group)

    // Grid dimensions
    int num_k_splits = gridDim.x;
    int k_start = bid_x * PITCH_SIZE;

    // Bounds check: ensure K tile is within bounds
    if (k_start >= K) return;

    // Global row index for this warp
    int row = wid + BLOCK_SIZE_M * bid_y;
    if (row >= M) return;

    float sum = 0.0f;

    // Process this K tile
    int tile_end = min(k_start + PITCH_SIZE, K);

    for (int k = k_start; k < tile_end; k += WARP_SIZE * 4) {

        int k_local = k - k_start + lid * 4;

        // Bounds check for float4 loading
        if (k_local + 3 < tile_end - k_start && k + lid * 4 + 3 < K) {
            float4 a_local = FLOAT4(&A[OFFSET2D(row, k + lid * 4, K)]);
            float4 x_local = FLOAT4(&x[k + lid * 4]);

            sum += a_local.x * x_local.x;
            sum += a_local.y * x_local.y;
            sum += a_local.z * x_local.z;
            sum += a_local.w * x_local.w;
        } else {
            // Partial load for boundary
            for (int i = 0; i < 4; ++i) {
                int col = k + lid * 4 + i;
                if (col < K) {
                    sum += A[OFFSET2D(row, col, K)] * x[col];
                }
            }
        }
        __syncthreads();
    }

    // Warp-level reduction
    sum = warp_reduce_add(sum);

    // Thread 0 in each warp atomic adds to the output
    if (lid == 0) {
        atomicAdd(&y[row], sum);
    }
}

// ============================================
// Wrapper Functions
// ============================================

/**
 * @brief GEMV with iterative K (warp-parallel)
 *
 * Each warp processes one row. Good for cases where M is large
 * and K is moderate (>= 128).
 *
 * @param A Matrix A (M x K)
 * @param x Vector x (K)
 * @param y Output vector y (M)
 * @param M Number of rows
 * @param K Number of columns
 */
void gemv_iterative_k(
    const float* A,
    const float* x,
    float* y,
    int M,
    int K
) {
    constexpr int THREADS = 128;  // 4 warps per block
    dim3 block(THREADS);
    dim3 grid((M + THREADS / WARP_SIZE - 1) / (THREADS / WARP_SIZE));

    gemv_iterative_k_kernel<THREADS><<<grid, block>>>(A, x, y, M, K);
    CUDA_KERNEL_CHECK();
}

/**
 * @brief GEMV with iterative K and custom thread count
 *
 * @param A Matrix A (M x K)
 * @param x Vector x (K)
 * @param y Output vector y (M)
 * @param M Number of rows
 * @param K Number of columns
 * @param threads Number of threads per block (must be multiple of 32)
 */
void gemv_iterative_k_custom(
    const float* A,
    const float* x,
    float* y,
    int M,
    int K,
    int threads
) {
    // Validate thread count
    if (threads % WARP_SIZE != 0) {
        threads = (threads / WARP_SIZE) * WARP_SIZE;
    }

    dim3 block(threads);
    dim3 grid((M + threads / WARP_SIZE - 1) / (threads / WARP_SIZE));

    if (threads == 128) {
        gemv_iterative_k_kernel<128><<<grid, block>>>(A, x, y, M, K);
    } else if (threads == 256) {
        gemv_iterative_k_kernel<256><<<grid, block>>>(A, x, y, M, K);
    } else if (threads == 64) {
        gemv_iterative_k_kernel<64><<<grid, block>>>(A, x, y, M, K);
    } else {
        // Fallback to 128
        gemv_iterative_k_kernel<128><<<grid, block>>>(A, x, y, M, K);
    }
    CUDA_KERNEL_CHECK();
}

/**
 * @brief GEMV with split-K parallelism
 *
 * Splits K dimension across blocks. Good for cases where K is very large.
 * IMPORTANT: y must be initialized to 0 before calling this function.
 *
 * @param A Matrix A (M x K)
 * @param x Vector x (K)
 * @param y Output vector y (M), must be pre-initialized to 0
 * @param M Number of rows
 * @param K Number of columns
 * @param pitch_size K tile size per block (default 128)
 */
void gemv_split_k(
    const float* A,
    const float* x,
    float* y,
    int M,
    int K,
    int pitch_size = 128
) {
    constexpr int THREADS = 128;

    // Calculate grid dimensions
    int num_k_splits = (K + pitch_size - 1) / pitch_size;
    int num_row_blocks = (M + THREADS / WARP_SIZE - 1) / (THREADS / WARP_SIZE);

    dim3 block(THREADS);
    dim3 grid(num_k_splits, num_row_blocks);

    // Select kernel based on pitch_size
    if (pitch_size == 128) {
        gemv_split_k_kernel<THREADS, 128><<<grid, block>>>(A, x, y, M, K);
    } else if (pitch_size == 256) {
        gemv_split_k_kernel<THREADS, 256><<<grid, block>>>(A, x, y, M, K);
    } else if (pitch_size == 512) {
        gemv_split_k_kernel<THREADS, 512><<<grid, block>>>(A, x, y, M, K);
    } else {
        // Fallback to 128
        gemv_split_k_kernel<THREADS, 128><<<grid, block>>>(A, x, y, M, K);
    }
    CUDA_KERNEL_CHECK();
}

/**
 * @brief GEMV split-K with automatic initialization
 *
 * Automatically initializes y to 0 before calling split-k kernel.
 *
 * @param A Matrix A (M x K)
 * @param x Vector x (K)
 * @param y Output vector y (M)
 * @param M Number of rows
 * @param K Number of columns
 * @param pitch_size K tile size per block
 */
void gemv_split_k_auto_init(
    const float* A,
    const float* x,
    float* y,
    int M,
    int K,
    int pitch_size = 128
) {
    // Initialize output to 0
    CUDA_CHECK(cudaMemset(y, 0, M * sizeof(float)));

    // Call split-k kernel
    gemv_split_k(A, x, y, M, K, pitch_size);
}

// ============================================
// Convenience aliases (for compatibility)
// ============================================

void gemv_naive(const float* A, const float* x, float* y, int M, int K) {
    gemv_iterative_k(A, x, y, M, K);
}

void gemv_shared(const float* A, const float* x, float* y, int M, int K) {
    gemv_split_k(A, x, y, M, K);
}
