/**
 * @file gemm.cu
 * @brief Naive General Matrix Multiplication (GEMM)
 *
 * Implements C = alpha * A * B + beta * C where:
 * - A is an M x K matrix
 * - B is a K x N matrix
 * - C is an M x N matrix
 *
 * This is the foundational implementation for understanding
 * more optimized GEMM algorithms.
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

/**
 * @brief Naive GEMM kernel
 * Each thread computes one element of C
 * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A and rows of B
 * @param alpha Scalar multiplier
 * @param beta Scalar for existing C
 */
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

// input A is col major
// input B is row major
// first dimision is always K
// 使用m16k16n8的mma指令
// M_MMA_ITERA : 一个block内一个warp需要在m方向做几次mma
// N_MMA_ITERA : 一个block内一个warp需要在n方向做几次mma
// K_MMA_ITERA : 一个block内一个warp需要在k方向做几次mma
template<int M_BLOCK_SIZE, int N_BLOCK_SIZE, int K_BLOCK_SIZE,
        int M_MMA_ITERA, int N_MMA_ITERA, int K_MMA_ITERA>
__global__ void gemm_sliced_k(const half* A, const half* B, half* C,
                                   int M, int N, int K){
    const int THREAD_NUM = threadDim.x;
    const int M_SUB_BLOCKS = M_BLOCK_SIZE / (16 * M_MMA_ITERA);
    const int N_SUB_BLOCKS = N_BLOCK_SIZE / (8  * N_MMA_ITERA);
    const int K_SUB_BLOCKS = K_BLOCK_SIZE / (16 * K_MMA_ITERA);
    // each warp should do mma along the K first
    // along the K can reduce by mma operator

    const int wid = WARP_ID();
    const int lid = LANE_ID();
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    __shared__ half sA[K_BLOCK_SIZE][M_BLOCK_SIZE];
    __shared__ half sB[K_BLOCK_SIZE][N_BLOCK_SIZE];

    __shared__ half sC[M_BLOCK_SIZE][N_BLOCK_SIZE];

    A = A + OFFSET2D(0, M_BLOCK_SIZE * by , M);
    B = B + OFFSET2D(0, N_BLOCK_SIZE * bx , N);

    // MMA registers
    // a[mm][nn]: A matrix fragment for sub-block (mm, nn)
    // b[mm][nn]: B matrix fragment for sub-block (mm, nn)
    // c[mm][nn]: Accumulator for sub-block (mm, nn)
    uint32_t a[M_MMA_ITERA][N_MMA_ITERA][4];  // m16n8k16: 4 registers per fragment
    uint32_t b[M_MMA_ITERA][N_MMA_ITERA][2];  // 2 registers per fragment
    uint32_t c[M_MMA_ITERA][N_MMA_ITERA][2];  // Accumulator registers

    // Initialize accumulator to zero
    #pragma unroll
    for (int mm = 0; mm < M_MMA_ITERA; mm++) {
        #pragma unroll
        for (int nn = 0; nn < N_MMA_ITERA; nn++) {
            c[mm][nn][0] = 0;
            c[mm][nn][1] = 0;
        }
    }

    // Determine which sub-block this thread/warp is responsible for
    // Each warp handles multiple sub-blocks based on wid
    int k_sub_block_id = wid % K_MMA_ITERA;
    int n_sub_block_id = wid / K_MMA_ITERA % N_SUB_BLOCKS;
    int m_sub_block_id = wid / (K_MMA_ITERA * N_SUB_BLOCKS) % M_SUB_BLOCKS;

    for(int k = 0; k < K; k += K_BLOCK_SIZE){
        // deal with [M_BLOCK_SIZE,K_BLOCK_SIZE,N_BLOCK_SIZE]
        // each warp use m16n8k16

        // Load A tile into shared memory
        // Cooperative loading: each thread loads elements
        #pragma unroll
        for (int i = 0; i < K_BLOCK_SIZE; i++) {
            int a_row = k + i;
            int a_col = tid;
            if (a_row < K && a_col < M_BLOCK_SIZE) {
                sA[i][a_col] = A[OFFSET2D(a_row, a_col, M)];
            }
        }

        // Load B tile into shared memory
        #pragma unroll
        for (int i = 0; i < K_BLOCK_SIZE; i++) {
            int b_row = k + i;
            int b_col = tid;
            if (b_row < K && b_col < N_BLOCK_SIZE) {
                sB[i][b_col] = B[OFFSET2D(b_row, b_col, N)];
            }
        }

        __syncthreads();


        half* subblock_A = &(sA[k_sub_block_id * 16][m_sub_block_id * 16]);
        half* subblock_B = &(sB[k_sub_block_id * 16][n_sub_block_id * 8]);

        // Perform MMA for each sub-block
        for(int mm = 0 ; mm < M_MMA_ITERA; mm++){
            for(int nn = 0; nn < N_MMA_ITERA; nn++){
                for(int kk = 0; kk < K_MMA_ITERA; kk++){
                    half* mma_block_A = subblock_A + OFFSET2D(kk * 16, mm * 16, M_BLOCK_SIZE);
                    half* mma_block_B = subblock_B + OFFSET2D(kk * 16, nn * 8, N_BLOCK_SIZE);

                    // Load A fragment using ldmatrix
                    // Each thread loads 8 consecutive elements
                    uint32_t a_frag[4];
                    ldmatrix.sync.aligned.m8n8.x4(a_frag, mma_block_A);

                    // Load B fragment using ldmatrix
                    uint32_t b_frag[2];
                    ldmatrix.sync.aligned.m8n8.x1(b_frag, mma_block_B);

                    // Perform MMA: c = a * b + c
                    // m16n8k16: 16 rows x 8 cols output
                    uint32_t c_frag[2] = {c[mm][nn][0], c[mm][nn][1]};
                    mma.sync.aligned.m16n8k16.f16.f16.f16.f16(
                        c[mm][nn][0], c[mm][nn][1],
                        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                        b_frag[0], b_frag[1],
                        c_frag[0], c_frag[1]
                    );
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: store result to global memory
    // Need to relayout accumulator to match global memory layout

    // Determine output position for this warp

    int c_row_base = by * M_BLOCK_SIZE + m_sub_block_id * 16;
    int c_col_base = bx * N_BLOCK_SIZE + n_sub_block_id * 8;
    
    // Store each sub-block result
    #pragma unroll
    for (int mm = 0; mm < M_MMA_ITERA; mm++) {
        #pragma unroll
        for (int nn = 0; nn < N_MMA_ITERA; nn++) {
            int c_row = c_row_base + mm * 16;
            int c_col = c_col_base + nn * 8;

            // Convert accumulator to half and store
            // Each thread stores its portion
            half* c_ptr = &sC[OFFSET2D(c_row, c_col, N_BLOCK_SIZE)];

            // Store using atomic add for multiple warps writing to same location
            half c_half_reg[2];
            // only support for m16n8k16
            *reinterpret_cast<uint32_t*>(&c_half_reg[0]) = c[mm][nn][0];
            // Use atomic add for correctness when multiple warps write to same location
            // c0
            atomicAdd(reinterpret_cast<half*>(&c_ptr[OFFSET2D(lid / 4, (lid % 4) * 2, N_BLOCK_SIZE)]), c_half_reg[0]);
            atomicAdd(reinterpret_cast<half*>(&c_ptr[OFFSET2D(lid / 4, (lid % 4) * 2 + 1, N_BLOCK_SIZE)]), c_half_reg[0]);

            *reinterpret_cast<uint32_t*>(&c_half_reg[0]) = c[mm][nn][1];
            // c1
            atomicAdd(reinterpret_cast<half*>(&c_ptr[OFFSET2D(lid / 4, (lid % 4) * 2, N_BLOCK_SIZE)]), c_half_reg[0]);
            atomicAdd(reinterpret_cast<half*>(&c_ptr[OFFSET2D(lid / 4, (lid % 4) * 2 + 1, N_BLOCK_SIZE)]), c_half_reg[0]);
        
            __syncthreads();
        }
    }

    // write back
    int transaction_c_size = (M_BLOCK_SIZE * N_BLOCK_SIZE * sizeof(half));

    const half* C_block = C + OFFSET2D(M_BLOCK_SIZE * by, N_BLOCK_SIZE * bx, N);

    // Cooperative store: all threads write their portion
    #pragma unroll
    for (int i = 0; i < M_BLOCK_SIZE * N_BLOCK_SIZE; i += THREAD_NUM) {
        int idx = tid + i;
        if (idx < M_BLOCK_SIZE * N_BLOCK_SIZE) {
            int row = idx / N_BLOCK_SIZE;
            int col = idx % N_BLOCK_SIZE;
            C_block[OFFSET2D(row, col, N)] = sC[row][col];
        }
    }
}


/**
 * @brief Shared memory tiled GEMM
 * Uses shared memory for block-level caching
 * @param A Matrix A (M x K)
 * @param B Matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A and rows of B
 * @param alpha Scalar multiplier
 * @param beta Scalar for existing C
 */
__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K, float alpha, float beta) {
    const int BLOCK_SIZE = 16;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Loop over K tiles
    for (int m = 0; m < K; m += BLOCK_SIZE) {
        // Load A into shared memory
        if (row < M && (m + tx) < K) {
            As[ty][tx] = A[row * K + m + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B into shared memory
        if (col < N && (m + ty) < K) {
            Bs[ty][tx] = B[(m + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Wrapper functions
void gemm_naive(const float* A, const float* B, float* C,
                int M, int N, int K, float alpha = 1.0f, float beta = 0.0f) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_KERNEL_CHECK();
}

void gemm_tiled(const float* A, const float* B, float* C,
                int M, int N, int K, float alpha = 1.0f, float beta = 0.0f) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_KERNEL_CHECK();
}
