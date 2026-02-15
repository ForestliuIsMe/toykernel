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

#include "utils.cuh"

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
// 制定block size之后，需要规定三个重要参数： M_MMA_ITERA, N_MMA_ITERA, K_MMA_ITERA
// 一般来说，需要满足以下constrain：  WARP_NUM = min(M_SUB_BLOCKS * K_SUB_BLOCKS, N_SUB_BLOCKS * K_SUB_BLOCKS)
// 所以根据规定的M_BLOCK_SIZE, N_BLOCK_SIZE和K_BLOCK_SIZE 是可以推断出合适的M_MMA_ITERA，N_MMA_ITERA， K_MMA_ITRERA以及WARP_NUM的组合关系的
// 至于grid的形状，则按输入的M，N和M_BLOCK_SIZE，N_BLOCK_SIZE来区分。
template<int M_BLOCK_SIZE, int N_BLOCK_SIZE, int K_BLOCK_SIZE,
        int M_MMA_ITERA, int N_MMA_ITERA, int K_MMA_ITERA>
__global__ void gemm_sliced_k(const half* A, const half* B, half* C,
                                   int M, int N, int K){
    const int THREAD_NUM = blockDim.x;
    const int tid = threadIdx.x;
    const int M_SUB_BLOCKS = M_BLOCK_SIZE / (16 * M_MMA_ITERA);
    const int N_SUB_BLOCKS = N_BLOCK_SIZE / (8  * N_MMA_ITERA);
    const int K_SUB_BLOCKS = K_BLOCK_SIZE / (16 * K_MMA_ITERA);
    // each warp should do mma along the K first
    // along the K can reduce by mma operator

    const int wid = WARP_ID();
    const int lid = LANE_ID();
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Shared memory with padding to avoid bank conflicts
    // Pad columns to odd number (e.g., +1) to avoid 32-bank conflicts
    __shared__ half sA[K_BLOCK_SIZE][M_BLOCK_SIZE + 8];
    __shared__ half sB[K_BLOCK_SIZE][N_BLOCK_SIZE + 8];

    __shared__ half sC[M_BLOCK_SIZE][N_BLOCK_SIZE + 8];

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
    int k_sub_block_id = wid % K_SUB_BLOCKS;
    int n_sub_block_id = (wid / K_SUB_BLOCKS) % N_SUB_BLOCKS;
    int m_sub_block_id = (wid / K_SUB_BLOCKS) % M_SUB_BLOCKS;

    for(int k = 0; k < K; k += K_BLOCK_SIZE){
        // deal with [M_BLOCK_SIZE,K_BLOCK_SIZE,N_BLOCK_SIZE]
        // each warp use m16n8k16

        // Load A tile into shared memory
        // Cooperative loading: each thread loads elements
        // A is col-major, so M is the stride between columns
        #pragma unroll
        for (int i = tid; i < K_BLOCK_SIZE * M_BLOCK_SIZE; i += THREAD_NUM) {
            int a_row = k + i / M_BLOCK_SIZE;
            int a_col = i % M_BLOCK_SIZE;
            if (a_row < K && a_col < M_BLOCK_SIZE) {
                sA[i / M_BLOCK_SIZE][a_col] = A[OFFSET2D(a_row, a_col, M)];
            }
        }

        // Load B tile into shared memory
        // B is row-major, so N is the stride between rows
        #pragma unroll
        for (int i = tid; i < K_BLOCK_SIZE * N_BLOCK_SIZE; i += THREAD_NUM) {
            int b_row = k + i / N_BLOCK_SIZE;
            int b_col = i % N_BLOCK_SIZE;
            if (b_row < K && b_col < N_BLOCK_SIZE) {
                sB[i / N_BLOCK_SIZE][b_col] = B[OFFSET2D(b_row, b_col, N)];
            }
        }

        __syncthreads();


        // Perform MMA for each sub-block
        // For each kk iteration, we process 16 elements in K dimension
        for(int kk = 0; kk < K_MMA_ITERA; kk++){
            // Calculate base pointers for this K iteration
            // ldmatrix requires 32-byte aligned address
            half* k_slice_A = &(sA[(kk + k_sub_block_id * K_MMA_ITERA) * 16][0]);
            half* k_slice_B = &(sB[(kk + k_sub_block_id * K_MMA_ITERA) * 16][0]);

            for(int mm = 0 ; mm < M_MMA_ITERA; mm++){
                for(int nn = 0; nn < N_MMA_ITERA; nn++){
                    // Calculate the starting position for this MMA tile
                    // Each warp handles specific sub-blocks based on warp id
                    int m_tile_offset = (m_sub_block_id * M_MMA_ITERA + mm) * 16;
                    int n_tile_offset = (n_sub_block_id * N_MMA_ITERA + nn) * 8;

                    // Pointers to the tile in shared memory
                    // Each thread provides its own pointer for ldmatrix
                    // ldmatrix will automatically handle the per-lane addressing
                    half* mma_tile_A = k_slice_A + m_tile_offset;
                    half* mma_tile_B = k_slice_B + n_tile_offset;

                    // Load A fragment using ldmatrix.x4 (loads 4 8x8 matrices = 128 bytes)
                    // For m16n8k16, A is M=16 x K=16, needs 4 registers (each holds 8 elements)
                    uint32_t a_frag[4];
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x4 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "l"(mma_tile_A)
                    );

                    // Load B fragment using ldmatrix.x2 (loads 2 8x8 matrices = 64 bytes)
                    // For m16n8k16, B is K=16 x N=8, needs 2 registers
                    uint32_t b_frag[2];
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x2 {%0, %1}, [%2];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1])
                        : "l"(mma_tile_B)
                    );

                    // Perform MMA: c = a * b + c
                    // m16n8k16.f16.f16.f16.f16: 16 rows x 8 cols output in fp16
                    // Input A: 4 registers, Input B: 2 registers, Accumulator: 2 registers
                    asm volatile (
                        "mma.sync.aligned.m16n8k16.f16.f16.f16.f16 "
                        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
                        : "=r"(c[mm][nn][0]), "=r"(c[mm][nn][1])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_frag[0]), "r"(b_frag[1]),
                          "r"(c[mm][nn][0]), "r"(c[mm][nn][1])
                    );
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: store MMA result to shared memory, then to global memory
    // For m16n8k16, each thread holds 2 elements of the 16x8 output tile
    // Thread layout in output tile (each thread holds 2 consecutive elements in a row):
    // - Group 0 (lanes 0-3, 8-11, 16-19, 24-27): cols 0-1, 4-5
    // - Group 1 (lanes 4-7, 12-15, 20-23, 28-31): cols 2-3, 6-7

    // Step 1: Each warp writes its MMA results to shared memory sC
    for (int mm = 0; mm < M_MMA_ITERA; mm++) {
        for (int nn = 0; nn < N_MMA_ITERA; nn++) {
            // Base position for this MMA tile in sC
            int c_row_base = (m_sub_block_id * M_MMA_ITERA + mm) * 16;
            int c_col_base = (n_sub_block_id * N_MMA_ITERA + nn) * 8;

            half c_half[2];
            uint32_t c_val;

            c_val = c[mm][nn][0];
            c_half[0] = __ushort_as_half(c_val & 0xffff);
            c_half[1] = __ushort_as_half(c_val >> 16);

            atomicAdd(sC[c_row_base + lid / 4][c_col_base + (lid % 4) * 2], c_half[0]);
            atomicAdd(sC[c_row_base + lid / 4][c_col_base + (lid % 4) * 2 + 1], c_half[1]);

            c_val = c[mm][nn][1];
            c_half[0] = __ushort_as_half(c_val & 0xffff);
            c_half[1] = __ushort_as_half(c_val >> 16);

            atomicAdd(sC[c_row_base + lid / 4 + 8][c_col_base + (lid % 4) * 2], c_half[0]);
            atomicAdd(sC[c_row_base + lid / 4 + 8][c_col_base + (lid % 4) * 2 + 1], c_half[1]);
        }
    }

    __syncthreads();

    // Step 2: Cooperative store from sC to global memory
    half* C_block = C + OFFSET2D(M_BLOCK_SIZE * by, N_BLOCK_SIZE * bx, N);

    // Each thread loads multiple elements from sC and writes to global C
    #pragma unroll
    for (int i = tid; i < M_BLOCK_SIZE * N_BLOCK_SIZE; i += THREAD_NUM) {
        int row = i / N_BLOCK_SIZE;
        int col = i % N_BLOCK_SIZE;

        // Check bounds
        int global_row = M_BLOCK_SIZE * by + row;
        int global_col = N_BLOCK_SIZE * bx + col;

        if (global_row < M && global_col < N) {
            C_block[OFFSET2D(row, col, N)] = sC[row][col];
        }
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

/**
 * @brief GEMM sliced-k wrapper with auto-selected parameters
 *
 * Based on the kernel constraints:
 * - MMA instruction: m16n8k16 (M=16, N=8, K=16)
 * - From kernel warp assignment:
 *     k_sub_block_id = wid % K_SUB_BLOCKS
 *     n_sub_block_id = (wid / K_SUB_BLOCKS) % N_SUB_BLOCKS
 *     m_sub_block_id = (wid / K_SUB_BLOCKS) % M_SUB_BLOCKS
 *   where:
 *     M_SUB_BLOCKS = M_BLOCK_SIZE / (16 * M_MMA_ITERA)
 *     N_SUB_BLOCKS = N_BLOCK_SIZE / (8 * N_MMA_ITERA)
 *     K_SUB_BLOCKS = K_BLOCK_SIZE / (16 * K_MMA_ITERA)
 *
 * This wrapper automatically selects optimal parameters based on matrix dimensions.
 */
void gemm_sliced_k(const half* A, const half* B, half* C, int M, int N, int K) {
    // Configuration selection based on matrix dimensions
    // Strategy: Set K_SUB_BLOCKS=1, then we need WARP_NUM >= max(M_SUB_BLOCKS, N_SUB_BLOCKS)
    // since both m_sub_block_id and n_sub_block_id use (wid / K_SUB_BLOCKS) = wid

    if (M >= 128 && N >= 64) {
        // Large matrices: 128x64x32
        // M_MMA_IT=2: M_SUB = 128/(16*2) = 4
        // N_MMA_IT=2: N_SUB = 64/(8*2) = 4
        // K_MMA_IT=2: K_SUB = 32/(16*2) = 1
        // Need WARP_NUM >= max(4, 4) = 4, use 8 for better occupancy
        const int M_BLOCK = 128, N_BLOCK = 64, K_BLOCK = 32;
        const int M_MMA_IT = 2, N_MMA_IT = 2, K_MMA_IT = 2;
        const int WARP_NUM = 8;

        dim3 block(WARP_NUM * 32);  // 256 threads
        dim3 grid((N + N_BLOCK - 1) / N_BLOCK, (M + M_BLOCK - 1) / M_BLOCK);

        gemm_sliced_k<M_BLOCK, N_BLOCK, K_BLOCK, M_MMA_IT, N_MMA_IT, K_MMA_IT>
            <<<grid, block>>>(A, B, C, M, N, K);
    } else if (M >= 64 && N >= 64) {
        // Medium matrices: 64x64x32
        // M_MMA_IT=2: M_SUB = 64/(16*2) = 2
        // N_MMA_IT=2: N_SUB = 64/(8*2) = 4
        // K_MMA_IT=2: K_SUB = 32/(16*2) = 1
        // Need WARP_NUM >= max(2, 4) = 4, use 8 for better occupancy
        const int M_BLOCK = 64, N_BLOCK = 64, K_BLOCK = 32;
        const int M_MMA_IT = 2, N_MMA_IT = 2, K_MMA_IT = 2;
        const int WARP_NUM = 8;

        dim3 block(WARP_NUM * 32);
        dim3 grid((N + N_BLOCK - 1) / N_BLOCK, (M + M_BLOCK - 1) / M_BLOCK);

        gemm_sliced_k<M_BLOCK, N_BLOCK, K_BLOCK, M_MMA_IT, N_MMA_IT, K_MMA_IT>
            <<<grid, block>>>(A, B, C, M, N, K);
    } else if (M >= 64 && N >= 32) {
        // Small-Medium matrices: 64x32x32
        // M_MMA_IT=2: M_SUB = 64/(16*2) = 2
        // N_MMA_IT=1: N_SUB = 32/(8*1) = 4
        // K_MMA_IT=2: K_SUB = 32/(16*2) = 1
        // Need WARP_NUM >= max(2, 4) = 4, use 8 for better occupancy
        const int M_BLOCK = 64, N_BLOCK = 32, K_BLOCK = 32;
        const int M_MMA_IT = 2, N_MMA_IT = 1, K_MMA_IT = 2;
        const int WARP_NUM = 8;

        dim3 block(WARP_NUM * 32);
        dim3 grid((N + N_BLOCK - 1) / N_BLOCK, (M + M_BLOCK - 1) / M_BLOCK);

        gemm_sliced_k<M_BLOCK, N_BLOCK, K_BLOCK, M_MMA_IT, N_MMA_IT, K_MMA_IT>
            <<<grid, block>>>(A, B, C, M, N, K);
    } else {
        // Small matrices: 32x32x32
        // M_MMA_IT=1: M_SUB = 32/16 = 2
        // N_MMA_IT=2: N_SUB = 32/16 = 2
        // K_MMA_IT=2: K_SUB = 32/32 = 1
        // Need WARP_NUM >= max(2, 2) = 2, use 4 for better parallelism
        const int M_BLOCK = 32, N_BLOCK = 32, K_BLOCK = 32;
        const int M_MMA_IT = 1, N_MMA_IT = 2, K_MMA_IT = 2;
        const int WARP_NUM = 4;

        dim3 block(WARP_NUM * 32);
        dim3 grid((N + N_BLOCK - 1) / N_BLOCK, (M + M_BLOCK - 1) / M_BLOCK);

        gemm_sliced_k<M_BLOCK, N_BLOCK, K_BLOCK, M_MMA_IT, N_MMA_IT, K_MMA_IT>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
    CUDA_KERNEL_CHECK();
}
