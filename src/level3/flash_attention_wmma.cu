/**
 * @file flash_attention_wmma.cu
 * @brief FlashAttention-2 with WMMA + cp.async (double buffer)
 *
 * Using:
 * - cp.async for async global -> shared memory copy
 * - WMMA for Tensor Core compute
 * - Double buffering to hide memory latency
 *
 * Pipeline:
 *   cp.async Q/K/V for iteration i+1  |
 *   WMMA compute for iteration i      | overlap
 *   cp.async_wait_group               |
 *
 * Copyright (c) 2024 ToyKernel Contributors
 */

#include "../include/utils.cuh"
#include <mma.h>

#define BLOCK_SIZE 64
#define HEAD_DIM 64
#define WARP_SIZE 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define CP_ASYNC_CA "@%0 cp.async.ca.shared.global [%1], [%2], %3;"
#define CP_ASYNC_CG "@%0 cp.async.cg.shared.global [%1], [%2], %3;"

using namespace nvcuda::wmma;

/**
 * @brief cp.async: async copy from global to shared memory
 * @param smem_ptr Shared memory address (128-bit aligned)
 * @param gmem_ptr Global memory address
 * @param bytes Number of bytes to copy (4, 8, or 16)
 */
__device__ __forceinline__ void cp_async_ca(void* smem_ptr, const void* gmem_ptr, int bytes) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(CP_ASYNC_CA
        :
        : "r"(1), "r"(smem_addr), "l"(gmem_ptr), "n"(bytes));
}

__device__ __forceinline__ void cp_async_cg(void* smem_ptr, const void* gmem_ptr, int bytes) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(CP_ASYNC_CG
        :
        : "r"(1), "r"(smem_addr), "l"(gmem_ptr), "n"(bytes));
}

/**
 * @brief Commit cp.async operations
 */
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

/**
 * @brief Wait for cp.async operations
 * @param n Number of pending groups to wait for (0 = wait all)
 */
__device__ __forceinline__ void cp_async_wait_group(int n) {
    asm volatile("cp.async.wait_group %0;" :: "n"(n) : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}

/**
 * @brief FlashAttention with WMMA + cp.async double buffering
 *
 * Pipeline structure:
 * - Buffer 0/1 for K,V double buffering
 * - Async copy next K,V while computing current
 */
__global__ void flash_attention_wmma_async_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int seqlen,
    int num_heads
) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int block_q = blockIdx.x;
    int head_id = blockIdx.y;

    // Shared memory layout with double buffering
    // Q[64, 64]: loaded once, no double buffer needed
    // K[2][64, 64]: double buffer for K (transposed layout)
    // V[2][64, 64]: double buffer for V
    // P[64, 64]: for softmax results
    // S[64, 64]: QK scores in float
    extern __shared__ half smem[];

    half* sQ = smem;
    half* sK = smem + BLOCK_SIZE * HEAD_DIM;           // K buffer 0
    half* sV = sK + 2 * BLOCK_SIZE * HEAD_DIM;         // V buffer 0
    half* sP = sV + 2 * BLOCK_SIZE * HEAD_DIM;         // P matrix
    float* sS = (float*)(sP + BLOCK_SIZE * BLOCK_SIZE); // QK scores

    // Double buffer pointers
    half* sK_buffers[2] = {sK, sK + BLOCK_SIZE * HEAD_DIM};
    half* sV_buffers[2] = {sV, sV + BLOCK_SIZE * HEAD_DIM};

    // ========== Phase 1: Load Q tile (sync, Q is reused) ==========
    // Q is loaded synchronously - we need it for all KV blocks
    int q_global_base = head_id * seqlen * HEAD_DIM + block_q * BLOCK_SIZE * HEAD_DIM;

    // Use cp.async for Q as well, but wait immediately since we need it now
    for (int i = tid; i < BLOCK_SIZE * HEAD_DIM / 4; i += blockDim.x) {
        // Copy 8 bytes (4 halfs) at a time
        int row = (i * 4) / HEAD_DIM;
        int col = (i * 4) % HEAD_DIM;
        int global_row = block_q * BLOCK_SIZE + row;

        if (global_row < seqlen) {
            const void* gmem_ptr = &Q[q_global_base + row * HEAD_DIM + col];
            void* smem_ptr = &sQ[row * HEAD_DIM + col];
            cp_async_cg(smem_ptr, gmem_ptr, 8);
        }
    }
    cp_async_commit_group();
    cp_async_wait_all();  // Wait for Q to be ready
    __syncthreads();

    // Per-warp accumulators
    float warp_m[2] = {-INFINITY, -INFINITY};
    float warp_l[2] = {0.0f, 0.0f};
    float warp_o[2][HEAD_DIM] = {{0}};

    // ========== Phase 2: Double buffered KV pipeline ==========
    int buffer_id = 0;

    // Preload first K,V block
    int first_kv_base = head_id * seqlen * HEAD_DIM;
    for (int i = tid; i < BLOCK_SIZE * HEAD_DIM / 4; i += blockDim.x) {
        int row = (i * 4) / HEAD_DIM;
        int col = (i * 4) % HEAD_DIM;
        int global_kv_row = row;

        if (global_kv_row < seqlen) {
            // K: transposed storage [HEAD_DIM, BLOCK_SIZE]
            int kv_offset = global_kv_row * HEAD_DIM + col;
            half* k_dst = sK_buffers[buffer_id] + col * BLOCK_SIZE + row;
            half* v_dst = sV_buffers[buffer_id] + row * HEAD_DIM + col;

            cp_async_cg(k_dst, &K[first_kv_base + kv_offset], 8);
            cp_async_cg(v_dst, &V[first_kv_base + kv_offset], 8);
        }
    }
    cp_async_commit_group();

    // Main KV loop with double buffering
    for (int kv_block = 0; kv_block < seqlen; kv_block += BLOCK_SIZE) {
        int next_kv_block = kv_block + BLOCK_SIZE;
        int next_buffer_id = 1 - buffer_id;

        // Wait for current buffer to be ready
        cp_async_wait_group(0);
        __syncthreads();

        // Start async load of next K,V (if not last iteration)
        if (next_kv_block < seqlen) {
            int next_kv_base = head_id * seqlen * HEAD_DIM + next_kv_block * HEAD_DIM;

            for (int i = tid; i < BLOCK_SIZE * HEAD_DIM / 4; i += blockDim.x) {
                int row = (i * 4) / HEAD_DIM;
                int col = (i * 4) % HEAD_DIM;
                int global_kv_row = next_kv_block + row;

                if (global_kv_row < seqlen) {
                    int kv_offset = global_kv_row * HEAD_DIM + col;
                    half* k_dst = sK_buffers[next_buffer_id] + col * BLOCK_SIZE + row;
                    half* v_dst = sV_buffers[next_buffer_id] + row * HEAD_DIM + col;

                    cp_async_cg(k_dst, &K[next_kv_base + kv_offset], 8);
                    cp_async_cg(v_dst, &V[next_kv_base + kv_offset], 8);
                }
            }
            cp_async_commit_group();
        }

        // Get current buffer pointers
        half* curr_K = sK_buffers[buffer_id];
        half* curr_V = sV_buffers[buffer_id];

        // ========== Compute with WMMA (overlaps with next async load) ==========
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            int q_row_start = warp_id * 32 + group * 16;

            // ----- Step 1: QK = Q @ K^T -----
            float tile_qk[8][MMA_M * MMA_N] = {{0}};

            #pragma unroll
            for (int k_step = 0; k_step < 4; ++k_step) {
                int k_offset = k_step * MMA_K;

                fragment<matrix_a, MMA_M, MMA_N, MMA_K, half, row_major> frag_q;
                load_matrix_sync(frag_q, sQ + q_row_start * HEAD_DIM + k_offset, HEAD_DIM);

                #pragma unroll
                for (int n_tile = 0; n_tile < 8; ++n_tile) {
                    fragment<matrix_b, MMA_M, MMA_N, MMA_K, half, col_major> frag_k;
                    fragment<accumulator, MMA_M, MMA_N, MMA_K, float> frag_acc;

                    // Load K from transposed layout
                    load_matrix_sync(frag_k, curr_K + k_offset * BLOCK_SIZE + n_tile * MMA_N, BLOCK_SIZE);

                    if (k_step == 0) {
                        fill_fragment(frag_acc, 0.0f);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < MMA_M * MMA_N; ++i) {
                            frag_acc.x[i] = tile_qk[n_tile][i];
                        }
                    }

                    mma_sync(frag_acc, frag_q, frag_k, frag_acc);

                    #pragma unroll
                    for (int i = 0; i < MMA_M * MMA_N; ++i) {
                        tile_qk[n_tile][i] = frag_acc.x[i];
                    }
                }
            }

            // Store QK to shared memory
            #pragma unroll
            for (int n_tile = 0; n_tile < 8; ++n_tile) {
                #pragma unroll
                for (int i = 0; i < MMA_M * MMA_N; ++i) {
                    int row = i / MMA_N;
                    int col = i % MMA_N;
                    sS[(q_row_start + row) * BLOCK_SIZE + n_tile * MMA_N + col] =
                        tile_qk[n_tile][i] * softmax_scale;
                }
            }
            __syncthreads();

            // ----- Step 2: Online Softmax -----
            float local_max = -INFINITY;
            #pragma unroll
            for (int col = lane_id; col < BLOCK_SIZE; col += WARP_SIZE) {
                if (kv_block + col < seqlen) {
                    #pragma unroll
                    for (int row = 0; row < MMA_M; ++row) {
                        local_max = fmaxf(local_max, sS[(q_row_start + row) * BLOCK_SIZE + col]);
                    }
                }
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
            }

            float new_m = fmaxf(warp_m[group], local_max);
            float scale = expf(warp_m[group] - new_m);
            warp_l[group] *= scale;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                warp_o[group][d] *= scale;
            }

            // Compute P = softmax(QK)
            float sum_exp = 0.0f;
            #pragma unroll
            for (int n_tile = 0; n_tile < 8; ++n_tile) {
                #pragma unroll
                for (int i = lane_id; i < MMA_M * MMA_N; i += WARP_SIZE) {
                    int row = i / MMA_N;
                    int col = i % MMA_N;
                    int global_col = kv_block + n_tile * MMA_N + col;

                    if (global_col < seqlen) {
                        float val = sS[(q_row_start + row) * BLOCK_SIZE + n_tile * MMA_N + col];
                        float p = expf(val - new_m);
                        sP[(q_row_start + row) * BLOCK_SIZE + n_tile * MMA_N + col] = __float2half(p);
                        sum_exp += p;
                    } else {
                        sP[(q_row_start + row) * BLOCK_SIZE + n_tile * MMA_N + col] = __float2half(0.0f);
                    }
                }
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
            }
            warp_l[group] += sum_exp;
            __syncthreads();

            // ----- Step 3: PV = P @ V -----
            float tile_pv[8][MMA_M * MMA_N] = {{0}};

            #pragma unroll
            for (int k_step = 0; k_step < 4; ++k_step) {
                int k_offset = k_step * MMA_K;

                fragment<matrix_a, MMA_M, MMA_N, MMA_K, half, row_major> frag_p;
                load_matrix_sync(frag_p, sP + q_row_start * BLOCK_SIZE + k_offset, BLOCK_SIZE);

                #pragma unroll
                for (int v_tile = 0; v_tile < 8; ++v_tile) {
                    fragment<matrix_b, MMA_M, MMA_N, MMA_K, half, row_major> frag_v;
                    fragment<accumulator, MMA_M, MMA_N, MMA_K, float> frag_acc;

                    load_matrix_sync(frag_v, curr_V + k_offset * HEAD_DIM + v_tile * MMA_N, HEAD_DIM);

                    if (k_step == 0) {
                        fill_fragment(frag_acc, 0.0f);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < MMA_M * MMA_N; ++i) {
                            frag_acc.x[i] = tile_pv[v_tile][i];
                        }
                    }

                    mma_sync(frag_acc, frag_p, frag_v, frag_acc);

                    #pragma unroll
                    for (int i = 0; i < MMA_M * MMA_N; ++i) {
                        tile_pv[v_tile][i] = frag_acc.x[i];
                    }
                }
            }

            // Accumulate to output
            #pragma unroll
            for (int v_tile = 0; v_tile < 8; ++v_tile) {
                #pragma unroll
                for (int i = 0; i < MMA_M * MMA_N; ++i) {
                    int col = i % MMA_N;
                    warp_o[group][v_tile * MMA_N + col] += tile_pv[v_tile][i];
                }
            }

            warp_m[group] = new_m;
        }

        // Swap buffer for next iteration
        buffer_id = next_buffer_id;
        __syncthreads();
    }

    // Wait for any remaining async operations
    cp_async_wait_all();

    // ========== Write Output ==========
    #pragma unroll
    for (int group = 0; group < 2; ++group) {
        int row = warp_id * 32 + group * 16 + (lane_id / 2);
        int global_row = block_q * BLOCK_SIZE + row;

        if (global_row < seqlen && (lane_id / 2) < 16) {
            int o_offset = head_id * seqlen * HEAD_DIM + global_row * HEAD_DIM;
            float inv_l = 1.0f / warp_l[group];

            int d_start = (lane_id % 2) * 32;
            #pragma unroll
            for (int d = 0; d < 32; ++d) {
                float val = warp_o[group][d_start + d] * inv_l;
                O[o_offset + d_start + d] = __float2half(val);
            }
        }
    }
}

/**
 * @brief Simple WMMA version without cp.async (for comparison)
 */
__global__ void flash_attention_wmma_simple_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int seqlen,
    int num_heads
) {
    // Same as async version but without cp.async - using regular loads
    // ... implementation similar to async version without cp.async calls

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int block_q = blockIdx.x;
    int head_id = blockIdx.y;

    extern __shared__ half smem[];
    half* sQ = smem;
    half* sK = smem + BLOCK_SIZE * HEAD_DIM;
    half* sV = smem + 2 * BLOCK_SIZE * HEAD_DIM;
    half* sP = smem + 3 * BLOCK_SIZE * HEAD_DIM;
    float* sS = (float*)(sP + BLOCK_SIZE * BLOCK_SIZE);

    // Load Q (regular sync load)
    for (int i = tid; i < BLOCK_SIZE * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int global_row = block_q * BLOCK_SIZE + row;
        int offset = head_id * seqlen * HEAD_DIM + global_row * HEAD_DIM + (i % HEAD_DIM);
        sQ[i] = (global_row < seqlen) ? Q[offset] : __float2half(0.0f);
    }
    __syncthreads();

    float warp_m[2] = {-INFINITY, -INFINITY};
    float warp_l[2] = {0.0f};
    float warp_o[2][HEAD_DIM] = {{0}};

    for (int kv_block = 0; kv_block < seqlen; kv_block += BLOCK_SIZE) {
        // Load K, V (regular)
        for (int i = tid; i < BLOCK_SIZE * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int global_kv_row = kv_block + row;
            int kv_offset = head_id * seqlen * HEAD_DIM + global_kv_row * HEAD_DIM + d;

            if (global_kv_row < seqlen) {
                sK[d * BLOCK_SIZE + row] = K[kv_offset];
                sV[i] = V[kv_offset];
            } else {
                sK[d * BLOCK_SIZE + row] = __float2half(0.0f);
                sV[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // WMMA compute (same as async version)
        for (int group = 0; group < 2; ++group) {
            int q_row_start = warp_id * 32 + group * 16;

            // QK
            float tile_qk[8][MMA_M * MMA_N] = {{0}};
            for (int k_step = 0; k_step < 4; ++k_step) {
                fragment<matrix_a, MMA_M, MMA_N, MMA_K, half, row_major> frag_q;
                load_matrix_sync(frag_q, sQ + q_row_start * HEAD_DIM + k_step * MMA_K, HEAD_DIM);

                for (int n_tile = 0; n_tile < 8; ++n_tile) {
                    fragment<matrix_b, MMA_M, MMA_N, MMA_K, half, col_major> frag_k;
                    fragment<accumulator, MMA_M, MMA_N, MMA_K, float> frag_acc;

                    load_matrix_sync(frag_k, sK + k_step * MMA_K * BLOCK_SIZE + n_tile * MMA_N, BLOCK_SIZE);

                    if (k_step == 0) fill_fragment(frag_acc, 0.0f);
                    else {
                        for (int i = 0; i < MMA_M * MMA_N; ++i) frag_acc.x[i] = tile_qk[n_tile][i];
                    }

                    mma_sync(frag_acc, frag_q, frag_k, frag_acc);

                    for (int i = 0; i < MMA_M * MMA_N; ++i) tile_qk[n_tile][i] = frag_acc.x[i];
                }
            }

            // Softmax and PV (abbreviated)
            // ... same logic as async version
        }
        __syncthreads();
    }

    // Write output
    for (int group = 0; group < 2; ++group) {
        int row = warp_id * 32 + group * 16 + (tid % WARP_SIZE / 2);
        int global_row = block_q * BLOCK_SIZE + row;
        if (global_row < seqlen) {
            int o_offset = head_id * seqlen * HEAD_DIM + global_row * HEAD_DIM;
            float inv_l = 1.0f / warp_l[group];
            for (int d = 0; d < HEAD_DIM; ++d) {
                O[o_offset + d] = __float2half(warp_o[group][d] * inv_l);
            }
        }
    }
}

/**
 * @brief Wrapper for WMMA + cp.async kernel
 */
void flash_attention_wmma(const half* Q, const half* K, const half* V,
                          half* O, float softmax_scale, int seqlen, int num_heads,
                          bool use_async = true) {
    dim3 block(64);
    dim3 grid((seqlen + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads);

    // Shared memory with double buffer:
    // Q: 64*64
    // K: 2*64*64 (double buffer)
    // V: 2*64*64 (double buffer)
    // P: 64*64
    // S: 64*64 float
    size_t shared_size = (BLOCK_SIZE * HEAD_DIM) * sizeof(half) +       // Q
                         (2 * BLOCK_SIZE * HEAD_DIM * 2) * sizeof(half) + // K,V double buffer
                         (BLOCK_SIZE * BLOCK_SIZE) * sizeof(half) +      // P
                         (BLOCK_SIZE * BLOCK_SIZE) * sizeof(float);       // S

    if (use_async) {
        flash_attention_wmma_async_kernel<<<grid, block, shared_size>>>(
            Q, K, V, O, softmax_scale, seqlen, num_heads);
    } else {
        flash_attention_wmma_simple_kernel<<<grid, block, shared_size>>>(
            Q, K, V, O, softmax_scale, seqlen, num_heads);
    }
    CUDA_KERNEL_CHECK();
}
