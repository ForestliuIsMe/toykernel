/**
 * @file flash_attention.cu
 * @brief FlashAttention-2 forward pass implementation
 *
 * Implements efficient attention:
 * O = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * Key optimizations (FlashAttention-2):
 * - Tiling: Process attention in blocks to fit in SRAM
 * - Recomputation: Recompute softmax stats during backward pass
 * - Forward pass stores m and l for backward
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"

#define BLOCK_SIZE 64
// #define HEAD_DIM 64


/**
 * @brief FlashAttention-2 forward kernel with PTX MMA optimization
 *
 * Implements efficient attention computation O = softmax(Q @ K^T / sqrt(d)) @ V
 * using the FlashAttention-2 algorithm with warp-level parallelism and TensorCore MMA.
 *
 * Algorithm Overview:
 * 1. Tile Q, K, V into SRAM blocks (Qj, Ki, Vi) to reduce HBM bandwidth
 * 2. Compute attention scores S = Qj @ Ki^T using PTX mma.sync.aligned.m16n8k16
 * 3. Apply online softmax with rescaling (FlashAttention-2 style)
 * 4. Compute output O = P @ V using MMA accumulation
 * 5. Normalize and write back to global memory
 *
 * Memory Layout:
 * - Q: [batch, num_heads, q_seq_len, head_dim] (row-major)
 * - K: [batch, num_heads, kv_seq_len, head_dim] (row-major)
 * - V: [batch, num_heads, kv_seq_len, head_dim] (row-major)
 * - O: [batch, num_heads, q_seq_len, head_dim] (row-major)
 *
 * Grid Configuration:
 * - gridDim.x: 1 (CHUNK_SIZE handled by warps)
 * - gridDim.y: num_heads
 * - gridDim.z: batch_size
 * - blockDim.x: WARP_NUM * 32 (threads per block)
 *
 * Shared Memory Usage per Block:
 * - Ki:  Bc * HEAD_DIM * sizeof(half)  // Key tile
 * - Vi:  Bc * HEAD_DIM * sizeof(half)  // Value tile
 * - Qj:  WARP_NUM * Br * HEAD_DIM * sizeof(half)  // Query tiles (per warp)
 * - Oj:  WARP_NUM * Br * HEAD_DIM * sizeof(half)  // Output accumulators (per warp)
 * - Si:  WARP_NUM * Br * Bc * sizeof(half)        // Attention scores (per warp)
 * - RowLi: WARP_NUM * Br * sizeof(half)           // Softmax sum L (for backward)
 *
 * @tparam WARP_NUM     Number of warps per block (typically 4 or 8)
 * @tparam HEAD_DIM     Head dimension (must be multiple of 16, typically 64 or 128)
 * @tparam Bc           KV sequence block size (must be multiple of 8, typically 64 or 128)
 * @tparam Br           Query sequence block size (must be multiple of 16, typically 64 or 128)
 * @tparam CHUNK_SIZE   Query sequence length processed per kernel launch
 *
 * @param[in]  Q              Query tensor [batch, heads, CHUNK_SIZE, HEAD_DIM]
 * @param[in]  K              Key tensor [batch, heads, kv_seq_len, HEAD_DIM]
 * @param[in]  V              Value tensor [batch, heads, kv_seq_len, HEAD_DIM]
 * @param[out] O              Output tensor [batch, heads, CHUNK_SIZE, HEAD_DIM]
 * @param[in]  softmax_scale  Pre-computed scale factor (1/sqrt(head_dim))
 * @param[in]  kv_seq_len     Length of key/value sequence (may differ from CHUNK_SIZE)
 * @param[in]  num_heads      Number of attention heads
 *
 * @note Requires Ampere architecture (sm_80+) for MMA and __pack_half2 support
 * @note Q/K/V/O must be aligned to 16 bytes for optimal global memory access
 * @note This kernel does not implement causal masking (future work)
 *
 * Example Launch:
 * @code
 * const int WARP_NUM = 4;
 * const int HEAD_DIM = 64;
 * const int Bc = 64;
 * const int Br = 64;
 * const int CHUNK_SIZE = 1024;
 *
 * dim3 block(WARP_NUM * 32);  // 128 threads
 * dim3 grid(1, num_heads, batch_size);
 * flash_attention_kernel<WARP_NUM, HEAD_DIM, Bc, Br, CHUNK_SIZE>
 *     <<<grid, block>>>(Q, K, V, O, 1.0f/sqrtf(HEAD_DIM), kv_seq_len, num_heads);
 * @endcode
 */
template<int WARP_NUM, int HEAD_DIM, int Bc, int Br, int CHUNK_SIZE>
__global__ void flash_attention_kernel(
    const half* __restrict Q,
    const half* __restrict K,
    const half* __restrict V,
    half* O,
    float softmax_scale,
    int kv_seq_len,
    int num_heads
){
    int tid = threadIdx.x;
    int warp_id = WARP_ID();
    int lane_id = LANE_ID();

    // Calculate batch and head indices
    int head_id = blockIdx.y;
    int batch_idx = blockIdx.z;

    __shared__ half Ki[Bc * HEAD_DIM];
    __shared__ half Vi[Bc * HEAD_DIM];
    __shared__ half Qj[WARP_NUM][Br * HEAD_DIM];
    __shared__ half Oj[WARP_NUM][Br * HEAD_DIM];
    
    __shared__ half Si[WARP_NUM][Br * Bc];
    __shared__ half RowLi[WARP_NUM][Br];

    const int rows_for_each_thread = Br / 32;

    for(int j = warp_id * Br; j < CHUNK_SIZE; j += WARP_NUM * Br){
        // Load Qj from global memory: Q[batch, head_id, j:j+Br, :]
        // Each warp loads its own Br * HEAD_DIM block
        int q_base = (batch_idx * num_heads + head_id) * CHUNK_SIZE * HEAD_DIM + j * HEAD_DIM;
        for(int row = tid; row < Br * HEAD_DIM; row += WARP_NUM * 32) {
            int r = row / HEAD_DIM;
            int c = row % HEAD_DIM;
            int global_row = j + r;
            if(global_row < CHUNK_SIZE) {
                // Load Q and apply sqrt(softmax_scale) to avoid post-MMA scaling
                float q_val = __half2float(Q[q_base + r * HEAD_DIM + c]);
                Qj[warp_id][row] = __float2half(q_val * sqrtf(softmax_scale));
            } else {
                Qj[warp_id][row] = __float2half(0.0f);
            }
        }
        // Initialize Oj to 0
        for(int row = tid; row < Br * HEAD_DIM; row += WARP_NUM * 32) {
            Oj[warp_id][row] = __float2half(0.0f);
        }
        __syncthreads();

        // Online softmax state (persistent across K/V blocks)
        half m_prev[rows_for_each_thread];
        half l_prev[rows_for_each_thread];
        #pragma unroll
        for (int ri = 0; ri < rows_for_each_thread; ri++) {
            m_prev[ri] = -INFINITY;
            l_prev[ri] = 0.0f;
        }

        // inner loop
        for(int i = 0; i < kv_seq_len; i += Bc){
        
            // Load Ki, Vi from global to shared memory (synchronous)
            int kv_base = (batch_idx * num_heads + head_id) * kv_seq_len * HEAD_DIM + i * HEAD_DIM;

            // Load Ki: K[i:i+Bc, :]
            for(int row = tid; row < Bc * HEAD_DIM; row += WARP_NUM * 32) {
                int r = row / HEAD_DIM;
                int c = row % HEAD_DIM;
                int global_row = i + r;
                if(global_row < kv_seq_len) {
                    Ki[row] = K[kv_base + r * HEAD_DIM + c];
                } else {
                    Ki[row] = __float2half(0.0f);
                }
            }

            // Load Vi: V[i:i+Bc, :]
            for(int row = tid; row < Bc * HEAD_DIM; row += WARP_NUM * 32) {
                int r = row / HEAD_DIM;
                int c = row % HEAD_DIM;
                int global_row = i + r;
                if(global_row < kv_seq_len) {
                    Vi[row] = V[kv_base + r * HEAD_DIM + c];
                } else {
                    Vi[row] = __float2half(0.0f);
                }
            }
            __syncthreads();

            // calculate S = QK using PTX MMA
            // Qj: [Br, HEAD_DIM], Ki: [Bc, HEAD_DIM] (transposed conceptually)
            // S: [warp_num][Br, Bc]
            for(int m = 0; m < Br ; m += 16){
                for(int k = 0; k < HEAD_DIM; k+= 16){
                    // Each warp computes 16xBc slice of S
                    for(int sub_mma = 0; sub_mma < Bc; sub_mma += 8){
                        // MMA m16n8k16: A[16,16] @ B[16,8] -> C[16,8]
                        // Load A from Qj: 16 rows from m, 16 cols from k
                        // Load B from Ki: 8 rows from sub_mma, 16 cols from k (K^T)

                        uint32_t a_reg[4];  // 16x16 f16 = 256 elements / 32 threads = 8 f16 = 4 uint32_t
                        uint32_t b_reg[2];  // 16x8 f16 = 128 elements / 32 threads = 4 f16 = 2 uint32_t
                        uint32_t c_reg[2];  // 16x8 f16 accumulator

                        // Initialize accumulator to 0
                        #pragma unroll
                        for(int i = 0; i < 2; i++) c_reg[i] = 0;

                        // Load A from shared memory (Qj)
                        // MMA A matrix: 16x16 row-major
                        // Each thread owns: row = lane_id % 16, col group = lane_id / 16
                        #pragma unroll
                        for(int i = 0; i < 4; i++){
                            int row = (lane_id % 16);  // 0-15
                            int col_group = (lane_id / 16);  // 0 or 1 (for 16x16)
                            int col = col_group * 8 + i * 2;  // 0,2,4,6 or 8,10,12,14
                            int q_offset = (m + row) * HEAD_DIM + (k + col);
                            asm volatile("ld.shared.b32 %0, [%1];" :
                                "=r"(a_reg[i]) :
                                "r"((uint32_t)__cvta_generic_to_shared(&Qj[warp_id][q_offset]))
                            );
                        }

                        // Load B from shared memory (Ki^T)
                        // MMA B matrix: 16x8 column-major (conceptually K^T)
                        // K is [Bc, HEAD_DIM] row-major, we need K^T[k:k+16, sub_mma:sub_mma+8]
                        // For col-major B in MMA: B[row, col] where row=0-15, col=0-7
                        // Thread mapping: 4 threads per column
                        #pragma unroll
                        for(int i = 0; i < 2; i++){
                            int col = (lane_id % 8);  // 0-7 (column in B/K^T)
                            int row_group = (lane_id / 8);  // 0-3
                            int row = row_group * 4 + i * 2;  // 0,2,4,6 or 0,2,4,6
                            // K[col][k+row] in row-major K = K^T[row][col]
                            int k_offset = (sub_mma + col) * HEAD_DIM + (k + row);
                            asm volatile("ld.shared.b32 %0, [%1];" :
                                "=r"(b_reg[i]) :
                                "r"((uint32_t)__cvta_generic_to_shared(&Ki[k_offset]))
                            );
                        }

                        // MMA instruction: m16n8k16
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                            "{%0, %1}, "    // D (accumulator)
                            "{%2, %3, %4, %5}, "  // A
                            "{%6, %7}, "    // B
                            "{%8, %9};"     // C
                            :
                            "=r"(c_reg[0]), "=r"(c_reg[1])
                            :
                            "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                            "r"(b_reg[0]), "r"(b_reg[1]),
                            "r"(c_reg[0]), "r"(c_reg[1])
                        );

                        // Store result to Si (no scale needed, applied during Q load)
                        #pragma unroll
                        for(int i = 0; i < 2; i++){
                            int row = (lane_id / 4);  // 0-7
                            int col = (lane_id % 4) * 2 + i * 8;  // 0,2 or 8,10
                            // c_reg contains packed half values from MMA
                            int s_offset = (m + row) * Bc + (sub_mma + col);
                            if(s_offset < Br * Bc){
                                *(uint32_t*)(&Si[warp_id][s_offset]) = c_reg[i];
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // calculate max
            // max = max(rowmax(S),oldmax)   \in [Br]
            // adjuster = expf(oldmax - max)
            // P*V \in [Br, HEAD_DIM]
            // O = O * adjuster + P*V
            half rowmax[rows_for_each_thread];
            #pragma unroll
            for (int ri = 0; ri < rows_for_each_thread; ri++) rowmax[ri] = -INFINITY;

            // Step 1: Find row-wise max of S
            for(int row = lane_id; row < Br; row += 32){
                int local_row = row / 32;
                for(int col = 0; col < Bc ; col ++){
                    rowmax[local_row] = max(Si[warp_id][Bc * row + col], rowmax[local_row]);
                }
            }

            // Step 2: Online softmax update
            for(int row = lane_id; row < Br; row += 32){
                int local_row = row / 32;
                half m_new = max(rowmax[local_row], m_prev[local_row]);
                half scale = __expf(m_prev[local_row] - m_new);

                // Rescale O: O = O * scale
                for(int col = 0; col < HEAD_DIM; col++){
                    Oj[warp_id][HEAD_DIM * row + col] *= scale;
                }

                // Compute P = exp(S - m_new), accumulate row_sum
                half row_sum = 0;
                for(int col = 0; col < Bc; col++){
                    half p = __expf(Si[warp_id][Bc * row + col] - m_new);
                    Si[warp_id][Bc * row + col] = p;  // Reuse Si to store P
                    row_sum += p;
                }

                // Update l: l_new = l_old * scale + row_sum
                half l_new = l_prev[local_row] * scale + row_sum;
                l_prev[local_row] = l_new;
                m_prev[local_row] = m_new;

                // Save Li for backward pass (optional)
                RowLi[warp_id][row] = l_new;
            }

            // update O = O + PV (MMA P @ V) using PTX
            // Si: [warp_num][Br, Bc] contains P, Vi: [Bc, HEAD_DIM]
            for(int m = 0; m < Br ; m += 16){
                for(int k = 0; k < Bc; k+= 16){
                    for(int sub_mma = 0; sub_mma < HEAD_DIM; sub_mma += 8){
                        // MMA m16n8k16: P[16,16] @ V[16,8] -> O[16,8]
                        uint32_t a_reg[4];  // P tile: 16x16
                        uint32_t b_reg[2];  // V tile: 16x8
                        uint32_t c_reg[2];  // O accumulator

                        // Load accumulator from Oj (load-add-store pattern)
                        #pragma unroll
                        for(int i = 0; i < 2; i++){
                            int row = (lane_id / 4);
                            int col = (lane_id % 4) * 2 + i * 8;
                            int o_offset = (m + row) * HEAD_DIM + (sub_mma + col);
                            half2 o_val = *(half2*)(&Oj[warp_id][o_offset]);
                            c_reg[i] = __pack_half2(o_val.x, o_val.y);
                        }

                        // Load A from Si (P matrix) [Br, Bc]
                        #pragma unroll
                        for(int i = 0; i < 4; i++){
                            int row = (lane_id % 16);
                            int col_group = (lane_id / 16);
                            int col = col_group * 8 + i * 2;
                            int p_offset = (m + row) * Bc + (k + col);
                            asm volatile("ld.shared.b32 %0, [%1];" :
                                "=r"(a_reg[i]) :
                                "r"((uint32_t)__cvta_generic_to_shared(&Si[warp_id][p_offset]))
                            );
                        }

                        // Load B from Vi (V matrix) [Bc, HEAD_DIM]
                        // V^T for MMA: V^T[k:k+16, sub_mma:sub_mma+8]
                        #pragma unroll
                        for(int i = 0; i < 2; i++){
                            int col = (lane_id % 8);  // 0-7
                            int row_group = (lane_id / 8);
                            int row = row_group * 4 + i * 2;
                            // V[col][sub_mma+row] = V^T[row][col]
                            int v_offset = (k + col) * HEAD_DIM + (sub_mma + row);
                            asm volatile("ld.shared.b32 %0, [%1];" :
                                "=r"(b_reg[i]) :
                                "r"((uint32_t)__cvta_generic_to_shared(&Vi[v_offset]))
                            );
                        }

                        // MMA: O += P @ V
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                            "{%0, %1}, "
                            "{%2, %3, %4, %5}, "
                            "{%6, %7}, "
                            "{%8, %9};"
                            :
                            "=r"(c_reg[0]), "=r"(c_reg[1])
                            :
                            "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                            "r"(b_reg[0]), "r"(b_reg[1]),
                            "r"(c_reg[0]), "r"(c_reg[1])
                        );

                        // Store result back to Oj
                        #pragma unroll
                        for(int i = 0; i < 2; i++){
                            int row = (lane_id / 4);
                            int col = (lane_id % 4) * 2 + i * 8;
                            int o_offset = (m + row) * HEAD_DIM + (sub_mma + col);
                            // Direct store of packed uint32_t to shared memory
                            *(uint32_t*)(&Oj[warp_id][o_offset]) = c_reg[i];
                        }
                    }
                }
            }

            __syncthreads();
        }

        // Final normalization: O = O / l
        for(int row = lane_id; row < Br; row += 32){
            int local_row = row / 32;
            half l_inv = 1.0f / l_prev[local_row];
            for(int col = 0; col < HEAD_DIM; col++){
                Oj[warp_id][HEAD_DIM * row + col] *= l_inv;
            }
        }

        // Write Oj back to global memory O
        int o_base = (batch_idx * num_heads + head_id) * CHUNK_SIZE * HEAD_DIM + j * HEAD_DIM;
        for(int row = tid; row < Br * HEAD_DIM; row += WARP_NUM * 32) {
            int r = row / HEAD_DIM;
            int c = row % HEAD_DIM;
            int global_row = j + r;
            if(global_row < CHUNK_SIZE) {
                O[o_base + r * HEAD_DIM + c] = Oj[warp_id][row];
            }
        }
        __syncthreads();
    }
}


/**
 * @brief Convenience wrapper for flash_attention_kernel with automatic template instantiation
 *
 * Launches the optimized FlashAttention-2 kernel with pre-configured tile sizes based on head dimension.
 * Automatically selects optimal Br/Bc block sizes for common configurations.
 *
 * @tparam HEAD_DIM     Head dimension (64 or 128 supported)
 * @tparam WARP_NUM     Number of warps (default: 4)
 *
 * @param[in]  Q              Query tensor [batch, heads, seq_len, head_dim] (fp16)
 * @param[in]  K              Key tensor [batch, heads, kv_seq_len, head_dim] (fp16)
 * @param[in]  V              Value tensor [batch, heads, kv_seq_len, head_dim] (fp16)
 * @param[out] O              Output tensor [batch, heads, seq_len, head_dim] (fp16)
 * @param[in]  softmax_scale  Pre-computed scale factor (typically 1/sqrt(head_dim))
 * @param[in]  batch_size     Batch size
 * @param[in]  num_heads      Number of attention heads
 * @param[in]  seq_len        Query/Output sequence length (CHUNK_SIZE)
 * @param[in]  kv_seq_len     Key/Value sequence length (may differ for cross-attention)
 *
 * @throws cudaError_t if kernel launch fails (checked via CUDA_KERNEL_CHECK)
 *
 * @note Q/K/V/O must be device pointers to fp16 (__half) data
 * @note seq_len must be <= CHUNK_SIZE template parameter
 * @note For seq_len > 256, consider using multiple kernel launches or increasing WARP_NUM
 *
 * Example:
 * @code
 * half *d_Q, *d_K, *d_V, *d_O;  // device pointers, allocated via cudaMalloc
 * float scale = 1.0f / sqrtf(64.0f);
 *
 * flash_attention_fwd_wrapper<64, 4>(
 *     d_Q, d_K, d_V, d_O, scale,
 *     batch_size=32, num_heads=8,
 *     seq_len=1024, kv_seq_len=1024
 * );
 * @endcode
 */
template<int HEAD_DIM = 64, int WARP_NUM = 4>
inline void flash_attention_fwd_wrapper(
    const half* __restrict Q,
    const half* __restrict K,
    const half* __restrict V,
    half* O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int kv_seq_len
) {
    // Select tile sizes based on head dimension
    constexpr int Bc = (HEAD_DIM <= 64) ? 64 : 32;   // KV block size
    constexpr int Br = (HEAD_DIM <= 64) ? 64 : 32;   // Query block size

    // Grid: (1, heads, batch) - each block handles CHUNK_SIZE via warps
    dim3 block(WARP_NUM * 32);
    dim3 grid(1, num_heads, batch_size);

    // Ensure seq_len fits within CHUNK_SIZE
    // For larger sequences, caller should partition into multiple launches
    const int CHUNK_SIZE = ((seq_len + Br - 1) / Br) * Br;  // Round up to Br multiple

    // Launch kernel with dynamic CHUNK_SIZE through template (must be compile-time constant)
    // For production use, consider multiple instantiations or dynamic scheduling
    if constexpr (HEAD_DIM == 64) {
        if (seq_len <= 64) {
            flash_attention_kernel<WARP_NUM, 64, 64, 64, 64>
                <<<grid, block>>>(Q, K, V, O, softmax_scale, kv_seq_len, num_heads);
        } else if (seq_len <= 128) {
            flash_attention_kernel<WARP_NUM, 64, 64, 64, 128>
                <<<grid, block>>>(Q, K, V, O, softmax_scale, kv_seq_len, num_heads);
        } else if (seq_len <= 256) {
            flash_attention_kernel<WARP_NUM, 64, 64, 64, 256>
                <<<grid, block>>>(Q, K, V, O, softmax_scale, kv_seq_len, num_heads);
        } else if (seq_len <= 512) {
            flash_attention_kernel<WARP_NUM, 64, 64, 64, 512>
                <<<grid, block>>>(Q, K, V, O, softmax_scale, kv_seq_len, num_heads);
        } else {
            flash_attention_kernel<WARP_NUM, 64, 64, 64, 1024>
                <<<grid, block>>>(Q, K, V, O, softmax_scale, kv_seq_len, num_heads);
        }
    } else if constexpr (HEAD_DIM == 128) {
        if (seq_len <= 64) {
            flash_attention_kernel<WARP_NUM, 128, 32, 32, 64>
                <<<grid, block>>>(Q, K, V, O, softmax_scale, kv_seq_len, num_heads);
        } else if (seq_len <= 128) {
            flash_attention_kernel<WARP_NUM, 128, 32, 32, 128>
                <<<grid, block>>>(Q, K, V, O, softmax_scale, kv_seq_len, num_heads);
        } else {
            flash_attention_kernel<WARP_NUM, 128, 32, 32, 256>
                <<<grid, block>>>(Q, K, V, O, softmax_scale, kv_seq_len, num_heads);
        }
    } else {
        static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "Unsupported HEAD_DIM");
    }

    CUDA_KERNEL_CHECK();
}

// Wrapper for the reference float kernel (legacy)
void flash_attention_fwd(const float* Q, const float* K, const float* V,
                          float* O, float softmax_scale, int seqlen, int num_heads) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(seqlen / BLOCK_SIZE + 1, num_heads);
    int shared_size = 3 * BLOCK_SIZE * HEAD_DIM * sizeof(float);
    flash_attention_fwd_kernel<<<grid, block, shared_size>>>(Q, K, V, O, softmax_scale, seqlen, num_heads);
    CUDA_KERNEL_CHECK();
}
