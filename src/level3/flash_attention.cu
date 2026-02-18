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


/*
    Q: [batch, heads, head_dim, m]
    K: [batch, heads, head_dim, n]
    V: [batch, heads, head_dim, n]

    Br is the block size of Qi      integer multiply of 16
    Bc is the block size of Ki,Vi.  integer multiply of 16
    CHUNK_SIZE is total seq len of Qi
    CHUNK_SIZE / (WARP_NUM * Br)   ==> seq dimention iteration number
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

    __shared__ half Ki[Bc * HEAD_DIM];
    __shared__ half Vi[Bc * HEAD_DIM];
    __shared__ half Qj[WARP_NUM][Br * HEAD_DIM];
    __shared__ half Oj[WARP_NUM][Br * HEAD_DIM];
    
    __shared__ half Si[WARP_NUM][Br * Bc];
    __shared__ half RowAjduster[WARP_NUM][Br];
    __shared__ half RowLi[WARP_NUM][Br];

    const int rows_for_each_thread = Br / 32;

    for(int j = warp_id * Br; j < CHUNK_SIZE; j += WARP_NUM * Br){
        // TODO: copy Qj first
        // TODO: initialize Oj to 0
        // Qj copy complete
        __syncthreads();

        // Online softmax state (persistent across K/V blocks)
        half m_prev[rows_for_each_thread];
        half l_prev[rows_for_each_thread];
        #pragma unroll
        for (int i = 0; i < rows_for_each_thread; i++) {
            m_prev[i] = -INFINITY;
            l_prev[i] = 0.0f;
        }

        // inner loop
        for(int i = 0; i < kv_seq_len; i += Bc){
        
            // calculate S = QK
            for(int m = 0; m < Br ; m += 16){
                for(int k = 0; k < HEAD_DIM; k+= 16){
                    // copy Ki
                    // for each sub k
                    for(int sub_mma = 0; sub_mma < Bc; sub_mma += 8){
                        // calculate mma and store data into Si
                    }
                }
            }

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

            // update O = O + SV (MMA P @ V)
            // Si now contains P[Br, Bc], Vi contains V[Bc, HEAD_DIM]
            for(int m = 0; m < Br ; m += 16){
                for(int k = 0; k < Bc; k+= 16){
                    for(int sub_mma = 0; sub_mma < HEAD_DIM; sub_mma += 8){
                        // calculate mma: Oj += Si @ Vi
                        // TODO: implement MMA here
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

        // TODO: Write Oj back to global memory O
        __syncthreads();
    }
}

/**
 * @brief FlashAttention-2 forward kernel
 * @param Q Query tensor [Batch, Num_heads, Seq_len, Head_dim]
 * @param K Key tensor
 * @param V Value tensor
 * @param O Output tensor
 * @param softmax_scale = 1/sqrt(d)
 * @param seqlen Sequence length
 * @param num_heads Number of attention heads
 */
__global__ void flash_attention_fwd_kernel(
    const float* Q, const float* K, const float* V, float* O,
    float softmax_scale, int seqlen, int num_heads
) {
    int bid = blockIdx.x;  // Block index along sequence
    int tid = threadIdx.x; // Thread index within block
    int head_id = blockIdx.y;

    // Shared memory for QK block and V block
    extern __shared__ float smem[];
    float* Qi = smem;                    // [BLOCK_SIZE, HEAD_DIM]
    float* Kj = smem + BLOCK_SIZE * HEAD_DIM;    // [BLOCK_SIZE, HEAD_DIM]
    float* Vi = smem + 2 * BLOCK_SIZE * HEAD_DIM; // [BLOCK_SIZE, HEAD_DIM]

    int qo_len = seqlen;

    // Thread-local softmax statistics
    float row_m = -INFINITY;  // max
    float row_l = 0.0f;      // sum of exp
    float row_o[HEAD_DIM];   // output accumulator

    for (int i = 0; i < HEAD_DIM; ++i) {
        row_o[i] = 0.0f;
    }

    // Load Q tile
    int q_row = bid * BLOCK_SIZE + tid;
    if (q_row < qo_len) {
        int q_offset = head_id * qo_len * HEAD_DIM + q_row * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d += BLOCK_SIZE) {
            int local_d = tid + d;
            if (local_d < HEAD_DIM) {
                Qi[tid * HEAD_DIM + local_d] = Q[q_offset + local_d];
            }
        }
    } else {
        for (int d = 0; d < HEAD_DIM; ++d) {
            Qi[tid * HEAD_DIM + d] = 0.0f;
        }
    }
    __syncthreads();

    // Loop over K/V blocks
    for (int block_j = 0; block_j < seqlen; block_j += BLOCK_SIZE) {
        // Load K tile
        int k_row = block_j + tid;
        if (k_row < seqlen) {
            int k_offset = head_id * seqlen * HEAD_DIM + k_row * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d += BLOCK_SIZE) {
                int local_d = tid + d;
                if (local_d < HEAD_DIM) {
                    Kj[tid * HEAD_DIM + local_d] = K[k_offset + local_d];
                    Vi[tid * HEAD_DIM + local_d] = V[k_offset + local_d];
                }
            }
        } else {
            for (int d = 0; d < HEAD_DIM; ++d) {
                Kj[tid * HEAD_DIM + d] = 0.0f;
                Vi[tid * HEAD_DIM + d] = 0.0f;
            }
        }
        __syncthreads();

        // Compute Q @ K^T for this block
        // Each thread computes dot product of Q[i] with K[j]
        float qk[BLOCK_SIZE];  // Per-thread temporary storage

        for (int j = 0; j < BLOCK_SIZE && (block_j + j) < seqlen; ++j) {
            float sum = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                sum += Qi[tid * HEAD_DIM + d] * Kj[j * HEAD_DIM + d];
            }
            qk[j] = sum * softmax_scale;
        }

        // Online softmax
        float row_m_new = row_m;
        for (int j = 0; j < BLOCK_SIZE && (block_j + j) < seqlen; ++j) {
            row_m_new = fmaxf(row_m_new, qk[j]);
        }

        float row_scale = expf(row_m - row_m_new);
        float row_l_new = 0.0f;
        for (int j = 0; j < BLOCK_SIZE && (block_j + j) < seqlen; ++j) {
            qk[j] = expf(qk[j] - row_m_new);
            row_l_new += qk[j];
        }

        // Rescale output
        for (int d = 0; d < HEAD_DIM; ++d) {
            row_o[d] = row_o[d] * row_scale;
        }

        // Accumulate O = O * scale + softmax(QK) @ V
        for (int j = 0; j < BLOCK_SIZE && (block_j + j) < seqlen; ++j) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                row_o[d] += qk[j] * Vi[j * HEAD_DIM + d];
            }
        }

        row_m = row_m_new;
        row_l = row_l * row_scale + row_l_new;

        __syncthreads();
    }

    // Normalize
    float row_l_inv = 1.0f / row_l;
    for (int d = 0; d < HEAD_DIM; ++d) {
        row_o[d] *= row_l_inv;
    }

    // Write output
    if (q_row < qo_len) {
        int o_offset = head_id * qo_len * HEAD_DIM + q_row * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d += BLOCK_SIZE) {
            int local_d = tid + d;
            if (local_d < HEAD_DIM) {
                O[o_offset + local_d] = row_o[local_d];
            }
        }
    }
}

// Wrapper
void flash_attention_fwd(const float* Q, const float* K, const float* V,
                          float* O, float softmax_scale, int seqlen, int num_heads) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(seqlen / BLOCK_SIZE + 1, num_heads);
    int shared_size = 3 * BLOCK_SIZE * HEAD_DIM * sizeof(float);
    flash_attention_fwd_kernel<<<grid, block, shared_size>>>(Q, K, V, O, softmax_scale, seqlen, num_heads);
    CUDA_KERNEL_CHECK();
}
