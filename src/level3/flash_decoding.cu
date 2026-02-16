/**
 * @file flash_decoding.cu
 * @brief FlashDecoding implementation for LLM inference
 *
 * FlashDecoding optimizes the decoding phase of autoregressive LLM inference:
 * - Split KV cache into smaller chunks for parallel processing
 * - Uses max-scoring token to identify relevant chunks
 * - Achieves near-linear speedup with longer sequences
 *
 * O = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * Copyright (c) 2024 ToyKernel Contributors
 * This file is for learning purposes only.
 * Unauthorized copying, distribution, or use is strictly prohibited.
 */

#include "../include/utils.cuh"


// Q : [heads, head_dim, 1]         : float32
// K : [heads, head_dim / 4, seq, 4] : float32    // borrow 4 from seqs
// V : [heads, head_dim / 4, seq, 4]
// O : [heads, head_dim, 1]
template<int HEAD_DIM, int THREAD_NUM>
__global__ void flash_decoding_gemv(
    const float* Q, const float* K, const float* V, float* O, float unified_max,
    const int q_head_stride, const int kv_head_stride, const int seq_len
){
    int tid = threadIdx.x;
    int thread_num = blockDim.x;
    int warp_num = thread_num / WARP_SIZE;
    int wid = WARP_ID();
    int lid = LANE_ID();

    int head_id = blockIdx.x;

    int token_per_thread = (seq_len + thread_num - 1) / thread_num;

    __shared__ float sQ[HEAD_DIM];
    __shared__ float sL[THREAD_NUM];

    __shared__ float sO[HEAD_DIM];
    __shared__ float sMax[HEAD_DIM];

    // Each head's offset
    Q = Q + head_id * q_head_stride;
    K = K + head_id * kv_head_stride;
    V = V + head_id * kv_head_stride;

    // memory bodun issue
    // __shared__ float sK[2][THREAD_NUM * 4]; // [2, ]. // buffering head_dim 

    // cooperative copy to Q (Q is [heads, head_dim, 1], stride = q_head_stride)
    for(int i = 4 * tid; i < HEAD_DIM; i += thread_num * 4){
        if(i + 3 < HEAD_DIM){
            // vec4 copy
            FLOAT4(sQ[i]) = CFLOAT4(Q[i]);
        }else{
            // copy by element
            #pragma unroll
            for(int j = 0; j < 4; j++){
                if(i + j < HEAD_DIM){
                    sQ[i + j] = Q[i + j];
                }
            }
        }
    }
    __syncthreads();

    float max_val = unified_max;
    
    // warp是按照seq分的，思路没问题
    // 问题是S=QK 也是seq，这个seq做reduce会很麻烦
    float l = 1.0f;
    
    for(int token = tid; token < seq_len; token += thread_num){
        float o = 0.0f;
        float kq = 0.0f;

        for(int kk = 0; kk < HEAD_DIM / 4; kk++){
            int col_offset = token * 4;
            float4 k_local = CFLOAT4(K[OFFSET2D(kk,col_offset,seq_len * 4)]);
            float4 q_local = FLOAT4(sQ[4 * kk]);
            kq +=  k_local.x * q_local.x 
                        + k_local.y * q_local.y
                        + k_local.z * q_local.z
                        + k_local.w * q_local.w;
        }
        
        float safe_kq = kq - unified_max;
        int pred = (safe_kq > 88.0f) ? 1 : 0;
        // warp 内任意线程的 pred 为真，则返回非零
        int need_recalculate = __any_sync(0xffffffff, pred);

        // thread diverge
        if(need_recalculate){
            // illegal value, need to recalucalte softmax 
            // update old one

            float new_unified_max = warp_reduce_max<float>(kq) - 88.0f;
            // update old
            safe_kq = kq - new_unified_max;
            l = l * __expf(unified_max - new_unified_max) + __expf(safe_kq);
            o = o * __expf(unified_max - new_unified_max);
            unified_max = new_unified_max;

        }else{
            l = l + __expf(safe_kq);
        }
        __syncthreads();

        for(int kk = 0; kk < HEAD_DIM / 4; kk++){
            int col_offset = token * 4;
            float4 v_local = CFLOAT4(V[OFFSET2D(kk,col_offset,seq_len * 4)]);
            o +=  (v_local.x + v_local.y + v_local.z + v_local.w) * __expf(safe_kq); 
        }

        sO[token] = o;
        sMax[token] = unified_max;
        __syncthreads();
    }

    // 当前的o是每个token的o，但是还没有除l
    // 当前的l是每个thread的和
    // 每个unifiedmax是每个thread的和
    // 现在需要计算一个整体的l，并且更新每个thread的max，
    // reduce max
    max_val = thread_block_reduce_max(unified_max);
    // update l
    l = l * __expf(unified_max - max_val);
    // reduce l
    l = thread_block_reduce_add(l);
    
    for(int token = tid; token < seq_len; token += thread_num){
        flaot scaler = __expf(sMax - max_val);
        O[token] = sO[token] * scaler / l;
    }
}


// input grid : (head_size, chunk_size)
/**
 * @brief FlashDecoding: Compute partial QK^T for all KV chunks
 * Each thread block processes one query token against multiple KV chunks
 * @param Q Query [num_heads, 1, head_dim] (single token)
 * @param K Key cache [num_heads, num_kv_chunks, chunk_size, head_dim]
 * @param softmax_scale Scaling factor
 * @param partial_max Partial max scores [num_heads, num_kv_chunks]
 * @param partial_sum Partial sum of exp [num_heads, num_kv_chunks]
 * @param num_heads Number of heads
 * @param num_kv_chunks Number of KV cache chunks
 * @param chunk_size Size of each chunk
 */
template<int DECODING_BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_decoding_partial_kernel(
    const float* Q, const float* K, float softmax_scale,
    float* partial_max, float* partial_sum,
    int num_heads, int num_kv_chunks, int chunk_size
) {
    int head_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;

    if (head_id >= num_heads || chunk_id >= num_kv_chunks) return;

    // Each thread computes partial QK for a subset of head_dim
    float partial_qk = 0.0f;
    for (int d = tid; d < HEAD_DIM; d += DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE) {
        partial_qk += Q[head_id * HEAD_DIM + d] * K[chunk_id * chunk_size * HEAD_DIM + d];
    }

    // Warp-level reduction for QK value
    for (int offset = DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        partial_qk += __shfl_down_sync(WARP_MASK, partial_qk, offset);
    }

    // Thread 0 writes the QK value
    if (tid == 0) {
        float qk = partial_qk * softmax_scale;
        float exp_qk = expf(qk);

        // Store partial results (will be reduced in next phase)
        int idx = head_id * num_kv_chunks + chunk_id;
        partial_max[idx] = qk;
        partial_sum[idx] = exp_qk;
    }
}

/**
 * @brief FlashDecoding: Reduce partial results and compute final attention
 * @param partial_max Partial max scores
 * @param partial_sum Partial sum of exp
 * @param V Value cache
 * @param O Output
 * @param num_heads, num_kv_chunks, chunk_size, head_dim
 */
__global__ void flash_decoding_reduce_kernel(
    const float* partial_max, const float* partial_sum,
    const float* V, float* O,
    int num_heads, int num_kv_chunks, int chunk_size
) {
    int head_id = blockIdx.x;
    int tid = threadIdx.x;

    if (head_id >= num_heads) return;

    // Compute global max across all chunks
    float global_max = -INFINITY;
    for (int c = 0; c < num_kv_chunks; ++c) {
        int idx = head_id * num_kv_chunks + c;
        global_max = fmaxf(global_max, partial_max[idx]);
    }

    // Compute global sum
    float global_sum = 0.0f;
    for (int c = 0; c < num_kv_chunks; ++c) {
        int idx = head_id * num_kv_chunks + c;
        global_sum += partial_sum[idx] * expf(partial_max[idx] - global_max);
    }

    // Compute weighted sum of V
    float output_val = 0.0f;
    for (int c = 0; c < num_kv_chunks; ++c) {
        int idx = head_id * num_kv_chunks + c;
        float weight = partial_sum[idx] * expf(partial_max[idx] - global_max) / global_sum;

        // Load V for this chunk
        for (int i = tid; i < chunk_size; i += DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE) {
            int v_idx = c * chunk_size * HEAD_DIM + i * HEAD_DIM + (tid % HEAD_DIM);
            output_val += weight * V[v_idx];
        }
    }

    // Warp reduction
    for (int offset = DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        output_val += __shfl_down_sync(WARP_MASK, output_val, offset);
    }

    if (tid == 0) {
        O[head_id * HEAD_DIM] = output_val;
    }
}

/**
 * @brief FlashDecoding wrapper
 */
void flash_decoding(const float* Q, const float* K, const float* V, float* O,
                    float softmax_scale, int num_heads, int num_kv_chunks, int chunk_size) {
    // Allocate partial results
    float *partial_max, *partial_sum;
    size_t partial_size = num_heads * num_kv_chunks * sizeof(float);
    CUDA_CHECK(cudaMalloc(&partial_max, partial_size));
    CUDA_CHECK(cudaMalloc(&partial_sum, partial_size));

    // Phase 1: Compute partial QK^T
    dim3 block1(DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE);
    dim3 grid1(num_heads, num_kv_chunks);
    flash_decoding_partial_kernel<<<grid1, block1>>>(
        Q, K, softmax_scale, partial_max, partial_sum, num_heads, num_kv_chunks, chunk_size
    );
    CUDA_KERNEL_CHECK();

    // Phase 2: Reduce and compute final attention
    dim3 block2(DECODING_BLOCK_SIZE * DECODING_BLOCK_SIZE);
    dim3 grid2(num_heads);
    flash_decoding_reduce_kernel<<<grid2, block2>>>(
        partial_max, partial_sum, V, O, num_heads, num_kv_chunks, chunk_size
    );
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaFree(partial_max));
    CUDA_CHECK(cudaFree(partial_sum));
}
