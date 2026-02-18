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
// V : [heads, head_dim , seq]
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

    extern __shared__ float sS[]; // extern seq_len

    __shared__ bool need_recalculate;

    if(tid == 0){
        need_recalculate = false;
    }
    __syncthreads();


    // Each head's offset
    Q = Q + head_id * q_head_stride;
    K = K + head_id * kv_head_stride;
    V = V + head_id * kv_head_stride;
    O = O + head_id * q_head_stride;

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
    float max_kq = unified_max;

    for(int token = tid; token < (seq_len + thread_num - 1) / thread_num; token += thread_num){
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
        int pred = (safe_kq > 88.0f) ? 1 : 0;   // overflow
        if(pred){
            need_recalculate = true;
            max_kq = kq - 88.0f;
        }

        sS[token] = safe_kq;
        __syncthreads();
    }

    // reduce max 
    if(need_recalculate){
        max_kq = thread_block_reduce_max<float>(max_kq);
    }
    __syncthreads();
    
    float softmax_diff_val = unified_max - max_kq;

    float l = 0.0f;
    for(int token = tid; token < seq_len; token += thread_num){
        sS[token] = __expf(sS[token] - softmax_diff_val);
        l += sS[token];
    }

    l = thread_block_reduce_add<float>(l);

    for(int token = tid; token < seq_len; token += thread_num){
        float s = sS[token] / l;
        for(int kk = 0; kk < HEAD_DIM; kk++){
            // no atomic issue, just write
            O[kk] += thread_block_reduce_add<float>(s * V[OFFSET2D(kk,token,seq_len)]);
        }
    }

    for(int kk = 0; kk < HEAD_DIM; kk++){
        float val = 0.0f;
        for(int token = tid; token < seq_len; token+=thread_num){
            float s = sS[token] / l;
            val += thread_block_reduce_add<float>(s * V[OFFSET2D(kk,token,seq_len)]);
        }
        // write back to kk
        if(tid == 0){
            O[kk] = val;
        }
        __syncthreads();
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
