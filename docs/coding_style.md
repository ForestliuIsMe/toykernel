# ToyKernel Coding Style

## Naming Conventions

### Files
- `.cu` for CUDA kernel implementations
- `.cuh` for CUDA header files
- Lowercase with underscores: `flash_attention_fwd.cu`

### Functions
- PascalCase for device kernels: `__global__ void FlashAttentionFwd`
- camelCase for helper functions: `load_tile_from_gmem`

### Variables
- snake_case for local variables
- Prefixes: `d_` for device, `h_` for host

## Kernel Structure

```cuda
// 1. Kernel declaration
__global__ void kernel_name(type* out, const type* in, size_t n) {
    // 2. Index calculation
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // 3. Main computation
    // ...
}

// 4. Launcher function
void launch_kernel(type* out, const type* in, size_t n, cudaStream_t stream) {
    dim3 blocks((n + 255) / 256);
    dim3 threads(256);
    kernel_name<<<blocks, threads, 0, stream>>>(out, in, n);
}
```

## Memory Access
- Coalesced global memory access
- Shared memory for frequently accessed data
- Use vector types (float2, float4) when possible

## Warp-Level
- Use __shfl_sync for warp communication
- Prefer warp reducers over loop reducers
- Handle divergent warps carefully
