# Architecture Documentation

## Kernel Implementation Order

### Level 1: Basics (Start Here)
1. Vector operations (add, mul, scale)
2. GEMV (matrix-vector multiply)
3. Softmax (warp-level optimization)
4. Normalization (LayerNorm, RMSNorm)
5. Activation functions (GeLU, Swish)

### Level 2: GEMM (Core)
1. Naive GEMM (row-major, col-major)
2. Sliced-K optimizations
3. Split-K parallelization
4. Persistent kernel pattern

### Level 3: LLM Core
1. FlashAttention (forward/backward)
2. Position encoding (RoPE, ALiBi)
3. Memory optimization (PagedAttention, KV Cache)
4. Decoding (Medusa, Speculative)

### Level 4: Quantization
1. Weight quantization (W8A16, W4A16, W4A4)
2. Quantized GEMM operations
3. Activation quantization (SmoothQuant, AWQ)

## Code Style
- Follow CUDA C Programming Guide best practices
- Use constexpr for compile-time constants
- Prefer warp-level primitives over shfl
- Profile with nvprof/nsight systems
