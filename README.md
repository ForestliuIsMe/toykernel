# ToyKernel

CUDA kernel implementations from scratch. 从零学习 CUDA 高性能算子实现。

## 🎯 Roadmap

### Level 1: 基础算子

| Status | Kernel | Description |
|--------|--------|-------------|
| ⬜ | Vector Add | 向量相加 |
| ⬜ | Vector Mul | 向量乘法 |
| ⬜ | Vector Scale | 向量缩放 |
| ⬜ | GEMV | 矩阵-向量乘法 |
| ⬜ | Softmax | Softmax 计算 |
| ⬜ | Softmax (Warp) | Warp 级 Softmax |
| ⬜ | Layernorm | 层归一化 |
| ⬜ | RMSNorm | RMS 归一化 |
| ⬜ | GeLU | 激活函数 |
| ⬜ | Swish | 激活函数 |

### Level 2: GEMM

| Status | Kernel | Description |
|--------|--------|-------------|
| ⬜ | GEMV | 矩阵-向量乘法 |
| ⬜ | Naive GEMM | 朴素矩阵乘法 |
| ⬜ | Sliced-K | K 维度分片优化 |
| ⬜ | Split-K | 跨块并行 |
| ⬜ | Persistent | 常驻线程模式 |

### Level 3: 大模型核心

| Status | Kernel | Description |
|--------|--------|-------------|
| ⬜ | FlashAttention-2 | 前向传播 |
| ⬜ | FlashDecoding | 推理解码 |
| ⬜ | RoPE | 旋转位置编码 |
| ⬜ | PagedAttention | vLLM 显存优化 |
| ⬜ | Sparse GEMM | 稀疏矩阵乘法 |
| ⬜ | Medusa | 多头并行解码 |

### Level 4: 量化

| Status | Kernel | Description |
|--------|--------|-------------|
| ⬜ | W8A16 Quant | INT8 权重量化 |
| ⬜ | W4A16 Quant | INT4 权重量化 |
| ⬜ | W8A16 GEMM | INT8 量化乘法 |
| ⬜ | W4A16 GEMM | INT4 量化乘法 |
| ⬜ | Quantize | 量化 kernel |
| ⬜ | Quantized GEMM | 量化矩阵乘法 |
| ⬜ | SmoothQuant | 激活平滑量化 |
| ⬜ | AWQ | 激活感知量化 |

---

## 📁 Structure

```
toykernel/
├── CMakeLists.txt
├── README.md
├── requirements.txt
├── scripts/
│   ├── build.sh
│   ├── test.sh
│   ├── benchmark.sh
│   └── clean.sh
├── include/
│   └── utils.cuh
└── src/
    ├── level1/              # 基础算子
    │   ├── vector_ops.cu
    │   ├── gemv.cu
    │   ├── softmax.cu
    │   ├── norm.cu
    │   └── activation.cu
    ├── level2/              # GEMM
    │   ├── gemv.cu
    │   ├── gemm.cu          # Naive
    │   ├── sliced_k.cu
    │   ├── split_k.cu
    │   └── persistent.cu
    ├── level3/              # LLM 核心
    │   ├── flash_attention.cu
    │   ├── flash_decoding.cu
    │   ├── rope.cu
    │   ├── paged_attention.cu
    │   ├── sparse_gemm.cu
    │   └── decoding.cu
    └── level4/              # 量化
        ├── weight_quant/
        │   └── w8a16_gemm.cu
        │   └── w4a16_gemm.cu
        ├── quantized_ops/
        │   ├── quantize.cu
        │   └── quantized_gemm.cu
        └── activation_quant/
            ├── smooth_quant.cu
            └── awq.cu
```

## 🚀 Build

```bash
./scripts/build.sh
```

## 📚 Ref

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashDecoding](https://arxiv.org/abs/2309.06169)
- [vLLM](https://github.com/vllm-project/vllm)
- [SmoothQuant](https://arxiv.org/abs/2308.15026)
- [AWQ](https://arxiv.org/abs/2306.00978)

---

*Learning by doing.*
