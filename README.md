# ToyKernel

Common CUDA kernel implementations from scratch. 从零学习 CUDA 高性能算子实现。

## 🎯 Roadmap

### 🌱 Level 1: 基础算子（入门）

| Kernel | 类型 | Description | Status |
|--------|------|-------------|--------|
| Vector Add | Basic | 向量相加 | ⬜ |
| Vector Mul | Basic | 向量乘法 | ⬜ |
| GEMV | Basic | 矩阵-向量乘法 | ⬜ |
| Softmax | Basic | Softmax 计算 | ⬜ |
| Layernorm | Basic | 层归一化 | ⬜ |
| RMSNorm | Basic | RMS 归一化 | ⬜ |

### 🚀 Level 2: 核心算子（进阶）

| Kernel | 类型 | Description | Status |
|--------|------|-------------|--------|
| GEMM (Naive) | MatMul | 朴素矩阵乘法 | ⬜ |
| GEMM (Tiled) | MatMul | 分块矩阵乘法 | ⬜ |
| GEMM (Shared Mem) | MatMul | 共享内存优化 | ⬜ |
| GEMM (Tensor Core) | MatMul | Tensor Core 加速 | ⬜ |
| GeLU | Activation | 激活函数 | ⬜ |
| RoPE | Position | 旋转位置编码 | ⬜ |

### 🔥 Level 3: 大模型核心（高阶）

| Kernel | 类型 | Description | Status |
|--------|------|-------------|--------|
| FlashAttention-2 | Attention | 前向传播 | ⬜ |
| FlashAttention-2 (BW) | Attention | 反向传播 | ⬜ |
| PagedAttention | Memory | vLLM 显存优化 | ⬜ |
| Medusa | Decoding | 多头并行解码 | ⬜ |

### ⚡ Level 4: 量化加速（精通）

| Kernel | 类型 | Description | Status |
|--------|------|-------------|--------|
| W8A16 Quant | Quantization | INT8 权重量化 | ⬜ |
| W8A16 GEMM | Quantization | INT8 量化乘法 | ⬜ |
| W4A16 Quant | Quantization | INT4 权重量化 | ⬜ |
| W4A16 GEMM | Quantization | INT4 量化乘法 | ⬜ |
| SmoothQuant | Quantization | 激活平滑量化 | ⬜ |
| AWQ Quant | Quantization | 激活感知量化 | ⬜ |

---

## 📁 Project Structure

```
toykernel/
├── README.md
├── src/
│   ├── level1/            # 基础算子
│   │   ├── vector_ops.cu
│   │   ├── gemv.cu
│   │   ├── softmax.cu
│   │   └── norm.cu
│   ├── level2/           # 核心算子
│   │   ├── gemm/
│   │   │   ├── naive.cu
│   │   │   ├── tiled.cu
│   │   │   ├── shared.cu
│   │   │   └── tensor_core.cu
│   │   ├── activation.cu
│   │   └── rope.cu
│   ├── level3/           # 大模型核心
│   │   ├── flash_attention.cu
│   │   └── paged_attention.cu
│   └── level4/           # 量化
│       ├── quantize.cu
│       ├── dequantize.cu
│       └── quantized_gemm.cu
├── include/
│   └── utils.cuh
├── tests/
├── benchmarks/
└── scripts/
```

## 🚀 Quick Start

### 环境要求

- CUDA Toolkit 12.x+
- CMake 3.18+
- GCC 11+
- NVIDIA GPU (sm_80+ for Tensor Cores)

### 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j$(nproc)
```

### 运行测试

```bash
./tests/basic_test
./benchmarks/gemm_benchmark
```

## 📚 Reference

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashDecoding Paper](https://arxiv.org/abs/2309.06169)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
- [SmoothQuant](https://arxiv.org/abs/2308.15026)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [GGML](https://github.com/ggerganov/ggml)

## 🤝 Contributing

1. Fork this repo
2. Create your feature branch (`git checkout -b feature/xxx`)
3. Commit with proper template (`git commit` will auto-use template)
4. Push to branch
5. Open a Pull Request

## 📝 License

MIT License

---

*Learning by doing. 纸上得来终觉浅，绝知此事要躬行。*
