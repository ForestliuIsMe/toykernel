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

### 🚀 Level 2: GEMM 核心（进阶）

#### 2.1 Naive GEMM（入门）
| Kernel | 类型 | Description | Status |
|--------|------|-------------|--------|
| GEMM (Row-Major) | Naive | 朴素行优先矩阵乘法 | ⬜ |
| GEMM (Col-Major) | Naive | 朴素列优先矩阵乘法 | ⬜ |
| GEMV | Naive | 矩阵-向量乘法（GEMM 简化版） | ⬜ |

#### 2.2 Sliced-K（分片策略）
| Kernel | 类型 | Description | Status |
|--------|------|-------------|--------|
| GEMM (Sliced-K Basic) | Sliced-K | 按 K 维度分片，减少共享内存 | ⬜ |
| GEMM (Sliced-K Warp) | Sliced-K | Warp 级分片并行 | ⬜ |
| GEMM (Sliced-K TensorCore) | Sliced-K | Tensor Core + Sliced-K 混合 | ⬜ |

#### 2.3 Split-K（跨块并行）
| Kernel | 类型 | Description | Status |
|--------|------|-------------|--------|
| GEMM (Split-K Basic) | Split-K | K 维度跨线程块并行 | ⬜ |
| GEMM (Split-K Reduce) | Split-K | Split-K + 跨块归约 | ⬜ |
| GEMM (Split-K Async) | Split-K | 异步执行优化 | ⬜ |

#### 2.4 Persistent（常驻线程）
| Kernel | 类型 | Description | Status |
|--------|------|-------------|--------|
| GEMM (Persistent Basic) | Persistent | 线程常驻，批处理复用 | ⬜ |
| GEMM (Persistent Stream) | Persistent | 多流并发执行 | ⬜ |
| GEMM (Persistent TensorCore) | Persistent | Tensor Core + Persistent 模式 | ⬜ |

**GEMM 学习路线：**
```
Naive → Sliced-K → Split-K → Persistent
(理解原理) → (内存优化) → (并行扩展) → (极致性能)
```

### 🔥 Level 3: 大模型核心（高阶）

| Kernel | 类型 | Description | Status |
|--------|------|-------------|--------|
| FlashAttention-2 | Attention | 前向传播 | ⬜ |
| FlashAttention-2 (BW) | Attention | 反向传播 | ⬜ |
| PagedAttention | Memory | vLLM 显存优化 | ⬜ |
| Medusa | Decoding | 多头并行解码 | ⬜ |

### ⚡ Level 4: 量化加速（精通）

| Kernel | 类型 | Description | Status | 参考 |
|--------|------|-------------|--------|------|
| W8A16 Quant | Quantization | INT8 权重量化，FP16 计算 | ⬜ | AWQ, GPTQ |
| W8A16 GEMM | Quantization | INT8 量化矩阵乘法 | ⬜ | BitBLAS, TensorRT |
| W4A16 Quant | Quantization | INT4 权重量化，FP16 计算 | ⬜ | GGUF, AWQ |
| W4A16 GEMM | Quantization | INT4 量化矩阵乘法 | ⬜ | GGML, AWQ |
| W4A4 Quant | Quantization | INT4 权重 + INT4 激活 | ⬜ | QLoRA, GPTQ |
| SmoothQuant | Quantization | 激活平滑，迁移量化难度 | ⬜ | Microsoft |
| Dequantize | Quantization | 反量化 kernel | ⬜ | 通用 |
| KV Cache Quant | Quantization | KV cache INT8/INT4 量化 | ⬜ | vLLM, SqueezeLLM |

**量化精度对比：**
```
FP16 > W8A16 > W4A16 > W4A4
显存占用：1x > 0.5x > 0.25x > 0.125x
```

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
│   ├── level2/           # GEMM 核心
│   │   ├── gemm/
│   │   │   ├── naive/
│   │   │   │   ├── row_major.cu
│   │   │   │   └── col_major.cu
│   │   │   ├── sliced_k/
│   │   │   │   ├── basic.cu
│   │   │   │   ├── warp.cu
│   │   │   │   └── tensor_core.cu
│   │   │   ├── split_k/
│   │   │   │   ├── basic.cu
│   │   │   │   ├── reduce.cu
│   │   │   │   └── async.cu
│   │   │   └── persistent/
│   │   │       ├── basic.cu
│   │   │       ├── stream.cu
│   │   │       └── tensor_core.cu
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
- [CUTLASS GEMM](https://github.com/NVIDIA/cutlass)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashDecoding Paper](https://arxiv.org/abs/2309.06169)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
- [SmoothQuant](https://arxiv.org/abs/2308.15026)
- [AWQ](https://arxiv.org/abs/2306.00978)
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
