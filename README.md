# ToyKernel

Common CUDA kernel implementations from scratch. 从零学习 CUDA 高性能算子实现。

## 🎯 Roadmap

### 🌱 Level 1: 基础算子（入门）

| Status | Kernel | 类型 | Description |
|--------|--------|------|-------------|
| ⬜ | Vector Add | Basic | 向量相加 |
| ⬜ | Vector Mul | Basic | 向量乘法 |
| ⬜ | Vector Scale | Basic | 向量缩放 |
| ⬜ | GEMV | Basic | 矩阵-向量乘法 |
| ⬜ | Softmax | Basic | Softmax 计算 |
| ⬜ | Softmax (Warp) | Basic | Warp 级 Softmax 优化 |
| ⬜ | Layernorm | Basic | 层归一化 |
| ⬜ | RMSNorm | Basic | RMS 归一化 |
| ⬜ | GeLU | Activation | GeLU 激活函数 |
| ⬜ | Swish | Activation | Swish 激活函数 |

### 🚀 Level 2: GEMM 核心（进阶）

#### 2.1 Naive GEMM
| Status | Kernel | 类型 | Description |
|--------|--------|------|-------------|
| ⬜ | GEMM (Row-Major) | Naive | 朴素行优先矩阵乘法 |
| ⬜ | GEMM (Col-Major) | Naive | 朴素列优先矩阵乘法 |
| ⬜ | GEMV | Naive | 矩阵-向量乘法 |

#### 2.2 Sliced-K（分片优化）
| Status | Kernel | 类型 | Description |
|--------|--------|------|-------------|
| ⬜ | GEMM (Sliced-K Basic) | Sliced-K | 按 K 维度分片 |
| ⬜ | GEMM (Sliced-K Warp) | Sliced-K | Warp 级分片并行 |
| ⬜ | GEMM (Sliced-K TensorCore) | Sliced-K | Tensor Core + Sliced-K |

#### 2.3 Split-K（跨块并行）
| Status | Kernel | 类型 | Description |
|--------|--------|------|-------------|
| ⬜ | GEMM (Split-K Basic) | Split-K | K 维度跨线程块并行 |
| ⬜ | GEMM (Split-K Reduce) | Split-K | Split-K + 跨块归约 |
| ⬜ | GEMM (Split-K Async) | Split-K | 异步执行优化 |

#### 2.4 Persistent（常驻线程）
| Status | Kernel | 类型 | Description |
|--------|--------|------|-------------|
| ⬜ | GEMM (Persistent Basic) | Persistent | 线程常驻，批处理复用 |
| ⬜ | GEMM (Persistent Stream) | Persistent | 多流并发执行 |
| ⬜ | GEMM (Persistent TensorCore) | Persistent | Tensor Core + Persistent |

**GEMM 学习路线：**
```
Naive → Sliced-K → Split-K → Persistent
(理解原理) → (内存优化) → (并行扩展) → (极致性能)
```

### 🔥 Level 3: 大模型核心（高阶）

#### 3.1 Attention 系列
| Status | Kernel | 类型 | Description |
|--------|--------|------|-------------|
| ⬜ | FlashAttention-2 | Attention | 前向传播 |
| ⬜ | FlashAttention-2 BW | Attention | 反向传播 |
| ⬜ | FlashDecoding | Attention | 推理解码优化 |
| ⬜ | FlashDecoding BW | Attention | 反向传播 |

#### 3.2 位置编码与归一化
| Status | Kernel | 类型 | Description |
|--------|--------|------|-------------|
| ⬜ | RoPE | Position | 旋转位置编码 |
| ⬜ | RoPE (Indexed) | Position | 索引优化版本 |
| ⬜ | ALiBi | Position | 线性偏置注意力 |

#### 3.3 推理优化
| Status | Kernel | 类型 | Description |
|--------|--------|------|-------------|
| ⬜ | PagedAttention | Memory | vLLM 显存优化 |
| ⬜ | KV Cache Quant | Memory | KV cache INT8/INT4 量化 |
| ⬜ | Medusa | Decoding | 多头并行解码 |
| ⬜ | Speculative Draft | Decoding | 推测解码草稿 |
| ⬜ | H2O Eviction | Memory | Heavy-Hitter  eviction |

### ⚡ Level 4: 量化加速（精通）

#### 4.1 权重量化
| Status | Kernel | 类型 | Description | 参考 |
|--------|--------|------|-------------|------|
| ⬜ | W8A16 Quant | Quantization | FP32 → INT8 量化 | AWQ, GPTQ |
| ⬜ | W4A16 Quant | Quantization | FP32 → INT4 量化 | GGUF, AWQ |
| ⬜ | W4A4 Quant | Quantization | INT4 权重 + INT4 激活 | QLoRA |
| ⬜ | GPTQ | Quantization | 逐层 GPTQ 量化 | GPTQ |

#### 4.2 量化算子
| Status | Kernel | 类型 | Description | 参考 |
|--------|--------|------|-------------|------|
| ⬜ | W8A16 GEMM | Quantization | INT8 量化矩阵乘法 | BitBLAS |
| ⬜ | W4A16 GEMM | Quantization | INT4 量化矩阵乘法 | GGML |
| ⬜ | W4A4 GEMM | Quantization | INT4×INT4 矩阵乘法 | QLoRA |
| ⬜ | Dequantize | Quantization | 反量化 kernel | 通用 |
| ⬜ | Quantize | Quantization | 量化 kernel | 通用 |

#### 4.3 激活量化
| Status | Kernel | 类型 | Description | 参考 |
|--------|--------|------|-------------|------|
| ⬜ | SmoothQuant | Quantization | 激活平滑量化 | Microsoft |
| ⬜ | AWQ Quant | Quantization | 激活感知权重量化 | AWQ |
| ⬜ | Static Quant | Quantization | 静态逐通道量化 | TensorRT |
| ⬜ | Dynamic Quant | Quantization | 动态逐 token 量化 | 通用 |

**量化精度对比：**
```
FP16 > W8A16 > W4A16 > W4A4
显存：1x    > 0.5x  > 0.25x > 0.125x
```

---

## 📁 Project Structure

```
toykernel/
├── README.md
├── LICENSE
├── .gitmessage
├── CMakeLists.txt
├── Makefile
├── requirements.txt
├── src/
│   ├── level1/
│   │   ├── vector_ops/
│   │   │   ├── vector_add.cu
│   │   │   ├── vector_mul.cu
│   │   │   ├── vector_scale.cu
│   │   │   └── CMakeLists.txt
│   │   ├── gemv/
│   │   │   ├── gemv.cu
│   │   │   └── CMakeLists.txt
│   │   ├── softmax/
│   │   │   ├── softmax.cu
│   │   │   ├── softmax_warp.cu
│   │   │   └── CMakeLists.txt
│   │   ├── norm/
│   │   │   ├── layernorm.cu
│   │   │   ├── rmsnorm.cu
│   │   │   └── CMakeLists.txt
│   │   ├── activation/
│   │   │   ├── gelu.cu
│   │   │   ├── swish.cu
│   │   │   └── CMakeLists.txt
│   │   └── CMakeLists.txt
│   ├── level2/
│   │   ├── gemm/
│   │   │   ├── naive/
│   │   │   │   ├── row_major.cu
│   │   │   │   ├── col_major.cu
│   │   │   │   └── CMakeLists.txt
│   │   │   ├── sliced_k/
│   │   │   │   ├── basic.cu
│   │   │   │   ├── warp.cu
│   │   │   │   ├── tensor_core.cu
│   │   │   │   └── CMakeLists.txt
│   │   │   ├── split_k/
│   │   │   │   ├── basic.cu
│   │   │   │   ├── reduce.cu
│   │   │   │   ├── async.cu
│   │   │   │   └── CMakeLists.txt
│   │   │   ├── persistent/
│   │   │   │   ├── basic.cu
│   │   │   │   ├── stream.cu
│   │   │   │   ├── tensor_core.cu
│   │   │   │   └── CMakeLists.txt
│   │   │   ├── common/
│   │   │   │   ├── gemm_common.cuh
│   │   │   │   ├── tile_config.cuh
│   │   │   │   └── CMakeLists.txt
│   │   │   └── CMakeLists.txt
│   │   └── CMakeLists.txt
│   ├── level3/
│   │   ├── attention/
│   │   │   ├── flash_attention_fwd.cu
│   │   │   ├── flash_attention_bwd.cu
│   │   │   ├── flash_decoding.cu
│   │   │   ├── flash_decoding_bwd.cu
│   │   │   └── CMakeLists.txt
│   │   ├── position/
│   │   │   ├── rope.cu
│   │   │   ├── rope_indexed.cu
│   │   │   ├── alibi.cu
│   │   │   └── CMakeLists.txt
│   │   ├── memory/
│   │   │   ├── paged_attention.cu
│   │   │   ├── kv_quant.cu
│   │   │   ├── h2o_eviction.cu
│   │   │   └── CMakeLists.txt
│   │   ├── decoding/
│   │   │   ├── medusa.cu
│   │   │   ├── speculative_draft.cu
│   │   │   └── CMakeLists.txt
│   │   └── CMakeLists.txt
│   ├── level4/
│   │   ├── weight_quant/
│   │   │   ├── w8a16_quant.cu
│   │   │   ├── w4a16_quant.cu
│   │   │   ├── w4a4_quant.cu
│   │   │   ├── gptq.cu
│   │   │   └── CMakeLists.txt
│   │   ├── quantized_ops/
│   │   │   ├── w8a16_gemm.cu
│   │   │   ├── w4a16_gemm.cu
│   │   │   ├── w4a4_gemm.cu
│   │   │   ├── dequantize.cu
│   │   │   ├── quantize.cu
│   │   │   └── CMakeLists.txt
│   │   ├── activation_quant/
│   │   │   ├── smooth_quant.cu
│   │   │   ├── awq_quant.cu
│   │   │   ├── static_quant.cu
│   │   │   ├── dynamic_quant.cu
│   │   │   └── CMakeLists.txt
│   │   ├── common/
│   │   │   ├── quant_common.cuh
│   │   │   ├── scales.cuh
│   │   │   └── CMakeLists.txt
│   │   └── CMakeLists.txt
│   └── CMakeLists.txt
├── include/
│   ├── utils/
│   │   ├── math.cuh
│   │   ├── type.cuh
│   │   ├── tensor.cuh
│   │   └── CMakeLists.txt
│   └── CMakeLists.txt
├── tests/
│   ├── level1/
│   │   ├── test_vector_ops.cu
│   │   ├── test_gemv.cu
│   │   ├── test_softmax.cu
│   │   ├── test_norm.cu
│   │   └── test_activation.cu
│   ├── level2/
│   │   ├── test_gemm_naive.cu
│   │   ├── test_gemm_sliced_k.cu
│   │   ├── test_gemm_split_k.cu
│   │   └── test_gemm_persistent.cu
│   ├── level3/
│   │   ├── test_flash_attention.cu
│   │   ├── test_rope.cu
│   │   └── test_paged_attention.cu
│   ├── level4/
│   │   ├── test_quantization.cu
│   │   └── test_quantized_gemm.cu
│   ├── catch2/
│   ├── unittest.cu
│   └── CMakeLists.txt
├── benchmarks/
│   ├── level1/
│   │   ├── bench_vector_ops.cu
│   │   └── bench_softmax.cu
│   ├── level2/
│   │   └── bench_gemm.cu
│   ├── level3/
│   │   ├── bench_flash_attention.cu
│   │   └── bench_paged_attention.cu
│   ├── level4/
│   │   └── bench_quantized_gemm.cu
│   └── CMakeLists.txt
├── scripts/
│   ├── build.sh
│   ├── test.sh
│   ├── benchmark.sh
│   └── clean.sh
├── docs/
│   ├── architecture.md
│   ├── coding_style.md
│   └── debugging.md
└── .gitignore
```

## 🚀 Quick Start

### 环境要求

- CUDA Toolkit 12.x+
- CMake 3.18+
- GCC 11+
- NVIDIA GPU (sm_80+ for Tensor Cores)

### 编译所有

```bash
./scripts/build.sh
```

### 运行测试

```bash
./scripts/test.sh          # 所有测试
./scripts/test.sh level1   # 只测 Level 1
```

### 运行基准

```bash
./scripts/benchmark.sh     # 所有基准
./scripts/benchmark.sh gemm # 只测 GEMM
```

## 📊 基准测试

| Kernel | TFLOPS (A100) | 显存带宽 |
|--------|---------------|---------|
| GEMM (Naive) | ~1-5 | 低 |
| GEMM (Tensor Core) | ~100-300 | 高 |
| FlashAttention-2 | ~80-120 | 高 |
| W8A16 GEMM | ~150-200 | 极高 |

## 📚 Reference

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashDecoding Paper](https://arxiv.org/abs/2309.06169)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
- [SmoothQuant](https://arxiv.org/abs/2308.15026)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [GGML](https://github.com/ggerganov/ggml)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

## 🤝 Contributing

1. Fork this repo
2. Create your feature branch (`git checkout -b feature/xxx`)
3. Commit with proper template (`git commit` will auto-use template)
4. Push to branch
5. Open a Pull Request

## 📝 Commit 规范

使用 `git commit` 会自动打开模板：

```bash
<type>: <subject>

# 详细说明（可选）

Author: elucat
Date:   2026-02-12
```

**Type 类型：**
- `feat` - 新功能
- `fix` - Bug 修复
- `refactor` - 重构
- `perf` - 性能优化
- `docs` - 文档更新
- `test` - 测试相关
- `chore` - 构建/工具
- `style` - 代码格式

## 📝 License

MIT License

---

*Learning by doing. 纸上得来终觉浅，绝知此事要躬行。*
