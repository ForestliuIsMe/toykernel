# ToyKernel

Common CUDA kernel implementations from scratch. 从零学习 CUDA 高性能算子实现。

## 🎯 Roadmap

| Phase | Operators | Description | Status |
|-------|-----------|-------------|--------|
| **Phase 1: Basics** | | 基础算子 | |
| | Vector Add | 向量相加 | ⬜ |
| | Matrix Multiply | 矩阵乘法 (Naive) | ⬜ |
| | Softmax | Softmax 计算 | ⬜ |
| **Phase 2: GEMM** | | 矩阵乘法优化 | |
| | GEMM (Tiled) | 分块矩阵乘法 | ⬜ |
| | GEMM (Shared Memory) | 共享内存优化 | ⬜ |
| | GEMM (Tensor Cores) | Tensor Core 加速 | ⬜ |
| **Phase 3: Attention** | | Attention 变体 | |
| | FlashAttention-2 | 前向传播 | ⬜ |
| | FlashAttention-2 (Backward) | 反向传播 | ⬜ |
| | FlashDecoding | 推理解码优化 | ⬜ |
| **Phase 4: Advanced** | | 进阶算子 | |
| | RoPE | 位置编码 | ⬜ |
| | LayerNorm | 层归一化 | ⬜ |
| | RMSNorm | RMS 归一化 | ⬜ |
| | GeLU | 激活函数 | ⬜ |

## 📁 Project Structure

```
toykernel/
├── README.md
├── src/                    # Kernel 实现
│   ├── basics/            # 基础算子
│   │   ├── vector_add.cu
│   │   └── softmax.cu
│   ├── gemm/               # 矩阵乘法
│   │   ├── naive_gemm.cu
│   │   ├── tiled_gemm.cu
│   │   └── tensor_core_gemm.cu
│   ├── attention/         # Attention 系列
│   │   ├── flash_attention.cu
│   │   └── flash_decoding.cu
│   └── norm/              # 归一化层
│       ├── layernorm.cu
│       └── rmsnorm.cu
├── include/               # 头文件
│   └── utils.cuh
├── tests/                 # 单元测试
├── benchmarks/           # 性能测试
└── scripts/              # 编译脚本
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

## 📊 Reference

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashDecoding Paper](https://arxiv.org/abs/2309.06169)
- [ CUTLASS](https://github.com/NVIDIA/cutlass)
- [TinyCUDA](https://github.com/eynnzerr/TinyCUDA)

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
