# ToyKernel Debugging Guide

## Common Issues

### Memory Errors
```bash
# Run with cuda-memcheck
cuda-memcheck ./unittest
```

### Performance Profiling
```bash
# Nsight Systems
nsight sys -o profile ./benchmark

# Nsight Compute
nsight compute -o profile ./benchmark
```

### Kernel Debugging
```bash
# Add debug info
nvcc -G -g -Xcompiler "-O0" kernel.cu
```

## Useful Commands

```bash
# Check GPU info
nvidia-smi

# Monitor GPU
nvtop

# CUDA version
nvcc --version
```

## Testing Checklist
- [ ] Correctness vs PyTorch reference
- [ ] Numerical accuracy (rtol, atol)
- [ ] Performance improvement
- [ ] Memory usage
- [ ] Edge cases (empty input, large input)
