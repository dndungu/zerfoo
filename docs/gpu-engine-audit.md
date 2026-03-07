# GPUEngine Transfer Behavior Audit (T69.1)

Phase 32 audit of all GPUEngine methods for H2D/D2H transfer patterns.

## GPU-Accelerated Methods (Output: GPUStorage)

All use `getDevicePtr` for input (zero-copy if GPUStorage, H2D if CPUStorage)
and `makeGPUResult` for output (always GPUStorage).

| Method | Input Path | Output | CPU Fallback Trigger |
|--------|-----------|--------|---------------------|
| MatMul (f32/BF16) | getDevicePtr(a,b) | GPUStorage | OOM |
| matMulQ4 | H2D always (Q4 bytes), getDevicePtr(b) | GPUStorage | shape/OOM |
| Add | getDevicePtr(a,b) | GPUStorage | !float32 or !sameShape |
| Sub | getDevicePtr(a,b) | GPUStorage | !float32 or !sameShape |
| Mul | getDevicePtr(a,b) | GPUStorage | !float32 or !sameShape |
| Div | getDevicePtr(a,b) | GPUStorage | !float32 or !sameShape |
| Pow | getDevicePtr(a,b) | GPUStorage | !float32 or !sameShape |
| TanhPrime | getDevicePtr(a,b) | GPUStorage | !float32 or !sameShape |
| Exp | getDevicePtr(a) | GPUStorage | !float32 |
| Log | getDevicePtr(a) | GPUStorage | !float32 |
| Sqrt | getDevicePtr(a) | GPUStorage | !float32 |
| Rsqrt | getDevicePtr(a) | GPUStorage | !float32 |
| Tanh | getDevicePtr(a) | GPUStorage | !float32 |
| AddScalar | getDevicePtr(a) | GPUStorage | !float32 |
| MulScalar | getDevicePtr(a) | GPUStorage | !float32 |
| DivScalar | getDevicePtr(a) | GPUStorage | !float32 |
| Fill | pool.Alloc + SetStorage | GPUStorage | !float32 |
| Sum/ReduceSum | getDevicePtr(a) | GPUStorage | !float32 or axis<0 |
| ReduceMean | gpuSum + gpuDivScalar | GPUStorage | !float32 or axis<0 |
| Softmax | getDevicePtr(a) | GPUStorage | !float32 |

## CPU Fallback Methods (Output: CPUStorage)

All delegate to `e.cpu.*`. When input has GPUStorage, `.Data()` triggers D2H.

| Method | Always CPU? | Notes |
|--------|------------|-------|
| Transpose | Yes | 8.1% of GPU inference time |
| Gather | Yes | Embedding lookup |
| UnaryOp | Yes | Custom function, cannot run on GPU |
| Zero | Yes | |
| Zeros | Yes | |
| Copy | Yes | |
| ScatterAdd | Yes | Training only |
| RandomUniform | Yes | Training only |
| Split | Yes | |
| Concat | Yes | |
| Repeat | Yes | |
| OneHot | Yes | Training only |
| Reshape | Yes | Metadata-only, no data copy |

## Root Causes of 43% cgocall Overhead

1. **Model weights are CPUStorage at load time.** Every GPU op's first use of a
   weight tensor triggers H2D via `getDevicePtr`. Fix: upload weights to GPU at
   load time (T69.3).

2. **CPU fallback methods break the GPU chain.** Transpose and Gather produce
   CPUStorage output. The next GPU op then does H2D to get data back on GPU.
   Fix: GPU Transpose (E70), GPU Gather (E72).

3. **Binary op broadcasting fallback.** When shapes differ, `sameShape()` guard
   sends ops to CPU. Fix: GPU broadcasting (E71).

4. **Q4 MatMul copies Q4 bytes every call.** Q4Storage.RawBytes() is always on
   host. Fix: upload Q4 weight bytes to GPU at load time (part of T69.3).

## Pre-existing GPU Residency

The GPU-accelerated methods already implement device-resident output:
- `makeGPUResult` creates tensors with GPUStorage.
- `getDevicePtr` returns device pointer directly for GPUStorage (zero-copy).
- Chained GPU ops (e.g., MatMul -> Add -> Softmax) already flow GPU->GPU
  when all inputs are GPUStorage.

The planned T69.2 (add GPU-resident tensor creation) is **already implemented**.
The planned T69.4 (logits D2H) is **already handled** by `sampleFromLogits`
calling `.Data()` which triggers implicit D2H for GPUStorage.
