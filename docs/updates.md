# Phase 34 -- Track D: NEON SIMD CPU Acceleration

## Status: In Progress (Wave D3 Complete)

## 2026-03-07 -- Wave D3: NEON Wiring + TensorArena

### Completed

| Task | File | Commit | Summary |
|------|------|--------|---------|
| T102.8 | compute/cpu_engine.go | 7ac9a35 | Wire NEON into CPUEngine: Softmax, Exp, Add/Sub/Mul/Div same-shape, scalar ops. neonBinaryF32 + scalarOp helpers. |
| T102.8 | compute/fused_*.go | 0afe430 | Wire xblas into FusedRMSNorm, FusedRoPE, FusedSiLUGate. |
| T103.1 | compute/tensor_arena.go | dc97cd7 | TensorArena: power-of-2 bucketed pool with per-bucket mutex. 5 tests + bench. |
| T103.2 | compute/cpu_engine.go | e3775a8 | Wire TensorArena into getOrCreateDest for float32 engines. |
| T101.4 | compute/ | (lint) | golangci-lint 0 issues. |
| T102.9 | compute/, internal/xblas/ | (lint) | golangci-lint 0 issues. |
| T103.3 | compute/ | (lint) | golangci-lint 0 issues. |

### Deviation

T103.1 was marked complete in Wave D1 but the commit (b4b5eb1) was empty.
Re-implemented TensorArena from scratch in Wave D3.

### Next: Wave D4 -- Benchmark Validation

- T104.1: CPU ARM64 benchmark on DGX Spark (target >= 10 tok/s)
- T104.2: Per-operation profiling
- T104.3: Output correctness verification
- T104.4: Final lint pass

---

## 2026-03-07 -- Wave D2: All 6 NEON Assembly Kernels

### Completed

| Task | File | Commit | Summary |
|------|------|--------|---------|
| T102.1 | internal/xblas/softmax_arm64.s | bc775d8 | 3-pass NEON Softmax: FMAX for max, inline exp polynomial, normalize by 1/sum. 253 lines. |
| T102.2 | internal/xblas/rmsnorm_arm64.s | a40a2f7 | NEON RMSNorm: dual accumulators, FRSQRTE + 2 Newton-Raphson iterations. 173 lines. Returns scale. |
| T102.3 | internal/xblas/silu_arm64.s | d766923 | SiLUF32 + SiLUGateF32: inline exp polynomial + FRECPE + 2 NR for sigmoid. 394 lines. |
| T102.4 | internal/xblas/rope_arm64.s | ef099ef | RoPE: 4-wide NEON for rotary dim, scalar tail, passthrough copy. 159 lines. |
| T102.5 | internal/xblas/elementwise_arm64.s | a40a2f7 | VaddF32/VmulF32/VsubF32/VdivF32: NEON load-4/op/store-4 loops. 158 lines. |
| T102.6 | internal/xblas/scalar_arm64.s | c751b5c | VmulScalarF32/VaddScalarF32/VdivScalarF32: VDUP broadcast + NEON loop. 129 lines. |
| (lint) | rmsnorm_generic.go, rope_test.go | 0faf769 | D->dim captLocal fix, math/rand/v2 depguard fix. |

### Wave D1 (Prior)

| Task | File | Commit | Summary |
|------|------|--------|---------|
| T101.1 | compute/cpu_engine.go | f733d15 | Same-shape fast path in binaryOp (7-8x speedup). |
| T101.2 | compute/cpu_engine.go | c28a529 | Pow x^2 specialization (13-15x speedup). |
| T101.3 | compute/cpu_engine_bench_test.go | 3d8c3d7 | Scalar op baseline benchmarks. |
| T102.7 | internal/xblas/vexp_arm64.s | 5931298 | VexpF32 shared exp polynomial (max error 8.98e-08). |
| T103.1 | internal/xblas/arena.go | b4b5eb1 | TensorArena with power-of-2 bucketed pooling. |

### Next: Wave D3

- T101.4: Run golangci-lint on compute/
- T102.8: Wire NEON functions into CPUEngine dispatch
- T102.9: Run golangci-lint on internal/xblas/
- T103.2: Wire TensorArena into CPUEngine
- T103.3: Run golangci-lint on compute/

---

# Phase 34 -- Track A: purego / dlopen

## Status: In Progress

## 2026-03-07 -- Track A Wave A1

Track A: Replace CGo CUDA bindings with purego dlopen loader.

### Completed

| Task | File | Commit | Summary |
|------|------|--------|---------|
| T87.1+T87.2+S87.2.1 | internal/cuda/purego*.go | c39ca30 | Zero-CGo CUDA dlopen loader with assembly trampolines. Darwin (syscall6/9), Linux arm64 (asmcgocall + AAPCS64 trampoline), other (stubs). CUDALib loads libcudart.so, resolves 13 function pointers. 7 tests pass. |
| T87.3+S87.3.1 | internal/cuda/runtime_purego.go | 40b4db0 | Purego CUDA runtime: all 14 functions (Malloc, Free, Memcpy, Stream, etc.) via ccall. Compile-time parity tests + graceful error tests. |
| T87.4 | internal/cuda/ | (verified) | golangci-lint 0 issues on entire internal/cuda/ package. |
| T88.1 | internal/cuda/purego.go, kernels/purego.go | 71d8174 | KernelLib dlopen loader: loads libkernels.so, resolves 28 kernel function pointers. Exports Ccall/Dlsym/DlopenKernels. |
| T88.2 | internal/cuda/kernels/elementwise_purego.go | b641607 | 22 elementwise kernel wrappers (binary, scalar, unary, broadcast, fill, sum_axis, softmax) via cuda.Ccall. |
| T88.3 | internal/cuda/kernels/*_purego.go | 01a6801 | RMSNorm, Gather, Transpose2D/ND, GemmQ4F32 purego wrappers. CUTLASS kernels remain CGo-only. |
| T88.4 | internal/cuda/mempool.go | fb4ccfd | Removed cuda build tag from MemPool (uses only purego-available APIs). |
| T89.1 (partial) | internal/gpuapi/cuda_*.go | 8829531 | Removed cuda tag from cuda_runtime, cuda_mempool, cuda_kernels. cuda_blas/cuda_dnn retain tag (cublas/cudnn dependency). |

### Deviation

Used purego-style assembly trampolines instead of golang.org/x/sys/unix.Dlopen
(which does not exist). User approved this approach. True zero CGo: calls bypass
runtime.cgocall entirely.

---

# Phase 34 -- Track 0: Composition Fixes

## Status: Complete (Priority 1+2), Blocked (Priority 3)

## 2026-03-07 -- Session Start

Track 0, Wave 0-1: Refactor violated layers to compose Engine primitives.

### Completed

| Task | File | Commit | Summary |
|------|------|--------|---------|
| T96.3 | layers/activations/gelu.go | ea3e04a | Gelu: replaced BaseActivation closure with explicit engine calls (Mul, MulScalar, Add, AddScalar, Tanh). Also fixed FastGelu to use engine.Tanh. |
| T96.2 | layers/attention/qk_norm.go | 8d0be2a | QKNorm: replaced manual Data() loops with engine.Mul, ReduceMean, AddScalar, Rsqrt, Mul. |
| T96.4 | layers/normalization/batch_norm.go | 0c9f356 | BatchNorm: replaced per-channel loops with engine.Reshape (broadcast) + Sub, Div, Mul, Add. |
| T96.5 | layers/attention/local_attention.go | 35b9cf6 | LocalAttention mask: build data slice directly, pass to tensor.New. No more Data() mutation. |
| T96.1 | layers/core/matmul_nbits.go | ec251e2 | MatMulNBits: eagerly dequantize at construction so Forward() only uses engine.MatMul. |

### Remaining Track 0 Tasks

Priority 3 (not on Gemma 3 path):
- T96.6 Conv2d: im2col + engine.MatMul decomposition (3h est.)
- T96.7/T96.8 MoE: needs engine.TopK or sort primitive
- T96.9 PolynomialExpansion: per-term engine.Pow + Mul
- T96.10 SpectralFingerprint (core): DFT -> MatMul with Fourier basis
- T96.11 S4: sequential scan, per-step engine calls
- T96.12 SpectralFeature (features): external Gonum FFT

---

# Phase 29 Updates

## 2026-03-06: Q4 B-Operand NEON + Parallel GEMV (6.5 tok/s)

### Summary

Implemented full Q4 B-operand fast path with NEON assembly and multi-core
parallelism. Throughput: **3.80 → 6.5 tok/s** (1.7x improvement).

### Changes

1. **NEON q4DotBlockSIMD** (`045ad78`, `df0fcbb`): ARM64 assembly for fused
   Q4 nibble extraction + float32 dot product in NEON registers. Replaces
   per-block scalar dequant.

2. **q4DotRowSIMD** (`78674e4`, `b0290c4`): Row-level assembly that processes
   an entire row of Q4 blocks in a single call. Eliminates per-block Go
   function call overhead (BlockScaleF32→float16.ToFloat32, BlockData,
   q4DotBlock = 4 calls × 248K blocks). Uses `LDR H + FCVT S,H` for
   float16→float32 scale conversion directly in NEON.

3. **GemmF32Q4NT** (`78674e4`): Computes C = A × B^T where B is [N,K] in
   Q4 format. Reads Q4 blocks directly from weight memory without transpose
   or dequantization. The Transpose layer passes Q4 storage through with
   transposed shape; the engine detects Q4 on B operand and dispatches here.

4. **Parallel Q4 GEMV** (`5c06704`): For M=1 GEMV with N*K >= 64K, splits
   the N dimension across runtime.NumCPU() goroutines. Each computes
   independent output elements via q4DotRow.

5. **Decode transpose short-circuit** (`c702bf7`): When attention transpose
   swaps a dimension of size 1 (seq_len=1 during decode), skip the blocked
   copy and use plain `copy()`. Reduced memclr from 0.47s→0.15s.

### Benchmark Results (DGX Spark GB10, Gemma 3 2B Q4_0)

| Config | tok/s | CPU util |
|--------|-------|----------|
| Phase 28 baseline | 3.80 | ~230% |
| + Transpose cache | 5.55 | ~230% |
| + NEON q4DotBlockSIMD | 3.51 | ~230% |
| + q4DotRowSIMD (row-level) | 4.04 | ~230% |
| + Parallel Q4 GEMV | 5.73 | ~236% |
| + Decode transpose short-circuit | **6.5** | ~236% |

### Profile Breakdown (post-optimization, 30 tokens)

| Component | % CPU | Wall ms/token | Notes |
|-----------|-------|---------------|-------|
| sgemmAccRowNeon (SGEMM) | 35.2% | ~12 | Float32 lm_head/embedding |
| q4DotRowSIMD (Q4 GEMV) | 34.6% | ~12 | Q4 weight layers |
| binaryOp (Mul/Add) | 6.7% | ~2 | Element-wise ops |
| Transpose | 3.9% | ~3 | Attention reshapes |
| GC/malloc | ~5% | ~15 | Reduced by TensorPool |
| Other | ~15% | ~130 | Graph traversal, scheduling |

### Bottleneck Analysis

**Compute is fast, overhead dominates.** The two GEMM paths (Q4 + float32)
account for ~70% of CPU but only ~24ms wall time per token (parallelized).
The remaining ~150ms/token is overhead:

1. **Graph traversal**: ~780 node executions per token (30+ nodes × 26 layers).
   Each node: interface dispatch, shape validation, pool acquire/release.
2. **Memory management**: TensorPool reduces allocations but not to zero.
   GC still significant.
3. **Goroutine scheduling**: Per-MatMul goroutine launch/sync (~130 MatMul
   calls per token, each spawning/joining 20 goroutines).

### Why 15 tok/s requires architectural changes

To reach 15 tok/s (67ms/token), we need to cut overhead from ~150ms to ~43ms.
This requires:
- Fused operation graphs (batch multiple ops per kernel launch)
- Worker pool instead of per-call goroutine creation
- Zero-copy tensor views for reshape/transpose-of-1D
- Graph compilation to eliminate per-node dispatch overhead

These are beyond the scope of Phase 29's planned tasks.

### FCVT encoding bug fix (`b0290c4`)

The WORD encoding `0x1E22E0E7` was `FCVT H7,S7` (single→half), not
`FCVT S7,H7` (half→single). Caused SIGILL on NVIDIA Grace (Neoverse V2).
Correct encoding: `0x1EE240E7` (ftype=11, opc=000100).

---

## Previous Updates

### E50 COMPLETE -- TensorPool wired into Generator

- `15bb955` T50.1+S50.1.1: Pool created in `NewGenerator`, attached via `graph.WithPool()`.
- `8e5678c` T50.2+S50.2.1: Allocation benchmark added.
- T50.3: golangci-lint passing.

### E51 COMPLETE -- KV cache decode already optimal + RoPE bug fixed

- Q/K/V projections: already optimal (single-token decode).
- KV cache: already optimal (append + return full sequence).
- `502986d` RoPE position offset fix (was always using position 0 during decode).
- `559b7e4` GQA sets RoPE offset to cache.SeqLen().

### T52.1 COMPLETE -- Constant transpose elimination

- `a004055` Transpose layer data-pointer cache (3.53 → 5.42 tok/s).
- `9dc3272` MatMul B-operand cache + LMHead tied weight cache.

### Q4 Storage Fix

- ZMF loader now keeps Q4 weights in Q4Storage (4x less memory).
- Q4Storage.Slice() caches dequantized result.

### Known Issues

- Q4 ZMF model produces garbage output (pre-existing, not caused by Phase 29).
