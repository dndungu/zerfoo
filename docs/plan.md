# Zerfoo Performance Optimization Plan -- Phases 25, 26, and 27

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with 40+ packages, a 34-method compute
Engine[T] interface, CPU and CUDA GPU backends, gRPC-based distributed
training, and comprehensive test coverage (95%+ across testable packages).
Phases 1-24 are complete (see docs/adr/001-019, docs/design.md).

Phase 25 built all core performance primitives: mmap loading, pre-allocated KV
cache, Q4/Q8 quantization, NEON/AVX2 SIMD GEMM, fused Q4 dequant+multiply,
CUDA Q4 dequant-GEMM (2383 GFLOPS on GB10), parallel graph execution, and
continuous batching.

Phase 26 validated end-to-end Q4 inference on Gemma 3 2B (3.60 tok/s on DGX
Spark ARM64 CPU), added PagedAttention, speculative decoding, GGUF parser, and
performance CI dashboard.

Phase 27 targets a 4.2x CPU throughput improvement (3.60 -> >= 15 tok/s) and
GPU inference enablement (>= 60 tok/s) by eliminating the dominant bottlenecks
identified in Phase 26 profiling:

CPU profile breakdown (after Phase 26 blocked 2D transpose optimization):

| Component | % CPU | Notes |
|-----------|-------|-------|
| Transpose (3D/4D) | 62% | Attention permute patterns, weight transposes on every forward pass |
| GEMM (sgemmAccRowNeon) | 16% | Actual compute -- the useful work |
| binaryOp | 3% | Element-wise add/mul |
| GC / malloc | 5% | 2.5M allocs per generation, 39GB total |
| Other | 14% | Model loading, tokenization, decoding |

Key bottlenecks in priority order:

1. **Redundant weight transposes.** Weight matrices are transposed on every
   forward pass even though weights never change. The ONNX-to-ZMF conversion
   emits explicit Transpose nodes for weight layout. Folding these at load
   time eliminates 62% of CPU time.

2. **Excessive allocation.** Every forward pass creates new tensors for all
   intermediate results (2.5M allocs for 32 tokens). A tensor arena that
   reuses buffers by shape eliminates GC pressure and cache thrashing.

3. **No GPU inference path.** The CUDA backend has cuBLAS SGEMM and a custom
   Q4 GEMM kernel (2383 GFLOPS) but has never been wired into end-to-end
   model inference. GPU should yield 10-50x over CPU.

4. **No GGUF end-to-end.** The GGUF parser/loader (E37) can read weights and
   metadata but cannot build a computation graph. Architecture-specific
   graph template builders are needed to run inference from GGUF files.

5. **No operator fusion.** Common multi-op patterns (RMSNorm, RoPE,
   SiLU-gate) execute as separate ops with intermediate tensor
   materialization.

### Phase 25 Summary (COMPLETE)

| Epic | Result |
|------|--------|
| E25: Mmap loading | Zero-copy model load via syscall.Mmap |
| E26: Pre-alloc KV cache | 0 allocs/token decode loop |
| E27: Q4/Q8 tensor storage | 8x/4.5x compression vs float32 |
| E28: Q4/Q8 CPU MatMul | Fused dequant+SIMD multiply, ~16% faster GEMV |
| E29: NEON/AVX2 SGEMM | ~2x faster than gonum BLAS |
| E30: Parallel graph executor | Concurrent independent branch execution |
| E31: Continuous batching | Channel-based batch scheduler in serve/ |
| E32: CUDA Q4 kernel | 2383 GFLOPS on DGX Spark GB10 (sm_121) |
| E33: Benchmark suite | tok/s, GFLOPS, memory allocs benchmarks |

### Phase 26 Summary (COMPLETE)

| Epic | Result |
|------|--------|
| E34: PagedAttention | Block pool + PagedKVCache + Generator integration (46% memory of pre-alloc) |
| E35: Speculative Decoding | SpeculativeGenerator with adaptive draft length |
| E36: End-to-End Q4 Pipeline | Gemma 3 2B Q4: 1.96 -> 3.60 tok/s (1.84x via blocked transpose) |
| E37: GGUF Model Import | Parser + loader + arch mapping (llama/gemma) |
| E38: Performance CI Dashboard | bench.sh + GH Actions workflow + DGX GPU job |

### Phase Summary (Previous Work)

| Phase | Description | ADRs |
|-------|-------------|------|
| 1-9 | Production readiness, distributed training, model import, inference library, multi-arch | 001-006 |
| 10-13 | Multi-GPU, cuDNN, TensorRT, CUTLASS flash attention | 007-010 |
| 14-19 | GRAL, ROCm, OpenCL, cuDNN backward, INT4/INT8 GEMM, TRT dynamic shapes | 011-016 |
| 20 | DGX Spark GB10 validation (66 packages, ARM64, sm_121, CUDA 13.0) | 017 |
| 21 | Model parity (18 PASS across 6 families, 18 ONNX fixes) | 018 |
| 22 | BF16 cuBLAS GEMM (1.5x faster), unified memory (200-5000x alloc speedup), SigLIP fix | 019 |
| 23 | Test coverage push (9 packages at 100%, 42 of 50 at >= 95%) | -- |
| 24 | FFN bias fix, embedding loading, cmd/zerfoo-predict refactor | -- |

### Objectives

Phase 25 (COMPLETE):
O39: Mmap-based model loading. Eliminate heap allocation for model weights.
O40: Pre-allocated KV cache. Zero allocation during autoregressive decode.
O41: Quantized inference pipeline (Q4_0, Q8_0). Run 4-bit models end-to-end.
O42: Parallel graph execution. Execute independent graph branches concurrently.
O43: Optimized CPU GEMM. Use NEON (ARM64) and AVX2 (x86-64) SIMD intrinsics.
O44: Continuous batching in the serve package.
O45: Benchmark suite with tok/s metric. Measure and track performance parity.

Phase 26 (COMPLETE):
O46: End-to-end quantized inference on a real model with measured tok/s.
O47: PagedAttention for efficient multi-sequence KV memory.
O48: Speculative decoding for 2-3x single-request decode speedup.
O49: GGUF model import for ecosystem compatibility.
O50: Automated performance regression tracking in CI.

Phase 27 (NEW):
O51: Gemma 3 2B Q4 >= 15 tok/s on DGX Spark CPU (ARM64).
O52: Gemma 3 2B Q4 >= 60 tok/s on DGX Spark GPU (GB10).
O53: Load and run inference from a GGUF file without any external conversion.
O54: Decode loop allocation < 100 allocs/token.
O55: Fused RMSNorm, RoPE, and SiLU-gate kernels.

### Non-Goals

- Training performance optimization (inference focus).
- Multi-node inference (requires second DGX Spark).
- Pipeline parallelism / tensor parallelism.
- FP4 kernels (blocked on upstream CUTLASS SM121 FP4 fixes).
- Vulkan or SYCL backends.
- New quantization formats (Q4_K_M, Q5_K, Q6_K) -- Phase 28 candidate.
- Prompt caching / prefix sharing (future phase, after PagedAttention).
- Vision model inference (focus on text-only LLMs).
- Breaking changes to the Engine[T] or Node[T] interfaces.
- Replacing gRPC with a different RPC framework.
- Adding third-party test frameworks (testify, etc.).
- SSM/Mamba architectures (Falcon Mamba, RWKV, Jamba).
- Multi-GPU inference orchestration.
- Mobile / WebAssembly targets.

### Constraints and Assumptions

- Use Go standard library only where possible. Minimize new dependencies.
- All CUDA code behind `//go:build cuda` build tags.
- All ROCm code behind `//go:build rocm` build tags.
- All OpenCL code behind `//go:build opencl` build tags.
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- Table-driven tests using the standard testing package.
- DGX Spark GB10 at ssh ndungu@192.168.86.250 for GPU validation.
- Target models: Gemma 3 2B (Q4 ZMF at ~/models/gemma3-q4/ on DGX), Llama 3.2 1B (GGUF).
- GGUF parser is pure Go (no CGo dependency on llama.cpp).
- PagedAttention block size: 16 tokens (matches vLLM default).
- Assembly (NEON/AVX2) files use Go's plan9 assembler syntax with build tags.

### Success Metrics

| Metric | Phase 26 Baseline | Phase 27 Target | How Measured |
|--------|-------------------|-----------------|-------------|
| CPU tok/s (Gemma 3 2B Q4) | 3.60 | >= 15 | BenchmarkGemma3Q4TokPerSec on DGX Spark |
| GPU tok/s (Gemma 3 2B Q4) | untested | >= 60 | GPU benchmark on DGX Spark GB10 |
| Allocs per token | ~80,000 | < 100 | go test -bench -benchmem |
| Transpose % of CPU | 62% | < 5% | pprof CPU profile |
| GGUF end-to-end | parser only | full inference | LoadGGUFModel -> Generate |
| Model load time (7B) | < 0.5s (mmap) | maintained | Benchmark in model/ |
| KV cache allocs/token | 0 per token | maintained | Go benchmark -benchmem |
| KV memory per sequence | ~used_tokens * dim * layers (paged) | maintained | PagedAttention benchmark |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D66 | Mmap ZMF loader | COMPLETE (Phase 25) |
| D67 | Pre-allocated KV cache with ring buffer | COMPLETE (Phase 25) |
| D68 | Q4_0 and Q8_0 tensor storage and dequant | COMPLETE (Phase 25) |
| D69 | Quantized MatMul kernel (CPU Q4_0 * FP32) | COMPLETE (Phase 25) |
| D70 | NEON SGEMM kernel (ARM64) | COMPLETE (Phase 25) |
| D71 | AVX2 SGEMM kernel (x86-64) | COMPLETE (Phase 25) |
| D72 | Parallel graph executor | COMPLETE (Phase 25) |
| D73 | Continuous batching in serve/ | COMPLETE (Phase 25) |
| D74 | tok/s benchmark suite | COMPLETE (Phase 25) |
| D75 | Quantized CUDA dequant-GEMM kernel | COMPLETE (Phase 25) |
| D76 | End-to-end Q4 inference pipeline | COMPLETE (Phase 26) |
| D77 | PagedAttention KV cache | COMPLETE (Phase 26) |
| D78 | Speculative decoding | COMPLETE (Phase 26) |
| D79 | GGUF model parser and loader | COMPLETE (Phase 26) |
| D80 | Performance CI dashboard | COMPLETE (Phase 26) |
| D81 | Weight transpose elimination | Transpose < 5% of CPU profile |
| D82 | Tensor arena / buffer pool | < 100 allocs/token in decode loop |
| D83 | GPU inference pipeline | >= 60 tok/s on Gemma 3 2B Q4 (GB10) |
| D84 | GGUF end-to-end inference | Load GGUF, generate text, no external tools |
| D85 | Fused operators (RMSNorm, RoPE, SiLU-gate) | Single-pass kernels, >= 2x micro-benchmark |

### Out of Scope

- Advanced quantization (Q4_K_M, Q5_K, Q6_K) -- Phase 28 candidate.
- Prompt caching / prefix sharing -- future phase after PagedAttention.
- Multi-GPU KV cache partitioning.
- Vision encoder inference.
- Training loop optimization.
- Multi-node inference.
- Mobile / WebAssembly targets.

---

## 3. Checkable Work Breakdown

### Phase 25 -- COMPLETE

### E25: Mmap-Based Model Loading (O39)

- [x] T25.1 Add mmap file reader to model/ package  Owner: TBD  Est: 2h  2026-03-05
- [x] S25.1.1 Unit tests for MmapReader  Owner: TBD  Est: 1h  2026-03-05
- [x] T25.2 Add streaming protobuf decoder for ZMF  Owner: TBD  Est: 3h  2026-03-05
- [x] S25.2.1 Unit tests for LoadZMFMmap  Owner: TBD  Est: 1h  2026-03-05
- [x] T25.3 Integrate mmap loader into inference.Load  Owner: TBD  Est: 1h  2026-03-05
- [x] S25.3.1 Tests for inference.Load with mmap  Owner: TBD  Est: 30m  2026-03-05
- [x] T25.4 Run golangci-lint on model/  Owner: TBD  Est: 15m  2026-03-05

### E26: Pre-Allocated KV Cache (O40)

- [x] T26.1 Implement ring-buffer KV cache  Owner: TBD  Est: 3h  2026-03-05
- [x] S26.1.1 Unit tests for ring-buffer KV cache  Owner: TBD  Est: 1.5h  2026-03-05
- [x] T26.2 Update generate.Generator to use pre-allocated cache  Owner: TBD  Est: 1.5h  2026-03-05
- [x] S26.2.1 Tests for generator with pre-allocated cache  Owner: TBD  Est: 1h  2026-03-05
- [x] T26.3 Update GQA attention to use view-based cache  Owner: TBD  Est: 2h  2026-03-05
- [x] S26.3.1 Tests for GQA with view cache  Owner: TBD  Est: 1h  2026-03-05
- [x] T26.4 Run golangci-lint on generate/ and layers/attention/  Owner: TBD  Est: 15m  2026-03-05

### E27: Quantized Tensor Storage (O41)

- [x] T27.1 Add Q4_0 block format to tensor/  Owner: TBD  Est: 3h  2026-03-05
- [x] S27.1.1 Unit tests for Q4_0 quantization  Owner: TBD  Est: 1.5h  2026-03-05
- [x] T27.2 Add Q8_0 block format to tensor/  Owner: TBD  Est: 2h  2026-03-05
- [x] S27.2.1 Unit tests for Q8_0 quantization  Owner: TBD  Est: 1h  2026-03-05
- [x] T27.3 Add quantized tensor loading to ZMF loader  Owner: TBD  Est: 2h  2026-03-05
- [x] S27.3.1 Tests for quantized ZMF loading  Owner: TBD  Est: 1h  2026-03-05
- [x] T27.4 Add zonnx quantization pass  Owner: TBD  Est: 2h  2026-03-05
- [x] S27.4.1 Tests for zonnx quantization  Owner: TBD  Est: 1h  2026-03-05
- [x] T27.5 Run golangci-lint on tensor/ and model/  Owner: TBD  Est: 15m  2026-03-05

### E28: Quantized CPU MatMul Kernel (O41, O43)

- [x] T28.1 Implement Q4_0 x FP32 MatMul in Go  Owner: TBD  Est: 3h  2026-03-05
- [x] S28.1.1 Unit and benchmark tests  Owner: TBD  Est: 1.5h  2026-03-05
- [x] T28.2 Implement Q8_0 x FP32 MatMul in Go  Owner: TBD  Est: 2h  2026-03-05
- [x] S28.2.1 Unit and benchmark tests  Owner: TBD  Est: 1h  2026-03-05
- [x] T28.3 Wire quantized MatMul into CPUEngine  Owner: TBD  Est: 2h  2026-03-05
- [x] S28.3.1 Integration tests for quantized MatMul via Engine  Owner: TBD  Est: 1h  2026-03-05
- [x] T28.4 Run golangci-lint on internal/xblas/ and compute/  Owner: TBD  Est: 15m  2026-03-05

### E29: SIMD-Optimized CPU GEMM (O43)

- [x] T29.1 NEON SGEMM kernel for ARM64  Owner: TBD  Est: 4h  2026-03-05
- [x] S29.1.1 Tests for NEON SGEMM  Owner: TBD  Est: 1.5h  2026-03-05
- [x] T29.2 AVX2 SGEMM kernel for x86-64  Owner: TBD  Est: 4h  2026-03-05
- [x] S29.2.1 Tests for AVX2 SGEMM  Owner: TBD  Est: 1.5h  2026-03-05
- [x] T29.3 Wire SIMD GEMM into xblas.GemmF32  Owner: TBD  Est: 1h  2026-03-05
- [x] S29.3.1 Integration tests  Owner: TBD  Est: 30m  2026-03-05
- [x] T29.4 Fused Q4 dequant+multiply kernel  Owner: TBD  Est: 3h  2026-03-05
- [x] S29.4.1 Tests for fused Q4 kernel  Owner: TBD  Est: 1h  2026-03-05
- [x] T29.5 Run golangci-lint on internal/xblas/  Owner: TBD  Est: 15m  2026-03-05

### E30: Parallel Graph Execution (O42)

- [x] T30.1 Build dependency-aware parallel executor  Owner: TBD  Est: 4h  2026-03-05
- [x] S30.1.1 Unit tests for parallel executor  Owner: TBD  Est: 2h  2026-03-05
- [x] T30.2 Add ForwardMode option to Graph  Owner: TBD  Est: 1h  2026-03-05
- [x] S30.2.1 Tests for ForwardMode  Owner: TBD  Est: 30m  2026-03-05
- [x] T30.3 Run golangci-lint on graph/  Owner: TBD  Est: 15m  2026-03-05

### E31: Continuous Batching (O44)

- [x] T31.1 Implement batch scheduler in serve/  Owner: TBD  Est: 4h  2026-03-05
- [x] S31.1.1 Unit tests for batch scheduler  Owner: TBD  Est: 2h  2026-03-05
- [x] T31.2 Implement batched forward pass in generate/  Owner: TBD  Est: 3h  2026-03-05
- [x] S31.2.1 Tests for batched generation  Owner: TBD  Est: 1.5h  2026-03-05
- [x] T31.3 Wire batch scheduler into serve HTTP handler  Owner: TBD  Est: 2h  2026-03-05
- [x] S31.3.1 Integration tests  Owner: TBD  Est: 1h  2026-03-05
- [x] T31.4 Run golangci-lint on serve/ and generate/  Owner: TBD  Est: 15m  2026-03-05

### E32: Quantized CUDA Kernel (O41)

- [x] T32.1 CUDA Q4_0 dequant-GEMM kernel  Owner: TBD  Est: 4h  2026-03-05
- [x] S32.1.1 Tests and benchmarks (CUDA-gated)  Owner: TBD  Est: 1.5h  2026-03-05
- [x] T32.2 Wire CUDA Q4 GEMM into GPUEngine  Owner: TBD  Est: 1.5h  2026-03-05
- [x] S32.2.1 Integration tests  Owner: TBD  Est: 1h  2026-03-05
- [x] T32.3 Run golangci-lint on compute/ and internal/cuda/  Owner: TBD  Est: 15m  2026-03-05

### E33: Benchmark Suite (O45)

- [x] T33.1 Create tok/s benchmark framework  Owner: TBD  Est: 2h  2026-03-05
- [x] S33.1.1 Benchmark validation tests  Owner: TBD  Est: 30m  2026-03-05
- [x] T33.2 Add CPU GEMM micro-benchmarks  Owner: TBD  Est: 1h  2026-03-05
- [x] S33.2.1 Verify benchmark correctness  Owner: TBD  Est: 30m  2026-03-05
- [x] T33.3 Add memory profiling benchmark  Owner: TBD  Est: 1h  2026-03-05
- [x] T33.4 Run golangci-lint on tests/benchmark/  Owner: TBD  Est: 15m  2026-03-05

### Phase 26 -- COMPLETE

### E36: End-to-End Quantized Inference Pipeline (O46)

- [x] T36.1 Create Q4 quantized ZMF model from Gemma 3 2B  Owner: TBD  Est: 2h  2026-03-05
  - Used `zmf-quantize` tool (zonnx) to quantize F32 ZMF to Q4. Fixed quantizer
    to skip norm/embed/bias/small tensors (caused NaN). Model: 4GB F32 -> 1.5GB Q4.
- [x] S36.1.1 Smoke test: load Q4 model and run single forward pass  Owner: TBD  Est: 1h  2026-03-05
- [x] T36.2 Profile and fix Q4 inference bottlenecks  Owner: TBD  Est: 4h  2026-03-05
  - Baseline: 1.96 tok/s (Q4), 2.23 tok/s (F32) on DGX Spark ARM64.
  - Top bottleneck: CPUEngine.Transpose at 90% of CPU time.
- [x] S36.2.1 Benchmark: tok/s for Q4 Gemma 3 2B on CPU and GPU  Owner: TBD  Est: 1h  2026-03-05
- [x] T36.3 Optimize hot path based on profiling  Owner: TBD  Est: 4h  2026-03-05
  - Added cache-friendly blocked 2D transpose (64x64 tiles) for axes=[1,0].
  - Result: 1.96 -> 3.60 tok/s (Q4), 2.23 -> 3.51 tok/s (F32) -- 1.84x speedup.
- [x] S36.3.1 Before/after benchmark comparison  Owner: TBD  Est: 30m  2026-03-05
- [x] T36.4 Run golangci-lint on affected packages  Owner: TBD  Est: 15m  2026-03-05

### E34: PagedAttention (O47)

- [x] T34.1 Design paged KV cache data structure  Owner: TBD  Est: 2h  2026-03-05
- [x] S34.1.1 Unit tests for PagedKVCache  Owner: TBD  Est: 2h  2026-03-05
- [x] T34.2 Block pool with configurable max memory  Owner: TBD  Est: 2h  2026-03-05
- [x] S34.2.1 Unit tests for BlockPool  Owner: TBD  Est: 1h  2026-03-05
- [x] T34.3 Integrate PagedKVCache into Generator  Owner: TBD  Est: 3h  2026-03-05
- [x] S34.3.1 Integration tests for paged generation  Owner: TBD  Est: 1.5h  2026-03-05
- [x] T34.4 Wire paged KV into attention layers  Owner: TBD  Est: 3h  2026-03-05
- [x] S34.4.1 GQA + paged KV correctness tests  Owner: TBD  Est: 1h  2026-03-05
- [x] T34.5 Memory efficiency benchmark  Owner: TBD  Est: 1h  2026-03-05
  - Paged uses 46% of pre-allocated (target <= 50%).
- [x] S34.5.1 Benchmark report  Owner: TBD  Est: 30m  2026-03-05
- [x] T34.6 Run golangci-lint on generate/ and layers/attention/  Owner: TBD  Est: 15m  2026-03-05

### E35: Speculative Decoding (O48)

- [x] T35.1 Implement draft-verify decode loop  Owner: TBD  Est: 4h  2026-03-05
- [x] S35.1.1 Unit tests for speculative decode  Owner: TBD  Est: 2h  2026-03-05
- [x] T35.2 Token verification with KV cache rollback  Owner: TBD  Est: 3h  2026-03-05
- [x] S35.2.1 KV rollback correctness tests  Owner: TBD  Est: 1h  2026-03-05
- [x] T35.3 Adaptive draft length  Owner: TBD  Est: 2h  2026-03-05
- [x] S35.3.1 Adaptive length tests  Owner: TBD  Est: 1h  2026-03-05
- [x] T35.4 Wire speculative decoding into serve/  Owner: TBD  Est: 2h  2026-03-05
- [x] S35.4.1 Integration tests for speculative serve  Owner: TBD  Est: 1h  2026-03-05
- [x] T35.5 Benchmark: speculative vs baseline decode  Owner: TBD  Est: 1h  2026-03-05
  - Baseline: 3.53 tok/s, Speculative (k=4, same model): 2.17 tok/s.
- [x] S35.5.1 Benchmark report  Owner: TBD  Est: 30m  2026-03-05
- [x] T35.6 Run golangci-lint on generate/ and serve/  Owner: TBD  Est: 15m  2026-03-05

### E37: GGUF Model Import (O49)

- [x] T37.1 Implement GGUF file parser  Owner: TBD  Est: 4h  2026-03-05
- [x] S37.1.1 Unit tests for GGUF parser  Owner: TBD  Est: 2h  2026-03-05
- [x] T37.2 GGUF tensor loader with Q4_0/Q8_0 support  Owner: TBD  Est: 3h  2026-03-05
- [x] S37.2.1 Tensor loading tests  Owner: TBD  Est: 1.5h  2026-03-05
- [x] T37.3 GGUF architecture mapping  Owner: TBD  Est: 3h  2026-03-05
- [x] S37.3.1 Architecture mapping tests  Owner: TBD  Est: 1h  2026-03-05
- [x] T37.4 Integrate GGUF loader into inference.Load  Owner: TBD  Est: 2h  2026-03-05
- [x] S37.4.1 End-to-end GGUF inference test  Owner: TBD  Est: 1h  2026-03-05
- [x] T37.5 Run golangci-lint on model/gguf/ and inference/  Owner: TBD  Est: 15m  2026-03-05

### E38: Performance CI Dashboard (O50)

- [x] T38.1 Create benchmark runner script  Owner: TBD  Est: 2h  2026-03-05
- [x] S38.1.1 Validate benchmark script output  Owner: TBD  Est: 30m  2026-03-05
- [x] T38.2 GitHub Actions workflow for benchmarks  Owner: TBD  Est: 2h  2026-03-05
- [x] S38.2.1 Workflow validation  Owner: TBD  Est: 30m  2026-03-05
- [x] T38.3 DGX Spark GPU benchmark integration  Owner: TBD  Est: 2h  2026-03-05
- [x] S38.3.1 GPU benchmark validation  Owner: TBD  Est: 30m  2026-03-05
- [x] T38.4 Run golangci-lint on scripts/  Owner: TBD  Est: 15m  2026-03-05

### Phase 27 -- NEW

### E39: Eliminate Redundant Transposes (O51)

- [x] T39.1 Fold weight transposes at model load time  Owner: TBD  Est: 4h
  - Created `graph/optimize.go` with `FoldConstantTransposes` pass.
  - Detects Transpose nodes with Parameter/Constant inputs, pre-applies the
    transpose at load time, rewires all consumers, removes dead nodes.
  - Handles any permutation (2D, 3D/4D). Wired into `model.BuildFromZMF`.
  - Commits: d4bd2df, b6ad94c.

- [x] S39.1.1 Unit tests for transpose folding  Owner: TBD  Est: 1h
  - 5 tests in `graph/optimize_test.go`: constant input folding, dynamic input
    not folded, output within 1e-6 tolerance, multiple consumers, no-op graph.

- [x] T39.2 Fast path for 3D/4D attention transposes  Owner: TBD  Est: 3h
  - Added blocked 4D transpose for axes=[0,2,1,3] using 32x32 tiling.
  - Copies contiguous D-element rows for cache efficiency.
  - Benchmark: 35x faster than generic path (1.6ms vs 55.8ms for 4x8x512x64).
  - Commit: 17dd56b.

- [x] S39.2.1 Benchmark: 3D/4D transpose speedup  Owner: TBD  Est: 30m
  - BenchmarkCPUEngineTranspose4D in compute/cpu_engine_bench_test.go.
  - Correctness test TestCPUEngine_Transpose4D in compute/cpu_engine_test.go.

- [ ] T39.3 End-to-end benchmark after transpose elimination  Owner: TBD  Est: 1h
  - Run Gemma 3 2B Q4 with transpose folding enabled on DGX Spark.
  - Profile with pprof to verify Transpose is < 5% of CPU.
  - Measure tok/s improvement.
  - Acceptance: measurable speedup, Transpose no longer dominant in profile.
  - Dependencies: T39.1, T39.2. **Requires DGX Spark access.**

- [ ] S39.3.1 Before/after profile comparison  Owner: TBD  Est: 30m

- [x] T39.4 Run golangci-lint on model/ and compute/  Owner: TBD  Est: 15m
  - 0 lint issues on graph/, model/, compute/.

### E40: Tensor Arena / Zero-Alloc Forward (O54)

- [x] T40.1 Design and implement TensorPool  Owner: TBD  Est: 4h
  - Created `compute/pool.go` with `TensorPool[T]` type.
  - Shape-keyed mutex-guarded free lists. Acquire zeroes returned buffers.
  - Commit: fab02f3.

- [x] S40.1.1 Unit tests for TensorPool  Owner: TBD  Est: 1h
  - 5 tests + 2 benchmarks in `compute/pool_test.go`.

- [x] T40.2 Wire pool into graph executor  Owner: TBD  Est: 4h
  - Added `WithPool(TensorReleaser)` to `graph.Graph`.
  - Forward computes reference counts, releases intermediates when refcount
    hits zero. Protects inputs, output, and constant/parameter nodes.
  - Commit: c2a7e6f.

- [x] S40.2.1 Allocation benchmark: before/after pool  Owner: TBD  Est: 30m
  - 5 integration tests in `graph/pool_integration_test.go`.
  - Pool benchmark in `compute/pool_test.go` shows 0 allocs for reuse cycle.

- [ ] T40.3 Benchmark tok/s with pool enabled  Owner: TBD  Est: 1h
  - Run Gemma 3 2B Q4 with tensor pool.
  - Verify tok/s improvement from reduced GC pressure.
  - Acceptance: < 100 allocs/token. Measurable tok/s improvement.
  - Dependencies: T40.2.

- [ ] S40.3.1 Allocation profile comparison  Owner: TBD  Est: 30m

- [x] T40.4 Run golangci-lint on compute/ and graph/  Owner: TBD  Est: 15m

### E41: GPU Inference Pipeline (O52)

- [ ] T41.1 Validate GPU forward pass on Gemma 3 2B  Owner: TBD  Est: 4h
  - Load Gemma 3 2B Q4 ZMF with `inference.Load(..., WithDevice("cuda"))`.
  - Run single forward pass. Verify output shape and non-NaN values.
  - Identify and fix any ops that fall back to CPU (log warnings for
    unsupported GPU ops).
  - Acceptance: forward pass completes on GPU without CPU fallback for
    core ops (MatMul, Add, Mul, Softmax, RMSNorm, RoPE, Gather).
  - Dependencies: none.
  - Risk: some ops may not have GPU implementations yet (R53).

- [ ] S41.1.1 GPU forward pass smoke test  Owner: TBD  Est: 1h

- [ ] T41.2 Profile and fix GPU bottlenecks  Owner: TBD  Est: 4h
  - Profile with CUDA events or nsys.
  - Typical issues: excessive CPU-GPU synchronization, small kernel launches,
    data transfers for unsupported ops, sub-optimal memory layout.
  - Fix top 3 bottlenecks.
  - Acceptance: GPU profile shows > 80% time in compute kernels.
  - Dependencies: T41.1.

- [ ] S41.2.1 GPU profile analysis  Owner: TBD  Est: 1h

- [ ] T41.3 GPU generation benchmark  Owner: TBD  Est: 2h
  - Run full generation (32 tokens) on GPU.
  - Measure tok/s. Target >= 60 tok/s.
  - Compare with CPU baseline.
  - Acceptance: GPU tok/s >= 10x CPU tok/s.
  - Dependencies: T41.2.

- [ ] S41.3.1 CPU vs GPU benchmark report  Owner: TBD  Est: 30m

- [ ] T41.4 Run golangci-lint on compute/ and inference/  Owner: TBD  Est: 15m

### E42: GGUF End-to-End Inference (O53)

- [ ] T42.1 Graph template builder for Llama architecture  Owner: TBD  Est: 6h
  - Create `inference/gguf_builder.go` with
    `BuildLlamaGraph(model *GGUFModel, engine Engine) (*graph.Graph, error)`.
  - Build the standard Llama graph: Embed -> [RMSNorm -> Attn -> RMSNorm ->
    MLP] x N -> RMSNorm -> LMHead.
  - Map GGUF tensor names (already canonical from E37 name mapping) to
    graph node parameters.
  - Support GQA (num_kv_heads < num_heads).
  - Acceptance: Llama 3.2 1B GGUF loads and produces non-NaN logits.
  - Dependencies: E37.

- [ ] S42.1.1 Llama GGUF forward pass test  Owner: TBD  Est: 1h

- [ ] T42.2 Graph template builder for Gemma architecture  Owner: TBD  Est: 3h
  - Extend for Gemma: shared embed/lm_head weight, GeGLU activation
    (GeLU instead of SiLU), embedding scaling by sqrt(hidden_size).
  - Acceptance: Gemma 3 2B GGUF loads and produces non-NaN logits.
  - Dependencies: T42.1.

- [ ] S42.2.1 Gemma GGUF forward pass test  Owner: TBD  Est: 1h

- [ ] T42.3 Tokenizer loading for GGUF models  Owner: TBD  Est: 3h
  - Extract tokenizer vocabulary from GGUF metadata (tokenizer.ggml.tokens,
    tokenizer.ggml.scores, tokenizer.ggml.merges).
  - Build a BPETokenizer from GGUF metadata without needing tokenizer.json.
  - Fallback: if GGUF tokenizer data is absent, look for tokenizer.json in
    the same directory as the GGUF file.
  - Acceptance: tokenizer encodes/decodes correctly for Llama and Gemma.
  - Dependencies: none.

- [ ] S42.3.1 GGUF tokenizer encode/decode tests  Owner: TBD  Est: 1h

- [ ] T42.4 Unified GGUF load function  Owner: TBD  Est: 2h
  - Create `inference.LoadGGUFModel(path string, opts ...Option) (*Model, error)`.
  - Dispatches to architecture-specific builder based on GGUF metadata
    `general.architecture`.
  - Wires tokenizer, graph, engine, and generator into a ready-to-use Model.
  - Acceptance: `LoadGGUFModel("model.gguf")` -> `model.Generate(ctx, prompt)`.
  - Dependencies: T42.1, T42.2, T42.3.

- [ ] S42.4.1 End-to-end GGUF generation test  Owner: TBD  Est: 1h

- [ ] T42.5 Run golangci-lint on inference/ and model/gguf/  Owner: TBD  Est: 15m

### E43: Operator Fusion (O55)

- [x] T43.1 Fused RMSNorm kernel  Owner: TBD  Est: 3h
  - Create `compute/fused_rmsnorm.go` that computes
    `x * rsqrt(mean(x^2) + eps) * weight` in a single pass over the data.
  - Avoids materializing the squared, mean, and rsqrt intermediate tensors.
  - Wire into the RMSNorm layer as an optimized path when engine supports it.
  - Acceptance: output matches unfused RMSNorm within 1e-5. Benchmark shows
    >= 2x improvement for typical hidden sizes (1152, 2048, 4096).
  - Dependencies: none.

- [x] S43.1.1 Fused RMSNorm correctness and benchmark tests  Owner: TBD  Est: 1h

- [x] T43.2 Fused RoPE kernel  Owner: TBD  Est: 3h
  - Create `compute/fused_rope.go` that applies rotary position embeddings
    in a single pass: interleave cos/sin multiply and rotate in one loop.
  - Avoid creating separate cos, sin, split, and rotate intermediate tensors.
  - Acceptance: output matches unfused RoPE within 1e-5. Benchmark shows
    >= 2x improvement.
  - Dependencies: none.

- [x] S43.2.1 Fused RoPE correctness and benchmark tests  Owner: TBD  Est: 1h

- [x] T43.3 Fused SiLU-gate kernel  Owner: TBD  Est: 2h
  - Create `compute/fused_silugate.go` that computes
    `silu(gate_proj(x)) * up_proj(x)` as a single element-wise pass after
    the two MatMuls.
  - Fuses: silu activation + element-wise multiply into one kernel.
  - Acceptance: output matches unfused path within 1e-5. Benchmark shows
    improvement for MLP forward.
  - Dependencies: none.

- [x] S43.3.1 Fused SiLU-gate correctness and benchmark tests  Owner: TBD  Est: 1h

- [ ] T43.4 End-to-end benchmark with all fusions enabled  Owner: TBD  Est: 1h
  - Run Gemma 3 2B Q4 with all fusions + transpose folding + tensor pool.
  - Measure final tok/s. Compare against Phase 26 baseline (3.60 tok/s).
  - Acceptance: >= 15 tok/s on DGX Spark CPU.
  - Dependencies: T43.1, T43.2, T43.3, E39, E40.

- [ ] S43.4.1 Final performance report  Owner: TBD  Est: 30m

- [x] T43.5 Run golangci-lint on compute/  Owner: TBD  Est: 15m

---

## 4. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M1: Zero-alloc inference | E25, E26 | none | COMPLETE -- model loads via mmap, decode loop has 0 allocs/token |
| M2: Quantized inference | E27, E28 | M1 | COMPLETE -- Q4_0 model loads, runs forward pass, output within tolerance |
| M3: Fast CPU GEMM | E29 | none | COMPLETE -- NEON/AVX2 SGEMM >= 2x gonum, Q4 dot fused |
| M4: Parallel and batched | E30, E31 | M1 | COMPLETE -- parallel graph, batch server |
| M5: GPU quantized + benchmarks | E32, E33 | M2 | COMPLETE -- CUDA Q4 GEMM 2383 GFLOPS, benchmark suite |
| M6: Validated pipeline | E36 | Phase 25 | COMPLETE -- Gemma 3 2B Q4 3.60 tok/s on DGX Spark |
| M7: Efficient serving | E34 | E36 | COMPLETE -- PagedAttention 46% memory of pre-alloc |
| M8: Fast decode | E35 | E36 | COMPLETE -- speculative decode pipeline validated |
| M9: Model ecosystem | E37 | none | COMPLETE -- GGUF files parse, load tensors, map names |
| M10: Regression tracking | E38 | E36 | COMPLETE -- CI tracks tok/s, alerts on regressions |
| M11: Zero-transpose inference | E39 | none | Transpose < 5% of CPU, >= 8 tok/s |
| M12: Zero-alloc decode | E40 | E39 | < 100 allocs/token, >= 10 tok/s |
| M13: GPU inference | E41 | none | >= 60 tok/s on GB10 |
| M14: GGUF ecosystem | E42 | E37 | GGUF load -> generate, no external tools |
| M15: Fused ops | E43 | E39, E40 | >= 15 tok/s CPU with all optimizations |

Recommended execution order for Phase 27:

1. **E39** -- transpose elimination (biggest single-item speedup, purely mechanical)
2. **E40** -- tensor arena (allocation reduction, multiplicative with E39)
3. **[E41, E42]** -- GPU pipeline + GGUF builders (independent, parallelizable)
4. **E43** -- operator fusion (polish, benefits from profiling after E39+E40)

---

## 5. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R51 | Transpose folding changes graph semantics for edge cases | Incorrect output | Medium | Conservative pattern matching: only fold when Transpose input is a graph parameter with no other consumers. Full parity test suite validates. |
| R52 | Tensor pool introduces use-after-release bugs | Data corruption, flaky tests | Medium | Reference counting with debug mode that poisons released buffers. Race detector in CI. |
| R53 | GPU ops missing for some ONNX-derived nodes | Fallback to CPU kills throughput | High | Audit all node types in Gemma 3 graph before starting. Implement missing GPU ops first. |
| R54 | GGUF tokenizer metadata varies across model families | Tokenizer fails for some models | Medium | Start with Llama (well-documented). Test against HuggingFace tokenizer.json as ground truth. |
| R55 | Fused kernels diverge numerically from unfused path | Parity test failures | Low | Use Kahan summation for RMSNorm. Accept 1e-5 tolerance for fused vs unfused. |
| R56 | 3D/4D transpose patterns vary by architecture | Fast path misses common cases | Medium | Profile both Llama and Gemma to catalog all transpose axis patterns before implementing. |
| R40 | Plan9 assembly for NEON has poor documentation | SIMD kernel development slows | Medium | RESOLVED -- NEON and AVX2 SGEMM implemented in Phase 25 |
| R41 | Mmap on protobuf may not save memory if proto.Unmarshal copies internally | Mmap benefit reduced | Medium | RESOLVED -- mmap loader works in Phase 25 |
| R42 | Quantization accuracy loss breaks model parity tests | False test failures | Medium | Use wider tolerance for quantized models. Separate parity thresholds for float32 vs Q4. |
| R43 | Parallel graph executor introduces non-determinism in floating point | Flaky tests | Low | Use sequential executor for tests. Only enable parallel for benchmarks and production. |
| R44 | Continuous batching padding wastes compute for varied prompt lengths | Throughput gain < 2x | Medium | Implement length-sorted batching to minimize padding. |
| R45 | Real model tok/s far below target | Phase 26 goals unmet | Medium | RESOLVED -- profiled and optimized in E36 (1.84x speedup) |
| R46 | GGUF format has undocumented quirks per model architecture | Loader breaks on some models | Medium | Start with well-documented Llama format, add architectures incrementally |
| R47 | PagedAttention gather overhead negates memory savings | No net benefit for small batch | Low | RESOLVED -- paged uses 46% of pre-alloc in Phase 26 |
| R48 | Speculative decoding acceptance rate too low | < 1.5x speedup | Medium | Adaptive draft length (T35.3) + careful draft model selection |
| R49 | Self-hosted GitHub runner on DGX Spark has network/security issues | GPU CI blocked | Medium | Fall back to manual GPU benchmarks, document results in PR |
| R50 | Attention layer changes for paged KV break existing parity tests | Regression | Medium | RESOLVED -- parity suite passes in Phase 26 |
| R17 | ROCm tests cannot run in CI | No AMD GPU in CI | High | Tests skip gracefully. Validate on AMD hardware manually. |
| R24 | Three GPU backends increase maintenance burden | Bug surface area grows | High | GRAL abstraction minimizes duplication. Only vendor-specific code is in internal/ packages. |
| R31 | Single-GPU DGX Spark cannot validate multi-GPU code | NCCL and multi-GPU tests remain unvalidated | High | Tests skip gracefully. Second DGX Spark unit needed. |
| R39 | Some code paths unreachable without GPU hardware | Cannot achieve 100% on GPU-tagged files locally | Medium | Accept 95% floor for packages with GPU build-tag code. Validate on DGX Spark. |

---

## 6. Operating Procedure

### Definition of Done

A task is done when:
1. Implementation matches the acceptance criteria.
2. All existing tests pass (`go test ./... -count=1`).
3. New code has unit tests with >= 95% coverage.
4. `golangci-lint run ./package/` reports 0 issues.
5. `go vet ./package/` reports no issues.
6. Tests pass with `-race` flag.
7. Non-CUDA build (`go build ./...` without any GPU tag) compiles.
8. CUDA build (`go build -tags cuda ./...`) compiles.
9. Changes are committed in a small commit touching one directory only.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(tensor): add Q4_0 block storage`.
- Always run linters and formatters before committing.

### Benchmark Protocol

- All performance claims must be backed by `go test -bench` output.
- Benchmarks run with `-benchtime=3s -count=3` minimum for statistical validity.
- Report median and p99 latency, not just mean.
- Track regressions by storing benchmark results in `tests/benchmark/baseline/`.
- Profile before and after every optimization with pprof.

---

## 7. Progress Log

### Change Summary -- 2026-03-05

Merged Phase 27 epics (E39-E43) from docs/phase27.md into docs/plan.md. Added
5 new epics with 30 tasks/subtasks targeting 4.2x CPU throughput improvement
and GPU inference enablement. Added deliverables D81-D85, milestones M11-M15,
risks R51-R56. Marked Phase 26 milestones M6-M10 as COMPLETE. Updated success
metrics with Phase 27 targets. Added Phase 27 objectives O51-O55.

### Previous Entries

| Date | Phase | Summary |
|------|-------|---------|
| 2026-03-05 | 26 | Phase 26 ALL COMPLETE. E34-E38 done. Gemma 3 2B Q4: 3.60 tok/s. |
| 2026-03-05 | 26 | Merged Phase 26 work breakdown. Added E34-E38, M6-M10, R45-R50, D76-D80. |
| 2026-03-05 | 25 | Phase 25 ALL COMPLETE. All epics E25-E33 done. PR #35 merged E32 CUDA Q4 kernel. |
| 2026-03-05 | 25 | Plan created. Performance optimization phase scoped. |
| 2026-03-05 | 22-24 | Phases 22-24 COMPLETE. BF16 GEMM, unified memory, SigLIP fix, coverage push. |
| 2026-03-04 | 21 | Phase 21 COMPLETE. 18 ONNX fixes, 18 PASS across 6 model families. |
| 2026-03-03 | 20 | Phase 20 COMPLETE. ARM64 build (10 fixes), GPU tests (66 pkgs), benchmarks. |
| 2026-03-03 | 14-19 | Phases 14-19 COMPLETE. GRAL, ROCm, OpenCL, cuDNN backward, INT4/INT8, TRT dynamic. |
| 2026-03-03 | 10-13 | Phases 10-13 COMPLETE. Multi-GPU, cuDNN, TensorRT, CUTLASS. |

---

## 8. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. It is the single reference
  document. Design decisions are in docs/adr/ (ADR-001 through ADR-019).
- **Phases 1-26:** All complete. No active development tasks.
- **Phase 27:** Inference throughput. This plan is the source of truth.
  See also docs/phase27.md for the standalone Phase 27 design document.
- **Quality:** See docs/QUALITY.md for test coverage report. 9 packages at 100%,
  42 of 50 at >= 95%.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - CUDA+CUTLASS: `go build -tags cuda,cutlass ./...`
  - CUDA on DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build -tags cuda,cutlass ./...`
  - ROCm: `go build -tags rocm ./...`
  - OpenCL: `go build -tags opencl ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key Phase 27 Starting Points

1. **E39 (start here):** Fold weight transposes at load time in the graph builder.
   Key files: `model/builder.go` (BuildFromZMF), `compute/cpu_engine.go` (Transpose),
   `graph/graph.go` (Forward). This is the biggest single-item speedup.
2. **E40:** TensorPool in `compute/pool.go`. Wire into `graph.Graph.Forward()` with
   reference counting. Target < 100 allocs/token.
3. **E41 (DGX Spark required):** GPU inference in `compute/gpu_engine.go`,
   `internal/cuda/`, `inference/inference.go`. Audit all node types first (R53).
4. **E42 (independent):** GGUF graph builders in `inference/gguf_builder.go`.
   Builds on E37 parser. Download Llama 3.2 1B GGUF for testing.
5. **E43:** Fused kernels in `compute/fused_*.go`. Profile after E39+E40 to
   confirm remaining bottlenecks.

### Baseline Performance Numbers

| Model | Params | Quant | CPU tok/s (DGX) | GPU tok/s (DGX) | CPU Target | GPU Target |
|-------|--------|-------|-----------------|-----------------|------------|------------|
| Gemma 3 2B | 2.6B | Q4_0 | 3.60 | untested | >= 15 | >= 60 |
| Gemma 3 2B | 2.6B | F32 | 3.51 | untested | -- | -- |

### External Dependencies

- **DGX Spark (ndungu@192.168.86.250, aitopatom-bfc8):**
  - Go 1.26.0 for linux/arm64, cuDNN 9.19.1, TensorRT 10.15.1,
    NCCL 2.29.7, CUTLASS 4.2, CUDA 13.0.2 / driver 580.126.09.
  - Models: ~/models/gemma3-q4/ (Q4 ZMF), ~/models/gemma3/ (F32 ZMF).
  - Repos: ~/zerfoo/, ~/zonnx/ (main), ~/zmf/ (fix/attribute-tensor branch).
- HIP SDK (>= 5.0) for AMD ROCm backend.
- OpenCL 2.0+ headers + CLBlast for OpenCL backend.
- Second DGX Spark unit (optional) for multi-GPU validation via ConnectX-7.

### Remaining Hardware-Blocked Items

1. **Multi-GPU parity test** -- requires >= 2 CUDA devices (second DGX Spark
   via ConnectX-7). 6 tests skip on single-GPU. See ADR-017.
2. **DeepSeek V3 parity** -- 671B MoE exceeds 128GB DGX Spark memory.

### Known Untestable Gaps

- health: EngineCheck takes concrete *CPUEngine type, preventing mock testing
- layers/attention: dupl linter blocks MLA Forward engine error test
- Most remaining gaps are tensor.New unreachable error paths
- cmd tools: main() and os.Exit paths

---

## 9. Appendix

### Performance Target Reference

| Model | Parameters | Quantization | Ollama CPU tok/s | Ollama GPU tok/s | Zerfoo Target CPU | Zerfoo Target GPU |
|-------|-----------|-------------|-----------------|-----------------|-------------------|-------------------|
| Llama 3.2 1B | 1.2B | Q4_0 | ~40 | ~120 | >= 30 | >= 100 |
| Gemma 3 2B | 2.6B | Q4_0 | ~20 | ~80 | >= 15 | >= 60 |
| Llama 3.1 8B | 8B | Q4_0 | ~10 | ~40 | >= 8 | >= 30 |

### GGUF Format Quick Reference

```
Header:
  magic: "GGUF" (4 bytes)
  version: uint32 (expect 3)
  tensor_count: uint64
  metadata_kv_count: uint64

Metadata KV:
  key: string (uint64 len + bytes)
  value_type: uint32 (enum)
  value: varies by type

Tensor Info:
  name: string
  n_dimensions: uint32
  dimensions: uint64[n_dimensions]
  type: uint32 (Q4_0=2, Q8_0=8, F32=0, F16=1)
  offset: uint64 (from start of tensor data)

Tensor Data:
  Aligned to tensor_data_alignment (default 32 bytes)
  Raw quantized/float data in GGML format
```

### PagedAttention Block Layout

```
Block (16 tokens):
  K: [num_layers, 16, head_dim]  -- key vectors for 16 positions
  V: [num_layers, 16, head_dim]  -- value vectors for 16 positions

Block Table (per sequence):
  [block_0_ptr, block_1_ptr, ..., block_N_ptr]
  Sequence of length L uses ceil(L/16) blocks.

Free Pool:
  Stack of available block pointers.
  Pre-allocated at startup based on max_memory config.
```

### Speculative Decoding Algorithm

```
Input: target model M, draft model D, prompt tokens
Output: generated tokens

loop:
  1. D generates N draft tokens autoregressively: d1, d2, ..., dN
  2. M runs single forward pass on [prompt + d1 + d2 + ... + dN]
     producing logits for all N+1 positions
  3. For i = 1 to N:
       if M accepts d_i (M's top-1 at position i-1 == d_i, or
           sampled from adjusted distribution):
         accept d_i
       else:
         reject d_i, sample replacement from M's distribution
         rollback D and M KV caches to position i-1
         break
  4. If all N accepted, sample one bonus token from M's logits at position N
  5. Emit accepted tokens
```

Expected speedup: `N * acceptance_rate / (1 + draft_cost/target_cost)`.
With N=4, acceptance=70%, draft 10x faster: `4 * 0.7 / (1 + 0.1) = 2.5x`.

### Transpose Folding Strategy

```
Before (current graph):
  weight_param -> Transpose(axes=[1,0]) -> MatMul(input, transposed_weight)

After folding:
  transposed_weight_param -> MatMul(input, transposed_weight_param)

Detection criteria:
  1. Node is Transpose
  2. Input is a constant parameter (model weight, not dynamic activation)
  3. Transpose node has exactly one consumer
  4. Consumer is MatMul

Folding action:
  1. Read weight data from parameter
  2. Apply transpose to weight data in-place
  3. Update parameter shape to transposed shape
  4. Replace MatMul input edge to point directly to parameter
  5. Remove Transpose node from graph
```

### Tensor Pool Strategy

```
Pool data structure:
  map[shapeHash]freeList  -- one free list per unique shape

Acquire(shape):
  hash = hashShape(shape)
  if freeList[hash] is non-empty:
    return pop(freeList[hash])  -- reuse existing buffer
  return allocate(shape)        -- new allocation

Release(tensor):
  hash = hashShape(tensor.Shape)
  push(freeList[hash], tensor)  -- return to pool

Reference counting in graph executor:
  Before forward: refcount[node] = len(node.Consumers)
  After each consume: refcount[node]--
  When refcount reaches 0: pool.Release(node.Output)
  Exception: graph output nodes are never released
```

### Production Readiness Scorecard (After Phase 26)

| Category | Score | How Achieved |
|----------|-------|-------------|
| Architecture | 10/10 | Multi-architecture config; MLA; multi-GPU device affinity |
| Core Functionality | 10/10 | 6 model families; multi-GPU inference; BF16 GEMM; unified memory |
| Testing | 10/10 | 18 model parity PASS; 42/50 packages >= 95% coverage |
| Error Handling | 9/10 | Structured logging, RPC validation, context deadlines |
| Security | 8/10 | TLS/mTLS for gRPC; HF_TOKEN for gated models |
| Observability | 8/10 | Logging, metrics, pprof endpoints |
| Configuration | 10/10 | Architecture-aware config parsing with HuggingFace field mapping |
| Operations | 10/10 | CLI pull/run/serve, OpenAI-compatible HTTP API |
| Documentation | 10/10 | Consolidated design.md + 19 ADRs |
| Performance | 7/10 | Mmap, pre-alloc KV, Q4/Q8, SIMD GEMM, CUDA Q4 2383 GFLOPS, PagedAttention, speculative decode, 3.60 tok/s CPU |

### Ollama/llama.cpp Feature Comparison (After Phase 26)

| Feature | Ollama | Zerfoo (After Phase 26) | Zerfoo (Phase 27 Target) |
|---------|--------|-------------------------|--------------------------|
| Model loading | mmap | mmap (E25) | mmap |
| KV cache | pre-allocated + paged | pre-allocated + paged (E26, E34) | + tensor arena |
| Quantization | Q4_0/Q4_K/Q8_0/Q5_K | Q4_0/Q8_0 (E27) | Q4_0/Q8_0 |
| CPU GEMM | hand-tuned SIMD | NEON/AVX2 (E29) | NEON/AVX2 |
| GPU GEMM | cuBLAS | cuBLAS + Q4 kernel (E32) | end-to-end GPU |
| Flash attention | yes | yes (CUTLASS) | yes |
| Continuous batching | yes | yes (E31) | yes |
| Speculative decoding | yes | yes (E35) | yes |
| PagedAttention | yes | yes (E34) | yes |
| GGUF import | native | parser/loader (E37) | full e2e (E42) |
| Streaming | yes | yes | yes |
| OpenAI API | yes | yes | yes |
| Perf CI tracking | internal | automated (E38) | automated |
| Transpose folding | yes | no | yes (E39) |
| Tensor arena | yes | no | yes (E40) |
| Fused ops | yes | no | yes (E43) |

### References

- llama.cpp quantization: Q4_0 block format = 18 bytes per 32 values.
- Go plan9 assembly: https://go.dev/doc/asm
- gonum BLAS: https://pkg.go.dev/gonum.org/v1/gonum/blas
- CUDA mixed-precision GEMM: cuBLAS cublasGemmEx with CUDA_R_8I.
- Go mmap: `syscall.Mmap` on unix, `golang.org/x/sys/windows` for Windows.
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- vLLM PagedAttention: https://arxiv.org/abs/2309.06180
- Speculative decoding: https://arxiv.org/abs/2211.17192
