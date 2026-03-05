# Zerfoo Performance Optimization Plan -- Phase 25

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with 40+ packages, a 34-method compute
Engine[T] interface, CPU and CUDA GPU backends, gRPC-based distributed
training, and comprehensive test coverage (95%+ across testable packages).
Phases 1-24 are complete (see docs/adr/001-019, docs/design.md).

The framework is functionally complete: 6 model families pass parity tests,
BF16 GEMM runs on GPU, flash attention is implemented, and KV caching exists
in the generate package. However, inference throughput lags behind production
frameworks like ollama/llama.cpp because of several architectural gaps:

1. **Model loading uses os.ReadFile**: The ZMF loader reads entire model files
   into memory via `os.ReadFile` + protobuf unmarshal. A 7B model (14GB) requires
   14GB of heap allocation before any tensor is created. Ollama uses mmap to
   memory-map model files, enabling instant "loading" with demand paging.

2. **KV cache allocates on every concat**: `generate/kvcache.go:ConcatAxis1`
   allocates a new slice and copies all previous KV data on every token. For a
   2048-token sequence with 32 layers, this is 32 * 2048 growing allocations.
   Ollama pre-allocates the KV cache to max sequence length.

3. **No quantized inference kernels**: While INT4/INT8 GEMM kernels exist via
   CUTLASS (Phase 15), the inference pipeline only runs float32. Ollama uses
   Q4_0/Q4_K_M/Q8_0 quantized weights with optimized dequant-multiply kernels.

4. **Graph execution is fully sequential**: `graph.Forward` holds a mutex and
   executes nodes one at a time (graph/graph.go:27-68). Independent branches
   (e.g., Q/K/V projections in attention) are not parallelized.

5. **CPU MatMul uses gonum BLAS only**: `internal/xblas/gemm.go` delegates to
   gonum's BLAS which is a pure Go implementation. No SIMD, no AVX2/NEON
   optimizations. Ollama uses hand-tuned SIMD kernels via ggml.

6. **No continuous batching**: The server handles one request at a time.
   Ollama batches multiple requests and shares prefill computation.

7. **Tokenizer is pure Go BPE**: No performance issues at small scale but no
   pre-compiled vocabulary trie or parallel tokenization.

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

O39: Mmap-based model loading. Eliminate heap allocation for model weights.
O40: Pre-allocated KV cache. Zero allocation during autoregressive decode.
O41: Quantized inference pipeline (Q4_0, Q8_0). Run 4-bit models end-to-end.
O42: Parallel graph execution. Execute independent graph branches concurrently.
O43: Optimized CPU GEMM. Use NEON (ARM64) and AVX2 (x86-64) SIMD intrinsics.
O44: Continuous batching in the serve package.
O45: Benchmark suite with tok/s metric. Measure and track performance parity.

### Non-Goals

- Breaking changes to the Engine[T] or Node[T] interfaces.
- Replacing gRPC with a different RPC framework.
- Adding third-party test frameworks (testify, etc.).
- SSM/Mamba architectures (Falcon Mamba, RWKV, Jamba).
- Pipeline parallelism (splitting layers across GPUs).
- Multi-GPU KV cache partitioning.
- Tensor parallelism within a single operation.
- Vulkan compute backend.
- SYCL/oneAPI backend (use OpenCL for Intel GPUs instead).
- FP4 kernel implementation (blocked on upstream CUTLASS SM121 FP4 fixes).
- ConnectX-7 multi-node inference (requires second DGX Spark unit).
- Speculative decoding (future phase).
- PagedAttention (future phase, after continuous batching).
- GGUF format support (use ZMF with quantized tensors instead).
- Training performance (focus is inference throughput).

### Constraints and Assumptions

- Use Go standard library only where possible. Minimize new dependencies.
- All CUDA code behind `//go:build cuda` build tags.
- All ROCm code behind `//go:build rocm` build tags.
- All OpenCL code behind `//go:build opencl` build tags.
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- Table-driven tests using the standard testing package.
- DGX Spark GB10 is ARM64 (aarch64), CUDA 13.0, sm_121 (Blackwell), 128GB
  unified LPDDR5X. Single GPU -- multi-GPU tests require two units via ConnectX-7.
- Assembly (NEON/AVX2) files use Go's plan9 assembler syntax with build tags.
- Benchmark target: >= 30 tok/s for Llama 3.2 1B on CPU (Apple M-series),
  >= 100 tok/s on DGX Spark GPU. Ollama achieves ~40 tok/s CPU, ~120 tok/s GPU
  for similar model sizes.

### Success Metrics

| Metric | Current | Target | How Measured |
|--------|---------|--------|-------------|
| Model load time (7B) | ~10s (full read + unmarshal) | < 0.5s (mmap) | Benchmark in model/ |
| Decode tok/s CPU (1B) | ~5 tok/s (est.) | >= 30 tok/s | Benchmark in generate/ |
| Decode tok/s GPU (1B) | ~20 tok/s (est.) | >= 100 tok/s | Benchmark in generate/ |
| KV cache allocs/token | O(seq_len) per token | 0 per token | Go benchmark -benchmem |
| Memory for 7B Q4_0 | 14GB (float32) | ~4GB (4-bit) | Runtime MemStats |
| Concurrent requests | 1 (sequential) | 8+ (batched) | Load test on serve/ |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D66 | Mmap ZMF loader | Eliminate model load heap allocation |
| D67 | Pre-allocated KV cache with ring buffer | Zero-alloc decode loop |
| D68 | Q4_0 and Q8_0 tensor storage and dequant | Run quantized models |
| D69 | Quantized MatMul kernel (CPU Q4_0 * FP32) | Core quantized inference op |
| D70 | NEON SGEMM kernel (ARM64) | Fast CPU MatMul on Apple Silicon and DGX Spark |
| D71 | AVX2 SGEMM kernel (x86-64) | Fast CPU MatMul on Intel/AMD |
| D72 | Parallel graph executor | Concurrent independent branch execution |
| D73 | Continuous batching in serve/ | Multi-request throughput |
| D74 | tok/s benchmark suite | Performance regression tracking |
| D75 | Quantized CUDA dequant-GEMM kernel | GPU quantized inference |

### Out of Scope

- GGUF format parser (use ZMF with quantized tensor protos).
- Speculative decoding.
- PagedAttention.
- Training loop optimization.
- Multi-node inference.

---

## 3. Checkable Work Breakdown

### E25: Mmap-Based Model Loading (O39)

- [x] T25.1 Add mmap file reader to model/ package  Owner: TBD  Est: 2h
  - Create `model/mmap.go` with `MmapReader` that memory-maps a file via `syscall.Mmap`
    (unix) and `syscall.CreateFileMapping` (windows, behind build tag).
  - Returns `[]byte` slice backed by mmap. `Close()` calls `syscall.Munmap`.
  - Acceptance: `MmapReader` opens a file, returns byte slice, Close unmaps.
  - Dependencies: none.

- [x] S25.1.1 Unit tests for MmapReader  Owner: TBD  Est: 1h
  - Write temp file, mmap it, verify contents match, close, verify no leak.
  - Test error cases: missing file, empty file.

- [x] T25.2 Add streaming protobuf decoder for ZMF  Owner: TBD  Est: 3h
  - Create `model/zmf_stream.go` with `LoadZMFMmap(path string) (*zmf.Model, error)`
    that mmaps the file and passes the mmap slice to `proto.Unmarshal`.
  - This avoids the `os.ReadFile` copy -- protobuf unmarshals from the mmap
    slice directly. The mmap slice stays mapped until the model is closed.
  - For tensor data: instead of copying tensor bytes into Go slices, store
    the mmap byte range reference and decode lazily on first access.
  - Acceptance: `LoadZMFMmap` loads a model with zero `os.ReadFile` calls.
    Memory usage delta is < 1MB for metadata (tensor data stays in mmap pages).
  - Dependencies: T25.1.

- [x] S25.2.1 Unit tests for LoadZMFMmap  Owner: TBD  Est: 1h
  - Round-trip test: save a small ZMF model, load via mmap, verify graph.
  - Benchmark: compare LoadZMF vs LoadZMFMmap for a 100MB test file.

- [x] T25.3 Integrate mmap loader into inference.Load  Owner: TBD  Est: 1h
  - Add `WithMmap(bool)` option to inference.Load. Default to true on unix.
  - When mmap is enabled, use `LoadZMFMmap` instead of `LoadModelFromZMF`.
  - Acceptance: `inference.Load("model-id", WithMmap(true))` uses mmap path.
  - Dependencies: T25.2.

- [x] S25.3.1 Tests for inference.Load with mmap  Owner: TBD  Est: 30m

- [x] T25.4 Run golangci-lint on model/  Owner: TBD  Est: 15m
  - Dependencies: T25.3.

### E26: Pre-Allocated KV Cache (O40)

- [x] T26.1 Implement ring-buffer KV cache  Owner: TBD  Est: 3h
  - Rewrite `generate/kvcache.go`: pre-allocate `[batch, maxSeqLen, dim]` tensors
    for each layer's K and V at construction time.
  - `Update(layer, newK, newV)` writes into the next position via `copy()` on the
    backing slice. No allocation. Track write position with an integer cursor.
  - `Get(layer)` returns a view (sub-slice) of the pre-allocated tensor covering
    `[0:cursor]` on the sequence axis.
  - `ConcatAxis1` is no longer needed -- remove it.
  - Acceptance: `go test -bench=. -benchmem` shows 0 allocs/op in Update.
  - Dependencies: none.
  - Risk: Changing KV cache shape semantics may break GQA attention layer's
    cache integration. Verify in S26.1.1.

- [x] S26.1.1 Unit tests for ring-buffer KV cache  Owner: TBD  Est: 1.5h
  - Test: create cache, update 100 times, verify data integrity at each step.
  - Test: cursor wraps correctly (if implementing ring buffer with overwrite).
  - Test: zero allocations via testing.B benchmark.
  - Test: concurrent reads during update (if applicable).

- [x] T26.2 Update generate.Generator to use pre-allocated cache  Owner: TBD  Est: 1.5h
  - Generator.Generate already creates KVCache with maxSeqLen from ModelConfig.
  - HeadDim/NumKVHeads not needed: cache detects batch/dim lazily on first Update.
  - Existing generator tests pass with pre-allocated cache.

- [x] S26.2.1 Tests for generator with pre-allocated cache  Owner: TBD  Est: 1h
  - All existing generator tests pass unchanged.

- [x] T26.3 Update GQA attention to use view-based cache  Owner: TBD  Est: 2h
  - GQA cache.Update/Get API unchanged. View tensors have correct shapes.
  - GQA cache tests (TestGQA_CachedForward, CacheLayerIndex) pass.

- [x] S26.3.1 Tests for GQA with view cache  Owner: TBD  Est: 1h

- [x] T26.4 Run golangci-lint on generate/ and layers/attention/  Owner: TBD  Est: 15m
  - 0 issues.

### E27: Quantized Tensor Storage (O41)

- [ ] T27.1 Add Q4_0 block format to tensor/  Owner: TBD  Est: 3h
  - Create `tensor/quantized.go` with `Q4Storage` type.
  - Q4_0 format: 32 values per block. Each block = 2 bytes scale (float16) +
    16 bytes data (32 x 4-bit packed). Total = 18 bytes per 32 values.
  - Implement `Dequantize(dst []float32)` that unpacks blocks to float32.
  - Implement `QuantizeQ4(src []float32) *Q4Storage` that quantizes float32 to Q4_0.
  - Q4Storage implements a read-only subset of Storage[T] (Len, DeviceType).
  - Slice() returns dequantized float32 (copies). No Set() -- quantized tensors
    are immutable weights.
  - Acceptance: round-trip quantize-dequantize has max error < 0.1 for values
    in [-1, 1] range. Compression ratio is 8x vs float32.
  - Dependencies: none.

- [ ] S27.1.1 Unit tests for Q4_0 quantization  Owner: TBD  Est: 1.5h
  - Test: quantize known values, dequantize, verify within tolerance.
  - Test: block boundary handling (not multiple of 32).
  - Test: extreme values (0, max float32, negative).
  - Benchmark: dequantize throughput (GB/s).

- [ ] T27.2 Add Q8_0 block format to tensor/  Owner: TBD  Est: 2h
  - Q8_0 format: 32 values per block. Each block = 4 bytes scale (float32) +
    32 bytes data (32 x int8). Total = 36 bytes per 32 values.
  - Same interface as Q4Storage.
  - Acceptance: round-trip error < 0.01 for values in [-1, 1].
    Compression ratio is ~4.5x vs float32.
  - Dependencies: none.

- [ ] S27.2.1 Unit tests for Q8_0 quantization  Owner: TBD  Est: 1h

- [ ] T27.3 Add quantized tensor loading to ZMF loader  Owner: TBD  Est: 2h
  - Extend `model/zmf_loader.go:DecodeTensor` to detect quantized tensor data
    in the ZMF proto and construct Q4Storage or Q8Storage instead of
    regular Storage[float32].
  - Add `QuantType` field to ZMF tensor proto (or use existing dtype field).
  - Acceptance: A ZMF model with quantized weights loads correctly and
    returns dequantized values when accessed.
  - Dependencies: T27.1, T27.2.

- [ ] S27.3.1 Tests for quantized ZMF loading  Owner: TBD  Est: 1h

- [ ] T27.4 Add zonnx quantization pass  Owner: TBD  Est: 2h
  - In the zonnx CLI, add `--quantize q4_0` and `--quantize q8_0` flags
    to the convert command that quantize weights during ONNX-to-ZMF conversion.
  - Acceptance: `zonnx convert --quantize q4_0 model.onnx model-q4.zmf` produces
    a ZMF file ~4x smaller than float32.
  - Dependencies: T27.3.
  - Risk: This touches the zonnx repo, not zerfoo. Coordinate changes.

- [ ] S27.4.1 Tests for zonnx quantization  Owner: TBD  Est: 1h

- [ ] T27.5 Run golangci-lint on tensor/ and model/  Owner: TBD  Est: 15m
  - Dependencies: T27.3.

### E28: Quantized CPU MatMul Kernel (O41, O43)

- [ ] T28.1 Implement Q4_0 x FP32 MatMul in Go  Owner: TBD  Est: 3h
  - Create `internal/xblas/gemm_quant.go`.
  - `GemmQ4F32(m, n, k int, a *Q4Storage, b []float32, c []float32)`:
    For each output row, dequantize the Q4 row into a local float32 buffer,
    then dot-product with the float32 column.
  - Optimize: dequantize one block (32 values) at a time into a stack buffer
    to minimize allocation and maximize cache locality.
  - Acceptance: correct output vs reference (dequant-then-GEMM). At least
    2x faster than dequant-all-then-GEMM for 4096x4096 matrices.
  - Dependencies: T27.1.

- [ ] S28.1.1 Unit and benchmark tests  Owner: TBD  Est: 1.5h
  - Correctness: compare against reference float32 GEMM within Q4 tolerance.
  - Benchmark: measure GFLOPS for 512, 1024, 2048, 4096 sizes.

- [ ] T28.2 Implement Q8_0 x FP32 MatMul in Go  Owner: TBD  Est: 2h
  - Same pattern as T28.1 but for Q8_0 blocks.
  - Acceptance: correct within Q8 tolerance, faster than dequant-all.
  - Dependencies: T27.2.

- [ ] S28.2.1 Unit and benchmark tests  Owner: TBD  Est: 1h

- [ ] T28.3 Wire quantized MatMul into CPUEngine  Owner: TBD  Est: 2h
  - Modify `compute/cpu_engine.go:MatMul` to detect when input A has quantized
    storage (via type assertion or interface check) and dispatch to GemmQ4F32
    or GemmQ8F32 instead of xblas.GemmF32.
  - Acceptance: `CPUEngine.MatMul` with a Q4 weight tensor produces correct
    output and is faster than the float32 path for same logical dimensions.
  - Dependencies: T28.1, T28.2.

- [ ] S28.3.1 Integration tests for quantized MatMul via Engine  Owner: TBD  Est: 1h

- [ ] T28.4 Run golangci-lint on internal/xblas/ and compute/  Owner: TBD  Est: 15m
  - Dependencies: T28.3.

### E29: SIMD-Optimized CPU GEMM (O43)

- [ ] T29.1 NEON SGEMM kernel for ARM64  Owner: TBD  Est: 4h
  - Create `internal/xblas/gemm_neon_arm64.s` in Go plan9 assembly.
  - Implement a 4x4 micro-kernel using NEON FMLA (fused multiply-add).
  - Outer loop tiles M and N in 4x4 blocks; inner loop iterates K.
  - Build-tagged with `//go:build arm64`.
  - Pure Go fallback in `gemm_neon_generic.go` with `//go:build !arm64`.
  - Acceptance: >= 2x faster than gonum BLAS for 1024x1024 SGEMM on
    Apple M1/M2 or DGX Spark ARM64.
  - Dependencies: none.
  - Risk: Plan9 assembly for ARM64 NEON is less documented than x86. Use
    Go's NEON instruction mnemonics (VFMLA, VLD1, VST1).

- [ ] S29.1.1 Tests for NEON SGEMM  Owner: TBD  Est: 1.5h
  - Correctness: compare against gonum BLAS for various sizes.
  - Benchmark: measure GFLOPS on ARM64.

- [ ] T29.2 AVX2 SGEMM kernel for x86-64  Owner: TBD  Est: 4h
  - Create `internal/xblas/gemm_avx2_amd64.s` in Go plan9 assembly.
  - Implement a 8x1 micro-kernel using AVX2 VFMADD231PS (8-wide FMA).
  - Build-tagged with `//go:build amd64`.
  - Pure Go fallback already exists (gonum BLAS).
  - Acceptance: >= 2x faster than gonum BLAS for 1024x1024 SGEMM on
    modern x86-64 (Intel 10th gen+, AMD Zen3+).
  - Dependencies: none.

- [ ] S29.2.1 Tests for AVX2 SGEMM  Owner: TBD  Est: 1.5h

- [ ] T29.3 Wire SIMD GEMM into xblas.GemmF32  Owner: TBD  Est: 1h
  - Modify `internal/xblas/gemm.go:GemmF32` to dispatch to the SIMD kernel
    when available (arm64 or amd64) and fall back to gonum for other archs.
  - Acceptance: all existing MatMul tests pass. Benchmark shows SIMD speedup.
  - Dependencies: T29.1 or T29.2.

- [ ] S29.3.1 Integration tests  Owner: TBD  Est: 30m

- [ ] T29.4 NEON Q4 dot product kernel  Owner: TBD  Est: 3h
  - Create `internal/xblas/qdot_neon_arm64.s` with a NEON-accelerated
    Q4_0 dequant-and-dot-product kernel.
  - Dequantize 32 values (one Q4 block) into 4 NEON registers, then FMLA
    with 4 registers from the float32 vector.
  - Acceptance: >= 3x faster than scalar Q4 dequant-dot for 4096-dim vectors.
  - Dependencies: T28.1, T29.1.

- [ ] S29.4.1 Tests for NEON Q4 dot product  Owner: TBD  Est: 1h

- [ ] T29.5 Run golangci-lint on internal/xblas/  Owner: TBD  Est: 15m
  - Dependencies: T29.3.

### E30: Parallel Graph Execution (O42)

- [ ] T30.1 Build dependency-aware parallel executor  Owner: TBD  Est: 4h
  - Create `graph/parallel.go` with `ParallelForward(ctx, inputs)`.
  - Compute in-degree for each node. Use a channel-based work queue.
  - Nodes with zero remaining in-degree are dispatched to a goroutine pool.
  - When a node completes, decrement in-degree of its dependents and enqueue
    newly-ready nodes.
  - Goroutine pool size = `runtime.GOMAXPROCS(0)` (CPU) or 1 (GPU, since
    CUDA ops are serialized on a single stream anyway).
  - Mutex on the memo map is replaced by per-node atomic or channel signaling.
  - Acceptance: identical output to sequential Forward. At least 1.3x faster
    on a 4-branch attention model on CPU.
  - Dependencies: none.
  - Risk: Non-deterministic execution order may cause subtle bugs if any node
    has hidden mutable state. All current layers are stateless in Forward
    (parameters are read-only). KV cache is per-context, not per-graph.

- [ ] S30.1.1 Unit tests for parallel executor  Owner: TBD  Est: 2h
  - Test: diamond graph (A -> B,C -> D). B and C must run in parallel.
  - Test: linear graph (no parallelism). Output matches sequential.
  - Test: context cancellation mid-execution.
  - Benchmark: 4-branch graph, sequential vs parallel.

- [ ] T30.2 Add ForwardMode option to Graph  Owner: TBD  Est: 1h
  - Add `WithParallel(bool)` option. Default false for backward compatibility.
  - When true, `Graph.Forward` delegates to `ParallelForward`.
  - Acceptance: existing tests pass with both modes.
  - Dependencies: T30.1.

- [ ] S30.2.1 Tests for ForwardMode  Owner: TBD  Est: 30m

- [ ] T30.3 Run golangci-lint on graph/  Owner: TBD  Est: 15m
  - Dependencies: T30.2.

### E31: Continuous Batching (O44)

- [ ] T31.1 Implement batch scheduler in serve/  Owner: TBD  Est: 4h
  - Create `serve/batch.go` with `BatchScheduler` that collects incoming
    requests into batches.
  - Requests wait up to `batchTimeoutMs` (default 10ms) for the batch to fill.
  - Maximum batch size is configurable (default 8).
  - The scheduler groups requests by phase: prefill vs decode.
  - Prefill requests with different prompt lengths are padded to max length
    in the batch (left-pad with pad token).
  - Decode requests all have input length 1 (single token).
  - Acceptance: multiple concurrent requests are served in fewer forward passes
    than sequential processing.
  - Dependencies: none.
  - Risk: Padding introduces wasted compute. For very different prompt lengths,
    the overhead may exceed the batching benefit.

- [ ] S31.1.1 Unit tests for batch scheduler  Owner: TBD  Est: 2h
  - Test: 4 requests arrive within timeout, batched into 1 forward pass.
  - Test: timeout expires with 2 requests, batch of 2 fires.
  - Test: max batch size enforced.

- [ ] T31.2 Implement batched forward pass in generate/  Owner: TBD  Est: 3h
  - Modify `Generator` to support batch dimension > 1 in input tensors.
  - KV cache must be per-request (not per-batch). Each request in the batch
    has its own KV cache and cursor position.
  - After the batched forward pass, split output logits by batch index and
    sample independently per request.
  - Acceptance: batched generation produces identical output per-request
    as sequential generation.
  - Dependencies: T26.1, T31.1.

- [ ] S31.2.1 Tests for batched generation  Owner: TBD  Est: 1.5h

- [ ] T31.3 Wire batch scheduler into serve HTTP handler  Owner: TBD  Est: 2h
  - Modify `serve/server.go` chat completions handler to submit requests
    to the batch scheduler instead of calling Generator directly.
  - SSE streaming must still work: each request gets its own stream channel.
  - Acceptance: `ab -n 100 -c 8` against the server shows > 2x throughput
    compared to sequential.
  - Dependencies: T31.2.

- [ ] S31.3.1 Integration tests  Owner: TBD  Est: 1h

- [ ] T31.4 Run golangci-lint on serve/ and generate/  Owner: TBD  Est: 15m
  - Dependencies: T31.3.

### E32: Quantized CUDA Kernel (O41)

- [ ] T32.1 CUDA Q4_0 dequant-GEMM kernel  Owner: TBD  Est: 4h
  - Create `internal/cuda/kernels/gemm_quant.cu` with a kernel that:
    1. Loads Q4_0 blocks from global memory into shared memory.
    2. Dequantizes 32 values per thread block.
    3. Performs tiled GEMM with the dequantized values.
  - Use the cuBLAS mixed-precision GEMM API if available, or custom kernel.
  - Acceptance: correct output vs CPU Q4 GEMM. At least 5x faster than
    CPU Q4 GEMM for 4096x4096 on DGX Spark.
  - Dependencies: T27.1.
  - Risk: Requires DGX Spark for testing. Skip tests gracefully on CPU-only.

- [ ] S32.1.1 Tests and benchmarks (CUDA-gated)  Owner: TBD  Est: 1.5h

- [ ] T32.2 Wire CUDA Q4 GEMM into GPUEngine  Owner: TBD  Est: 1.5h
  - Modify `compute/gpu_kernels.go` to dispatch quantized MatMul to the
    CUDA Q4 kernel.
  - Acceptance: GPUEngine.MatMul with Q4 input produces correct output.
  - Dependencies: T32.1.

- [ ] S32.2.1 Integration tests  Owner: TBD  Est: 1h

- [ ] T32.3 Run golangci-lint on compute/ and internal/cuda/  Owner: TBD  Est: 15m
  - Dependencies: T32.2.

### E33: Benchmark Suite (O45)

- [ ] T33.1 Create tok/s benchmark framework  Owner: TBD  Est: 2h
  - Create `tests/benchmark/` with `BenchmarkTokPerSec` that:
    1. Loads a model (configurable via env var `BENCH_MODEL_PATH`).
    2. Runs generation for a fixed prompt and max_tokens.
    3. Measures wall time and computes tok/s.
    4. Reports via `b.ReportMetric(toksPerSec, "tok/s")`.
  - Include separate benchmarks for prefill (prompt processing) and decode
    (token generation).
  - Acceptance: `go test -bench=BenchmarkTokPerSec -run=^$ ./tests/benchmark/`
    produces tok/s metrics.
  - Dependencies: none.

- [ ] S33.1.1 Benchmark validation tests  Owner: TBD  Est: 30m

- [ ] T33.2 Add CPU GEMM micro-benchmarks  Owner: TBD  Est: 1h
  - Add benchmarks to `internal/xblas/` comparing gonum, NEON, and AVX2
    GEMM at sizes 128, 512, 1024, 2048, 4096.
  - Also benchmark Q4 and Q8 dequant-GEMM.
  - Acceptance: `go test -bench=. ./internal/xblas/` shows all variants.
  - Dependencies: T29.3.

- [ ] S33.2.1 Verify benchmark correctness  Owner: TBD  Est: 30m

- [ ] T33.3 Add memory profiling benchmark  Owner: TBD  Est: 1h
  - Benchmark that tracks `runtime.MemStats` during model load and generation.
  - Reports peak heap, total allocs, and allocs/token.
  - Acceptance: allocs/token is reported for KV cache performance tracking.
  - Dependencies: T26.1.

- [ ] T33.4 Run golangci-lint on tests/benchmark/  Owner: TBD  Est: 15m
  - Dependencies: T33.3.

---

## 4. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M1: Zero-alloc inference | E25, E26 | none | Model loads via mmap, decode loop has 0 allocs/token |
| M2: Quantized inference | E27, E28 | M1 | Q4_0 model loads, runs forward pass, output within tolerance |
| M3: Fast CPU GEMM | E29 | none | NEON/AVX2 SGEMM >= 2x gonum, Q4 dot >= 3x scalar |
| M4: Parallel and batched | E30, E31 | M1 | Parallel graph >= 1.3x speedup, batch server >= 2x throughput |
| M5: GPU quantized + benchmarks | E32, E33 | M2 | CUDA Q4 GEMM works, tok/s benchmarks report metrics |

Recommended execution order (parallelizable groups in brackets):

1. [E25, E26, E29, E33.T33.1] -- independent foundations
2. [E27, E30] -- quantized storage + parallel graph (independent)
3. [E28, E32] -- quantized kernels (depend on E27)
4. E31 -- continuous batching (depends on E26)
5. E33.T33.2-T33.4 -- final benchmarks (depends on everything)

---

## 5. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R40 | Plan9 assembly for NEON has poor documentation | SIMD kernel development slows | Medium | Reference Go stdlib crypto/aes arm64 assembly. Start with simple dot product before full GEMM. |
| R41 | Mmap on protobuf may not save memory if proto.Unmarshal copies internally | Mmap benefit reduced | Medium | Profile with pprof. If protobuf copies, use a flat binary format for tensor data alongside protobuf for metadata. |
| R42 | Quantization accuracy loss breaks model parity tests | False test failures | Medium | Use wider tolerance for quantized models. Separate parity thresholds for float32 vs Q4. |
| R43 | Parallel graph executor introduces non-determinism in floating point | Flaky tests | Low | Use sequential executor for tests. Only enable parallel for benchmarks and production. |
| R44 | Continuous batching padding wastes compute for varied prompt lengths | Throughput gain < 2x | Medium | Implement length-sorted batching to minimize padding. |
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
- Benchmarks run with `-benchtime=3s -count=5` for statistical validity.
- Report median and p99 latency, not just mean.
- Track regressions by storing benchmark results in `tests/benchmark/baseline/`.

---

## 7. Progress Log

| Date | Phase | Summary |
|------|-------|---------|
| 2026-03-05 | 25 | Plan created. Performance optimization phase scoped: mmap loading, pre-alloc KV cache, quantized inference, SIMD GEMM, parallel graph, continuous batching, benchmark suite. |
| 2026-03-05 | 22-24 | Phases 22-24 COMPLETE. BF16 GEMM, unified memory, SigLIP fix, coverage push, TODO fixes. ADR-019 written. Plan trimmed. |
| 2026-03-04 | 21 | Phase 21 COMPLETE. 18 ONNX fixes, 18 PASS across 6 model families. ADR-018 written. |
| 2026-03-03 | 20 | Phase 20 COMPLETE. ARM64 build (10 fixes), GPU tests (66 pkgs), benchmarks. ADR-017 written. |
| 2026-03-03 | 14-19 | Phases 14-19 COMPLETE. GRAL, ROCm, OpenCL, cuDNN backward, INT4/INT8, TRT dynamic. ADRs 011-016. |
| 2026-03-03 | 10-13 | Phases 10-13 COMPLETE. Multi-GPU, cuDNN, TensorRT, CUTLASS. ADRs 007-010. |

---

## 8. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. It is the single reference
  document. Design decisions are in docs/adr/ (ADR-001 through ADR-019).
- **Phases 1-24:** All complete. No active development tasks.
- **Phase 25:** Performance optimization. This plan is the source of truth.
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

### Key Performance Bottlenecks (Current)

1. `model/zmf_loader.go:21-37` -- `os.ReadFile` loads entire model into heap.
2. `generate/kvcache.go:100-139` -- `ConcatAxis1` allocates on every token.
3. `internal/xblas/gemm.go:15-21` -- gonum BLAS is pure Go, no SIMD.
4. `graph/graph.go:27-68` -- sequential node execution under mutex.
5. `inference/inference.go:191` -- float32 only, no quantized weight support.
6. `serve/` -- single-request processing, no batching.

### External Dependencies

- **DGX Spark (ndungu@192.168.86.250, aitopatom-bfc8):**
  - Go 1.26.0 for linux/arm64, cuDNN 9.19.1, TensorRT 10.15.1,
    NCCL 2.29.7, CUTLASS 4.2, CUDA 13.0.2 / driver 580.126.09.
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

### Production Readiness Scorecard (After Phase 24)

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
| Performance | 4/10 | BF16 GEMM and flash attention exist, but no quantization, no SIMD, no mmap |

### Ollama/llama.cpp Feature Comparison

| Feature | Ollama | Zerfoo (Current) | Zerfoo (Phase 25 Target) |
|---------|--------|-------------------|--------------------------|
| Model loading | mmap | os.ReadFile | mmap (E25) |
| KV cache | pre-allocated | concat-per-token | pre-allocated ring (E26) |
| Quantization | Q4_0/Q4_K/Q8_0/Q5_K | float32 only | Q4_0/Q8_0 (E27) |
| CPU GEMM | hand-tuned SIMD | gonum (pure Go) | NEON/AVX2 (E29) |
| GPU GEMM | cuBLAS | cuBLAS | cuBLAS + Q4 kernel (E32) |
| Flash attention | yes | yes (CUTLASS) | yes |
| Continuous batching | yes | no | yes (E31) |
| Speculative decoding | yes | no | future phase |
| PagedAttention | yes | no | future phase |
| Streaming | yes | yes | yes |
| OpenAI API | yes | yes | yes |

### References

- llama.cpp quantization: Q4_0 block format = 18 bytes per 32 values.
- Go plan9 assembly: https://go.dev/doc/asm
- gonum BLAS: https://pkg.go.dev/gonum.org/v1/gonum/blas
- CUDA mixed-precision GEMM: cuBLAS cublasGemmEx with CUDA_R_8I.
- Go mmap: `syscall.Mmap` on unix, `golang.org/x/sys/windows` for Windows.
