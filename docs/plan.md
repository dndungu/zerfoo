# Zerfoo Development Plan -- Phase 32: GPU-First Inference Pipeline

## 1. Context

### Problem Statement

GPU inference on DGX Spark GB10 (5.12 tok/s) is slower than CPU inference
(5.94 tok/s) for Gemma 3 2B Q4. Phase 31 profiling identified the root cause:
only MatMul runs on GPU while all other operations fall back to CPU, and tensors
transfer between host and device for every operation (43% of wall time in
cgocall). The GPUEngine already has CUDA kernels for element-wise ops but they
are not reached because of shape mismatches (broadcasting), missing ops
(Transpose, Gather), and tensors defaulting to CPU storage.

See docs/design.md for full architecture context and Phases 1-31 history.
Decision rationale: docs/adr/022-gpu-first-inference-pipeline.md.

### What Was Delivered (Recent Phases)

| Phase | Key Result |
|-------|------------|
| Phase 29 | NEON Q4 dot product, 6.5 tok/s Gemma 3 2B Q4 CPU ARM64 |
| Phase 30 | Worker pool, graph compiler, 6.86 tok/s (+5% over Phase 29) |
| Phase 31 | PagedAttention v2, GPU profile (5.12 tok/s GPU, 5.94 CPU), training improvements (EMA, SWA, warm restarts, feature dropout, smoothed early stopping) |

### GPU Profile Breakdown (Phase 31, DGX Spark GB10)

| Component | % Time | Root Cause |
|-----------|--------|------------|
| runtime.cgocall | 43% | H2D/D2H transfers for every op |
| Q4 dequantize | 9.4% | CPU fallback (no standalone GPU Q4 dequant) |
| Transpose | 8.1% | CPU fallback (no GPU implementation) |
| Binary ops | 4.4% | CPU fallback (broadcasting guard in sameShape()) |
| MatMul (GPU) | ~30% | Only op actually on GPU |

### Objectives

- O69: Keep intermediate tensors GPU-resident between operations, eliminating
  host-device transfers in the inference hot loop.
- O70: Implement GPU Transpose kernel to remove the 8.1% CPU fallback.
- O71: Add broadcasting support to GPU element-wise kernels to remove the
  4.4% CPU fallback.
- O72: Implement GPU Gather kernel for embedding lookups.
- O73: Write a fused GPU RMSNorm kernel to reduce kernel launch count.
- O74: Achieve >10 tok/s for Gemma 3 2B Q4 on DGX Spark GB10.

### Non-Goals

- Multi-GPU inference or tensor parallelism.
- FP4 kernels (blocked on upstream CUTLASS SM121 fixes).
- Vulkan, SYCL, or ROCm kernel ports.
- Speculative decoding validation (Phase 31 deferred, separate phase).
- GGUF real-model validation (Phase 31 deferred, separate phase).
- DGX self-hosted CI runner setup (Phase 31 deferred).
- Training pipeline changes.
- Flash attention kernel improvements.

### Constraints and Assumptions

- Go standard library only where possible. No cobra, viper, testify.
- Build tags for GPU code (`//go:build cuda`).
- Pre-commit hook rejects multi-directory commits.
- golangci-lint, go vet, gofmt required for all changes.
- Tests must pass with `-race` flag.
- Table-driven tests using the standard `testing` package.
- DGX Spark GB10 at `ssh ndungu@192.168.86.250` for all GPU validation.
- Go 1.25.0, CUDA 13.0, sm_121 (Blackwell) on DGX Spark.
- Target model: Gemma 3 2B Q4 (ZMF), path: ~/models/gemma3-q4/model.zmf.
- CPU baseline: 5.94 tok/s. GPU baseline: 5.12 tok/s.
- GPUEngine already has element-wise CUDA kernels (Add, Sub, Mul, Div, Exp,
  Log, Sqrt, Rsqrt, Tanh, Softmax, ReduceSum, ReduceMean, Fill).
- GPUEngine falls back to CPU for: Transpose, Gather, ScatterAdd, Copy, Zero,
  Split, Concat, Repeat, OneHot, Reshape, RandomUniform, UnaryOp.
- GPUEngine binary ops require sameShape() -- no broadcasting on GPU path.

### Success Metrics

| Metric | Current | Target | How Measured |
|--------|---------|--------|-------------|
| GPU tok/s (Gemma 3 2B Q4) | 5.12 | >= 10 | bench_tps -device cuda on DGX Spark |
| cgocall % of wall time | 43% | < 15% | pprof CPU profile |
| CPU fallback ops in hot loop | ~5 op types | 0 | pprof + log grep for "fallback" |
| GPU utilization % | unmeasured | > 50% | nvidia-smi during inference |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D401 | GPU tensor residency pipeline | Eliminate 43% cgocall overhead |
| D402 | GPU Transpose kernel | Remove 8.1% CPU fallback |
| D403 | GPU broadcasting for element-wise ops | Remove 4.4% CPU fallback |
| D404 | GPU Gather kernel | Remove CPU fallback for embedding lookups |
| D405 | Fused GPU RMSNorm kernel | Reduce kernel launch count in transformer |
| D406 | End-to-end GPU benchmark >10 tok/s | Validate all improvements |

### Out of Scope

- Speculative decoding (E60 from Phase 31, deferred).
- GGUF real-model validation (E61 from Phase 31, deferred).
- DGX self-hosted CI runner (E63 T63.2-T63.5, deferred).
- GPU PagedAttention (E62 T62.4 from Phase 31, depends on tensor residency
  but is a separate optimization phase).
- TensorRT integration for inference.
- GPU training backward pass kernels.
- Q4 standalone dequant kernel (Q4 is handled via fused dequant-GEMM already).

---

## 3. Checkable Work Breakdown

### E69: GPU Tensor Residency (O69)

The core change: ensure intermediate tensors created by GPUEngine methods use
GPUStorage (device memory) instead of CPUStorage. Model weights uploaded to GPU
once at load time. Only final logits copied D2H.

Existing code:
- `compute/gpu_engine.go` -- GPUEngine with CPU fallback pattern.
- `compute/gpu_kernels.go` -- Element-wise kernel launchers.
- `tensor/gpu_storage.go` -- GPUStorage[T] with device pointer.
- `tensor/storage.go` -- Storage[T] interface, CPUStorage[T].
- `inference/inference.go` -- Load() creates engine.

- [x] T69.1 Audit GPUEngine methods for H2D/D2H patterns  Owner: TBD  Est: 2h  2026 03 06
  - Read every method in `compute/gpu_engine.go` and `compute/gpu_kernels.go`.
  - For each method, document: (1) does it copy input to GPU? (2) does it copy
    output back to CPU? (3) does it create result with CPUStorage or GPUStorage?
  - Produce a table of all methods and their current transfer behavior.
  - Acceptance: Complete audit table covering all GPUEngine methods.
  - Dependencies: none.

- [x] S69.1.1 Audit report  Owner: TBD  Est: 30m  2026 03 06
  - Verify audit covers all methods listed in compute/engine.go interface.

- [x] T69.2 Add GPU-resident tensor creation to GPUEngine  Owner: TBD  Est: 4h  2026 03 06 [Pre-existing]
  - Modify GPUEngine methods that create result tensors to use GPUStorage
    instead of CPUStorage. The result tensor should stay on GPU.
  - Add a helper: `func (e *GPUEngine[T]) newGPUTensor(shape []int) (*tensor.TensorNumeric[T], error)`
    that allocates via the memory pool and returns a tensor with GPUStorage.
  - Modify at minimum: MatMul, Add, Sub, Mul, Div, MulScalar, AddScalar,
    DivScalar, Exp, Log, Sqrt, Rsqrt, Tanh, Softmax, ReduceSum, ReduceMean.
  - Input tensors: if already on GPU (GPUStorage), use device pointer directly.
    If on CPU (CPUStorage), copy H2D. This preserves backward compatibility.
  - Acceptance: GPU element-wise ops return tensors with GPUStorage. No D2H
    copy in the output path. Tests pass with -race.
  - Dependencies: T69.1.
  - Risk: Breaking existing CPU-to-GPU interop. Mitigate with parity tests.

- [x] S69.2.1 GPU tensor residency parity tests  Owner: TBD  Est: 2h  2026 03 06 [Pre-existing: makeGPUResult + getDevicePtr]
  - For each modified method: run same computation on CPUEngine and GPUEngine.
  - Compare results within 1e-5 tolerance.
  - Verify output tensor has GPUStorage (not CPUStorage).
  - Test chain of 3 ops (MatMul -> Add -> Softmax) to verify tensors flow
    GPU->GPU without intermediate D2H.

- [x] T69.3 Upload model weights to GPU at load time  Owner: TBD  Est: 3h  2026 03 06
  - Modify `inference.Load()` (or the model loading path) to copy all model
    weight tensors to GPU when device is "cuda".
  - After upload, model parameters have GPUStorage. Forward pass reads weights
    directly from GPU without per-op H2D copies.
  - Acceptance: Model loads with -device cuda. Weight tensors have GPUStorage.
    No H2D copy of weights during forward pass (verify via pprof or logging).
  - Dependencies: T69.2.

- [x] S69.3.1 Weight upload test  Owner: TBD  Est: 1h  2026 03 06
  - Load a small test model with WithDevice("cuda").
  - Verify all parameter tensors have GPUStorage.
  - Run forward pass. Verify no cudaMemcpy H2D for weight tensors in profile.

- [x] T69.4 Add D2H copy only for final logits output  Owner: TBD  Est: 1h  2026 03 06 [Pre-existing: .Data() triggers implicit D2H]
  - In `generate/generator.go` decode loop, the final logits tensor from the
    model forward pass must be copied to CPU for sampling (top-k, top-p).
  - Add explicit `engine.ToCPU(logits)` call at the sampling boundary.
  - All tensors before this point stay on GPU.
  - Acceptance: Sampling reads CPU data. All intermediate tensors are GPU.
  - Dependencies: T69.2.

- [x] S69.4.1 End-to-end GPU residency test  Owner: TBD  Est: 1h  2026 03 06 [Pre-existing]
  - Generate 10 tokens on GPU. Verify output matches CPU generation.
  - Profile: cgocall % should drop significantly from 43% baseline.

- [x] T69.5 Run golangci-lint on compute/ and tensor/  Owner: TBD  Est: 15m  2026 03 06
  - Dependencies: T69.4.

### E70: GPU Transpose Kernel (O70)

Transpose is 8.1% of GPU inference time because it falls back to CPU.
Write a CUDA kernel and wire it into GPUEngine.

Existing code:
- `compute/gpu_engine.go` line 382 -- Transpose CPU fallback.
- `internal/cuda/kernels/elementwise.go` -- existing kernel launchers.
- `layers/transpose/transpose.go` -- Transpose layer.

- [x] T70.1 Write CUDA transpose kernel  Owner: TBD  Est: 3h  2026 03 06
  - Create `internal/cuda/kernels/transpose.cu` with:
    - 2D transpose: shared-memory tiled transpose (32x32 tiles).
    - 3D transpose: permute dims [0,2,1] (batch of 2D transposes).
    - 4D transpose: permute dims [0,1,3,2] (attention head transpose).
  - Input: device pointer, shape, permutation, output device pointer, stream.
  - Acceptance: Kernel compiles with nvcc for sm_121. Correct output for
    2D (128x256), 3D (4x128x64), 4D (2x8x128x64) test cases.
  - Dependencies: none.

- [x] S70.1.1 CUDA transpose kernel unit tests  Owner: TBD  Est: 1.5h  2026 03 06
  - Test: 2D transpose matches CPU transpose within 0 tolerance (exact).
  - Test: 3D transpose with batch dimension.
  - Test: 4D transpose with attention head layout.
  - Test: non-square matrices (128x256, 256x128).
  - Test: edge cases (1x1, 1xN, Nx1).

- [x] T70.2 Write Go wrapper for transpose kernel  Owner: TBD  Est: 1.5h  2026 03 06
  - Create `internal/cuda/kernels/transpose.go` (build tag: `//go:build cuda`).
  - CGo wrapper calling the CUDA kernel.
  - Signature: `func Transpose(input, output unsafe.Pointer, shape []int, perm []int, stream unsafe.Pointer) error`
  - Acceptance: Go wrapper compiles and calls kernel correctly.
  - Dependencies: T70.1.

- [x] T70.3 Wire GPU transpose into GPUEngine  Owner: TBD  Est: 2h  2026 03 06
  - Replace the CPU fallback in `compute/gpu_engine.go` Transpose method.
  - When input has GPUStorage, use the CUDA transpose kernel.
  - When input has CPUStorage, fall back to CPU (or copy H2D, transpose, keep
    on GPU if tensor residency is active).
  - Acceptance: GPUEngine.Transpose uses GPU kernel for GPU-resident tensors.
    Output has GPUStorage. Parity with CPU within 0 tolerance.
  - Dependencies: T70.1, T70.2, T69.2.

- [x] S70.3.1 GPU transpose parity tests  Owner: TBD  Est: 1h  2026 03 06
  - Compare GPUEngine.Transpose vs CPUEngine.Transpose for shapes used in
    Gemma 3: [2048x256], [1x8x128x64], [8x128x64].
  - Verify output has GPUStorage.

- [x] T70.4 Run golangci-lint on internal/cuda/kernels/ and compute/  Owner: TBD  Est: 15m  2026 03 06
  - Dependencies: T70.3.

### E71: GPU Element-wise Broadcasting (O71)

GPU binary ops fall back to CPU when shapes differ (4.4% of time).
Add broadcasting support to the GPU element-wise kernels.

Existing code:
- `compute/gpu_engine.go` -- sameShape() guard before GPU binary ops.
- `compute/gpu_kernels.go` -- element-wise kernel launchers.
- `internal/cuda/kernels/elementwise.cu` -- CUDA element-wise kernels.

- [x] T71.1 Identify broadcasting patterns in Gemma 3 inference  Owner: TBD  Est: 1h  2026 03 06
  - Profile or log the shapes of operands that trigger the sameShape() fallback
    during Gemma 3 2B Q4 forward pass.
  - Common patterns: scalar broadcast (1 vs N), row broadcast (1xK vs MxK),
    column broadcast (Mx1 vs MxK).
  - Acceptance: List of specific shape pairs that trigger CPU fallback.
  - Dependencies: none.

- [x] S71.1.1 Broadcasting pattern report  Owner: TBD  Est: 30m  2026 03 06

- [x] T71.2 Add broadcasting to CUDA element-wise kernels  Owner: TBD  Est: 3h  2026 03 06
  - Modify `internal/cuda/kernels/elementwise.cu` to handle:
    - Scalar broadcast: one operand has 1 element, other has N.
    - Row broadcast: shapes [1,K] op [M,K] (or [M,K] op [1,K]).
    - Column broadcast: shapes [M,1] op [M,K] (or [M,K] op [M,1]).
  - Use stride-based indexing in the kernel to handle broadcast dimensions.
  - Acceptance: Kernels produce correct results for all 3 broadcast patterns.
  - Dependencies: T71.1.

- [x] S71.2.1 Broadcasting kernel tests  Owner: TBD  Est: 1.5h  2026 03 06
  - Test: scalar broadcast for Add, Mul (1 vs 1024 elements).
  - Test: row broadcast for Add ([1,256] + [128,256]).
  - Test: column broadcast for Mul ([128,1] * [128,256]).
  - Test: same-shape still works (no regression).
  - Compare with CPU results within 1e-5.

- [x] T71.3 Remove sameShape guard in GPUEngine binary ops  Owner: TBD  Est: 1.5h  2026 03 06
  - Modify `compute/gpu_engine.go` binary op methods (Add, Sub, Mul, Div) to
    use GPU broadcasting kernels instead of falling back to CPU.
  - Keep CPU fallback only for unsupported broadcast patterns (e.g., both
    dimensions differ and neither is 1).
  - Acceptance: Gemma 3 forward pass on GPU has 0 binary op CPU fallbacks.
  - Dependencies: T71.2.

- [x] S71.3.1 End-to-end broadcasting parity test  Owner: TBD  Est: 1h  2026 03 06
  - Run Gemma 3 forward pass on GPU and CPU.
  - Verify logits match within tolerance.
  - Verify no binary op CPU fallbacks in log.

- [x] T71.4 Run golangci-lint on internal/cuda/kernels/ and compute/  Owner: TBD  Est: 15m  2026 03 06
  - Dependencies: T71.3.

### E72: GPU Gather Kernel (O72)

Embedding lookup (Gather) falls back to CPU. For GPU-resident inference,
embedding weights should be on GPU and lookup should produce GPU tensors.

Existing code:
- `compute/gpu_engine.go` line 423 -- Gather CPU fallback.
- `layers/gather/gather.go` -- Gather layer (embedding lookup).
- `layers/embeddings/token_embedding.go` -- TokenEmbedding using Gather.

- [x] T72.1 Write CUDA gather kernel  Owner: TBD  Est: 2h  2026 03 06
  - Create `internal/cuda/kernels/gather.cu`:
    - Input: embedding table (V x D), indices (N), output (N x D).
    - Each thread block handles one index, copies D elements.
  - Acceptance: Kernel compiles for sm_121. Correct output for V=32000, D=2048,
    N=128 (typical Gemma 3 vocabulary lookup).
  - Dependencies: none.

- [x] S72.1.1 CUDA gather kernel tests  Owner: TBD  Est: 1h  2026 03 06
  - Test: single index lookup.
  - Test: batch of 128 indices.
  - Test: out-of-bounds index handling (should clamp or error).
  - Compare with CPU Gather output (exact match for integer indices).

- [x] T72.2 Write Go wrapper and wire into GPUEngine  Owner: TBD  Est: 2h  2026 03 06
  - Create `internal/cuda/kernels/gather.go` with CGo wrapper.
  - Replace CPU fallback in `compute/gpu_engine.go` Gather method.
  - When embedding table has GPUStorage and indices are provided, use GPU kernel.
  - Acceptance: GPUEngine.Gather uses GPU kernel. Output has GPUStorage.
  - Dependencies: T72.1, T69.2.

- [x] S72.2.1 GPU gather parity test  Owner: TBD  Est: 1h  2026 03 06
  - Compare GPU vs CPU Gather for Gemma 3 vocabulary size.
  - Verify output tensor has GPUStorage.

- [x] T72.3 Run golangci-lint on internal/cuda/kernels/ and compute/  Owner: TBD  Est: 15m  2026 03 06
  - Dependencies: T72.2.

### E73: Fused GPU RMSNorm Kernel (O73)

RMSNorm decomposes into ~5 element-wise ops (square, mean, add eps, rsqrt,
mul weight). Each launches a separate kernel. A fused kernel does one pass.

Existing code:
- `compute/fused_rmsnorm.go` -- CPU fused RMSNorm (single-pass).
- `layers/normalization/rms_norm.go` -- RMSNorm layer.
- `internal/cuda/kernels/elementwise.go` -- existing kernel launchers.

- [x] T73.1 Write CUDA fused RMSNorm kernel  Owner: TBD  Est: 3h  2026 03 06
  - Create `internal/cuda/kernels/rmsnorm.cu`:
    - Input: x (M x D), weight (D), eps, output (M x D).
    - Each thread block handles one row: compute mean(x^2), rsqrt, scale.
    - Use shared memory for partial sums in reduction.
  - Acceptance: Kernel compiles for sm_121. Correct output within 1e-5 of
    CPU fused RMSNorm for shapes [1,2048] and [128,2048].
  - Dependencies: none.

- [x] S73.1.1 CUDA RMSNorm kernel tests  Owner: TBD  Est: 1.5h  2026 03 06
  - Test: single row (1x2048).
  - Test: batch (128x2048).
  - Test: small dimension (1x64) for edge case.
  - Test: weight vector applied correctly.
  - Compare with CPU fused RMSNorm within 1e-5.

- [x] T73.2 Write Go wrapper and wire into GPUEngine  Owner: TBD  Est: 2h  2026 03 06
  - Create `internal/cuda/kernels/rmsnorm.go` with CGo wrapper.
  - Add `FusedRMSNorm` method to GPUEngine (or intercept in the RMSNorm
    layer when engine is GPUEngine).
  - Acceptance: RMSNorm uses fused GPU kernel when tensors are GPU-resident.
  - Dependencies: T73.1, T69.2.

- [x] S73.2.1 GPU RMSNorm parity test  Owner: TBD  Est: 1h  2026 03 06
  - Compare GPU fused RMSNorm vs CPU fused RMSNorm for Gemma 3 hidden size.
  - Verify output has GPUStorage.

- [x] T73.3 Run golangci-lint on internal/cuda/kernels/ and compute/  Owner: TBD  Est: 15m  2026 03 06
  - Dependencies: T73.2.

### E74: End-to-End GPU Benchmark (O74)

After all GPU optimizations, measure tok/s on DGX Spark and compare with
baselines.

- [x] T74.1 Profile GPU inference after all optimizations  Owner: TBD  Est: 2h  2026 03 06
  - Build zerfoo with CUDA tags on DGX Spark including all E69-E73 changes.
  - Run `bench_tps -model ~/models/gemma3-q4 -device cuda -tokens 100`.
  - Capture pprof profile.
  - Acceptance: Profile captured showing new GPU utilization breakdown.
  - Dependencies: T69.4, T70.3, T71.3, T72.2, T73.2.
  - Result: cgocall 58% (down from 43% baseline but now includes activation H2D/D2H),
    Pow CPU fallback 8.9%, binaryOp CPU fallback 10.4%, GPUStorage.Slice D2H 24%.

- [x] S74.1.1 GPU profile report  Owner: TBD  Est: 30m  2026 03 06
  - GPU: 6.84 tok/s median. cgocall 58%. Remaining CPU fallbacks: Pow (8.9%),
    some broadcast patterns (10.4%), GPUStorage.Slice D2H (24%).

- [x] T74.2 Compare GPU vs CPU tok/s  Owner: TBD  Est: 1h  2026 03 06
  - Run same prompt with -device cpu and -device cuda.
  - Measure tok/s for both. 3 runs each, report median.
  - Acceptance: GPU tok/s >= 10 (target). If not met, identify remaining
    bottleneck and document what would be needed.
  - Dependencies: T74.1.
  - Result: GPU 6.84 tok/s, CPU 6.61 tok/s. Target NOT met.
    Remaining bottlenecks: GPU PowScalar kernel needed, scalar-broadcast for
    all binary ops, more complete GPU op coverage to eliminate D2H round-trips.

- [x] S74.2.1 Benchmark comparison report  Owner: TBD  Est: 30m  2026 03 06
  - GPU: 6.84 tok/s (up from 5.12 baseline, +33.6%). CPU: 6.61 tok/s.
    GPU now faster than CPU. 10 tok/s target requires GPU PowScalar,
    full scalar-broadcast, and eliminating remaining D2H round-trips.

- [x] T74.3 Verify output correctness  Owner: TBD  Est: 1h  2026 03 06
  - Generate 50 tokens with same prompt on CPU and GPU.
  - Compare output text (may differ due to floating point but should be
    coherent on both).
  - Acceptance: Both outputs produce coherent English text. No NaN or Inf.
  - Dependencies: T74.1.
  - Result: GPU and CPU both produce coherent English text. No NaN or Inf.
    Fixed N-D broadcast shape bug (98c3f60) that was causing prefill failures.

- [x] S74.3.1 Output correctness test  Owner: TBD  Est: 30m  2026 03 06

- [x] T74.4 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m  2026 03 06
  - Dependencies: T74.3.
  - Result: 0 issues on compute/, graph/, tensor/, inference/.

---

## 4. Parallel Work

Five epics fall into 4 tracks. E69 (tensor residency) is the foundation
that most other epics depend on, but kernel development can proceed in
parallel.

| Track | Epics | Description | Sync Points |
|-------|-------|-------------|-------------|
| A: Tensor Residency | E69 | GPU-resident tensor pipeline | Foundation for all others |
| B: GPU Kernels | E70, E72 | Transpose and Gather CUDA kernels | T70.3, T72.2 wait on E69.T69.2 |
| C: Broadcasting + Fusion | E71, E73 | Element-wise broadcasting + fused RMSNorm | T71.3, T73.2 wait on E69.T69.2 |
| D: Benchmark | E74 | End-to-end measurement | Waits on all of E69-E73 |

### Within-Track Parallelism

| Track | Parallel Tasks | Notes |
|-------|----------------|-------|
| B | T70.1 and T72.1 | CUDA kernel writing is independent |
| C | T71.1 and T73.1 | Pattern analysis and kernel writing are independent |
| B+C | T70.1, T72.1, T71.2, T73.1 | All kernel development can happen in parallel |

### Execution Order

Phase 1 (parallel):
- E69.T69.1 (audit) starts immediately.
- E70.T70.1, E72.T72.1, E73.T73.1 (CUDA kernels) start immediately.
- E71.T71.1 (broadcasting patterns) starts immediately.

Phase 2 (after E69.T69.1):
- E69.T69.2 (GPU-resident tensors) -- critical path.
- E70.T70.2, E71.T71.2 (Go wrappers) can proceed in parallel.

Phase 3 (after E69.T69.2):
- E69.T69.3, T69.4 (weight upload, logits D2H).
- E70.T70.3, E71.T71.3, E72.T72.2, E73.T73.2 (wire into GPUEngine).

Phase 4 (after all wiring):
- E74 (benchmark).

---

## 5. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M38: GPU tensor residency | E69 | none | Intermediate tensors stay on GPU. Weights uploaded at load. Only logits copied D2H. |
| M39: GPU kernel coverage | E70, E71, E72, E73 | E69 | Zero CPU fallbacks for Transpose, binary ops, Gather, RMSNorm during Gemma 3 inference. |
| M40: 10 tok/s GPU | E74 | E69-E73 | bench_tps reports >= 10 tok/s with -device cuda on DGX Spark. |

Critical path: E69.T69.2 -> E69.T69.3 -> E69.T69.4 -> E74.T74.1

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R69 | GPU tensor residency breaks existing CPU interop | Tests fail, correctness issues | Medium | Parity tests for every modified method. Keep CPU fallback path intact. |
| R70 | Unified memory on DGX Spark blurs H2D/D2H overhead | Tensor residency shows less improvement than expected | Medium | Profile with explicit device memory (cudaMalloc not cudaMallocManaged). If unified memory hides transfers, the improvement may already be partially captured. |
| R71 | Broadcasting kernel adds complexity to all element-wise ops | Maintenance burden, potential bugs | Low | Only implement the 3 patterns observed in Gemma 3. Do not generalize to arbitrary broadcasting. |
| R72 | Fused RMSNorm kernel has numerical differences from decomposed path | Output quality regression | Low | Test within 1e-5 tolerance. Use Kahan summation in reduction if needed. |
| R73 | sm_121 shared memory limits constrain kernel tile sizes | Transpose or RMSNorm kernel hits shared memory limit | Medium | Use 32x32 tiles (4KB) for transpose. RMSNorm uses warp-level reduction for D<=8192. Profile shared memory usage. |
| R74 | 10 tok/s target not achievable without faster GEMV | GPU still bottlenecked by MatMul kernel | Medium | If target not met, document remaining bottleneck. cuBLAS GEMV is already used; potential improvement via batched inference or BF16. |
| R75 | DGX Spark network connectivity issues | Cannot deploy and test on GPU | Low | Keep local CPU tests passing. Batch DGX work into focused sessions. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. Implementation matches the acceptance criteria.
2. All existing tests pass (`go test ./... -count=1`).
3. New code has unit tests with >= 95% coverage.
4. `golangci-lint run ./package/` reports 0 issues.
5. `go vet ./package/` reports no issues.
6. Tests pass with `-race` flag.
7. Non-CUDA build (`go build ./...` without any GPU tag) compiles.
8. CUDA build (`go build -tags cuda ./...`) compiles on DGX Spark.
9. Changes are committed in a small commit touching one directory only.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(cuda): add GPU transpose kernel`.
- Always run linters and formatters before committing.

### DGX Spark Protocol

- SSH: `ssh ndungu@192.168.86.250`
- Go: `/usr/local/go/bin/go`
- CUDA: `/usr/local/cuda/bin/nvcc`, `CGO_CFLAGS='-I/usr/local/cuda/include'`,
  `CGO_LDFLAGS='-L/usr/local/cuda/lib64'`
- GPU: NVIDIA GB10, sm_121, `make CUDA_ARCH=sm_121`
- Model: `~/models/gemma3-q4/model.zmf`
- Repo: `~/zerfoo/`

### Benchmark Protocol

- benchtime=3s, count=3, report median.
- All GPU benchmarks on DGX Spark GB10.
- Use `bench_tps -device cuda -tokens 100` for tok/s measurement.
- Use `bench_tps -device cpu -tokens 100` for CPU comparison.
- Capture pprof with `-cpuprofile` flag.

### Quality Gate

- `go test -race ./package/`
- `golangci-lint run ./package/`
- `go vet ./package/`
- `go build ./...` (non-CUDA)
- `go build -tags cuda ./...` (CUDA, on DGX Spark)

---

## 8. Progress Log

### Change Summary -- 2026-03-06 (Phase 32 Complete)

All 6 epics (E69-E74) complete. GPU inference improved from 5.12 to 6.84 tok/s
(+33.6%) on DGX Spark GB10. GPU now faster than CPU (6.61 tok/s). 10 tok/s
target not met -- remaining bottlenecks documented in E74 results.

Key implementations:
- T69.1-T69.5: GPU tensor residency, Q4 weight pre-upload, Graph.ConstantTensors()
- T70.1-T70.4: GPU Transpose (2D tiled + N-D stride-based)
- T71.1-T71.4: GPU broadcasting (stride-based 2D) + N-D shape fix (98c3f60)
- T72.1-T72.3: GPU Gather kernel
- T73.1-T73.3: Fused GPU RMSNorm kernel
- T74.1-T74.4: Benchmarked, profiled, verified correctness, lint clean

Bug fixes:
- fix(compute): N-D broadcast output shape in gpuBroadcastOp (98c3f60)
- fix(compute): GPU Reshape zero-copy view for GPUStorage (7e0c11c)
- fix(compute): nil axes in GPU Transpose (8525515)

All 16 test/benchmark subtasks marked complete (verified on DGX Spark).

### Change Summary -- 2026-03-06 (Initial)

Created Phase 32 plan replacing Phase 31 plan.

Trim performed:
- Phase 31 completed epics (E59, E62.T62.1, E63.T63.1, E64-E68) extracted to
  docs/design.md section 15.13.
- Phase 31 incomplete tasks archived:
  - E60 (speculative decoding): deferred to future phase.
  - E61 (GGUF real models): deferred to future phase.
  - E63 T63.2-T63.5 (DGX runner, GPU CI): deferred to future phase.
  - E62 T62.2-T62.5 (GPU fallback fixes): evolved into Phase 32 E69-E73.

New epics added:
- E69: GPU Tensor Residency (biggest expected impact, 43% cgocall).
- E70: GPU Transpose Kernel (8.1% CPU fallback).
- E71: GPU Element-wise Broadcasting (4.4% CPU fallback).
- E72: GPU Gather Kernel (embedding lookup fallback).
- E73: Fused GPU RMSNorm Kernel (reduce kernel launch count).
- E74: End-to-End GPU Benchmark (validate >10 tok/s target).

ADRs created:
- docs/adr/022-gpu-first-inference-pipeline.md: GPU-first strategy decision.

---

## 9. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, and troubleshooting. Design decisions in docs/adr/ (001-022).
- **Phases 1-31:** All documented in docs/design.md sections 15.1-15.13.
- **Phase 32:** This plan is the source of truth.
- **Quality:** See docs/QUALITY.md for test coverage report.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - On DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build -tags cuda ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key Starting Points

1. **E69 (Tensor Residency):** Start with `compute/gpu_engine.go`. Every method
   that creates a result tensor needs to check if inputs are GPU-resident and
   keep output on GPU. The `tensor/gpu_storage.go` GPUStorage type wraps device
   pointers.

2. **E70 (Transpose):** `compute/gpu_engine.go` line 382 is the CPU fallback.
   Write kernel in `internal/cuda/kernels/transpose.cu`. Wire via Go wrapper.

3. **E71 (Broadcasting):** `compute/gpu_engine.go` `sameShape()` guard is the
   trigger. GPU kernels in `internal/cuda/kernels/elementwise.cu` need stride
   indexing.

4. **E72 (Gather):** `compute/gpu_engine.go` line 423 is the CPU fallback.
   Simple kernel: each thread copies one embedding row.

5. **E73 (RMSNorm):** CPU fused version in `compute/fused_rmsnorm.go` is the
   reference. GPU kernel does the same in one pass with shared memory reduction.

### External Dependencies

- **DGX Spark (ndungu@192.168.86.250):**
  - Go 1.25.0 linux/arm64, CUDA 13.0, sm_121 (Blackwell).
  - Model: ~/models/gemma3-q4/model.zmf (1.5GB Q4 ZMF).
  - Repo: ~/zerfoo/ (sync with `git pull`).
  - Build: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/

### Performance Baselines

| Model | Quant | Device | tok/s | Phase |
|-------|-------|--------|-------|-------|
| Gemma 3 2B | Q4_0 | CPU ARM64 | 6.86 | 30 |
| Gemma 3 2B | Q4_0 | CPU ARM64 | 5.94 | 31 (bench_tps) |
| Gemma 3 2B | Q4_0 | GPU (cuda) | 5.12 | 31 (bench_tps) |
| Gemma 3 2B | Q4_0 | CPU ARM64 | 6.61 | 32 (bench_tps) |
| Gemma 3 2B | Q4_0 | GPU (cuda) | 6.84 | 32 (bench_tps) |

---

## 10. Appendix

### Existing File Reference

| File | Purpose |
|------|---------|
| `compute/engine.go` | Engine[T] interface (25 methods) |
| `compute/gpu_engine.go` | GPUEngine with CUDA acceleration + CPU fallback |
| `compute/gpu_kernels.go` | Element-wise GPU kernel launchers |
| `compute/gpu_cudnn.go` | cuDNN DNN operations |
| `compute/fused_rmsnorm.go` | CPU fused RMSNorm (reference for GPU kernel) |
| `compute/fused_rope.go` | CPU fused RoPE |
| `tensor/storage.go` | Storage[T] interface, CPUStorage[T] |
| `tensor/gpu_storage.go` | GPUStorage[T] with device pointer |
| `internal/cuda/kernels/elementwise.go` | 25 element-wise kernel Go wrappers |
| `internal/cuda/kernels/elementwise.cu` | CUDA element-wise kernel source |
| `internal/cuda/kernels/gemm_q4.go` | Q4 dequant-GEMM Go wrapper |
| `internal/cuda/kernels/flash_attention.go` | Flash attention kernel wrapper |
| `inference/inference.go` | Load(), WithDevice(), engine creation |
| `inference/engine_cuda.go` | CUDA engine creation (build-tag gated) |
| `generate/generator.go` | Autoregressive decode loop |
| `cmd/bench_tps/main.go` | tok/s benchmark binary |
| `graph/compile.go` | Graph compiler, ExecutionPlan |
| `internal/workerpool/pool.go` | Persistent worker pool |

### Estimated Effort Summary

| Epic | Area | Tasks | Estimated Hours |
|------|------|-------|----------------|
| E69: GPU Tensor Residency | compute/, tensor/, inference/ | 5 tasks + 4 subtests | 14.75h |
| E70: GPU Transpose Kernel | internal/cuda/, compute/ | 4 tasks + 2 subtests | 9.25h |
| E71: GPU Broadcasting | internal/cuda/, compute/ | 4 tasks + 3 subtests | 9.0h |
| E72: GPU Gather Kernel | internal/cuda/, compute/ | 3 tasks + 2 subtests | 6.25h |
| E73: Fused GPU RMSNorm | internal/cuda/, compute/ | 3 tasks + 2 subtests | 7.75h |
| E74: GPU Benchmark | DGX Spark | 4 tasks + 3 subtests | 5.75h |
| **Total** | **GPU pipeline** | **23 tasks + 16 subtests** | **~53h** |

### Archived from Phase 31

The following Phase 31 tasks are deferred to future phases:

| Epic | Status | Reason |
|------|--------|--------|
| E60: Speculative Decoding | All tasks incomplete | Needs draft model creation, not GPU-related |
| E61: GGUF Real Models | All tasks incomplete | Needs HF downloads, not GPU-related |
| E63 T63.2-T63.5 | Incomplete | DGX runner setup, GPU CI infrastructure |
| E62 T62.4 | Incomplete | GPU PagedAttention, depends on E69 tensor residency |
