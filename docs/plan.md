# Zerfoo Enterprise Production Readiness Plan

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with 40+ packages, a 34-method compute
Engine[T] interface, CPU and CUDA GPU backends, gRPC-based distributed
training, and comprehensive test coverage (95%+ across testable packages).

Phases 1-9 brought the framework to production grade: observability, security,
reliability, configuration, CI/CD enforcement, open-weights model import (6
model families), embeddable inference library with BPE tokenizer, KV cache,
generation loop, streaming, model registry, high-level API, CLI commands, and
OpenAI-compatible HTTP server.

Phases 10-13 added multi-GPU and NVIDIA library integrations: device affinity,
NCCL, cuDNN, TensorRT, CUTLASS flash attention. See ADRs 007-010.

Phases 14-19 added GPU portability and advanced features: GRAL abstraction,
AMD ROCm backend, OpenCL backend, cuDNN backward pass, CUTLASS INT4/INT8 GEMM,
TensorRT dynamic shapes. See ADRs 011-016.

Phase 20 validated the full GPU stack on DGX Spark GB10 (Blackwell sm_121,
ARM64, CUDA 13.0): 66 packages pass, benchmarks captured, feature gaps
documented. See ADR-017.

Phase 21 resolved model parity test gaps (17 PASS, 5 SKIP across 7 model
families) and documented the multi-GPU test coverage gap. See ADR-018.

Phase 22 addresses three gaps identified during Phase 20-21 validation:
1. BF16 GPU compute: cuBLAS supports BF16 GEMM via `cublasGemmEx`, but
   zerfoo only wraps `cublasSgemm` (float32). The `float16.BFloat16` type
   exists in the `float16` package with full arithmetic, and `tensor.Numeric`
   already includes `float16.BFloat16`, but GPUEngine falls back to CPU for
   all non-float32 types.
2. Unified memory: The DGX Spark GB10 has NVLink-C2C hardware-coherent access
   to 128 GB shared LPDDR5X. `cudaMallocManaged` avoids explicit H2D copies.
   Currently, MemPool only uses `cudaMalloc` (discrete device memory).
3. SigLIP Concat: The SigLIP vision model fails at node 1462 with
   "Concat shape mismatch [1] vs [1 1]" -- a rank mismatch between 1D and 2D
   tensors at a Concat input. Root cause is likely a missing Unsqueeze or
   incorrect shape propagation in a preceding node.

Architecture, design, GPU details, operations, and troubleshooting are
documented in docs/design.md (the single reference document). Stable design
decisions are extracted into docs/adr/ (see [ADR index](design.md#14-architectural-decision-records)).

### Objectives

O12-O31: COMPLETE (Phases 10-20). See ADRs 007-017.
O32-O33: COMPLETE (Phase 21). See ADR-018.

- O34: Add BF16 cuBLAS GEMM support so `GPUEngine[float16.BFloat16].MatMul`
  runs on GPU instead of falling back to CPU. **(IN PROGRESS)**
- O35: Add `cudaMallocManaged` allocator option in MemPool for zero-copy model
  loading on DGX Spark GB10 unified memory. **(IN PROGRESS)**
- O36: Fix SigLIP vision model Concat shape mismatch so the SigLIP parity test
  passes. **(IN PROGRESS)**

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
- FP16 training loop (FP16 inference via TensorRT is in scope).
- ROCm TensorRT equivalent (MIGraphX integration deferred).
- OpenCL multi-GPU collective communications (no NCCL equivalent).
- OpenCL flash attention (too complex for OpenCL kernel model).
- FP4 kernel implementation (blocked on upstream CUTLASS SM121 FP4 fixes).
- BF16 training loop (only inference GEMM is in scope for Phase 22).
- ConnectX-7 multi-node inference (requires second DGX Spark unit).
- ROCm or OpenCL BF16 GEMM (CUDA only for Phase 22).
- Unified memory for ROCm or OpenCL (CUDA only for Phase 22).
- Full FP16 GPU kernel support (only MatMul via cuBLAS for Phase 22).

### Constraints and Assumptions

- Use Go standard library only where possible. Minimize new dependencies.
- All CUDA code behind `//go:build cuda` build tags.
- All ROCm code behind `//go:build rocm` build tags.
- All OpenCL code behind `//go:build opencl` build tags.
- NCCL code behind `//go:build cuda` (requires libnccl2).
- cuDNN code behind `//go:build cuda` (requires libcudnn8 or libcudnn9).
- TensorRT code behind `//go:build cuda` (requires libnvinfer).
- CUTLASS requires nvcc and CUTLASS headers at build time; kernels compile into
  the existing `libkernels.a` static library. CUTLASS >= 4.2 required for sm_121.
- ROCm requires HIP SDK >= 5.0, rocBLAS, and MIOpen.
- OpenCL requires OpenCL 2.0+ headers and ICD loader (libOpenCL.so).
  CLBlast required for BLAS operations.
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- Table-driven tests using the standard testing package.
- No breaking changes to the Engine[T] interface. All GPU backends implement
  the same interface; the GRAL abstraction is internal.
- DGX Spark GB10 is ARM64 (aarch64), not x86_64. CUDA 13.0 pre-installed.
  Compute capability sm_121 (Blackwell). 128GB unified LPDDR5X memory.
  Single GPU -- multi-GPU tests require two units linked via ConnectX-7.
- The `float16` package (../float16) already provides a complete `BFloat16`
  type with IEEE 754 compliance, conversions, arithmetic, and rounding modes.
- `tensor.Numeric` already includes `float16.BFloat16` in its type constraint.
- `model/tensor_decoder.go` already decodes BFloat16 from ZMF model files.
- `cublasGemmEx` supports `CUDA_R_16BF` data type for BF16 GEMM on Blackwell.

### Success Metrics

All metrics for Phases 10-21 ACHIEVED. See ADRs 007-018 and design.md Section 15.

Phase 22 metrics:

| Metric | Target | Status |
|--------|--------|--------|
| BF16 MatMul on GPU | GPUEngine[BFloat16].MatMul dispatches to cublasGemmEx | NOT STARTED |
| BF16 benchmark | BF16 GEMM benchmark on DGX Spark vs CPU baseline | NOT STARTED |
| Unified memory allocator | MemPool.AllocManaged returns cudaMallocManaged pointer | NOT STARTED |
| Unified memory model load | Model loaded via managed memory skips explicit H2D copy | NOT STARTED |
| SigLIP parity test | TestSigLIPForwardPass PASS on DGX Spark | NOT STARTED |

---

## 2. Scope and Deliverables

D11-D55: COMPLETE. See ADRs 007-018.

| ID | Description | Status |
|----|-------------|--------|
| D56 | BF16 cuBLAS GEMM | cublasGemmEx binding + GPUEngine BF16 MatMul dispatch |
| D57 | Unified memory allocator | cudaMallocManaged in MemPool + GPUStorage integration |
| D58 | SigLIP Concat fix | Concat handles rank-mismatched inputs; SigLIP parity PASS |

---

## 3. Work Breakdown

### Completed Phases (1-21)

Phase 1 (Test Coverage), Phase 2 (GPU Engine), Phase 3 (GPU Production
Readiness), Phase 4 (Enterprise Production Readiness), Phase 5 (Distributed
Training Protocol), Phase 6 (Open Weights Model Import), Phase 7 (Architecture
Cleanup), Phase 8 (Embeddable Inference Library), Phase 9 (Multi-Architecture
Support), Phase 10 (Multi-GPU), Phase 11 (cuDNN), Phase 12 (TensorRT),
Phase 13 (CUTLASS Flash Attention), Phase 14 (GRAL), Phase 15 (ROCm),
Phase 16 (OpenCL), Phase 17 (cuDNN Backward), Phase 18 (CUTLASS INT4/INT8),
Phase 19 (TensorRT Dynamic Shapes), Phase 20 (DGX Spark Validation),
Phase 21 (Model Parity + Multi-GPU Gap) are all complete.
See docs/adr/ for design decisions.

### Phase 22: BF16 GEMM, Unified Memory, SigLIP Fix

#### Phase 22 Context

Phase 20 identified BF16 GEMM and unified memory as low-effort improvements for
DGX Spark (ADR-017 Sections "BF16 Tensor Operations" and "Unified Memory").
Phase 21 identified SigLIP Concat shape mismatch as a bug to fix (ADR-018).

Key codebase facts for implementers:
- `internal/cublas/cublas.go`: Only wraps `cublasSgemm`. Needs `cublasGemmEx`.
- `internal/gpuapi/blas.go`: BLAS interface has only `Sgemm`. Needs `GemmEx` or
  `BFloat16Gemm` method.
- `compute/gpu_engine.go` line 166-169: MatMul type-checks for float32 and falls
  back to CPU for everything else. Needs BFloat16 dispatch path.
- `compute/gpu_engine.go` line 246: `elemSize` is hardcoded to `float32(0)`.
  Must use `unsafe.Sizeof(zero)` for the actual type T.
- `internal/cuda/mempool.go`: Cache key is `(deviceID, byteSize)`. Alloc calls
  `cudaMalloc` on cache miss. Needs `AllocManaged` variant using
  `cudaMallocManaged`.
- `internal/gpuapi/mempool.go`: MemPool interface has `Alloc`, `Free`, `Drain`,
  `Stats`. Needs `AllocManaged` method.
- `layers/core/concat.go`: Delegates to `engine.Concat()`.
- `compute/cpu_engine.go` lines 1250-1267: Concat validates all non-axis
  dimensions are equal. Fails when inputs have different ranks.
- SigLIP error: "node[1462] Concat shape mismatch [1] vs [1 1]" -- one input is
  1D shape [1], another is 2D shape [1,1]. A preceding Unsqueeze or Reshape is
  likely not producing the expected rank.

#### E117: BF16 cuBLAS GEMM

Bind cublasGemmEx for BFloat16 GEMM and wire it into GPUEngine.MatMul.

- [x] T117.1 Add cublasGemmEx CGo binding  2026-03-04
  - Dependencies: None
  - Files: internal/cublas/cublas.go
  - Commits: 33a97e5 (binding), 251c336 (tests)
  - [x] S117.1.1 Define CudaDataType enum (CUDA_R_32F=0, CUDA_R_16BF=14, CUDA_R_16F=2)  Est: 10m
  - [x] S117.1.2 Add GemmEx method with CGo call to cublasGemmEx  Est: 30m
  - [x] S117.1.3 Add CublasComputeType enum (CUBLAS_COMPUTE_32F)  Est: 10m
  - [x] S117.1.4 Write table-driven tests: FP32 via GemmEx (BF16 validated on DGX Spark)  Est: 30m
  - [x] S117.1.5 Run golangci-lint and go test -cover  Est: 10m

- [x] T117.2 Add BFloat16Gemm to BLAS interface and CUDA adapter  2026-03-04
  - Dependencies: T117.1
  - Files: internal/gpuapi/blas.go, internal/gpuapi/cuda_blas.go,
    internal/gpuapi/rocm_blas.go, internal/gpuapi/opencl_blas.go,
    internal/gpuapi/gpuapi_test.go
  - Commit: 51facd7
  - [x] S117.2.1 Add BFloat16Gemm to BLAS interface  Est: 10m
  - [x] S117.2.2 Implement BFloat16Gemm in CUDABlas  Est: 15m
  - [x] S117.2.3 Add stub returning error in RocmBlas and OpenCLBlas  Est: 10m
  - [x] S117.2.4 Run golangci-lint on internal/gpuapi/  Est: 5m

- [x] T117.3 Add BFloat16 dispatch to GPUEngine.MatMul  2026-03-04
  - Dependencies: T117.2
  - Files: compute/gpu_engine.go, compute/gpu_kernels.go
  - Commit: fc15197
  - Note: BFloat16 MatMul tests require CUDA hardware; will be validated on DGX
    Spark during T117.4.
  - [x] S117.3.1 Fix elemSize to use unsafe.Sizeof(zero) for generic T  Est: 15m
  - [x] S117.3.2 Fix getDevicePtr to use unsafe.Sizeof(zero) and direct pointer  Est: 15m
  - [x] S117.3.3 Add BFloat16 type check and dispatch in MatMul  Est: 30m
  - [ ] S117.3.4 Write tests: BFloat16 MatMul 2x2, 4x4, batched  Est: 20m  (DGX Spark)
  - [x] S117.3.5 Run golangci-lint and go test -cover  Est: 10m

- [ ] T117.4 Benchmark BF16 GEMM on DGX Spark  Owner: TBD  Est: 30m
  - Dependencies: T117.3
  - Files: compute/gpu_engine_test.go (benchmark function)
  - Acceptance: BenchmarkBF16MatMul runs 128x128, 512x512, 1024x1024 BF16 GEMM
    on GPU. Results recorded in ADR-019. Compare with float32 Sgemm latency.
  - [ ] S117.4.1 Write BenchmarkBF16MatMul table-driven benchmark  Est: 15m
  - [ ] S117.4.2 Run on DGX Spark and record results  Est: 15m

- [ ] T117.5 Run linters and verify coverage for E117  Owner: TBD  Est: 15m
  - Dependencies: T117.4
  - Acceptance: golangci-lint 0 issues. go test -tags cuda -cover -race passes.
    Coverage >= 95% on internal/cublas/ and compute/ BF16 paths.
  - [ ] S117.5.1 Run golangci-lint, go vet, go test -tags cuda -cover -race  Est: 10m
  - [ ] S117.5.2 Fix any remaining issues  Est: 5m

#### E118: Unified Memory Allocator

Add cudaMallocManaged support to MemPool for zero-copy model loading on the
DGX Spark GB10 unified memory architecture.

- [x] T118.1 Add MallocManaged CGo binding  2026-03-04
  - Dependencies: None
  - Files: internal/cuda/runtime.go
  - Commit: 52a2500
  - [x] S118.1.1 Add MallocManaged CGo binding  Est: 20m
  - [x] S118.1.2 Write test: MallocManaged, write from host, read from host  Est: 15m
  - [x] S118.1.3 Run golangci-lint and go test -tags cuda -cover  Est: 10m

- [x] T118.2 Add AllocManaged to MemPool  2026-03-04
  - Dependencies: T118.1
  - Files: internal/cuda/mempool.go
  - Commit: 52c4660
  - [x] S118.2.1 Add managedCache field to MemPool  Est: 10m
  - [x] S118.2.2 Implement AllocManaged with cache lookup and MallocManaged fallback  Est: 20m
  - [x] S118.2.3 Add FreeManaged method to return to managed cache  Est: 10m
  - [x] S118.2.4 Update Drain to free managed cache entries  Est: 10m
  - [x] S118.2.5 Write tests: AllocManaged/FreeManaged round-trip, Drain clears both caches  Est: 15m
  - [x] S118.2.6 Run golangci-lint and go test -tags cuda -cover -race  Est: 5m

- [x] T118.3 Add AllocManaged to gpuapi.MemPool interface and CUDAMemPool  2026-03-04
  - Dependencies: T118.2
  - Files: internal/gpuapi/mempool.go, internal/gpuapi/cuda_mempool.go,
    internal/gpuapi/rocm_mempool.go, internal/gpuapi/opencl_mempool.go,
    internal/gpuapi/gpuapi_test.go
  - Commit: b15bf8c
  - [x] S118.3.1 Add AllocManaged and FreeManaged to MemPool interface  Est: 10m
  - [x] S118.3.2 Implement in CUDAMemPool  Est: 10m
  - [x] S118.3.3 Add stubs in ROCm and OpenCL pool implementations  Est: 10m

- [x] T118.4 Add ManagedGPUStorage option to GPUStorage  2026-03-04
  - Dependencies: T118.3
  - Files: tensor/gpu_storage.go, tensor/gpu_storage_test.go
  - Commit: 5a6350a
  - Note: NewManagedGPUStorage accepts gpuapi.MemPool (not runtime) since
    AllocManaged lives on the MemPool interface. Tests require CUDA hardware.
  - [x] S118.4.1 Add managed bool field to GPUStorage  Est: 5m
  - [x] S118.4.2 Add NewManagedGPUStorage constructor  Est: 15m
  - [x] S118.4.3 Update TrySlice to skip Memcpy for managed storage  Est: 15m
  - [x] S118.4.4 Update TrySet to skip Memcpy for managed storage  Est: 10m
  - [x] S118.4.5 Write tests: create managed storage, write, read, verify  Est: 15m
  - [x] S118.4.6 Run golangci-lint and go test -tags cuda -cover -race  Est: 5m

- [ ] T118.5 Benchmark managed vs discrete allocation on DGX Spark  Owner: TBD  Est: 30m
  - Dependencies: T118.4
  - Files: tensor/gpu_storage_test.go (benchmark function)
  - Acceptance: BenchmarkManagedVsDiscrete compares allocation+H2D copy time for
    discrete vs managed memory at sizes 1MB, 10MB, 100MB. Results in ADR-019.
  - [ ] S118.5.1 Write benchmark function  Est: 15m
  - [ ] S118.5.2 Run on DGX Spark and record results  Est: 15m

- [ ] T118.6 Run linters and verify coverage for E118  Owner: TBD  Est: 15m
  - Dependencies: T118.5
  - Acceptance: golangci-lint 0 issues. go test -tags cuda -cover -race passes.
    Coverage >= 95% on unified memory paths.
  - [ ] S118.6.1 Run golangci-lint, go vet, go test -tags cuda -cover -race  Est: 10m
  - [ ] S118.6.2 Fix any remaining issues  Est: 5m

#### E119: SigLIP Concat Shape Mismatch Fix

Fix the Concat layer to handle rank-mismatched inputs by broadcasting or
erroring clearly, and fix the SigLIP model's shape propagation so the parity
test passes.

- [ ] T119.1 Reproduce and diagnose SigLIP Concat failure  Owner: TBD  Est: 1.5h
  - Dependencies: None
  - Files: tests/parity/siglip_test.go, graph/graph.go
  - Steps:
    1. Run TestSigLIPForwardPass on DGX Spark to confirm the exact error
    2. Add temporary debug logging at graph.go line 61 to print node 1462
       input shapes and dependency ops
    3. Trace backward from node 1462 to find which node produces the [1]
       shape and which produces [1 1]
    4. Identify the root cause: missing Unsqueeze, incorrect Reshape target
       shape, or Concat not handling rank broadcast
  - Acceptance: Root cause documented with the specific ONNX node type and
    index that produces the wrong shape. Fix strategy identified.
  - [ ] S119.1.1 Run SigLIP test on DGX Spark and capture full error  Est: 15m
  - [ ] S119.1.2 Add debug logging to trace node 1462 inputs  Est: 15m
  - [ ] S119.1.3 Identify root cause node and shape mismatch origin  Est: 30m
  - [ ] S119.1.4 Document fix strategy  Est: 15m

- [ ] T119.2 Fix shape propagation or Concat rank handling  Owner: TBD  Est: 1.5h
  - Dependencies: T119.1
  - Files: TBD based on diagnosis (likely one of: layers/core/concat.go,
    compute/cpu_engine.go, model/builder.go, or a specific layer's Forward)
  - Acceptance: The fix addresses the root cause identified in T119.1. If the
    issue is in Concat itself, add rank broadcasting (Unsqueeze lower-rank
    inputs to match the highest rank). If the issue is in a preceding node,
    fix that node's output shape. The fix must not break existing Concat tests.
  - [ ] S119.2.1 Write a failing test that reproduces the shape mismatch  Est: 20m
  - [ ] S119.2.2 Implement the fix  Est: 30m
  - [ ] S119.2.3 Verify the failing test now passes  Est: 10m
  - [ ] S119.2.4 Run existing Concat tests to verify no regressions  Est: 10m
  - [ ] S119.2.5 Run golangci-lint  Est: 5m

- [ ] T119.3 Run SigLIP parity test on DGX Spark  Owner: TBD  Est: 30m
  - Dependencies: T119.2
  - Files: tests/parity/siglip_test.go
  - Acceptance: TestSigLIPForwardPass PASS on DGX Spark with the SigLIP ZMF
    model at ~/models/siglip/model.zmf. Update ADR-018 results table.
  - [ ] S119.3.1 Deploy fix to DGX Spark  Est: 10m
  - [ ] S119.3.2 Run SigLIP parity test  Est: 10m
  - [ ] S119.3.3 Update ADR-018 results (SKIP -> PASS)  Est: 10m

- [ ] T119.4 Run linters and verify coverage for E119  Owner: TBD  Est: 15m
  - Dependencies: T119.3
  - Acceptance: golangci-lint 0 issues. All existing tests pass. SigLIP parity
    PASS.
  - [ ] S119.4.1 Run golangci-lint, go vet, go test -cover -race  Est: 10m
  - [ ] S119.4.2 Fix any remaining issues  Est: 5m

#### E120: Phase 22 Final Verification

- [ ] T120.1 Update documentation  Owner: TBD  Est: 30m
  - Dependencies: E117, E118, E119
  - Files: docs/plan.md, docs/design.md, docs/adr/019-phase22-bf16-unified-siglip.md
  - Steps:
    1. Mark all Phase 22 tasks complete with results
    2. Create ADR-019 with BF16 benchmark results, unified memory benchmark,
       and SigLIP fix details
    3. Update design.md Section 15 with BF16 and unified memory results
    4. Update ADR-018 results table (SigLIP SKIP -> PASS)
  - Acceptance: All docs reflect actual results. ADR-019 written.

---

## 4. Timeline and Milestones

M72-M96: All ACHIEVED (Phases 10-21). See ADRs 007-018.

| ID | Milestone | Dependencies | Exit Criteria |
|----|-----------|--------------|---------------|
| M97 | BF16 cuBLAS GEMM | E117 | GPUEngine[BFloat16].MatMul dispatches to GPU; benchmark recorded |
| M98 | Unified memory allocator | E118 | cudaMallocManaged available in MemPool; managed GPUStorage works |
| M99 | SigLIP parity PASS | E119 | TestSigLIPForwardPass PASS on DGX Spark |
| M100 | Phase 22 complete | E120 | All three features implemented, tested, benchmarked, documented |

Sequencing:
- E117, E118, and E119 are independent and can proceed in parallel.
- E120 depends on all three completing.
- E117 T117.1-T117.3 can be developed locally (macOS). T117.4 requires DGX Spark.
- E118 T118.1-T118.4 can be developed locally. T118.5 requires DGX Spark.
- E119 T119.1 and T119.3 require DGX Spark. T119.2 may be doable locally
  depending on the root cause.

---

## 5. Risk Register

Active risks only. Resolved/mitigated risks (R1-R13, R26) removed.

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R14 | GRAL abstraction adds indirection overhead | GPU performance regression | Medium | Benchmark before/after; inline critical paths if needed. GRAL interfaces are internal, not virtual dispatch in hot loop. |
| R15 | HIP API divergence from CUDA | Unexpected incompatibilities | Medium | HIP is designed as CUDA-compatible. Test on actual AMD hardware. |
| R16 | MIOpen API differs significantly from cuDNN | Extra development time for DNN ops | High | MIOpen has different workspace management. Budget extra time for conv forward/backward. |
| R17 | ROCm tests cannot run in CI | No AMD GPU in CI | High | Tests skip gracefully. Validate on AMD hardware manually (Instinct MI250 or RX 7900). |
| R18 | OpenCL buffer model vs pointer model mismatch | GRAL interface awkward for OpenCL | High | GRAL Runtime uses unsafe.Pointer to wrap cl_mem handles. Document the convention. |
| R19 | CLBlast performance worse than cuBLAS/rocBLAS | Slow OpenCL MatMul | Medium | CLBlast is the best available. Document expected performance gap. |
| R20 | OpenCL kernel compilation at runtime is slow | Slow first inference | Medium | Cache compiled kernels (clGetProgramInfo + binary). |
| R21 | cuDNN backward workspace larger than forward | GPU OOM during training | Medium | Pool workspace buffers. Fall back to CPU on OOM (existing pattern). |
| R22 | CUTLASS INT4 packing format varies by model | Incompatible quantization formats | High | Support ONNX MatMulNBits format (block quantization, group_size). Document supported formats. |
| R23 | TRT dynamic shapes slower than fixed shapes | Performance regression | Medium | Optimization profile's "opt" dimension guides kernel selection. Document tradeoff. |
| R24 | Three GPU backends increase maintenance burden | Bug surface area grows | High | GRAL abstraction minimizes duplication. Only vendor-specific code is in internal/ packages. |
| R25 | OpenCL DNN ops missing (no cuDNN/MIOpen equivalent) | Incomplete OpenCL support | High | Document that OpenCL does not support Conv2d/BatchNorm on GPU. CPU fallback is acceptable. |
| R27 | CUTLASS sm_121 requires version >= 4.2 | Flash attention and INT4 GEMM kernels may not compile | High | Install CUTLASS 4.2+. If unavailable, skip cutlass-tagged tests; CPU fallback works. |
| R28 | ARM64 memory ordering differs from x86 | Subtle concurrency bugs in CGo code | Low | Go runtime handles memory barriers. Monitor for flaky tests on ARM64. |
| R30 | Gonum BLAS slower on ARM64 (no SIMD assembly) | CPU fallback operations significantly slower | Medium | Document perf gap. Long-term: link ARM-optimized BLAS (OpenBLAS with NEON). |
| R31 | Single-GPU DGX Spark cannot validate multi-GPU code | NCCL and multi-GPU tests remain unvalidated | High | Tests skip gracefully. Second DGX Spark unit needed for full multi-GPU validation. |
| R36 | cublasGemmEx BF16 compute precision differs from CPU | Numerical mismatches in BF16 MatMul parity | Medium | Use CUBLAS_COMPUTE_32F (accumulate in FP32) for maximum precision. Document tolerance. |
| R37 | cudaMallocManaged slower than cudaMalloc on PCIe GPUs | Performance regression on non-unified-memory hardware | Low | Managed memory is opt-in, not default. Only beneficial on NVLink-C2C (DGX Spark). Document. |
| R38 | SigLIP Concat fix may require zonnx converter changes | Fix spans two repositories | Medium | Check if root cause is in zerfoo (shape propagation) or zonnx (ONNX import). Fix in the correct repo. |

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
9. ROCm build (`go build -tags rocm ./...`) compiles.
10. OpenCL build (`go build -tags opencl ./...`) compiles.
11. Changes are committed in a small commit touching one directory only.

### Review and QA Steps

1. Read existing implementation before writing code.
2. Write tests first or alongside implementation. Use table-driven tests.
3. After implementation, run `go test -cover ./package/` to verify coverage.
4. Run `golangci-lint run --fix ./package/` to fix lint issues.
5. Run `gofmt -w .` to ensure formatting.
6. Run `go test ./... -count=1` to verify no regressions.
7. Run `go build ./...` (without GPU tags) to verify non-GPU build.
8. Run `go build -tags cuda ./...` to verify CUDA build.
9. Multi-GPU tests must skip gracefully when fewer than 2 GPUs are available.
10. cuDNN tests must skip when libcudnn is not available.
11. TensorRT tests must skip when libnvinfer is not available.
12. CUTLASS tests must skip when cutlass build tag is not set.
13. ROCm tests must skip when rocm build tag is not set or HIP SDK absent.
14. OpenCL tests must skip when opencl build tag is not set or ICD loader absent.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(cublas): add BF16 GemmEx binding`.
- Never allow changes to pile up. Commit after each completed subtask.
- Always run linters and formatters before committing.

---

## 7. Progress Log

| Date | Phase | Summary |
|------|-------|---------|
| 2026-03-04 | 22 | Change Summary: Created Phase 22 plan with 3 epics (E117-E119) and final verification (E120). Added BF16 cuBLAS GEMM (5 tasks), unified memory allocator (6 tasks), SigLIP Concat fix (4 tasks). Added risks R36-R38. New deliverables D56-D58, milestones M97-M100. |
| 2026-03-04 | 21 | Phase 21 COMPLETE. T116.1 docs updated (27a94c8). All tasks marked complete. O32 COMPLETE. |
| 2026-03-04 | 21 | Gemma 3 parity PASS (FP/GD/Gen). Fixes: tokenizer merges format, Gather embedded-indices, Slice hybrid mode, zonnx initializer promotion. Model parity now 11 PASS, 10 SKIP. |
| 2026-03-04 | 21 | E115 COMPLETE. Multi-GPU test gap documented in ADR-017 (6 tests, hardware prereqs). Test runner script created (scripts/dgx-spark-multigpu.sh). Plan trimmed 1411->522 lines. ADR-018 written. |
| 2026-03-03 | 20 | Phase 20 COMPLETE. ARM64 build (10 fixes), GPU tests (66 pkgs), benchmarks, feature gaps. ADR-017 written. |
| 2026-03-03 | 14-19 | Phases 14-19 all COMPLETE. GRAL, ROCm, OpenCL, cuDNN backward, INT4/INT8 GEMM, TRT dynamic shapes. ADRs 011-016 written. |
| 2026-03-03 | 10-13 | Phases 10-13 COMPLETE. Multi-GPU, cuDNN, TensorRT, CUTLASS. ADRs 007-010 written. |

---

## 8. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. It is the single reference
  document. Design decisions are in docs/adr/.
- **Phases 1-21:** Complete. See ADRs 001-018.
- **Phase 22 (Current):** BF16 GEMM + unified memory + SigLIP fix. Three
  independent epics (E117, E118, E119) that can proceed in parallel.
- **BF16 context:** The `float16` package (../float16) has a complete BFloat16
  type. `tensor.Numeric` includes it. `model/tensor_decoder.go` decodes it from
  ZMF files. What is missing: GPU compute dispatch in `compute/gpu_engine.go`
  and cuBLAS BF16 binding in `internal/cublas/cublas.go`.
- **Unified memory context:** DGX Spark GB10 has NVLink-C2C unified memory.
  `cudaMallocManaged` enables zero-copy. Currently `MemPool` only uses
  `cudaMalloc`. The Runtime interface in `internal/gpuapi/runtime.go` needs a
  `MallocManaged` method.
- **SigLIP context:** Concat fails at node 1462 with rank mismatch [1] vs [1 1].
  Debug on DGX Spark with the SigLIP ZMF at ~/models/siglip/model.zmf.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - CUDA+CUTLASS: `go build -tags cuda,cutlass ./...`
  - CUDA on DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build -tags cuda,cutlass ./...`
  - ROCm: `go build -tags rocm ./...`
  - OpenCL: `go build -tags opencl ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### External Dependencies

- **DGX Spark (ndungu@192.168.86.250, aitopatom-bfc8):**
  - Go 1.26.0 for linux/arm64 -- INSTALLED (~/.local/go).
  - cuDNN 9.19.1 for CUDA 13.0 -- INSTALLED.
  - TensorRT 10.15.1 -- INSTALLED.
  - NCCL 2.29.7 -- INSTALLED.
  - CUTLASS 4.2 headers -- INSTALLED (~/cutlass).
  - CUDA 13.0.2 and driver 580.126.09 -- INSTALLED.
- HIP SDK (>= 5.0) for AMD ROCm backend.
- OpenCL 2.0+ headers and ICD loader for OpenCL backend.
- CLBlast library for OpenCL BLAS operations.
- Second DGX Spark unit (optional) for multi-GPU validation via ConnectX-7.

---

## 9. Appendix

### Production Readiness Scorecard (After Phase 21)

| Category | Score | How Achieved |
|----------|-------|-------------|
| Architecture | 10/10 | Multi-architecture config; MLA; multi-GPU device affinity |
| Core Functionality | 10/10 | 6 model families; multi-GPU inference; NCCL gradient exchange |
| Testing | 10/10 | Parity tests for all architectures; multi-GPU integration tests |
| Error Handling | 9/10 | Structured logging, RPC validation, context deadlines |
| Security | 8/10 | TLS/mTLS for gRPC; HF_TOKEN for gated models |
| Observability | 8/10 | Logging, metrics, pprof endpoints |
| Configuration | 10/10 | Architecture-aware config parsing with HuggingFace field mapping |
| Operations | 10/10 | CLI pull/run/serve, OpenAI-compatible HTTP API |
| Documentation | 10/10 | Consolidated design.md + 18 ADRs |
| CI/CD | 9/10 | Blocking tests, coverage gate, benchmark gate |
| GPU Performance | 10/10 | cuBLAS + cuDNN + TensorRT (dynamic shapes) + CUTLASS flash attention + INT4/INT8 GEMM |
| GPU Portability | 8/10 | NVIDIA (CUDA/cuDNN/TensorRT), AMD (ROCm/HIP/MIOpen), OpenCL (CLBlast) |

### Key Files for Phase 22

| File | Purpose | Epic |
|------|---------|------|
| internal/cublas/cublas.go | cuBLAS CGo bindings (add GemmEx) | E117 |
| internal/gpuapi/blas.go | BLAS interface (add BFloat16Gemm) | E117 |
| internal/gpuapi/cuda_blas.go | CUDA BLAS adapter (implement BFloat16Gemm) | E117 |
| compute/gpu_engine.go | GPUEngine MatMul (add BF16 dispatch) | E117 |
| compute/gpu_kernels.go | GPU kernel helpers (fix elemSize) | E117 |
| internal/cuda/cuda.go | CUDA runtime (add MallocManaged) | E118 |
| internal/cuda/mempool.go | MemPool (add AllocManaged) | E118 |
| internal/gpuapi/mempool.go | MemPool interface (add AllocManaged) | E118 |
| internal/gpuapi/cuda_mempool.go | CUDA pool adapter (add AllocManaged) | E118 |
| tensor/gpu_storage.go | GPUStorage (add managed mode) | E118 |
| layers/core/concat.go | Concat layer (fix or trace) | E119 |
| compute/cpu_engine.go | Concat shape validation (possible fix) | E119 |
| tests/parity/siglip_test.go | SigLIP parity test | E119 |
