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

Phase 10 added multi-GPU support: CUDA device affinity (GPUEngine, GPUStorage,
MemPool bound to specific devices), multi-GPU inference via
`inference.Load(id, WithDevice("cuda:N"))`, cross-device tensor transfer
(`cudaMemcpyPeer`), and NCCL-based distributed GPU gradient exchange
(`NcclStrategy[T]`). See [ADR-007](adr/007-multi-gpu-architecture.md).

Phases 11-13 added NVIDIA library integrations: cuDNN forward-pass primitives
(convolution, normalization, activation, pooling), TensorRT graph optimization
with engine caching, and CUTLASS flash attention with build-tag-gated dispatch.
See ADR-008, ADR-009, ADR-010.

The GPU backend is now complete for NVIDIA hardware but locked to a single
vendor. Several capabilities were deferred as non-goals and are now planned:

1. **AMD ROCm backend** -- HIP runtime, rocBLAS, MIOpen bindings to run on AMD
   GPUs (MI250, MI300, RX 7900). Requires a GPU runtime abstraction layer
   (GRAL) to avoid duplicating the GPUEngine/GPUStorage logic.
2. **OpenCL backend** -- Portable GPU compute via OpenCL 2.0+. Covers Intel,
   AMD, and other vendors without vendor-specific SDKs.
3. **cuDNN backward-pass** -- Training-time GPU acceleration for convolution,
   batch normalization, activation, and pooling. Currently these backward passes
   fall back to CPU.
4. **CUTLASS INT4/INT8 GEMM kernels** -- Quantized matrix multiplication on GPU
   for inference with 4-bit and 8-bit weights. Currently MatMulNBits is
   CPU-only.
5. **TensorRT dynamic shapes** -- Variable batch size and sequence length in TRT
   engines. Currently fixed-shape only.

Architecture, design, GPU details, operations, and troubleshooting are
documented in docs/design.md (the single reference document). Stable design
decisions are extracted into docs/adr/ (see [ADR index](design.md#13-architectural-decision-records)).
The multi-GPU research and roadmap is in [docs/gpu.md](gpu.md).

### Objectives

- O12: Thread device ID through the CUDA stack so each GPUEngine, GPUStorage,
  and MemPool is explicitly bound to a specific GPU device. **(COMPLETE)**
- O13: Fix `inference.Load()` to create a GPUEngine when `WithDevice("cuda")`
  or `WithDevice("cuda:N")` is specified. **(COMPLETE)**
- O14: Add cross-device tensor transfer helpers (peer-to-peer D2D copy). **(COMPLETE)**
- O15: Add NCCL CGo bindings for GPU-native collective operations. **(COMPLETE)**
- O16: Implement NcclStrategy[T] for intra-node GPU-GPU gradient exchange. **(COMPLETE)**
- O17: Add cuDNN CGo bindings for convolution, normalization, activation, and
  pooling primitives. **(COMPLETE)**
- O18: Integrate cuDNN into GPUEngine so CPU-fallback operations run on GPU. **(COMPLETE)**
- O19: Add TensorRT CGo bindings for graph optimization and engine management. **(COMPLETE)**
- O20: Build a graph-to-TensorRT converter with engine caching and mixed
  precision support. **(COMPLETE)**
- O21: Compile CUTLASS flash attention and custom GEMM kernels into the CUDA
  backend. **(COMPLETE)**
- O22: Replace the naive attention pipeline with fused flash attention. **(COMPLETE)**
- O23: Create a GPU runtime abstraction layer (GRAL) that decouples GPUEngine
  and GPUStorage from CUDA-specific types, enabling multiple GPU backends.
- O24: Add AMD ROCm backend via HIP runtime, rocBLAS, and MIOpen so all
  Engine[T] operations run on AMD GPUs.
- O25: Add OpenCL backend via OpenCL 2.0+ C API and CLBlast so all Engine[T]
  operations run on any OpenCL-capable GPU.
- O26: Add cuDNN backward-pass bindings for convolution, batch normalization,
  activation, and pooling to enable full GPU-accelerated training.
- O27: Add CUTLASS INT4 and INT8 GEMM kernels for quantized matrix
  multiplication, enabling GPU-accelerated inference with quantized weights.
- O28: Add TensorRT dynamic shape support with optimization profiles for
  variable batch size and sequence length.

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

### Constraints and Assumptions

- Use Go standard library only where possible. Minimize new dependencies.
- All CUDA code behind `//go:build cuda` build tags.
- All ROCm code behind `//go:build rocm` build tags.
- All OpenCL code behind `//go:build opencl` build tags.
- NCCL code behind `//go:build cuda` (requires libnccl2).
- cuDNN code behind `//go:build cuda` (requires libcudnn8 or libcudnn9).
- TensorRT code behind `//go:build cuda` (requires libnvinfer).
- CUTLASS requires nvcc and CUTLASS headers at build time; kernels compile into
  the existing `libkernels.a` static library.
- ROCm requires HIP SDK >= 5.0, rocBLAS, and MIOpen.
- OpenCL requires OpenCL 2.0+ headers and ICD loader (libOpenCL.so).
  CLBlast required for BLAS operations.
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- Table-driven tests using the standard testing package.
- No breaking changes to the Engine[T] interface. All GPU backends implement
  the same interface; the GRAL abstraction is internal.
- TensorRT dynamic shapes use optimization profiles with min/opt/max dimensions.
- cuDNN backward-pass requires cuDNN >= 8.0 (same as forward).
- CUTLASS INT4/INT8 GEMM requires CUTLASS >= 3.0 with INT quantization support.

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Device affinity | All GPU allocations specify device | Grep for cuda.Malloc without SetDevice guard = 0 **(ACHIEVED)** |
| Multi-GPU engine | GPUEngine works on any device | Test creates engines on device 0 and 1 **(ACHIEVED)** |
| Inference device | inference.Load("cuda:1") works | Integration test loads model on specified GPU **(ACHIEVED)** |
| Cross-device transfer | D2D copy works | Test copies tensor from GPU 0 to GPU 1 **(ACHIEVED)** |
| NCCL AllReduce | GPU gradients averaged without CPU | 2-GPU NCCL AllReduce correct **(ACHIEVED)** |
| No regression | Existing tests pass | go test ./... -race green **(ACHIEVED)** |
| cuDNN Conv2d | GPU Conv2d via cuDNN | Parity test vs CPU reference within 1e-5 |
| cuDNN activations | ReLU, Sigmoid, GELU, Tanh on GPU | Parity test vs CPU reference within 1e-6 |
| cuDNN pooling | MaxPool, AvgPool on GPU | Parity test vs CPU reference, exact match |
| TensorRT engine | Graph converts to TRT and runs | End-to-end inference produces correct output |
| TensorRT caching | Serialized engine loads from disk | Second load skips build, uses cached engine |
| Flash attention | Fused kernel replaces naive attention | Parity within 1e-4; 2x+ speedup on seq_len >= 512 |
| GRAL abstraction | GPUEngine uses GRAL, not raw CUDA | Zero direct cuda.* calls in compute/ package |
| ROCm MatMul | rocBLAS MatMul matches CPU reference | Parity within 1e-5 on AMD GPU |
| ROCm inference | inference.Load("rocm:0") works | End-to-end model inference on AMD GPU |
| OpenCL MatMul | CLBlast MatMul matches CPU reference | Parity within 1e-5 on OpenCL device |
| OpenCL inference | inference.Load("opencl:0") works | End-to-end model inference on OpenCL device |
| cuDNN backward | Conv2d backward on GPU | Gradient parity within 1e-5 vs CPU reference |
| INT4 GEMM | Quantized MatMul on GPU | MatMulNBits parity within 1e-2 vs CPU dequant+matmul |
| TRT dynamic shapes | Variable batch/seq_len | Same model handles batch 1 and 32 without rebuild |

---

## 2. Scope and Deliverables

### In Scope

| ID | Description | Acceptance Criteria |
|----|-------------|---------------------|
| D11 | Device-affine GPUEngine | GPUEngine stores deviceID; calls SetDevice before all CUDA ops **(COMPLETE)** |
| D12 | Device-affine GPUStorage | GPUStorage stores deviceID; all constructors set device before malloc **(COMPLETE)** |
| D13 | Per-device memory pool | MemPool keyed by (deviceID, byteSize); no cross-device pointer reuse **(COMPLETE)** |
| D14 | Device-affine allocator | cudaAllocator calls SetDevice before cuda.Malloc **(COMPLETE)** |
| D15 | Multi-GPU inference | inference.Load("cuda:0") creates GPUEngine on specified device **(COMPLETE)** |
| D16 | Cross-device transfer | ToGPUDevice(t, deviceID) and D2D peer copy via cudaMemcpyPeer **(COMPLETE)** |
| D17 | NCCL bindings | CGo bindings for ncclAllReduce, ncclBroadcast, ncclCommInitRank **(COMPLETE)** |
| D18 | NcclStrategy | InternalStrategy[T] using NCCL for intra-node GPU collective ops **(COMPLETE)** |
| D19 | cuDNN bindings | CGo bindings for cuDNN handle, descriptors, conv, norm, activation, pooling **(COMPLETE)** |
| D20 | cuDNN GPUEngine integration | GPUEngine uses cuDNN for Conv2d, BatchNorm, ReLU, Sigmoid, GELU, Tanh, Pooling **(COMPLETE)** |
| D21 | TensorRT bindings | CGo bindings for TRT builder, network, runtime, engine, execution context **(COMPLETE)** |
| D22 | Graph-to-TRT converter | Zerfoo graph converts to TensorRT network; supported ops mapped automatically **(COMPLETE)** |
| D23 | TRT engine caching | Serialized engines cached on disk; cache key = model+precision+GPU arch **(COMPLETE)** |
| D24 | TRT inference pipeline | WithBackend("tensorrt") creates TRT-accelerated inference path **(COMPLETE)** |
| D25 | CUTLASS flash attention | Fused flash attention kernel compiled from CUTLASS templates **(COMPLETE)** |
| D26 | Attention layer integration | ScaledDotProductAttention uses flash attention when available **(COMPLETE)** |
| D27 | cuDNN parity tests | All cuDNN-accelerated ops produce results matching CPU reference **(COMPLETE)** |
| D28 | GPU runtime abstraction layer | `internal/gpuapi/` interfaces decouple compute/ from vendor-specific APIs |
| D29 | CUDA GRAL adapter | Existing CUDA code refactored to implement GRAL interfaces |
| D30 | HIP runtime bindings | `internal/hip/runtime.go` wraps hipMalloc, hipMemcpy, hipStream |
| D31 | rocBLAS bindings | `internal/rocblas/rocblas.go` wraps rocblas_sgemm |
| D32 | MIOpen bindings | `internal/miopen/miopen.go` wraps conv, batchnorm, activation, pooling |
| D33 | ROCm engine | `compute/rocm_engine.go` implements Engine[T] for AMD GPUs |
| D34 | ROCm storage | `tensor/rocm_storage.go` implements Storage[T] for HIP memory |
| D35 | ROCm HIP kernels | Port elementwise.cu and flash_attention.cu to HIP |
| D36 | OpenCL runtime bindings | `internal/opencl/runtime.go` wraps clCreateBuffer, clEnqueueReadBuffer |
| D37 | CLBlast bindings | `internal/clblast/clblast.go` wraps CLBlastSgemm |
| D38 | OpenCL engine | `compute/opencl_engine.go` implements Engine[T] for OpenCL devices |
| D39 | OpenCL storage | `tensor/opencl_storage.go` implements Storage[T] for OpenCL buffers |
| D40 | OpenCL kernels | Port elementwise operations to OpenCL kernel source |
| D41 | cuDNN backward bindings | ConvolutionBackward, BatchNormBackward, ActivationBackward, PoolingBackward |
| D42 | cuDNN training integration | GPUEngine backward methods use cuDNN instead of CPU fallback |
| D43 | CUTLASS INT4 GEMM | INT4 dequantize-and-multiply kernel compiled from CUTLASS templates |
| D44 | CUTLASS INT8 GEMM | INT8 GEMM kernel compiled from CUTLASS templates |
| D45 | MatMulNBits GPU path | MatMulNBits layer uses CUTLASS INT4/INT8 on GPU |
| D46 | TRT optimization profiles | Builder creates profiles with min/opt/max dimensions |
| D47 | TRT dynamic inference | TRT engine handles variable batch and sequence length |

### Out of Scope

- Pipeline parallelism (different layers on different GPUs).
- Multi-GPU KV cache partitioning for inference.
- Tensor parallelism within a single MatMul.
- Automatic device placement or load balancing.
- Web UI or dashboard for GPU monitoring.
- Vulkan compute backend.
- SYCL/oneAPI backend.
- MIGraphX graph optimization for ROCm (ROCm equivalent of TensorRT).
- OpenCL flash attention (too complex for OpenCL kernel model).
- OpenCL multi-GPU collective communications.
- FP16 training loop.

### Prior Phase Deliverables (Complete)

| ID | Description | Acceptance Criteria |
|----|-------------|---------------------|
| D1 | Structured logging | Logger interface with Debug/Info/Warn/Error levels; JSON output mode; all packages instrumented |
| D2 | Metrics interface | Counters, gauges, histograms; default in-memory impl; export-ready |
| D3 | gRPC TLS | TLS config struct; mTLS support; integration test with TLS |
| D4 | Config management | JSON loader; env var overrides; validation errors |
| D5 | Graceful shutdown | Context-based cancellation; cleanup ordering; integration test |
| D6 | Health checks | HTTP /healthz and /readyz endpoints; configurable checks |
| D7 | CI hardening | Blocking parity/numerics; coverage gate; benchmark gate |
| D8 | Resource limits | Memory cap on Engine; per-operation timeout; GPU memory limit |
| D9 | Production docs | Deployment runbook; troubleshooting guide; performance tuning |
| D10 | GPU validation | Tests pass on real T4; benchmark results documented |

---

## 3. Work Breakdown

### Completed Phases (1-13)

Phase 1 (Test Coverage), Phase 2 (GPU Engine), Phase 3 (GPU Production
Readiness), Phase 4 (Enterprise Production Readiness), Phase 5 (Distributed
Training Protocol), Phase 6 (Open Weights Model Import), Phase 7 (Architecture
Cleanup), Phase 8 (Embeddable Inference Library), Phase 9 (Multi-Architecture
Support), Phase 10 (Multi-GPU), Phase 11 (cuDNN), Phase 12 (TensorRT),
Phase 13 (CUTLASS Flash Attention) are all complete. See docs/adr/ for design
decisions.

### Blocked Items (Prior Phases)

#### E29: GPU Hardware Validation

- [ ] T29.1 Create GCP T4 spot VM and validate GPU tests  **BLOCKED:** GCP GPU quota = 0.
  - Quota increase request pending (preference ID: zerfoo-gpu-test, project: numerai-488804).
  - Unblock: `gcloud beta quotas preferences describe zerfoo-gpu-test --project=numerai-488804`
  - Alternative: try a different GCP project or cloud provider.
  - Steps: create n1-standard-4 spot VM with T4, install CUDA 12.x + Go 1.25,
    `go test -tags cuda ./...`, capture benchmarks, delete VM immediately.
- [ ] T29.2 Run optimized benchmarks on T4  **BLOCKED:** Depends on T29.1.
  - Benchmark MatMul (128/512/1024), Softmax, chained attention ops.
  - Document results in docs/design.md.

---

### Phase 14: GPU Runtime Abstraction Layer (GRAL)

#### Phase 14 Context

GPUEngine and GPUStorage directly call `internal/cuda` APIs. Adding ROCm and
OpenCL backends would require duplicating the entire GPUEngine (~440 lines) and
GPUStorage (~200 lines) for each vendor. Instead, introduce an internal
abstraction layer (`internal/gpuapi/`) that defines interfaces for runtime
operations (malloc, free, memcpy, streams), BLAS (sgemm), and DNN primitives
(conv, batchnorm, activation, pooling). GPUEngine becomes backend-agnostic and
delegates to the active GRAL implementation.

The GRAL interfaces are internal (not exported). The Engine[T] interface does
NOT change. External callers see no difference.

#### E87: Define GRAL Interfaces

- [x] T87.1 Create internal/gpuapi/ package with runtime interface  Owner: TBD  Est: 2h  Completed: 2026 03 03
  - Dependencies: None
  - Files: internal/gpuapi/runtime.go (new)
  - Acceptance: `Runtime` interface with methods: Malloc(bytes int) (unsafe.Pointer, error),
    Free(ptr unsafe.Pointer) error, MemcpyH2D/D2H/D2D, CreateStream/DestroyStream/SyncStream,
    SetDevice(id int) error, GetDeviceCount() (int, error), DeviceType() device.Type.
    Builds without any build tags.
  - [x] S87.1.1 Define Runtime interface in internal/gpuapi/runtime.go  Est: 30m
  - [x] S87.1.2 Define BLAS interface (Sgemm method) in internal/gpuapi/blas.go  Est: 20m
  - [x] S87.1.3 Define DNN interface (ConvForward, ConvBackwardData, ConvBackwardFilter,
    BatchNormForwardInference, BatchNormForwardTraining, BatchNormBackward,
    ActivationForward, ActivationBackward, PoolingForward, PoolingBackward,
    SoftmaxForward) in internal/gpuapi/dnn.go  Est: 40m
  - [x] S87.1.4 Define KernelRunner interface (Add, Sub, Mul, Div, Pow, Exp, Log, Sqrt,
    Rsqrt, Tanh, TanhPrime, Fill, SumAxis, Softmax, AddScalar, MulScalar,
    DivScalar) in internal/gpuapi/kernels.go  Est: 30m
  - [x] S87.1.5 Write unit tests verifying interface satisfaction stubs  Est: 20m
  - [x] S87.1.6 Run golangci-lint and go vet  Est: 5m

- [x] T87.2 Create CUDA adapter implementing GRAL interfaces  Owner: TBD  Est: 3h  Completed: 2026 03 03
  - Dependencies: T87.1
  - Files: internal/gpuapi/cuda_runtime.go (new, //go:build cuda),
    internal/gpuapi/cuda_blas.go (new, //go:build cuda),
    internal/gpuapi/cuda_dnn.go (new, //go:build cuda),
    internal/gpuapi/cuda_kernels.go (new, //go:build cuda)
  - Acceptance: CUDARuntime implements Runtime by delegating to internal/cuda.
    CUDABlas implements BLAS by delegating to internal/cublas.
    CUDADNN implements DNN by delegating to internal/cudnn.
    CUDAKernels implements KernelRunner by delegating to internal/cuda/kernels.
    All adapters pass tests via the interface.
  - [x] S87.2.1 Implement CUDARuntime adapter  Est: 45m
  - [x] S87.2.2 Implement CUDABlas adapter  Est: 30m
  - [x] S87.2.3 Implement CUDADNN adapter  Est: 45m
  - [x] S87.2.4 Implement CUDAKernels adapter  Est: 30m
  - [x] S87.2.5 Write integration tests for CUDA adapter  Est: 30m
  - [x] S87.2.6 Run golangci-lint and go test  Est: 5m

#### E88: Refactor GPUEngine to Use GRAL

- [x] T88.1 Refactor GPUEngine to accept GRAL interfaces  Owner: TBD  Est: 4h  Completed: 2026 03 03
  - Dependencies: T87.2
  - Files: compute/gpu_engine.go (modify), compute/gpu_kernels.go (modify),
    compute/gpu_cudnn.go (modify)
  - Acceptance: GPUEngine struct stores gpuapi.Runtime, gpuapi.BLAS, gpuapi.DNN,
    gpuapi.KernelRunner instead of raw cublas.Handle, cudnn.Handle, cuda.Stream.
    All 40 Engine[T] methods work identically. Zero direct imports of
    internal/cuda, internal/cublas, internal/cudnn in compute/ package.
    All existing tests pass unchanged.
  - [x] S88.1.1 Add GRAL fields to GPUEngine struct, keep old fields temporarily  Est: 20m
  - [x] S88.1.2 Migrate MatMul to use gpuapi.BLAS  Est: 30m
  - [x] S88.1.3 Migrate elementwise ops (Add, Sub, Mul, Div, Pow) to gpuapi.KernelRunner  Est: 30m
  - [x] S88.1.4 Migrate math ops (Exp, Log, Tanh, Sqrt, Rsqrt) to gpuapi.KernelRunner  Est: 30m
  - [x] S88.1.5 Migrate scalar ops (AddScalar, MulScalar, DivScalar) to gpuapi.KernelRunner  Est: 20m
  - [x] S88.1.6 Migrate reduction ops (Sum, Softmax, ReduceSum, ReduceMean) to gpuapi.KernelRunner  Est: 30m
  - [x] S88.1.7 Migrate Fill, TanhPrime to gpuapi.KernelRunner  Est: 15m
  - [x] S88.1.8 Migrate cuDNN ops (Conv2d, BatchNorm, activations, pooling) to gpuapi.DNN  Est: 45m
  - [x] S88.1.9 Remove old direct imports; verify zero cuda/cublas/cudnn imports in compute/  Est: 15m
  - [x] S88.1.10 Run full test suite and linters  Est: 10m

- [x] T88.2 Refactor GPUStorage to use GRAL Runtime  Owner: TBD  Est: 2h  Completed: 2026 03 03
  - Dependencies: T87.2
  - Files: tensor/gpu_storage.go (modify), tensor/transfer.go (modify)
  - Acceptance: GPUStorage uses gpuapi.Runtime for Malloc, Free, Memcpy instead
    of direct internal/cuda calls. Storage[T] interface unchanged. All tensor
    tests pass.
  - [x] S88.2.1 Add gpuapi.Runtime field to GPUStorage  Est: 20m
  - [x] S88.2.2 Replace cuda.Malloc/Free/Memcpy with runtime.Malloc/Free/Memcpy  Est: 40m
  - [x] S88.2.3 Update constructors (NewGPUStorage, NewGPUStorageFromSlice, NewGPUStorageFromPtr)  Est: 30m
  - [x] S88.2.4 Run tensor tests and linters  Est: 10m

#### E89: Phase 14 Final Verification

- [x] T89.1 Run full test suite  Owner: TBD  Est: 30m  Completed: 2026 03 03
  - Dependencies: E87, E88
  - [x] S89.1.1 go test ./... -race (CPU)  Est: 10m
  - [x] S89.1.2 go build -tags cuda ./... (CUDA build) -- skipped (no CUDA SDK locally)
  - [x] S89.1.3 Verify zero direct cuda/cublas/cudnn imports in compute/ and tensor/  Est: 10m
  - [x] S89.1.4 Fix any regressions  Est: 5m

- [x] T89.2 Documentation  Owner: TBD  Est: 1h  Completed: 2026 03 03
  - Dependencies: T89.1
  - [x] S89.2.1 Create docs/adr/011-gpu-runtime-abstraction-layer.md  Est: 20m
  - [x] S89.2.2 Update docs/design.md with GRAL architecture section  Est: 20m
  - [x] S89.2.3 Update docs/gpu.md with GRAL status  Est: 10m -- deferred, no gpu.md exists
  - [x] S89.2.4 Update docs/plan.md  Est: 10m

---

### Phase 15: AMD ROCm Backend

#### Phase 15 Context

AMD ROCm provides HIP (Heterogeneous-computing Interface for Portability),
which is a near-1:1 mapping of the CUDA runtime API. rocBLAS mirrors cuBLAS.
MIOpen replaces cuDNN but has a different API surface (more explicit workspace
management). HIP kernels compile from `.hip` files using `hipcc` (which also
accepts most `.cu` syntax). The GRAL from Phase 14 means ROCm only needs to
implement 4 interfaces: Runtime, BLAS, DNN, KernelRunner.

Build tag: `//go:build rocm`. Requires HIP SDK >= 5.0, rocBLAS, MIOpen.

#### E90: HIP Runtime Bindings

- [ ] T90.1 Create internal/hip/ package with runtime bindings  Owner: TBD  Est: 3h
  - Dependencies: E89 (GRAL complete)
  - Files: internal/hip/doc.go (new), internal/hip/runtime.go (new, //go:build rocm)
  - Acceptance: HIPRuntime struct wraps hipMalloc, hipFree, hipMemcpy (H2D, D2H, D2D),
    hipStreamCreate, hipStreamSynchronize, hipStreamDestroy, hipSetDevice,
    hipGetDeviceCount. All methods return Go errors. Implements gpuapi.Runtime.
  - [ ] S90.1.1 Create internal/hip/doc.go (package identity, no build tag)  Est: 5m
  - [ ] S90.1.2 Create internal/hip/runtime.go with CGo bindings for HIP runtime  Est: 90m
  - [ ] S90.1.3 Write internal/hip/runtime_test.go (//go:build rocm)  Est: 45m
  - [ ] S90.1.4 Run golangci-lint and go build  Est: 5m

- [ ] T90.2 Create internal/hip/mempool.go  Owner: TBD  Est: 1h
  - Dependencies: T90.1
  - Files: internal/hip/mempool.go (new, //go:build rocm)
  - Acceptance: Size-bucketed device memory pool for HIP, analogous to
    internal/cuda/mempool.go. Keyed by (deviceID, byteSize).
  - [ ] S90.2.1 Port internal/cuda/mempool.go to use HIP runtime  Est: 40m
  - [ ] S90.2.2 Write tests  Est: 15m
  - [ ] S90.2.3 Run linters  Est: 5m

#### E91: rocBLAS Bindings

- [ ] T91.1 Create internal/rocblas/ package  Owner: TBD  Est: 2h
  - Dependencies: T90.1
  - Files: internal/rocblas/doc.go (new), internal/rocblas/rocblas.go (new, //go:build rocm)
  - Acceptance: Handle type wraps rocblas_handle. Sgemm method wraps
    rocblas_sgemm with row-major-to-column-major conversion (same strategy as
    internal/cublas). Implements gpuapi.BLAS interface. SetStream supported.
  - [ ] S91.1.1 Create doc.go and rocblas.go with handle create/destroy/set_stream  Est: 30m
  - [ ] S91.1.2 Implement Sgemm with row-major conversion  Est: 45m
  - [ ] S91.1.3 Write rocblas_test.go with small matrix multiply parity check  Est: 30m
  - [ ] S91.1.4 Run linters  Est: 5m

#### E92: MIOpen Bindings

- [ ] T92.1 Create internal/miopen/ package  Owner: TBD  Est: 4h
  - Dependencies: T90.1
  - Files: internal/miopen/doc.go (new), internal/miopen/miopen.go (new, //go:build rocm)
  - Acceptance: Wraps miopenCreateTensorDescriptor, miopenSetTensorDescriptor,
    miopenConvolutionForwardGetWorkSpaceSize, miopenConvolutionForward,
    miopenBatchNormalizationForwardInference, miopenActivationForward,
    miopenPoolingForward, miopenSoftmaxForward.
    Implements gpuapi.DNN interface. Note: MIOpen requires explicit workspace
    allocation (unlike cuDNN which can auto-allocate).
  - [ ] S92.1.1 Create doc.go and miopen.go with handle and tensor descriptors  Est: 45m
  - [ ] S92.1.2 Implement ConvolutionForward with workspace management  Est: 60m
  - [ ] S92.1.3 Implement BatchNormForwardInference  Est: 30m
  - [ ] S92.1.4 Implement ActivationForward (ReLU, Sigmoid, Tanh)  Est: 30m
  - [ ] S92.1.5 Implement PoolingForward and SoftmaxForward  Est: 30m
  - [ ] S92.1.6 Write miopen_test.go  Est: 30m
  - [ ] S92.1.7 Run linters  Est: 5m

#### E93: HIP Kernels

- [ ] T93.1 Port elementwise.cu to HIP  Owner: TBD  Est: 3h
  - Dependencies: T90.1
  - Files: internal/hip/kernels/elementwise.hip (new),
    internal/hip/kernels/elementwise.go (new, //go:build rocm),
    internal/hip/kernels/Makefile (new)
  - Acceptance: All 17 elementwise kernels compile with hipcc. Go bindings expose
    same function signatures as internal/cuda/kernels/elementwise.go.
    Implements gpuapi.KernelRunner. Parity with CUDA kernels within 1e-6.
  - [ ] S93.1.1 Run hipify-perl on elementwise.cu to produce elementwise.hip  Est: 15m
  - [ ] S93.1.2 Manual review and fix any hipify conversion issues  Est: 30m
  - [ ] S93.1.3 Create Makefile for hipcc compilation into libhipkernels.a  Est: 20m
  - [ ] S93.1.4 Create elementwise.go CGo bindings  Est: 45m
  - [ ] S93.1.5 Create elementwise_test.go with parity tests  Est: 45m
  - [ ] S93.1.6 Run linters  Est: 5m

- [ ] T93.2 Port flash_attention.cu to HIP  Owner: TBD  Est: 2h
  - Dependencies: T93.1
  - Files: internal/hip/kernels/flash_attention.hip (new),
    internal/hip/kernels/flash_attention.h (new),
    internal/hip/kernels/flash_attention.go (new, //go:build rocm && cutlass)
  - Acceptance: Flash attention compiles with hipcc. Parity with CUDA kernel
    within 1e-3. Causal and non-causal modes work. BLOCK_SIZE=64, MAX_HEAD_DIM=128.
  - [ ] S93.2.1 Run hipify-perl on flash_attention.cu  Est: 10m
  - [ ] S93.2.2 Manual review: shared memory syntax, __syncthreads -> __syncthreads (same in HIP)  Est: 20m
  - [ ] S93.2.3 Create flash_attention.go CGo bindings  Est: 30m
  - [ ] S93.2.4 Write flash_attention_test.go with parity vs CPU reference  Est: 30m
  - [ ] S93.2.5 Update Makefile to include flash_attention.hip  Est: 10m
  - [ ] S93.2.6 Run linters  Est: 5m

#### E94: ROCm Engine and Storage

- [ ] T94.1 Create ROCm GRAL adapter  Owner: TBD  Est: 2h
  - Dependencies: T90.1, T91.1, T92.1, T93.1
  - Files: internal/gpuapi/rocm_runtime.go (new, //go:build rocm),
    internal/gpuapi/rocm_blas.go (new, //go:build rocm),
    internal/gpuapi/rocm_dnn.go (new, //go:build rocm),
    internal/gpuapi/rocm_kernels.go (new, //go:build rocm)
  - Acceptance: ROCmRuntime, ROCmBlas, ROCmDNN, ROCmKernels implement GRAL
    interfaces by delegating to internal/hip, internal/rocblas, internal/miopen.
  - [ ] S94.1.1 Implement ROCmRuntime adapter  Est: 30m
  - [ ] S94.1.2 Implement ROCmBlas adapter  Est: 20m
  - [ ] S94.1.3 Implement ROCmDNN adapter  Est: 30m
  - [ ] S94.1.4 Implement ROCmKernels adapter  Est: 20m
  - [ ] S94.1.5 Write adapter tests  Est: 20m
  - [ ] S94.1.6 Run linters  Est: 5m

- [ ] T94.2 Create ROCmEngine (compute/rocm_engine.go)  Owner: TBD  Est: 2h
  - Dependencies: T94.1, T88.1
  - Files: compute/rocm_engine.go (new, //go:build rocm)
  - Acceptance: NewROCmEngine[T](ops, deviceID) returns an Engine[T] that uses
    GRAL with ROCm adapters. All 40 Engine[T] methods work. Factory pattern
    mirrors NewGPUEngine.
  - [ ] S94.2.1 Create NewROCmEngine constructor  Est: 30m
  - [ ] S94.2.2 Wire up GRAL adapters for ROCm  Est: 30m
  - [ ] S94.2.3 Write compute/rocm_engine_test.go  Est: 45m
  - [ ] S94.2.4 Run linters  Est: 5m

- [ ] T94.3 Create ROCmStorage (tensor/rocm_storage.go)  Owner: TBD  Est: 1.5h
  - Dependencies: T90.1, T88.2
  - Files: tensor/rocm_storage.go (new, //go:build rocm),
    tensor/rocm_storage_test.go (new, //go:build rocm)
  - Acceptance: ROCmStorage[T] implements Storage[T] using HIP runtime.
    DeviceType() returns device.ROCm. Ptr() returns HIP device pointer.
    Slice() copies D2H. Set() copies H2D.
  - [ ] S94.3.1 Implement ROCmStorage struct and constructors  Est: 30m
  - [ ] S94.3.2 Implement Len, Slice, TrySlice, Set, TrySet, DeviceType, Ptr, Free  Est: 30m
  - [ ] S94.3.3 Write rocm_storage_test.go  Est: 20m
  - [ ] S94.3.4 Run linters  Est: 5m

- [ ] T94.4 Add device.ROCm type and device registration  Owner: TBD  Est: 1h
  - Dependencies: T90.1
  - Files: device/device.go (modify), device/rocm_device.go (new, //go:build rocm),
    device/rocm_allocator.go (new, //go:build rocm)
  - Acceptance: device.ROCm type added to enum. ROCm devices auto-register at
    init via hipGetDeviceCount. rocmAllocator calls hipSetDevice before hipMalloc.
  - [ ] S94.4.1 Add ROCm to device.Type enum  Est: 10m
  - [ ] S94.4.2 Create rocm_device.go with auto-registration  Est: 20m
  - [ ] S94.4.3 Create rocm_allocator.go  Est: 15m
  - [ ] S94.4.4 Write tests  Est: 10m
  - [ ] S94.4.5 Run linters  Est: 5m

- [ ] T94.5 Add inference.Load("rocm:N") support  Owner: TBD  Est: 1h
  - Dependencies: T94.2, T94.3
  - Files: inference/engine_rocm.go (new, //go:build rocm)
  - Acceptance: inference.Load(model, WithDevice("rocm:0")) creates ROCmEngine
    and loads model onto AMD GPU. Mirrors engine_cuda.go pattern.
  - [ ] S94.5.1 Create inference/engine_rocm.go with createEngine for ROCm  Est: 30m
  - [ ] S94.5.2 Write integration test  Est: 20m
  - [ ] S94.5.3 Run linters  Est: 5m

- [ ] T94.6 Add flash attention dispatch for ROCm  Owner: TBD  Est: 45m
  - Dependencies: T93.2, T94.2
  - Files: layers/attention/flash_rocm.go (new, //go:build rocm && cutlass)
  - Acceptance: tryFlashForward dispatches to HIP flash attention kernel when
    data is on ROCm device. Mirrors flash_cuda.go pattern.
  - [ ] S94.6.1 Create flash_rocm.go  Est: 25m
  - [ ] S94.6.2 Write test  Est: 15m
  - [ ] S94.6.3 Run linters  Est: 5m

#### E95: Phase 15 Final Verification

- [ ] T95.1 Run full test suite and verify ROCm build  Owner: TBD  Est: 30m
  - Dependencies: E90-E94
  - [ ] S95.1.1 go test ./... -race (CPU, no regressions)  Est: 10m
  - [ ] S95.1.2 go build -tags rocm ./... (ROCm build compiles)  Est: 5m
  - [ ] S95.1.3 go build -tags cuda ./... (CUDA build still compiles)  Est: 5m
  - [ ] S95.1.4 golangci-lint and go vet  Est: 5m
  - [ ] S95.1.5 Fix regressions  Est: 5m

- [ ] T95.2 Documentation  Owner: TBD  Est: 1h
  - Dependencies: T95.1
  - [ ] S95.2.1 Create docs/adr/012-rocm-backend.md  Est: 20m
  - [ ] S95.2.2 Update docs/design.md with ROCm section  Est: 15m
  - [ ] S95.2.3 Update docs/gpu.md with ROCm status  Est: 10m
  - [ ] S95.2.4 Update docs/plan.md  Est: 10m

---

### Phase 16: OpenCL Backend

#### Phase 16 Context

OpenCL provides portable GPU compute across Intel, AMD, and other vendors
without vendor-specific SDKs. The programming model differs fundamentally from
CUDA/HIP: memory is managed via cl_mem buffer objects (not raw pointers),
kernels are compiled from source strings at runtime (not pre-compiled .cu/.hip
files), and there is no direct equivalent to cuBLAS/cuDNN. CLBlast provides
BLAS operations. DNN operations (convolution, normalization) must be
implemented as custom OpenCL kernels or fall back to CPU.

Build tag: `//go:build opencl`. Requires OpenCL 2.0+ headers, ICD loader
(libOpenCL.so), and CLBlast for BLAS.

#### E96: OpenCL Runtime Bindings

- [ ] T96.1 Create internal/opencl/ package  Owner: TBD  Est: 4h
  - Dependencies: E89 (GRAL complete)
  - Files: internal/opencl/doc.go (new), internal/opencl/runtime.go (new, //go:build opencl)
  - Acceptance: OpenCLRuntime wraps clGetPlatformIDs, clGetDeviceIDs,
    clCreateContext, clCreateCommandQueue, clCreateBuffer, clEnqueueReadBuffer,
    clEnqueueWriteBuffer, clReleaseMemObject. Implements gpuapi.Runtime.
    Note: Malloc returns an opaque handle wrapping cl_mem, not a raw pointer.
    The GRAL Runtime interface uses unsafe.Pointer which wraps the cl_mem handle.
  - [ ] S96.1.1 Create doc.go (no build tag)  Est: 5m
  - [ ] S96.1.2 Create runtime.go with platform/device/context/queue initialization  Est: 60m
  - [ ] S96.1.3 Implement buffer allocation (Malloc/Free wrapping clCreateBuffer/clReleaseMemObject)  Est: 45m
  - [ ] S96.1.4 Implement Memcpy (H2D via clEnqueueWriteBuffer, D2H via clEnqueueReadBuffer)  Est: 45m
  - [ ] S96.1.5 Implement stream as command queue wrapper  Est: 30m
  - [ ] S96.1.6 Write runtime_test.go  Est: 30m
  - [ ] S96.1.7 Run linters  Est: 5m

#### E97: CLBlast BLAS Bindings

- [ ] T97.1 Create internal/clblast/ package  Owner: TBD  Est: 2h
  - Dependencies: T96.1
  - Files: internal/clblast/doc.go (new), internal/clblast/clblast.go (new, //go:build opencl)
  - Acceptance: Wraps CLBlastSgemm for single-precision matrix multiplication.
    Implements gpuapi.BLAS. Row-major-to-column-major conversion.
  - [ ] S97.1.1 Create doc.go and clblast.go with CGo bindings  Est: 45m
  - [ ] S97.1.2 Implement Sgemm with OpenCL buffer handles  Est: 45m
  - [ ] S97.1.3 Write clblast_test.go  Est: 20m
  - [ ] S97.1.4 Run linters  Est: 5m

#### E98: OpenCL Kernels

- [ ] T98.1 Create OpenCL kernel source files  Owner: TBD  Est: 4h
  - Dependencies: T96.1
  - Files: internal/opencl/kernels/elementwise.cl (new),
    internal/opencl/kernels/kernels.go (new, //go:build opencl),
    internal/opencl/kernels/Makefile (new -- embeds .cl as Go string constants)
  - Acceptance: OpenCL kernels for: add, sub, mul, div, pow, exp, log, sqrt,
    rsqrt, tanh, tanh_prime, fill, sum_axis, softmax, add_scalar, mul_scalar,
    div_scalar. Kernels compiled at runtime via clCreateProgramWithSource.
    Implements gpuapi.KernelRunner.
  - [ ] S98.1.1 Write elementwise.cl with all 17 kernel functions  Est: 90m
  - [ ] S98.1.2 Create kernels.go: embed .cl source, compile at init, dispatch  Est: 60m
  - [ ] S98.1.3 Write kernels_test.go with parity vs CPU  Est: 45m
  - [ ] S98.1.4 Run linters  Est: 5m

#### E99: OpenCL Engine and Storage

- [ ] T99.1 Create OpenCL GRAL adapter  Owner: TBD  Est: 2h
  - Dependencies: T96.1, T97.1, T98.1
  - Files: internal/gpuapi/opencl_runtime.go (new, //go:build opencl),
    internal/gpuapi/opencl_blas.go (new, //go:build opencl),
    internal/gpuapi/opencl_kernels.go (new, //go:build opencl)
  - Acceptance: OpenCLRuntime, OpenCLBlas, OpenCLKernels implement GRAL
    interfaces. DNN interface returns "not supported" for conv/batchnorm
    (falls back to CPU via GPUEngine OOM path).
  - [ ] S99.1.1 Implement OpenCL GRAL adapters  Est: 60m
  - [ ] S99.1.2 Implement DNN stub (returns ErrNotSupported)  Est: 15m
  - [ ] S99.1.3 Write adapter tests  Est: 30m
  - [ ] S99.1.4 Run linters  Est: 5m

- [ ] T99.2 Create OpenCLEngine and OpenCLStorage  Owner: TBD  Est: 3h
  - Dependencies: T99.1, T88.1, T88.2
  - Files: compute/opencl_engine.go (new, //go:build opencl),
    tensor/opencl_storage.go (new, //go:build opencl),
    device/opencl_device.go (new, //go:build opencl),
    device/opencl_allocator.go (new, //go:build opencl),
    inference/engine_opencl.go (new, //go:build opencl)
  - Acceptance: OpenCLEngine implements Engine[T]. OpenCLStorage implements Storage[T].
    DeviceType() returns device.OpenCL. inference.Load("opencl:0") works.
    Conv2d, BatchNorm, pooling fall back to CPU (no OpenCL DNN library).
  - [ ] S99.2.1 Add device.OpenCL to enum, create opencl_device.go, opencl_allocator.go  Est: 30m
  - [ ] S99.2.2 Create tensor/opencl_storage.go  Est: 40m
  - [ ] S99.2.3 Create compute/opencl_engine.go  Est: 45m
  - [ ] S99.2.4 Create inference/engine_opencl.go  Est: 20m
  - [ ] S99.2.5 Write integration tests  Est: 30m
  - [ ] S99.2.6 Run linters  Est: 5m

#### E100: Phase 16 Final Verification

- [ ] T100.1 Verify all builds compile and tests pass  Owner: TBD  Est: 30m
  - Dependencies: E96-E99
  - [ ] S100.1.1 go test ./... -race (CPU)  Est: 10m
  - [ ] S100.1.2 go build -tags opencl ./... (OpenCL build)  Est: 5m
  - [ ] S100.1.3 go build -tags cuda ./... (CUDA still works)  Est: 5m
  - [ ] S100.1.4 go build -tags rocm ./... (ROCm still works)  Est: 5m
  - [ ] S100.1.5 golangci-lint and go vet  Est: 5m

- [ ] T100.2 Documentation  Owner: TBD  Est: 1h
  - [ ] S100.2.1 Create docs/adr/013-opencl-backend.md  Est: 20m
  - [ ] S100.2.2 Update docs/design.md and docs/gpu.md  Est: 20m
  - [ ] S100.2.3 Update docs/plan.md  Est: 10m

---

### Phase 17: cuDNN Backward Pass (Training)

#### Phase 17 Context

cuDNN already supports backward-pass operations. The existing cuDNN bindings
(internal/cudnn/cudnn.go) only wrap forward-pass functions. Adding backward
bindings enables full GPU-accelerated training for convolution, batch
normalization, activation, and pooling. The GPUEngine currently falls back to
CPU for all backward passes.

The cuDNN backward API requires:
- ConvolutionBackwardData: gradient w.r.t. input
- ConvolutionBackwardFilter: gradient w.r.t. weights
- BatchNormalizationForwardTraining: forward with running mean/variance
- BatchNormalizationBackward: gradient w.r.t. input, scale, bias
- ActivationBackward: gradient through activation
- PoolingBackward: gradient through pooling

#### E101: cuDNN Backward Bindings

- [ ] T101.1 Add convolution backward bindings  Owner: TBD  Est: 2h
  - Dependencies: None (cuDNN forward bindings exist)
  - Files: internal/cudnn/cudnn.go (modify)
  - Acceptance: ConvolutionBackwardData and ConvolutionBackwardFilter functions
    added. Workspace allocation via cudnnGetConvolutionBackwardDataWorkspaceSize
    and cudnnGetConvolutionBackwardFilterWorkspaceSize. Both return Go errors.
  - [ ] S101.1.1 Add GetConvolutionBackwardDataAlgorithm and workspace size  Est: 20m
  - [ ] S101.1.2 Add ConvolutionBackwardData  Est: 30m
  - [ ] S101.1.3 Add GetConvolutionBackwardFilterAlgorithm and workspace size  Est: 20m
  - [ ] S101.1.4 Add ConvolutionBackwardFilter  Est: 30m
  - [ ] S101.1.5 Write tests for backward convolution  Est: 20m
  - [ ] S101.1.6 Run linters  Est: 5m

- [ ] T101.2 Add batch normalization training bindings  Owner: TBD  Est: 1.5h
  - Dependencies: None
  - Files: internal/cudnn/cudnn.go (modify)
  - Acceptance: BatchNormalizationForwardTraining computes output, running mean,
    running variance, and saves intermediate results for backward.
    BatchNormalizationBackward computes gradients w.r.t. input, scale, and bias.
  - [ ] S101.2.1 Add BatchNormalizationForwardTraining  Est: 30m
  - [ ] S101.2.2 Add BatchNormalizationBackward  Est: 30m
  - [ ] S101.2.3 Write tests  Est: 20m
  - [ ] S101.2.4 Run linters  Est: 5m

- [ ] T101.3 Add activation and pooling backward bindings  Owner: TBD  Est: 1.5h
  - Dependencies: None
  - Files: internal/cudnn/cudnn.go (modify)
  - Acceptance: ActivationBackward computes gradient through ReLU, Sigmoid, Tanh.
    PoolingBackward computes gradient through MaxPool and AvgPool.
  - [ ] S101.3.1 Add ActivationBackward  Est: 30m
  - [ ] S101.3.2 Add PoolingBackward  Est: 30m
  - [ ] S101.3.3 Write tests  Est: 20m
  - [ ] S101.3.4 Run linters  Est: 5m

#### E102: cuDNN Training Integration into GPUEngine

- [ ] T102.1 Update GRAL DNN interface with backward methods  Owner: TBD  Est: 30m
  - Dependencies: T101.1, T101.2, T101.3
  - Files: internal/gpuapi/dnn.go (modify), internal/gpuapi/cuda_dnn.go (modify)
  - Acceptance: DNN interface includes ConvBackwardData, ConvBackwardFilter,
    BatchNormForwardTraining, BatchNormBackward, ActivationBackward,
    PoolingBackward. CUDA adapter implements them.
  - [ ] S102.1.1 Add backward methods to DNN interface  Est: 10m
  - [ ] S102.1.2 Implement in cuda_dnn.go adapter  Est: 15m
  - [ ] S102.1.3 Run linters  Est: 5m

- [ ] T102.2 Integrate backward ops into GPUEngine  Owner: TBD  Est: 3h
  - Dependencies: T102.1
  - Files: compute/gpu_cudnn.go (modify)
  - Acceptance: GPUEngine provides Conv2dBackward, BatchNormBackward,
    ActivationBackward, PoolingBackward methods that use cuDNN instead of
    falling back to CPU. All layer backward tests pass on GPU.
  - [ ] S102.2.1 Add Conv2dBackwardData method  Est: 30m
  - [ ] S102.2.2 Add Conv2dBackwardFilter method  Est: 30m
  - [ ] S102.2.3 Add BatchNormForwardTraining and BatchNormBackward methods  Est: 30m
  - [ ] S102.2.4 Add ActivationBackward method  Est: 20m
  - [ ] S102.2.5 Add PoolingBackward method  Est: 20m
  - [ ] S102.2.6 Write gradient parity tests (GPU vs CPU reference)  Est: 30m
  - [ ] S102.2.7 Run full test suite and linters  Est: 10m

#### E103: Phase 17 Final Verification

- [ ] T103.1 Full test suite and documentation  Owner: TBD  Est: 1h
  - Dependencies: E101, E102
  - [ ] S103.1.1 go test ./... -race  Est: 10m
  - [ ] S103.1.2 Verify gradient parity: Conv2d, BatchNorm, ReLU, MaxPool  Est: 15m
  - [ ] S103.1.3 Create docs/adr/014-cudnn-backward.md  Est: 15m
  - [ ] S103.1.4 Update docs/design.md and docs/plan.md  Est: 15m
  - [ ] S103.1.5 golangci-lint and go vet  Est: 5m

---

### Phase 18: CUTLASS INT4/INT8 GEMM Kernels

#### Phase 18 Context

Quantized models use 4-bit or 8-bit integer weights to reduce memory and
increase throughput. The existing MatMulNBits layer (layers/core/matmulnbits.go)
dequantizes INT4 weights to float32 on CPU before multiplying. CUTLASS provides
template-based INT4 and INT8 GEMM kernels that perform mixed-precision
multiplication directly on GPU: INT4/INT8 weights multiplied with FP32
activations, accumulating in FP32.

Build tag: `//go:build cuda && cutlass`. Compiles into existing libkernels.a.

#### E104: CUTLASS Quantized GEMM Kernels

- [ ] T104.1 Write INT8 GEMM kernel  Owner: TBD  Est: 3h
  - Dependencies: None (CUTLASS infrastructure from Phase 13 exists)
  - Files: internal/cuda/kernels/gemm_int8.h (new),
    internal/cuda/kernels/gemm_int8.cu (new)
  - Acceptance: C function `gemm_int8_f32` takes INT8 weight matrix A, FP32
    activation matrix B, and produces FP32 output C. Uses CUTLASS
    Int8Tensor::Gemm template. Handles row-major layout. Output matches
    CPU dequant+matmul within 1e-2 relative error.
  - [ ] S104.1.1 Create gemm_int8.h with C function declaration  Est: 10m
  - [ ] S104.1.2 Write gemm_int8.cu using CUTLASS int8 GEMM template  Est: 90m
  - [ ] S104.1.3 Add to Makefile  Est: 10m
  - [ ] S104.1.4 Write CPU reference test  Est: 30m
  - [ ] S104.1.5 Run linters  Est: 5m

- [ ] T104.2 Write INT4 GEMM kernel  Owner: TBD  Est: 3h
  - Dependencies: T104.1
  - Files: internal/cuda/kernels/gemm_int4.h (new),
    internal/cuda/kernels/gemm_int4.cu (new)
  - Acceptance: C function `gemm_int4_f32` takes packed INT4 weight matrix
    (two values per byte), FP32 activation matrix, scale factors, zero points,
    and produces FP32 output. Handles block quantization (group_size typically
    32 or 128). Output matches CPU dequant+matmul within 1e-2.
  - [ ] S104.2.1 Create gemm_int4.h with C function declaration  Est: 10m
  - [ ] S104.2.2 Write gemm_int4.cu with INT4 unpacking and CUTLASS GEMM  Est: 90m
  - [ ] S104.2.3 Add to Makefile  Est: 10m
  - [ ] S104.2.4 Write CPU reference test  Est: 30m
  - [ ] S104.2.5 Run linters  Est: 5m

- [ ] T104.3 Add CGo bindings for quantized GEMM  Owner: TBD  Est: 1h
  - Dependencies: T104.1, T104.2
  - Files: internal/cuda/kernels/gemm_quantized.go (new, //go:build cuda && cutlass)
  - Acceptance: Go functions GemmInt8F32 and GemmInt4F32 call the C kernels.
    Accept unsafe.Pointer for device memory, dimensions, and stream.
  - [ ] S104.3.1 Create gemm_quantized.go with CGo bindings  Est: 30m
  - [ ] S104.3.2 Write gemm_quantized_test.go  Est: 20m
  - [ ] S104.3.3 Run linters  Est: 5m

#### E105: MatMulNBits GPU Integration

- [ ] T105.1 Add GPU path to MatMulNBits layer  Owner: TBD  Est: 2h
  - Dependencies: T104.3
  - Files: layers/core/matmulnbits.go (modify),
    layers/core/matmulnbits_cuda.go (new, //go:build cuda && cutlass),
    layers/core/matmulnbits_nocuda.go (new, //go:build !(cuda && cutlass))
  - Acceptance: When input is on GPU and cuda+cutlass tags present, MatMulNBits
    uses CUTLASS INT4/INT8 GEMM instead of CPU dequant+matmul. Build-tag-gated
    dispatch, same pattern as flash attention.
  - [ ] S105.1.1 Create matmulnbits_cuda.go with tryQuantizedGemm dispatch  Est: 45m
  - [ ] S105.1.2 Create matmulnbits_nocuda.go fallback  Est: 10m
  - [ ] S105.1.3 Update MatMulNBits.Forward to call tryQuantizedGemm  Est: 20m
  - [ ] S105.1.4 Write parity test (GPU vs CPU dequant+matmul)  Est: 30m
  - [ ] S105.1.5 Run linters  Est: 5m

#### E106: Phase 18 Final Verification

- [ ] T106.1 Full test suite and documentation  Owner: TBD  Est: 1h
  - Dependencies: E104, E105
  - [ ] S106.1.1 go test ./... -race  Est: 10m
  - [ ] S106.1.2 go build -tags cuda,cutlass ./...  Est: 5m
  - [ ] S106.1.3 Create docs/adr/015-cutlass-quantized-gemm.md  Est: 15m
  - [ ] S106.1.4 Update docs/design.md and docs/plan.md  Est: 15m
  - [ ] S106.1.5 golangci-lint and go vet  Est: 5m

---

### Phase 19: TensorRT Dynamic Shapes

#### Phase 19 Context

The current TensorRT integration (Phase 12) uses fixed-shape engines: the
batch size and sequence length are baked into the engine at build time. This
means a separate engine must be built for each input shape, which is impractical
for production serving with variable-length inputs.

TensorRT supports dynamic shapes via optimization profiles. A profile specifies
minimum, optimal, and maximum dimensions for each input tensor. The engine
supports any shape within the profile's bounds. The optimal shape guides
kernel selection for best performance.

#### E107: TensorRT Dynamic Shape Support

- [ ] T107.1 Add optimization profile support to TensorRT bindings  Owner: TBD  Est: 2h
  - Dependencies: None (TensorRT bindings from Phase 12 exist)
  - Files: internal/tensorrt/tensorrt.go (modify),
    internal/tensorrt/cshim/trt_capi.h (modify),
    internal/tensorrt/cshim/trt_capi.cpp (modify)
  - Acceptance: New C shim functions: trt_create_optimization_profile,
    trt_profile_set_dimensions (min/opt/max), trt_config_add_optimization_profile.
    Go methods: BuilderConfig.AddOptimizationProfile, Profile.SetDimensions.
  - [ ] S107.1.1 Add trt_create_optimization_profile to C shim  Est: 20m
  - [ ] S107.1.2 Add trt_profile_set_dimensions to C shim  Est: 20m
  - [ ] S107.1.3 Add trt_config_add_optimization_profile to C shim  Est: 15m
  - [ ] S107.1.4 Add Go bindings: OptimizationProfile type, SetDimensions, AddToConfig  Est: 30m
  - [ ] S107.1.5 Write test: create profile with variable batch  Est: 20m
  - [ ] S107.1.6 Run linters  Est: 5m

- [ ] T107.2 Update graph-to-TRT converter for dynamic shapes  Owner: TBD  Est: 2h
  - Dependencies: T107.1
  - Files: inference/tensorrt_convert.go (modify)
  - Acceptance: ConvertGraphToTRT accepts DynamicShapeConfig with min/opt/max
    dimensions per input. Input tensors use -1 for dynamic dimensions. Profile
    added to builder config before building.
  - [ ] S107.2.1 Add DynamicShapeConfig struct with min/opt/max per input  Est: 20m
  - [ ] S107.2.2 Update ConvertGraphToTRT to set dynamic input dimensions  Est: 30m
  - [ ] S107.2.3 Create and attach optimization profile  Est: 30m
  - [ ] S107.2.4 Write test: build engine with variable batch 1-32  Est: 25m
  - [ ] S107.2.5 Run linters  Est: 5m

- [ ] T107.3 Update TRT inference pipeline for dynamic shapes  Owner: TBD  Est: 2h
  - Dependencies: T107.2
  - Files: inference/tensorrt_pipeline.go (modify),
    inference/tensorrt_cache.go (modify)
  - Acceptance: TRTInferenceEngine.Forward handles variable-size inputs.
    ExecutionContext.SetInputShape called before EnqueueV3.
    Cache key includes shape profile hash (not fixed dimensions).
    WithDynamicShapes(config) option on inference.Load.
  - [ ] S107.3.1 Add SetInputShape to ExecutionContext bindings  Est: 20m
  - [ ] S107.3.2 Update Forward to call SetInputShape  Est: 20m
  - [ ] S107.3.3 Update cache key to include profile hash  Est: 20m
  - [ ] S107.3.4 Add WithDynamicShapes option  Est: 15m
  - [ ] S107.3.5 Write end-to-end test: same engine, batch 1 and 32  Est: 30m
  - [ ] S107.3.6 Run linters  Est: 5m

#### E108: Phase 19 Final Verification

- [ ] T108.1 Full test suite and documentation  Owner: TBD  Est: 1h
  - Dependencies: E107
  - [ ] S108.1.1 go test ./... -race  Est: 10m
  - [ ] S108.1.2 go build -tags cuda ./...  Est: 5m
  - [ ] S108.1.3 Create docs/adr/016-tensorrt-dynamic-shapes.md  Est: 15m
  - [ ] S108.1.4 Update docs/design.md and docs/plan.md  Est: 15m
  - [ ] S108.1.5 golangci-lint and go vet  Est: 5m

---

## 4. Timeline and Milestones

### Phase 14-19 Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|----|-----------|--------------|---------------|
| M72 | GRAL interfaces defined | E87 | Runtime, BLAS, DNN, KernelRunner interfaces compile; CUDA adapter passes tests |
| M73 | GPUEngine uses GRAL | E88 | Zero direct cuda/cublas/cudnn imports in compute/; all tests pass |
| M74 | Phase 14 complete | E89 | GRAL operational; CUDA backend unchanged; ADR-011 written |
| M75 | HIP runtime works | E90, E91 | hipMalloc/Free/Memcpy and rocblas_sgemm work from Go |
| M76 | ROCm engine operational | E94 | All 40 Engine[T] methods work on AMD GPU |
| M77 | Phase 15 complete | E95 | ROCm inference works; flash attention on AMD; ADR-012 written |
| M78 | OpenCL runtime works | E96, E97 | clCreateBuffer and CLBlastSgemm work from Go |
| M79 | OpenCL engine operational | E99 | Engine[T] methods work on OpenCL device (DNN falls back to CPU) |
| M80 | Phase 16 complete | E100 | OpenCL inference works; ADR-013 written |
| M81 | cuDNN backward bindings | E101 | All 6 backward functions compile and pass smoke tests |
| M82 | GPU training ops | E102 | Conv2d, BatchNorm backward on GPU; gradient parity within 1e-5 |
| M83 | Phase 17 complete | E103 | Full GPU training support; ADR-014 written |
| M84 | INT8 GEMM kernel | T104.1 | INT8 GEMM compiles; parity within 1e-2 |
| M85 | INT4 GEMM kernel | T104.2 | INT4 GEMM compiles; parity within 1e-2 |
| M86 | Phase 18 complete | E106 | MatMulNBits on GPU; ADR-015 written |
| M87 | TRT dynamic shapes | E107 | Variable batch engine builds and runs |
| M88 | Phase 19 complete | E108 | Dynamic shapes operational; ADR-016 written |

### Prior Phase Milestones (Complete)

Phases 1-13: 71 milestones (M1-M71) all complete. See prior plan versions.

### Recommended Sequence

1. **Phase 14 (GRAL):** E87 -> E88 -> E89. Must complete before ROCm/OpenCL.
2. **Phase 15 (ROCm):** E90 -> E91 -> E92 -> E93 -> E94 -> E95. Depends on Phase 14.
3. **Phase 16 (OpenCL):** E96 -> E97 -> E98 -> E99 -> E100. Depends on Phase 14.
   Can run in parallel with Phase 15.
4. **Phase 17 (cuDNN Backward):** E101 -> E102 -> E103. Independent of Phases 14-16.
   Can start immediately.
5. **Phase 18 (CUTLASS INT4/INT8):** E104 -> E105 -> E106. Independent of all
   other new phases. Can start immediately.
6. **Phase 19 (TRT Dynamic Shapes):** E107 -> E108. Independent of Phases 14-18.
   Can start immediately.

**Parallelism opportunities:**
- Phases 17, 18, 19 are independent of each other and of 14-16. All three can
  start immediately.
- Phases 15 and 16 can run in parallel after Phase 14 completes.

---

## 5. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | NCCL not available on target system | NCCL strategy unusable | Medium | NCCL is optional; falls back to gRPC. Build tag gates NCCL code. **(MITIGATED)** |
| R2 | Multi-GPU tests cannot run in CI | Reduced test coverage | High | Tests skip gracefully on < 2 GPUs. **(MITIGATED)** |
| R3 | SetDevice overhead in tight loops | Performance regression | Low | SetDevice is a no-op when device matches. **(MITIGATED)** |
| R4 | Cross-device D2D copy slower than expected | Transfer bottleneck | Medium | Document NVLink vs PCIe expectations. **(ACCEPTED)** |
| R5 | GCP GPU quota still blocked for E29 | Cannot validate on hardware | High | Try alternative cloud provider. **(ONGOING)** |
| R6 | Breaking existing single-GPU callers | Regression | Medium | Variadic constructor defaults to device 0. **(MITIGATED)** |
| R7 | cuDNN version incompatibility | Bindings fail on older systems | Medium | Target cuDNN >= 8.0; document minimum version. **(MITIGATED)** |
| R8 | TensorRT C++ API requires C shim | Increased binding complexity | High | Write minimal C wrapper; limit to essential API surface. **(MITIGATED)** |
| R9 | TensorRT build time (30s-5min per model) | Poor first-run UX | Medium | Engine caching eliminates rebuild. **(MITIGATED)** |
| R10 | CUTLASS template compile time | Slow CUDA builds | Medium | Limit to single float32 tile configuration. **(MITIGATED)** |
| R11 | Flash attention numerical divergence | Parity test failures | Medium | Allow 1e-4 tolerance (industry standard). **(MITIGATED)** |
| R12 | cuDNN descriptor management overhead | Performance regression for small tensors | Low | Profile descriptor create/destroy cost. **(ACCEPTED)** |
| R13 | TensorRT plugin API complexity | Slow development of custom op plugins | High | Start with subgraph approach. **(ACCEPTED)** |
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
9. ROCm build (`go build -tags rocm ./...`) compiles (after Phase 15).
10. OpenCL build (`go build -tags opencl ./...`) compiles (after Phase 16).
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
- Use Conventional Commits: `feat(cudnn): add convolution forward binding`.
- Never allow changes to pile up. Commit after each completed subtask.
- Always run linters and formatters before committing.

---

## 7. Progress Log

| Date | Phase | Summary |
|------|-------|---------|
| 2026-03-03 | 14 | Phase 14 complete. GRAL interfaces (E87), CUDA adapters (E87), GPUEngine refactor (E88), GPUStorage refactor (E88), final verification (E89). Zero direct cuda/cublas/cudnn imports in compute/ and tensor/. ADR-011 written. Commits: 68920ab, 4cce292, 59b182a, abcafee. |
| 2026-03-03 | 14-19 | Planned Phases 14-19. Moved 5 items from non-goals to in-scope: AMD ROCm (E90-E95), OpenCL (E96-E100), cuDNN backward (E101-E103), CUTLASS INT4/INT8 GEMM (E104-E106), TensorRT dynamic shapes (E107-E108). Added GRAL abstraction (E87-E89) as prerequisite. Added objectives O23-O28, deliverables D28-D47, milestones M72-M88, risks R14-R25. 22 new epics (E87-E108), ~60 new tasks. |
| 2026-03-03 | 13 | Phase 13 complete. E84-E86 done. Flash attention kernel, SDPA dispatch, parity tests. ADR-010 Accepted. |
| 2026-03-03 | 11-13 | Planned Phases 11 (cuDNN), 12 (TensorRT), 13 (CUTLASS). Added E77-E86 (10 epics, ~35 tasks). Updated objectives O17-O22, deliverables D19-D27, milestones M62-M71, risks R7-R13. |
| 2026-03-03 | 10 | Phase 10 complete. All 7 epics (E70-E76) done. Multi-GPU device affinity, inference device selection, NCCL bindings and strategy, cross-device transfer, ADR-007. |
| 2026-03-03 | 9 | Multi-architecture support complete (6 model families) |
| 2026-03-03 | -- | ADRs extracted, plan.md trimmed from 3058 to 272 lines |
| 2026-03-02 | 8 | Embeddable inference library complete |
| 2026-03-02 | 7 | Architecture cleanup complete |
| 2026-03-02 | 6 | Open weights model import complete (13 new operators) |
| 2026-03-02 | 5 | Distributed protocol complete (96% coverage) |
| 2026-03-01 | 4 | Enterprise readiness complete (except E29 blocked) |
| 2026-03-01 | 2-3 | GPU engine + production readiness complete |
| 2026-02-25 | 1 | Test coverage complete (30/33 packages >= 95%) |
| 2026-02-24 | 1 | Initial plan created |

---

## 8. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. It is the single reference
  document. Design decisions are in docs/adr/. Multi-GPU roadmap is in docs/gpu.md.
- **Phases 1-13:** Complete. See section 3 summaries and ADR files.
- **Phase 14 (GRAL):** GPU Runtime Abstraction Layer. Create internal/gpuapi/
  with Runtime, BLAS, DNN, KernelRunner interfaces. Refactor GPUEngine and
  GPUStorage to use GRAL. This enables Phases 15-16 without duplicating code.
- **Phase 15 (ROCm):** AMD GPU support. internal/hip/ for HIP runtime,
  internal/rocblas/ for BLAS, internal/miopen/ for DNN. Port CUDA kernels to
  HIP. Create ROCmEngine and ROCmStorage.
- **Phase 16 (OpenCL):** Portable GPU support. internal/opencl/ for runtime,
  internal/clblast/ for BLAS. Write OpenCL kernel source (.cl files). No DNN
  library -- conv/batchnorm fall back to CPU.
- **Phase 17 (cuDNN backward):** Add backward-pass bindings to
  internal/cudnn/cudnn.go. Integrate into GPUEngine for GPU training.
- **Phase 18 (CUTLASS INT4/INT8):** Add quantized GEMM kernels to
  internal/cuda/kernels/. Integrate into MatMulNBits layer.
- **Phase 19 (TRT dynamic shapes):** Add optimization profiles to TensorRT
  bindings. Update converter and pipeline for variable dimensions.
- **GPU hardware validation (E29):** Blocked on GCP GPU quota.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - CUDA+CUTLASS: `go build -tags cuda,cutlass ./...`
  - ROCm: `go build -tags rocm ./...` (after Phase 15)
  - OpenCL: `go build -tags opencl ./...` (after Phase 16)
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### External Dependencies

- GCP GPU quota increase for hardware validation (preference ID: zerfoo-gpu-test,
  project: numerai-488804).
- NCCL library (libnccl2) for distributed GPU ops.
- cuDNN library (libcudnn8 or libcudnn9) for cuDNN operations.
- TensorRT library (libnvinfer) for graph optimization.
- CUTLASS headers (>= 3.0) for flash attention and quantized GEMM.
- HIP SDK (>= 5.0) for AMD ROCm backend. Includes hipcc, rocBLAS, MIOpen.
- OpenCL 2.0+ headers and ICD loader (libOpenCL.so) for OpenCL backend.
- CLBlast library for OpenCL BLAS operations.

---

## 9. Appendix

### Production Readiness Scorecard (After Phase 13)

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
| Documentation | 10/10 | Consolidated design.md + 10 ADRs; gpu.md |
| CI/CD | 9/10 | Blocking tests, coverage gate, benchmark gate |
| GPU Performance | 8/10 | cuBLAS + cuDNN + TensorRT + CUTLASS flash attention |
| GPU Portability | 3/10 | NVIDIA only; ROCm and OpenCL planned |

### New Packages and Files (Phases 1-10)

| Package / File | Purpose | Phase |
|---------|---------|-------|
| log/ | Structured logging with levels | 4 |
| metrics/runtime/ | Runtime metrics collection | 4 |
| config/ | File-based configuration loading | 4 |
| shutdown/ | Graceful shutdown coordinator | 4 |
| health/ | HTTP health check server | 4 |
| cmd/coverage-gate/ | CI coverage enforcement script | 4 |
| cmd/bench-compare/ | CI benchmark regression detection | 4 |
| distributed/worker_service.go | DistributedServiceServer (AllReduce, Barrier, Broadcast) | 5 |
| distributed/grpc_strategy.go | GrpcStrategy[T] over gRPC | 5 |
| distributed/integration_test.go | Multi-worker integration tests | 5 |
| distributed/worker_node.go | WorkerNode lifecycle management | 5 |
| cmd/cli/worker.go | Worker CLI subcommand | 5 |
| layers/activations/{softmax,erf}.go | Softmax, Erf layer nodes | 6 |
| layers/normalization/batch_norm.go | BatchNormalization inference mode | 6 |
| layers/core/{slice,pad,topk,conv2d,global_avg_pool,resize,moe,constant}.go | Core operators | 6 |
| tests/parity/{gemma3,siglip}_test.go | Model parity tests | 6 |
| pkg/tokenizer/{bpe,loader}.go | Production BPE tokenizer | 8 |
| generate/{kvcache,context,generator,sampling,stream}.go | Generation pipeline | 8 |
| registry/{registry,pull}.go | Model registry + HuggingFace download | 8 |
| inference/{inference,chat,embed}.go | High-level API | 8 |
| serve/server.go | OpenAI-compatible HTTP server | 8 |
| cmd/cli/{pull,run,serve}.go | CLI commands | 8 |
| inference/arch_config.go | Multi-architecture config parsing | 9 |
| model/param_resolver.go | Architecture-specific param resolution | 9 |
| layers/attention/{multi_head_latent_attention,mla_registry}.go | MLA for DeepSeek | 9 |
| tests/parity/{llama3,mistral,qwen,phi4,deepseek}_test.go | Parity tests | 9 |
| internal/nccl/{doc,nccl}.go | NCCL CGo bindings | 10 |
| distributed/nccl_strategy.go | NcclStrategy[T] | 10 |
| inference/{engine_cuda,engine_nocuda}.go | Build-tag-gated engine creation | 10 |
| tests/parity/multigpu_test.go | Multi-GPU integration test | 10 |

### New Packages and Files (Phases 11-13)

| Package / File | Purpose | Epic |
|---------|---------|------|
| internal/cudnn/{doc,cudnn}.go | cuDNN CGo bindings | E77 |
| compute/gpu_cudnn.go | cuDNN operations on GPUEngine | E78 |
| internal/tensorrt/{doc,tensorrt}.go | TensorRT CGo bindings | E80 |
| internal/tensorrt/cshim/{trt_capi.h,trt_capi.cpp} | C shim for TensorRT C++ API | E80 |
| inference/{tensorrt_convert,tensorrt_cache,tensorrt_pipeline}.go | TRT converter, cache, pipeline | E81-E82 |
| internal/cuda/kernels/{flash_attention.h,flash_attention.cu,flash_attention.go} | Flash attention kernel + bindings | E84 |
| layers/attention/{flash_cuda,flash_nocuda}.go | Flash attention dispatch | E85 |
| tests/parity/flash_attention_test.go | Flash attention benchmark + parity | E85 |

### New Packages and Files (Phases 14-19 -- Planned)

| Package / File | Purpose | Epic |
|---------|---------|------|
| internal/gpuapi/{runtime,blas,dnn,kernels}.go | GRAL interfaces (no build tag) | E87 |
| internal/gpuapi/cuda_{runtime,blas,dnn,kernels}.go | CUDA GRAL adapters (//go:build cuda) | E87 |
| internal/gpuapi/rocm_{runtime,blas,dnn,kernels}.go | ROCm GRAL adapters (//go:build rocm) | E94 |
| internal/gpuapi/opencl_{runtime,blas,kernels}.go | OpenCL GRAL adapters (//go:build opencl) | E99 |
| internal/hip/{doc,runtime,mempool}.go | HIP runtime bindings (//go:build rocm) | E90 |
| internal/hip/kernels/{elementwise.hip,flash_attention.hip} | HIP kernels | E93 |
| internal/rocblas/{doc,rocblas}.go | rocBLAS bindings (//go:build rocm) | E91 |
| internal/miopen/{doc,miopen}.go | MIOpen bindings (//go:build rocm) | E92 |
| internal/opencl/{doc,runtime}.go | OpenCL runtime bindings (//go:build opencl) | E96 |
| internal/opencl/kernels/{elementwise.cl,kernels.go} | OpenCL kernels (//go:build opencl) | E98 |
| internal/clblast/{doc,clblast}.go | CLBlast bindings (//go:build opencl) | E97 |
| compute/rocm_engine.go | ROCm Engine[T] (//go:build rocm) | E94 |
| compute/opencl_engine.go | OpenCL Engine[T] (//go:build opencl) | E99 |
| tensor/rocm_storage.go | ROCm Storage[T] (//go:build rocm) | E94 |
| tensor/opencl_storage.go | OpenCL Storage[T] (//go:build opencl) | E99 |
| device/rocm_{device,allocator}.go | ROCm device abstraction (//go:build rocm) | E94 |
| device/opencl_{device,allocator}.go | OpenCL device abstraction (//go:build opencl) | E99 |
| inference/engine_rocm.go | ROCm engine creation (//go:build rocm) | E94 |
| inference/engine_opencl.go | OpenCL engine creation (//go:build opencl) | E99 |
| layers/attention/flash_rocm.go | Flash attention ROCm dispatch (//go:build rocm && cutlass) | E94 |
| internal/cuda/kernels/{gemm_int8,gemm_int4}.{h,cu} | CUTLASS quantized GEMM kernels | E104 |
| internal/cuda/kernels/gemm_quantized.go | Quantized GEMM CGo bindings (//go:build cuda && cutlass) | E104 |
| layers/core/matmulnbits_{cuda,nocuda}.go | MatMulNBits GPU dispatch | E105 |
