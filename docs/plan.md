# Zerfoo Development Plan -- Phase 34: Close the Gap with llama.cpp

## 1. Context

### Problem Statement

Phase 33 achieved 10.32 tok/s peak (7.78 median) for Gemma 3 2B Q4 on DGX Spark
GB10. Ollama/llama.cpp on the same hardware achieves:

| Model | Quant | Ollama tok/s | Zerfoo tok/s | Gap |
|-------|-------|-------------|-------------|-----|
| Gemma 3 1B | Q4 | 205 | n/a | n/a |
| Gemma 3 4B | Q4 | 77.7 | n/a | n/a |
| Gemma 3 2B | Q4 | ~100 (est.) | 7.78 median | ~13x |

The DGX Spark GB10 has 273 GB/s LPDDR5x bandwidth. For a 1.5GB Q4 model, the
theoretical max is ~182 tok/s. Zerfoo at 7.78 median is 4.3% of theoretical;
llama.cpp reaches 13-55% depending on model size.

The gap has three causes:

1. **CGo overhead and build complexity**: 8 files use `import "C"` with
   `//go:build cuda` tags. Each CGo call costs ~100-200ns. With 650+ calls per
   token, that is ~65-130us per token. More importantly, CGo forces compile-time
   GPU detection and slow builds.

2. **Per-op kernel launches**: Each forward pass dispatches 25+ GPU operations
   individually. For single-token decode with small tensors (batch=1), kernel
   launch overhead (~5-10us each) dominates compute. 650 launches x 5us = 3.2ms
   overhead per token.

3. **Global memory round-trips**: Every intermediate activation is written to
   and read from global GPU memory between ops. For a 26-layer transformer with
   ~25 ops per layer, that is ~650 global memory round-trips per token. llama.cpp
   uses fused kernels and CUDA graphs to minimize this.

Phase 34 addresses these with three tracks executed sequentially:

- **Track 0 (composition fixes)**: A 5-agent audit found 12 layers that do
  complex math inline instead of composing Engine primitives. These must be
  fixed first -- the megakernel code generator only sees ops in the instruction
  tape. See docs/adr/027-composition-prerequisite.md.

- **Track A (purego)**: Replace CGo with dlopen-based pure Go bindings. Quick
  win: cleaner build, runtime GPU detection, single binary, minor perf gain.
  Benefits ALL layer types and architectures.

- **Track B (megakernel)**: Generate a single CUDA kernel from the compiled
  ExecutionPlan instruction tape. Because Zerfoo is compositional (all complex
  layers decompose into primitive Engine ops), the generator is architecture-
  agnostic: it maps each primitive op (Add, MatMul, RMSNorm, Softmax, etc.) to
  a register-resident CUDA device function and chains them in instruction order.
  Any model that compiles into an ExecutionPlan gets a megakernel automatically.
  Maximum performance: one launch per token, all intermediates in registers/
  shared memory.

See docs/design.md for full architecture context and Phases 1-33 history.
Decision rationale: docs/adr/024-cuda-graph-fused-kernels.md (original approach),
docs/adr/025-purego-cuda-bindings.md (purego), docs/adr/026-megakernel-decode.md
(megakernel).

### What Was Delivered (Phase 33)

| Area | Key Result |
|------|------------|
| PowScalar GPU kernel | Eliminated 8.9% Pow CPU fallback |
| SubScalar GPU kernel | Completed scalar-op coverage |
| Scalar-broadcast detection | Eliminated 10.4% binary op CPU fallback |
| GPU Split/Concat (D2D memcpy) | Eliminated 24% D2H from GPUStorage.Slice |
| Float32 weight upload | All weights GPU-resident at load time |
| Benchmark | 10.32 tok/s peak / 7.78 median GPU on DGX Spark GB10 |

### Current Architecture

The decode loop in `generate/stream.go` calls `ExecutionPlan.Run()` per token.
`Run()` iterates a flat `[]Instruction` array. Each instruction calls
`node.Forward(ctx, inputs)` which dispatches to `GPUEngine` methods. Each method
calls a CGo wrapper which launches a CUDA kernel.

```
Token loop (Go)
  -> ExecutionPlan.Run() (Go, per-instruction loop)
    -> node.Forward() (Go)
      -> GPUEngine.Add() (Go)
        -> pool.Alloc() (Go)
        -> C.launch_add() (CGo -> CUDA kernel launch)
        -> pool.Free() for temporaries (Go)
```

After Track A (purego), the CGo layer is replaced:
```
Token loop (Go)
  -> ExecutionPlan.Run()
    -> node.Forward()
      -> GPUEngine.Add()
        -> pool.Alloc()
        -> dlcall(launch_add, ...) (pure Go -> dlopen -> CUDA kernel launch)
```

After Track B (megakernel), the entire inner loop collapses:
```
Token loop (Go)
  -> megakernel.Launch(input_token_embedding, output_logits)
     (one CUDA kernel does everything: all 26 layers, all ops, all in registers)
```

### Objectives

- O96: Refactor all layers with inline math to compose Engine primitives.
- O87: Replace all CGo CUDA bindings with dlopen-based pure Go bindings.
- O88: Eliminate `//go:build cuda` tags. Single binary, runtime GPU detection.
- O89: Generate a single-kernel decode for Gemma 3 2B transformer.
- O90: Achieve >= 50 tok/s median for Gemma 3 2B Q4 on DGX Spark GB10.

### Non-Goals

- Multi-GPU inference or tensor parallelism.
- Megakernel for batch > 1 (activations must fit in registers/shared memory).
- Training pipeline changes.
- ROCm/OpenCL kernel implementations (stubs remain, dlopen pattern extends).
- Vulkan, SYCL, or Metal backends.
- Prefill optimization (focus is on per-token decode latency).
- Quantized KV cache (separate phase).

### Constraints and Assumptions

- Go standard library plus golang.org/x/sys/unix for dlopen/dlsym. No other
  third-party dependencies.
- Pre-commit hook rejects multi-directory commits.
- golangci-lint, go vet, gofmt required for all changes.
- Tests must pass with `-race` flag.
- Table-driven tests using the standard `testing` package.
- DGX Spark GB10 at `ssh ndungu@192.168.86.250` for all GPU validation.
- Go 1.25.0, CUDA 13.0, sm_121 (Blackwell) on DGX Spark.
- Target model: Gemma 3 2B Q4 (ZMF), path: ~/models/gemma3-q4/model.zmf.
- GPU baseline from Phase 33: 10.32 tok/s peak, 7.78 tok/s median.
- CUDA cooperative launch requires sm_60+ and CUDA 12+ (DGX Spark qualifies).
- Megakernel is Gemma 3 specific but code generator is designed for extensibility.
- Existing .cu kernel source files and Makefile are unchanged for Track A.

### Success Metrics

| Metric | Current | Track A Target | Track B Target | How Measured |
|--------|---------|---------------|---------------|-------------|
| GPU tok/s median | 7.78 | >= 10 | >= 50 | bench_tps -device cuda, 7 runs, median |
| GPU tok/s peak | 10.32 | >= 12 | >= 60 | bench_tps -device cuda, best of 7 |
| Build tags required | cuda | none | none | go build ./... compiles on any machine |
| CGo calls per token | ~650 | 0 (dlopen) | 0 | runtime.cgocall count |
| Kernel launches per token | ~650 | ~650 | 1 | nsys profile |
| Global mem round-trips | ~650 | ~650 | ~26 (weight reads only) | nsys profile |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Track | Rationale |
|----|-------------|-------|-----------|
| D430 | All layers compose Engine primitives | 0 | Prerequisite for megakernel instruction-tape coverage |
| D421 | dlopen/dlsym CUDA runtime loader | A | Load libcudart.so at runtime |
| D422 | dlopen kernel function wrappers | A | Replace all 8 CGo files |
| D423 | Remove build tags, runtime GPU detection | A | Single binary for CPU+GPU |
| D424 | Benchmark Track A improvements | A | Validate purego gains |
| D425 | ExecutionPlan-to-CUDA code generator | B | Emit model-specific .cu |
| D426 | Megakernel for Gemma 3 2B decode | B | One kernel, all 26 layers |
| D427 | JIT/AOT compilation and loading | B | nvrtc or cached nvcc |
| D428 | Megakernel integration in generate loop | B | Wire into token generation |
| D429 | End-to-end benchmark >= 50 tok/s | B | Validate all improvements |

### Out of Scope

- Megakernel templates for RNN, S4, HRM, convolution architectures (future).
- CUDA graph capture (subsumed by megakernel; if megakernel underperforms,
  revisit as fallback -- see archived E80-E82 below).
- Individual fused kernels (SwiGLU, Scale+Softmax, dequant+GEMV) as separate
  ops (subsumed by megakernel; the megakernel fuses everything).
- BF16 elementwise kernels.
- Flash attention changes.
- Multi-sequence batch decode.

### Archived Epics (from previous Phase 34 plan)

The following epics from the original Phase 34 plan are archived. They are
superseded by the megakernel approach but can be revived as fallback if the
megakernel proves infeasible for certain ops.

- E80: Pre-Allocated Buffer Pool -- subsumed by megakernel (no intermediate
  buffers at all). Could be useful as standalone optimization if megakernel
  is deferred.
- E81: CUDA Graph CGo Wrappers -- subsumed by megakernel (single launch).
- E82: CUDA Graph Capture -- subsumed by megakernel (single launch).
- E83: Fused SwiGLU Kernel -- subsumed by megakernel (fused inline).
- E84: Fused Scale+Softmax -- subsumed by megakernel (fused inline).
- E85: Fused Dequant+GEMV Q4 -- subsumed by megakernel (fused inline).
- E86: End-to-End Benchmark -- replaced by E93 and E95.

---

## 3. Checkable Work Breakdown

### Track 0: Composition Fixes (Prerequisite for Megakernel)

A 5-agent parallel audit of all 18 layer sub-packages found 12 composition
violations: layers that do complex math inline (direct tensor.Data() access,
manual loops, external library calls) instead of composing Engine primitives.
These ops are invisible to the ExecutionPlan instruction tape and would not
appear in the megakernel.

Decision rationale: docs/adr/027-composition-prerequisite.md.

#### E96: Refactor Violated Layers to Compose Engine Primitives (O96)

##### Priority 1: Gemma 3 Inference Path (blocks megakernel for Gemma 3)

- [x] T96.1 Refactor MatMulNBits to compose engine dequant + MatMul  Owner: TBD  Est: 3h  Done: 2026 03 07
  - File: `layers/core/matmul_nbits.go` lines 118-172.
  - Violation: Inline Q4 dequantization with manual numeric.Unpack4BitSlice
    loops and manual scaling.
  - Fix: Add `engine.DequantQ4(ctx, q4Tensor) (*TensorNumeric[T], error)` to
    Engine interface if not present, or use existing engine.UnaryOp with a
    dequant function. Then call engine.MatMul on the dequantized result.
  - If engine.DequantQ4 does not exist: create it as a new Engine method that
    delegates to the existing dequant logic in CPUEngine, and to the existing
    GPU dequant kernel in GPUEngine.
  - Acceptance: MatMulNBits.Forward() contains zero tensor.Data() calls. Output
    matches current implementation within 1e-5. All existing MatMulNBits tests pass.
  - Dependencies: none.

- [x] S96.1.1 MatMulNBits composition parity test  Owner: TBD  Est: 1h  Done: 2026 03 07
  - Existing tests pass. Forward() no longer calls tensor.Data() (dequant at construction).

- [x] T96.2 Refactor QKNorm to compose engine primitives  Owner: TBD  Est: 2h  Done: 2026 03 07
  - File: `layers/attention/qk_norm.go` lines 30-62.
  - Violation: Manual tensor.Data() loops computing RMS normalization.
  - Fix: Replace with engine.Mul -> engine.ReduceMean -> engine.AddScalar ->
    engine.Rsqrt -> engine.Mul (same pattern as RMSNorm layer).
  - Acceptance: QKNorm.Forward() contains zero tensor.Data() calls. Output
    matches within 1e-5. Existing tests pass.
  - Dependencies: none.

- [x] S96.2.1 QKNorm composition parity test  Owner: TBD  Est: 45m  Done: 2026 03 07
  - Compare old vs new for Q and K shapes used in Gemma 3. Existing tests pass.

- [x] T96.3 Refactor Gelu to use engine.Tanh instead of math.Tanh  Owner: TBD  Est: 1h  Done: 2026 03 07
  - File: `layers/activations/gelu.go` lines 21-40.
  - Violation: Uses math.Tanh inside a closure passed to engine.UnaryOp.
    The UnaryOp applies element-wise, but the closure internals are opaque
    to the instruction tape.
  - Fix: Decompose into explicit engine calls: engine.Pow (x^3),
    engine.MulScalar (0.044715), engine.Add, engine.MulScalar (sqrt(2/pi)),
    engine.Tanh, engine.AddScalar (1), engine.Mul, engine.MulScalar (0.5).
    This matches the FastGelu pattern already used in the codebase.
  - Acceptance: Gelu.Forward() uses only engine method calls. Output matches
    within 1e-5. Existing tests pass.
  - Dependencies: none.

- [x] S96.3.1 Gelu composition parity test  Owner: TBD  Est: 45m  Done: 2026 03 07
  - Compare old vs new for input shapes [1,2048] and [1,8192]. Existing tests pass.

##### Priority 2: Other Model Variants

- [x] T96.4 Refactor BatchNormalization to compose engine primitives  Owner: TBD  Est: 2h  Done: 2026 03 07
  - File: `layers/normalization/batch_norm.go` lines 35-93.
  - Violation: Manual per-channel loops with tensor.Data() access.
  - Fix: Use engine.Sub (subtract mean), engine.AddScalar (epsilon),
    engine.Sqrt, engine.Div (normalize), engine.Mul (scale), engine.Add (bias).
  - Acceptance: Zero tensor.Data() in Forward(). Output within 1e-5.
  - Dependencies: none.

- [x] S96.4.1 BatchNorm composition parity test  Owner: TBD  Est: 45m  Done: 2026 03 07

- [x] T96.5 Refactor LocalAttention mask to compose engine primitives  Owner: TBD  Est: 1.5h  Done: 2026 03 07
  - File: `layers/attention/local_attention.go` lines 97-125.
  - Violation: createLocalAttentionMask() manually fills mask via tensor.Data().
  - Fix: Use engine.Fill to create base mask, engine.AddScalar or engine ops
    for mask pattern. Or create mask as a Constant node in the graph.
  - Acceptance: Zero tensor.Data() in mask creation. Output matches.
  - Dependencies: none.

- [x] S96.5.1 LocalAttention mask parity test  Owner: TBD  Est: 45m  Done: 2026 03 07

- [ ] T96.6 Refactor Conv2d to im2col + engine.MatMul  Owner: TBD  Est: 3h
  - File: `layers/core/conv2d.go` lines 60-144.
  - Violation: 6-nested loop convolution with direct data access.
  - Fix: Implement im2col transform (unfold input patches into columns),
    then engine.MatMul(weight_matrix, col_matrix), then reshape output.
    Add engine.Im2Col if needed, or implement as engine.Reshape + engine.Gather.
  - Acceptance: Zero nested compute loops in Forward(). Output within 1e-5.
  - Dependencies: none.

- [ ] S96.6.1 Conv2d composition parity test  Owner: TBD  Est: 1h

##### Priority 3: Specialized Layers (not on Gemma 3 path)

- [ ] T96.7 Refactor MoEGate to compose engine primitives  Owner: TBD  Est: 2h
  - File: `layers/core/moe.go` lines 43-100.
  - Violation: Manual tensor.Data() loops for expert routing, sorting, normalization.
  - Fix: Use engine.Softmax for routing probs, add engine.TopK if not present
    (or compose from engine.Sort + engine.Slice), engine.Div for normalization.
  - Acceptance: Zero tensor.Data() in Forward(). Output matches.
  - Dependencies: none.

- [ ] T96.8 Refactor MixtureOfExperts to compose engine primitives  Owner: TBD  Est: 2h
  - File: `layers/core/moe.go` lines 217-282.
  - Violation: Manual token extraction and weighted sum with tensor.Data().
  - Fix: Use engine.Gather for token extraction, compose expert Forward calls,
    engine.MulScalar for weighting, engine.Add for accumulation.
  - Acceptance: Zero tensor.Data() in Forward(). Output matches.
  - Dependencies: T96.7.

- [ ] S96.8.1 MoE composition parity test  Owner: TBD  Est: 1h
  - Test both MoEGate and MixtureOfExperts together.

- [ ] T96.9 Refactor PolynomialExpansion to compose engine primitives  Owner: TBD  Est: 1.5h
  - File: `layers/core/polynomial.go` lines 191-249.
  - Violation: Manual tensor.Data() loops for power computation.
  - Fix: Use engine.Pow for each term, engine.MulScalar for coefficients,
    engine.Add for accumulation.
  - Acceptance: Zero tensor.Data() in Forward(). Output within 1e-5.
  - Dependencies: none.

- [ ] S96.9.1 Polynomial composition parity test  Owner: TBD  Est: 45m

- [ ] T96.10 Refactor SpectralFingerprint to compose engine primitives  Owner: TBD  Est: 2h
  - File: `layers/core/spectral_fingerprint.go` lines 96-157.
  - Violation: Inline DFT with manual cos/sin loops.
  - Fix: Use engine.UnaryOp with cos/sin, or add engine.DFT primitive.
    Alternatively, decompose DFT into engine.MatMul with a precomputed
    Fourier basis matrix.
  - Acceptance: Zero tensor.Data() compute loops. Output within 1e-4.
  - Dependencies: none.

- [ ] T96.11 Refactor S4 layer to compose engine primitives  Owner: TBD  Est: 3h
  - File: `layers/sequence/s4.go` lines 184-222 (forward), 264-346 (backward).
  - Violation: 4-nested loop diagonal SSM scan with tensor.Data() access.
  - Fix: Decompose scan into per-step engine calls: engine.Exp (discrete A),
    engine.Mul (a * x_prev), engine.Add (+ b * u), engine.Mul (c * x) for
    output. This makes each scan step visible in the instruction tape.
  - Risk: Per-step engine calls add overhead vs fused scan. For inference
    correctness this is acceptable; optimization can add an engine.SSMScan
    primitive later.
  - Acceptance: Zero tensor.Data() in Forward(). Output within 1e-5.
  - Dependencies: none.

- [ ] S96.11.1 S4 composition parity test  Owner: TBD  Est: 1h
  - Compare old vs new for sequence lengths 32, 128, 512.

- [ ] T96.12 Refactor SpectralFeature to remove Gonum FFT  Owner: TBD  Est: 2h
  - File: `layers/features/spectral.go` lines 57-77.
  - Violation: External Gonum FFT library call + manual tensor.Data().
  - Fix: Same approach as SpectralFingerprint (T96.10). Use engine.MatMul
    with precomputed Fourier basis, or add engine.FFT primitive.
  - Acceptance: Zero external library calls. Zero tensor.Data() access.
  - Dependencies: none.

- [ ] S96.12.1 SpectralFeature composition parity test  Owner: TBD  Est: 45m

##### Quality Gate

- [ ] T96.13 Run golangci-lint on all modified layer packages  Owner: TBD  Est: 30m
  - Dependencies: T96.1-T96.12.

- [ ] T96.14 Run full test suite and verify no regressions  Owner: TBD  Est: 1h
  - `go test ./... -race -count=1`
  - Verify zero tensor.Data() calls in any layer Forward() method (grep).
  - Dependencies: T96.13.

- [ ] S96.14.1 Composition audit verification  Owner: TBD  Est: 30m
  - Grep all layers/*/**.go Forward() methods for tensor.Data(), .Float32s(),
    .Data(). Verify zero hits in compute paths (data access for shape/metadata
    is acceptable, data access for computation is not).

---

### Track A: purego / dlopen (Quick Win, All Architectures)

#### E87: CUDA Runtime dlopen Loader (O87, O88)

Replace `internal/cuda/runtime.go` (CGo) with a pure Go implementation that
loads libcudart.so via dlopen and calls CUDA runtime functions via dlsym
function pointers. This is the foundation -- all other CUDA calls depend on the
runtime (streams, malloc, memcpy, sync).

Existing code to replace:
- `internal/cuda/runtime.go` -- 8 CGo functions (Malloc, Free, Memcpy, etc.)

Approach:
- Create `internal/cuda/dlopen.go` (no build tag) with a CUDALib struct that
  holds dlopen handle and dlsym function pointers for each CUDA runtime func.
- Use `golang.org/x/sys/unix.Dlopen`, `unix.Dlsym`, `unix.Dlclose`.
- On init, attempt to open libcudart.so. If not found, set a package-level
  `Available() bool` to false. All GPU codepaths check this.
- Each function (Malloc, Free, etc.) calls the dlsym pointer using
  `syscall.Syscall` with the correct ABI.

Decision rationale: docs/adr/025-purego-cuda-bindings.md.

- [ ] T87.1 Add golang.org/x/sys dependency  Owner: TBD  Est: 30m
  - `go get golang.org/x/sys` and update go.mod/go.sum.
  - Acceptance: `go build ./...` succeeds with new dependency.
  - Dependencies: none.

- [ ] T87.2 Create dlopen loader for libcudart.so  Owner: TBD  Est: 3h
  - Create `internal/cuda/dlopen.go` (no build tag).
  - Implement `type CUDALib struct` with handle and function pointers:
    cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyAsync, cudaMallocManaged,
    cudaStreamCreate, cudaStreamSynchronize, cudaStreamDestroy,
    cudaDeviceSynchronize, cudaGetErrorString.
  - Implement `Open() (*CUDALib, error)` that does dlopen + dlsym for each.
  - Implement `Close()` that calls dlclose.
  - Implement `Available() bool` package-level function.
  - Acceptance: On DGX Spark, `Open()` succeeds and all function pointers
    are non-nil. On a CPU-only machine, `Open()` returns an error and
    `Available()` returns false.
  - Dependencies: T87.1.

- [ ] S87.2.1 dlopen loader unit test  Owner: TBD  Est: 1h
  - Test: Open() succeeds when libcudart.so exists (DGX Spark).
  - Test: Open() fails gracefully when libcudart.so is absent.
  - Test: All function pointers are non-nil after successful Open().

- [ ] T87.3 Replace runtime.go CGo functions with dlopen calls  Owner: TBD  Est: 3h
  - Rewrite `internal/cuda/runtime.go` to remove `import "C"` and
    `//go:build cuda`.
  - Each function (Malloc, Free, Memcpy, etc.) calls the corresponding
    function pointer from the package-level CUDALib instance.
  - Package init: attempt Open(). If fails, set available=false.
  - All existing callers (mempool.go, gpuapi/) continue to work unchanged.
  - Acceptance: `go build ./...` compiles without `-tags cuda`. All
    runtime functions work correctly on DGX Spark.
  - Dependencies: T87.2.
  - Risk: ABI mismatch between Go calling convention and CUDA C functions.
    Test each function individually with known inputs.

- [ ] S87.3.1 Runtime function parity test  Owner: TBD  Est: 1.5h
  - Test: Malloc + Memcpy(H2D) + Memcpy(D2H) round-trip preserves data.
  - Test: StreamCreate + StreamSync works.
  - Test: MallocManaged returns accessible pointer.
  - Compare behavior with the old CGo implementation.

- [ ] T87.4 Run golangci-lint on internal/cuda/  Owner: TBD  Est: 15m
  - Dependencies: T87.3.

#### E88: Kernel Function dlopen Wrappers (O87, O88)

Replace the 7 remaining CGo kernel wrapper files with dlopen-based equivalents.
Each file currently does `import "C"` with extern declarations for the kernel
launcher functions, then wraps them in Go functions. The new approach loads the
kernel launcher function pointers from libkernels.so via dlsym.

Existing code to replace (7 files, all in internal/cuda/kernels/):
- `elementwise.go` -- 20+ kernel launchers (Add, Sub, Mul, Div, Pow, etc.)
- `transpose.go` -- Transpose kernel launcher
- `rmsnorm.go` -- RMSNorm fused kernel launcher
- `gather.go` -- Gather (embedding lookup) kernel launcher
- `gemm_q4.go` -- Q4 dequant+GEMM kernel launcher
- `gemm_quantized.go` -- Quantized GEMM kernel launcher
- `flash_attention.go` -- Flash attention kernel launcher

Approach:
- Create `internal/cuda/kernels/dlopen.go` with a KernelLib struct that loads
  libkernels.so and resolves all launcher function pointers.
- Replace each CGo wrapper function with a dlsym-based call.
- Remove `//go:build cuda` from all kernel .go files.

- [ ] T88.1 Create KernelLib dlopen loader  Owner: TBD  Est: 2h
  - Create `internal/cuda/kernels/dlopen.go` (no build tag).
  - Load libkernels.so via dlopen.
  - Resolve all launcher function pointers (launch_add, launch_sub, etc.).
  - Acceptance: All ~30 function pointers resolved on DGX Spark.
  - Dependencies: T87.2.

- [ ] T88.2 Replace elementwise.go CGo wrappers  Owner: TBD  Est: 3h
  - Rewrite all 20+ functions (Add, Sub, Mul, Div, Pow, Exp, Log, Sqrt,
    etc.) to call dlsym function pointers instead of C functions.
  - Remove `import "C"` and `//go:build cuda`.
  - Acceptance: All elementwise kernel tests pass on DGX Spark.
  - Dependencies: T88.1.

- [ ] S88.2.1 Elementwise kernel parity test  Owner: TBD  Est: 1.5h
  - Run existing elementwise_test.go tests.
  - Verify output matches CGo version within 1e-7.

- [ ] T88.3 Replace remaining kernel CGo wrappers  Owner: TBD  Est: 3h
  - Rewrite transpose.go, rmsnorm.go, gather.go, gemm_q4.go,
    gemm_quantized.go, flash_attention.go.
  - Remove `import "C"` and `//go:build cuda` from each.
  - Acceptance: All kernel tests pass. `go build ./...` compiles without
    any build tags.
  - Dependencies: T88.1.

- [ ] S88.3.1 Full kernel test suite  Owner: TBD  Est: 2h
  - Run all tests in internal/cuda/kernels/ on DGX Spark.
  - Run full test suite: `go test ./... -count=1` (no -tags cuda needed).
  - Verify flash_attention, gemm_q4, rmsnorm outputs match within tolerance.

- [ ] T88.4 Remove build tag from internal/cuda/mempool.go  Owner: TBD  Est: 30m
  - mempool.go may have `//go:build cuda`. Remove it.
  - Ensure it compiles when CUDA is not available (guard with Available()).
  - Dependencies: T88.3.

- [ ] T88.5 Run golangci-lint on internal/cuda/  Owner: TBD  Est: 15m
  - Dependencies: T88.4.

#### E89: Runtime GPU Detection and Build Tag Removal (O88)

Remove `//go:build cuda` from all files outside internal/cuda/. Replace
compile-time GPU detection with runtime `cuda.Available()` checks. The goal:
`go build ./...` always compiles, GPU is used when available.

Existing code affected:
- `internal/gpuapi/cuda_runtime.go` -- `//go:build cuda`
- `internal/gpuapi/cuda_kernels.go` -- `//go:build cuda`
- `compute/gpu_engine.go` -- `//go:build cuda`
- `compute/gpu_kernels.go` -- `//go:build cuda`
- Any other files with `//go:build cuda`

- [ ] T89.1 Remove build tags from gpuapi CUDA files  Owner: TBD  Est: 2h
  - Remove `//go:build cuda` from cuda_runtime.go and cuda_kernels.go.
  - Guard initialization with `if !cuda.Available() { return }`.
  - Acceptance: Files compile without build tags. GPU path activates only
    when CUDA is present.
  - Dependencies: E88.

- [ ] T89.2 Remove build tags from compute/ GPU files  Owner: TBD  Est: 2h
  - Remove `//go:build cuda` from gpu_engine.go and gpu_kernels.go.
  - Guard GPU engine creation with `cuda.Available()`.
  - Acceptance: `go build ./...` compiles on any machine. GPUEngine is
    created only when CUDA is detected at runtime.
  - Dependencies: T89.1.

- [ ] T89.3 Update all remaining build-tagged files  Owner: TBD  Est: 1.5h
  - Grep for `//go:build cuda` across the entire codebase.
  - Remove each one, adding runtime guards as needed.
  - Acceptance: Zero files with `//go:build cuda`. `go build ./...` works
    on macOS (no CUDA) and DGX Spark (with CUDA).
  - Dependencies: T89.2.

- [ ] S89.3.1 Cross-platform build verification  Owner: TBD  Est: 1h
  - `go build ./...` on macOS (no CUDA) -- must compile and run CPU-only.
  - `go build ./...` on DGX Spark (CUDA present) -- must compile and use GPU.
  - `go test ./... -count=1` on macOS -- all tests pass (GPU tests skip).

- [ ] T89.4 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m
  - Dependencies: T89.3.

#### E90: Track A Benchmark (O87, O88)

Validate purego improvements on DGX Spark.

- [ ] T90.1 Benchmark purego inference  Owner: TBD  Est: 2h
  - Build zerfoo on DGX Spark (no build tags needed).
  - Run `bench_tps -model ~/models/gemma3-q4 -device cuda -tokens 100`.
  - 7 runs, report median and peak.
  - Acceptance: At least parity with Phase 33 (>= 7.78 median). No regression.
  - Dependencies: E89.

- [ ] S90.1.1 Track A benchmark report  Owner: TBD  Est: 30m
  - Document: tok/s (median + peak), build time comparison, binary size.
  - Verify `go build ./...` works on macOS without CUDA.

- [ ] T90.2 Verify output correctness  Owner: TBD  Est: 1h
  - Generate 50 tokens with same prompt on CPU and GPU (purego).
  - Compare outputs. Must produce coherent text, no NaN/Inf.
  - Dependencies: T90.1.

- [ ] T90.3 Run golangci-lint on all packages  Owner: TBD  Est: 15m
  - Dependencies: T90.2.

---

### Track B: Megakernel (Maximum Performance, All Architectures)

#### E91: Instruction-Tape-to-CUDA Code Generator (O89)

Build a code generator that reads a compiled ExecutionPlan's flat instruction
list and emits a single .cu file containing a megakernel. Because Zerfoo is
compositional (all complex layers decompose into primitive Engine ops), the
generator does not need architecture-specific templates. It maps each OpName
to a register-resident CUDA device function and chains them in instruction order.

This works for ANY model that compiles into an ExecutionPlan -- transformers,
RNNs, CNNs, S4, HRM, or any future architecture.

Decision rationale: docs/adr/026-megakernel-decode.md.

Existing code:
- `graph/compile.go` -- ExecutionPlan with []Instruction. Each Instruction has
  OpName (string), InputIdx ([]int), OutputIdx (int). Slot shapes are known
  from the warmup Forward() pass during Compile().
- `internal/cuda/kernels/` -- Individual CUDA kernels (reference implementations
  for each primitive op).

The key insight: ExecutionPlan already IS the instruction tape. The code
generator walks it and emits one CUDA device function call per instruction.
No pattern matching for "transformer layers" or "attention blocks" is needed.

Primitive op -> CUDA device function mapping:
- Add, Sub, Mul, Div -> elementwise in registers
- MatMul (Q4 GEMV) -> dequant + dot product from global memory
- MatMul (F32 GEMV) -> dot product from global memory
- RMSNorm -> shared memory reduction + register normalize
- Softmax -> shared memory max/sum reduction + register exp/div
- Exp, Log, Sqrt, Rsqrt, Tanh -> register unary
- SiLU -> register (x * sigmoid(x))
- Gather -> global memory read to registers
- Concat, Split -> register reindexing
- RoPE -> register rotation
- MulScalar, AddScalar, etc. -> register scalar ops

- [ ] T91.1 Export instruction metadata from ExecutionPlan  Owner: TBD  Est: 2h
  - Add exported methods to ExecutionPlan:
    - `Instructions() []InstructionMeta` -- returns OpName, InputIdx,
      OutputIdx, and slot shapes for each instruction.
    - `SlotShapes() [][]int` -- returns the shape of each slot (from warmup).
    - `FrozenSlots() []FrozenSlot` -- returns slot index + GPU pointer for
      frozen data (model weights).
  - Define `InstructionMeta` and `FrozenSlot` types in graph/compile.go.
  - Acceptance: After Compile(), InstructionMeta list has correct OpNames
    matching the node.OpType() for each compute instruction.
  - Dependencies: none.

- [ ] S91.1.1 Instruction metadata test  Owner: TBD  Est: 1h
  - Compile a simple graph (Add two inputs). Verify Instructions() returns
    one entry with OpName="Add", correct InputIdx, correct OutputIdx.
  - Compile with a MatMul. Verify OpName="MatMulNBits" and shapes match.

- [ ] T91.2 Create op-to-CUDA device function table  Owner: TBD  Est: 3h
  - Create `internal/codegen/optable.go`.
  - Define `type OpEmitter func(op InstructionMeta, slots []SlotInfo) string`
    that returns the CUDA device function call code for one instruction.
  - Implement emitters for each primitive op:
    - Elementwise (Add, Sub, Mul, Div): `slot_N[tid] = slot_A[tid] + slot_B[tid]`
    - Unary (Exp, Log, Sqrt, etc.): `slot_N[tid] = expf(slot_A[tid])`
    - Scalar ops: `slot_N[tid] = slot_A[tid] * scalar`
    - RMSNorm: shared memory reduction call
    - Softmax: shared memory reduction call
    - MatMul/GEMV: device function call with weight pointer
    - Gather: `slot_N[tid] = weight[index][tid]`
  - Register an `Unsupported` emitter that marks ops the megakernel cannot
    handle (triggers fallback to ExecutionPlan.Run()).
  - Acceptance: All ops in Gemma 3 2B's instruction list have emitters.
  - Dependencies: T91.1.

- [ ] S91.2.1 Op emitter unit tests  Owner: TBD  Est: 1.5h
  - Test each emitter produces syntactically correct CUDA code fragments.
  - Test Unsupported emitter returns error for unknown ops.

- [ ] T91.3 CUDA megakernel emitter  Owner: TBD  Est: 3h
  - Create `internal/codegen/emit.go`.
  - Walk the instruction list. For each instruction, call the corresponding
    OpEmitter. Chain them in order.
  - Emit a complete .cu file:
    - Slot declarations (register arrays or shared memory, sized by SlotShapes).
    - Frozen slot parameters (GPU pointers passed as kernel args).
    - `__global__ void megakernel(input_ptr, output_ptr, frozen_ptrs..., pos)`.
    - Body: load input to registers, emit ops in order, write output.
    - Grid sync between ops that need cross-block communication.
  - Use Go `text/template` for the boilerplate.
  - Acceptance: Emits compilable .cu for a simple Add+Mul graph.
  - Dependencies: T91.2.

- [ ] S91.3.1 Emitted CUDA compilation test  Owner: TBD  Est: 1.5h
  - Emit .cu for simple graph. Compile with nvcc on DGX Spark.
  - Verify: compiles for sm_121 without errors.

- [ ] T91.4 Emit megakernel for full model  Owner: TBD  Est: 3h
  - Load Gemma 3 2B Q4, compile ExecutionPlan, emit megakernel .cu.
  - Handle the full instruction list (~650 instructions, 26 layers).
  - Verify compilation with nvcc on DGX Spark.
  - Acceptance: Generated .cu compiles. All ops have emitters (no Unsupported).
  - Dependencies: T91.3.

- [ ] S91.4.1 Full model emit test  Owner: TBD  Est: 2h
  - Verify: instruction count matches, all slot shapes accounted for,
    all frozen weights referenced, compiled binary loads.

- [ ] T91.5 Run golangci-lint on graph/ and internal/codegen/  Owner: TBD  Est: 15m
  - Dependencies: T91.4.

#### E92: Register-Resident Device Functions for Primitive Ops (O89)

Implement CUDA `__device__` functions for each primitive op in the op table.
These are the building blocks the emitter chains together. Each device function
operates on data in registers or shared memory, never writing intermediates to
global memory. They are architecture-agnostic -- the same device functions are
used whether the model is a transformer, RNN, or anything else.

All device functions go in `internal/cuda/kernels/megakernel_ops.cu` (a header-
style .cu file included by the generated megakernel).

- [ ] T92.1 Elementwise device functions  Owner: TBD  Est: 2h
  - Write `__device__` functions for: add, sub, mul, div, add_scalar,
    mul_scalar, sub_scalar, div_scalar, pow_scalar.
  - Each operates on register arrays: `out[tid] = a[tid] + b[tid]`.
  - Acceptance: Each matches the standalone CUDA kernel output within 1e-7.
  - Dependencies: none.

- [ ] S92.1.1 Elementwise device function test  Owner: TBD  Est: 1h
  - Test kernel: load data to registers, call device functions, write back.
  - Compare with reference standalone kernel output.

- [ ] T92.2 Unary device functions  Owner: TBD  Est: 1.5h
  - Write `__device__` functions for: exp, log, sqrt, rsqrt, tanh, silu
    (x * sigmoid(x)), neg.
  - Acceptance: Each matches standalone kernel output within 1e-6.
  - Dependencies: none.

- [ ] T92.3 RMSNorm device function  Owner: TBD  Est: 3h
  - Write `__device__ void dev_rmsnorm(float* out, const float* in,
    const float* weight, int dim)`.
  - Uses shared memory for variance reduction across threads.
  - Acceptance: Matches RMSNorm CUDA kernel output within 1e-5.
  - Dependencies: none.

- [ ] S92.3.1 RMSNorm device function test  Owner: TBD  Est: 1h
  - Test with dim=2048. Compare with standalone rmsnorm kernel.

- [ ] T92.4 Softmax device function  Owner: TBD  Est: 2.5h
  - Write `__device__ void dev_softmax(float* out, const float* in,
    int rows, int cols)`.
  - Uses shared memory for max and sum reductions.
  - Acceptance: Rows sum to 1.0 within 1e-6. Matches standalone softmax.
  - Dependencies: none.

- [ ] T92.5 Q4 GEMV device function  Owner: TBD  Est: 4h
  - Write `__device__ void dev_gemv_q4(float* out, const void* q4_weight,
    const float* activation, int M, int K)`.
  - Reads Q4 weight blocks from global memory, dequantizes in registers,
    dot-products with activation vector, accumulates in float32.
  - This is the hot path -- reads 1.5GB of weights per token.
  - Optimize for memory bandwidth: coalesced reads, warp-level shuffles.
  - Acceptance: Output matches GemmQ4F32 within 1e-3.
  - Dependencies: none.
  - Risk: Register pressure for large K. Tile and use shared memory for
    partial activation vectors if needed.

- [ ] S92.5.1 Q4 GEMV device function test  Owner: TBD  Est: 1.5h
  - Test with M=2048, K=2048 (Gemma 3 hidden dim).
  - Compare with separate dequant+GEMM.

- [ ] T92.6 F32 GEMV device function  Owner: TBD  Est: 2h
  - Write `__device__ void dev_gemv_f32(float* out, const float* weight,
    const float* activation, int M, int K)`.
  - For non-quantized weight matrices (embeddings, small projections).
  - Acceptance: Output matches F32 GEMM within 1e-5.
  - Dependencies: none.

- [ ] T92.7 Gather device function  Owner: TBD  Est: 1.5h
  - Write `__device__ void dev_gather(float* out, const float* table,
    int index, int dim)`.
  - Reads one row from an embedding table in global memory to registers.
  - Acceptance: Output matches standalone Gather kernel.
  - Dependencies: none.

- [ ] T92.8 Cooperative grid sync wrapper  Owner: TBD  Est: 1.5h
  - Write `__device__ void grid_sync()` using cooperative groups
    (`cooperative_groups::this_grid().sync()`).
  - Write Go dlopen wrapper for `cudaLaunchCooperativeKernel` (driver API).
  - Acceptance: Grid sync works across all thread blocks on DGX Spark.
  - Dependencies: none.

- [ ] S92.8.1 Grid sync test  Owner: TBD  Est: 1h
  - Launch kernel with cooperative launch. Verify all blocks synchronize.

- [ ] T92.9 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Dependencies: T92.8.

#### E93: Megakernel Integration and Compilation (O89, O90)

Wire the code generator and megakernel into the inference pipeline. At model
load time, walk the ExecutionPlan instruction tape, emit a megakernel .cu,
compile it (JIT or cached), and use it for decode tokens.

- [ ] T93.1 JIT compilation with nvrtc  Owner: TBD  Est: 3h
  - Create `internal/cuda/jit.go` with dlopen wrappers for nvrtc:
    nvrtcCreateProgram, nvrtcCompileProgram, nvrtcGetPTX, nvrtcDestroyProgram.
  - Add cuModuleLoadData, cuModuleGetFunction, cuLaunchCooperativeKernel
    wrappers (driver API via dlopen).
  - Acceptance: Can compile a .cu string and launch the resulting kernel.
  - Dependencies: E87 (dlopen infrastructure).

- [ ] S93.1.1 JIT compilation smoke test  Owner: TBD  Est: 1h
  - Compile a trivial kernel via nvrtc. Launch it. Verify correct output.

- [ ] T93.2 Cached compilation with nvcc  Owner: TBD  Est: 2h
  - Alternative to JIT: emit .cu to temp file, call nvcc to compile to .so,
    cache the .so alongside the model file.
  - If cached .so exists and model hash matches, skip compilation.
  - Acceptance: First load compiles (slow). Second load uses cache (fast).
  - Dependencies: T91.3 (emitter).

- [ ] S93.2.1 Cache hit/miss test  Owner: TBD  Est: 1h
  - First load: verify compilation happens.
  - Second load: verify cache is used (no nvcc invocation).

- [ ] T93.3 Wire megakernel into decode loop  Owner: TBD  Est: 3h
  - In `generate/stream.go`, after model load and plan compilation:
    1. Call plan.Instructions() to get the instruction tape.
    2. Pass instruction tape to the emitter (T91.3) to generate .cu.
    3. Compile via nvrtc or cached nvcc (T93.1 or T93.2).
    4. On each decode token: launch megakernel instead of plan.Run().
  - Pass: input token embedding pointer, output logits pointer, all frozen
    slot GPU pointers (model weights), position index.
  - Acceptance: Generates coherent text using megakernel path.
  - Dependencies: E91, E92, T93.1 or T93.2.

- [ ] S93.3.1 Megakernel decode correctness test  Owner: TBD  Est: 2h
  - Generate 50 tokens with megakernel path.
  - Compare output with ExecutionPlan.Run() path within tolerance.
  - Verify coherent text output.

- [ ] T93.4 Fallback to ExecutionPlan on unsupported ops  Owner: TBD  Est: 1.5h
  - If any instruction has an Unsupported op emitter (op not in the table),
    fall back to ExecutionPlan.Run() with a warning log listing which ops
    are unsupported.
  - Acceptance: Models with unsupported ops still work via the standard path.
    The warning log tells the user exactly which ops need emitters.
  - Dependencies: T93.3.

- [ ] T93.5 Run golangci-lint on generate/ and internal/codegen/  Owner: TBD  Est: 15m
  - Dependencies: T93.4.

#### E94: Megakernel Performance Tuning (O90)

Optimize the megakernel for maximum throughput on DGX Spark GB10.

- [ ] T94.1 Profile megakernel with nsys  Owner: TBD  Est: 2h
  - Run megakernel decode with nsys profiler.
  - Identify: memory bandwidth utilization, occupancy, register usage,
    shared memory usage, stall reasons.
  - Acceptance: Profile report with actionable bottleneck identification.
  - Dependencies: E93.

- [ ] T94.2 Optimize memory access patterns  Owner: TBD  Est: 3h
  - Based on nsys profile, optimize:
    - Coalesced weight reads (Q4 block layout alignment).
    - Shared memory bank conflict avoidance.
    - Warp-level reduction using __shfl_down_sync.
    - Register spill reduction (adjust tiling).
  - Acceptance: Memory bandwidth utilization >= 70% of theoretical.
  - Dependencies: T94.1.

- [ ] T94.3 Tune thread block and grid dimensions  Owner: TBD  Est: 2h
  - Experiment with block sizes (128, 256, 512) and grid dimensions.
  - Use cooperative launch occupancy calculator.
  - Acceptance: Optimal configuration documented and hardcoded in emitter.
  - Dependencies: T94.1.

- [ ] T94.4 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Dependencies: T94.3.

#### E95: End-to-End Benchmark (O90)

Final validation after all optimizations.

- [ ] T95.1 Profile GPU inference after all optimizations  Owner: TBD  Est: 2h
  - Run `bench_tps -model ~/models/gemma3-q4 -device cuda -tokens 100`.
  - 7 runs, report median and peak.
  - Capture nsys profile.
  - Acceptance: Profile shows single kernel launch per token, high bandwidth
    utilization.
  - Dependencies: E94.

- [ ] S95.1.1 GPU profile report  Owner: TBD  Est: 30m
  - Document: tok/s (median + peak), kernel launch count, bandwidth
    utilization %, memory traffic per token.

- [ ] T95.2 Compare all configurations  Owner: TBD  Est: 1.5h
  - Run bench_tps with:
    - `-device cpu` (CPU baseline)
    - `-device cuda` with ExecutionPlan.Run() (Track A purego baseline)
    - `-device cuda` with megakernel (Track B)
  - 7 runs each, report median.
  - Compare with Phase 33 baselines and Ollama numbers.
  - Acceptance: Megakernel tok/s >= 50 median. If not met, identify
    remaining bottleneck and document what would close the gap.
  - Dependencies: T95.1.

- [ ] S95.2.1 Benchmark comparison report  Owner: TBD  Est: 30m

- [ ] T95.3 Verify output correctness across all paths  Owner: TBD  Est: 1h
  - Generate 50 tokens with same prompt on CPU, GPU (purego/plan.Run),
    GPU (megakernel).
  - Compare outputs. All should produce coherent text.
  - Acceptance: No NaN or Inf. Coherent output on all paths.
  - Dependencies: T95.1.

- [ ] S95.3.1 Output correctness report  Owner: TBD  Est: 30m

- [ ] T95.4 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m
  - Dependencies: T95.3.

---

## 4. Parallel Work

Phase 34 has three sequential tracks (0 then A then B), with internal parallelism.

| Track | Epics | Description | Prerequisite |
|-------|-------|-------------|-------------|
| 0: composition | E96 | Fix 12 layers with inline math. All architectures. | none |
| A: purego | E87, E88, E89, E90 | Replace CGo with dlopen. All architectures. | Track 0 Priority 1 (T96.1-T96.3) |
| B: megakernel | E91, E92, E93, E94, E95 | Single-kernel decode from instruction tape. All architectures. | Track 0 + Track A complete |

Track A can start after Track 0 Priority 1 is done (the Gemma 3 critical fixes).
Track 0 Priority 2-3 can run in parallel with Track A.
Track B depends on all of Track 0 and Track A.

### Track 0 Internal Parallelism

| Wave | Tasks | Notes |
|------|-------|-------|
| 0-1 | T96.1, T96.2, T96.3, T96.4, T96.5 | All independent: different files in different packages |
| 0-2 | T96.6, T96.7, T96.9, T96.10, T96.11, T96.12 | All independent: different files |
| 0-3 | T96.8 | MixtureOfExperts depends on T96.7 (MoEGate) |
| 0-4 | T96.13, T96.14 | Quality gate after all fixes |

### Track A Internal Parallelism

| Wave | Tasks | Notes |
|------|-------|-------|
| A1 | T87.1, T87.2 | Add x/sys dep, create dlopen loader |
| A2 | T87.3, T88.1 | Replace runtime CGo, create kernel loader |
| A3 | T88.2, T88.3 | Replace all kernel CGo wrappers (can split across files) |
| A4 | T88.4, T89.1, T89.2, T89.3 | Remove build tags everywhere |
| A5 | E90 | Benchmark and validate |

### Track B Internal Parallelism

| Wave | Tasks | Notes |
|------|-------|-------|
| B1 | T91.1, T92.1, T92.2, T92.3, T92.4, T92.5, T92.6, T92.7, T92.8 | Export metadata + all device functions (independent) |
| B2 | T91.2, T91.3, T91.4, T93.1, T93.2 | Op table + emitter + full model emit + JIT/cache (after B1) |
| B3 | T93.3, T93.4 | Wire into decode loop |
| B4 | E94 | Performance tuning |
| B5 | E95 | Final benchmark |

### Execution Order

Track A waves (A1-A5) execute first. After A5 completes, Track B waves
(B1-B5) execute. Within each wave, tasks are parallelizable.

Key sync points:
- A5 -> B1: Track A must be validated before starting Track B.
- B1 -> B2: Register ops and IR definition needed before builder/emitter.
- B3 -> B4: Integration needed before tuning.
- B4 -> B5: Tuning needed before final benchmark.

---

## 5. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M49: Composition clean | E96 | none | All 12 violated layers compose Engine primitives. Zero tensor.Data() in Forward() compute paths. grep verification passes. |
| M50: dlopen runtime | E87 | M49 (P1) | libcudart.so loaded via dlopen. Malloc/Memcpy work without CGo. |
| M51: All CGo eliminated | E88, E89 | M50 | Zero `import "C"` in codebase. `go build ./...` compiles anywhere. |
| M52: purego validated | E90 | M51 | tok/s >= Phase 33 baseline on DGX Spark. No regression. |
| M53: Code generator | E91 | M52 | Walks instruction tape, emits compilable .cu for any model. |
| M54: Megakernel works | E92, E93 | M53 | Decode produces correct output via single kernel. |
| M55: 50 tok/s | E94, E95 | M54 | bench_tps reports >= 50 tok/s median on DGX Spark GB10. |

Critical path: T96.1 -> T96.2 -> T96.14 -> T87.1 -> T87.2 -> T87.3 -> T88.1
  -> T88.2 -> T89.1 -> T90.1 -> T91.1 -> T91.2 -> T91.3 -> T91.4 -> T93.3
  -> T94.1 -> T95.1

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R99 | Composition refactoring introduces numerical differences or performance regression | Incorrect output or slower CPU inference | Medium | Parity tests with 1e-5 tolerance for each refactored layer. Benchmark before/after on CPU. Accept small overhead from extra engine dispatch. |
| R89 | dlopen ABI mismatch: Go calling convention does not match CUDA C ABI for some functions (varargs, struct returns) | Segfault or incorrect results | Medium | Test each function individually with known inputs. Use syscall.Syscall6 for functions with many args. Verify with address sanitizer. |
| R90 | dlopen not available on all platforms (Windows, some embedded Linux) | Reduced portability | Low | Use golang.org/x/sys/unix which handles platform differences. Windows can use LoadLibrary via x/sys/windows. |
| R91 | Performance regression from dlopen vs CGo (function pointer indirection) | Slower than CGo baseline | Low | dlopen function calls are typically faster than CGo (no goroutine stack switch). Benchmark to confirm. |
| R92 | Megakernel register pressure: hidden_dim=2048 activation vectors do not fit in registers | Must use shared memory, slower | High | Tile the hidden dimension. Each thread handles a slice. Use shared memory for cross-thread reductions. Profile register usage with nvcc --ptxas-options=-v. |
| R93 | Cooperative grid sync not supported or too slow on GB10 | Cannot synchronize between layers | Medium | GB10 supports cooperative launch (sm_121, CUDA 13.0). If grid sync is slow, consider multi-kernel approach with minimal global memory (just between layers). |
| R94 | nvrtc JIT compilation too slow (>10s) for model load | Bad user experience on first load | Medium | Cache compiled PTX/cubin alongside model file. Only recompile when model hash changes. AOT compilation via nvcc as fallback. |
| R95 | KV cache access pattern limits bandwidth utilization | Cannot reach theoretical max | High | KV cache reads are unavoidable for attention. For long contexts, KV cache may dominate. For short contexts (< 512), weight reads dominate and megakernel helps most. Document the crossover point. |
| R96 | Code generator produces incorrect CUDA for edge cases (e.g., unusual tensor shapes, broadcasting) | Silent correctness bugs | Medium | Extensive parity testing: compare megakernel output with ExecutionPlan.Run() for every token. Add assertion checks inside generated kernel (debug mode). The instruction-tape approach reduces risk vs pattern-matching because it maps 1:1 from known primitive ops. |
| R97 | DGX Spark GB10 thermal throttling causes high variance | Unreliable benchmark results | Medium | Run benchmarks after 5-minute cooldown. Use median of 7+ runs. Monitor GPU temperature via nvidia-smi. |
| R98 | 50 tok/s target not achieved | Unknown bottleneck | Medium | If megakernel reaches 30+ tok/s, investigate with nsys. Common limiters: L2 cache thrashing, shared memory bank conflicts, warp divergence. Fall back to 20 tok/s target with fused kernels (revive archived E83-E85). |

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
7. `go build ./...` compiles on any machine (no build tags required after E89).
8. On DGX Spark: GPU path activates and produces correct results.
9. Changes are committed in a small commit touching one directory only.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(cuda): replace CGo with dlopen for runtime`.
- Always run linters and formatters before committing.

### DGX Spark Protocol

- SSH: `ssh ndungu@192.168.86.250`
- Go: `/usr/local/go/bin/go`
- CUDA: `/usr/local/cuda/bin/nvcc`, `/usr/local/cuda/lib64/libcudart.so`
- nvrtc: `/usr/local/cuda/lib64/libnvrtc.so`
- GPU: NVIDIA GB10, sm_121, `make CUDA_ARCH=sm_121`
- Model: `~/models/gemma3-q4/model.zmf`
- Repo: `~/zerfoo/`

### Benchmark Protocol

- 7 runs minimum, report median and peak.
- All GPU benchmarks on DGX Spark GB10.
- Use `bench_tps -device cuda -tokens 100` for tok/s measurement.
- Use `bench_tps -device cpu -tokens 100` for CPU comparison.
- Capture nsys profile for kernel launch and bandwidth analysis.
- 5-minute GPU cooldown between benchmark sessions.

### Quality Gate

- `go test -race ./package/`
- `golangci-lint run ./package/`
- `go vet ./package/`
- `go build ./...` (must compile without any build tags)

---

## 8. Progress Log

### Change Summary -- 2026-03-07 (v4)

Added Track 0 (E96: Composition Fixes) as prerequisite before Tracks A and B.

5-agent parallel audit scanned all 18 layer sub-packages and found 12 composition
violations -- layers that do inline math instead of composing Engine primitives:

Priority 1 (Gemma 3 path): MatMulNBits (T96.1), QKNorm (T96.2), Gelu (T96.3)
Priority 2 (other models): BatchNorm (T96.4), LocalAttention mask (T96.5),
  Conv2d (T96.6)
Priority 3 (specialized): MoEGate (T96.7), MixtureOfExperts (T96.8),
  Polynomial (T96.9), SpectralFingerprint (T96.10), S4 (T96.11),
  SpectralFeature (T96.12)

Layers confirmed COMPOSED (no issues): SwiGLU, FastGelu, Softmax, TokenEmbedding,
  RotaryPositionalEmbedding, TransformerBlock, FFN, Dense, FiLM, Linear, Add,
  Sub, Mul, Div, MatMul, Reshape, Concat, Split, GroupQueryAttention, SDPA,
  AttentionHead, GlobalAttention, MultiHeadLatentAttention, LayerNorm, RMSNorm,
  SimplifiedLayerNorm, SkipSimplifiedLayerNorm, SimpleRNN, HModule, LModule,
  Transpose, ReduceSum, MatrixMultiplier, GradientComputer.

ADRs created:
- docs/adr/027-composition-prerequisite.md: Enforce layer composition before
  megakernel code generation.

New milestone M49 (Composition clean) added before M50.
New risk R99 (composition refactoring regression) added.

### Change Summary -- 2026-03-06 (v3)

Revised Track B megakernel approach to be architecture-agnostic, leveraging
Zerfoo's compositional design. Key change: instead of architecture-specific
templates (transformer, RNN, etc.), the code generator walks the ExecutionPlan
instruction tape directly and maps each primitive OpName to a CUDA device
function. Any model that compiles into an ExecutionPlan gets a megakernel.

Changes:
- E91: Replaced architecture-specific IR (KernelIR, LayerIR) with instruction-
  tape export (T91.1) + op-to-CUDA table (T91.2) + generic emitter (T91.3) +
  full-model emit (T91.4).
- E92: Renamed from "Megakernel CUDA Implementation" to "Register-Resident
  Device Functions for Primitive Ops". Restructured as per-primitive-op device
  functions instead of per-architecture-component functions. Added elementwise
  (T92.1), unary (T92.2), F32 GEMV (T92.6), Gather (T92.7).
- E93: Updated to reference instruction tape instead of IR builder.
- ADR 026 revised to document composition-based approach.
- Removed "Transformer-Specific" from Track B heading.
- Updated non-goals: removed "megakernel templates for non-transformer
  architectures" (no longer needed).

### Change Summary -- 2026-03-06 (v2)

Rewrote Phase 34 plan with two-track approach:
- Track A: purego/dlopen (E87-E90) -- replaces CGo, benefits all architectures.
- Track B: megakernel (E91-E95) -- single-kernel decode.

Archived original Phase 34 epics E80-E86 (CUDA graph capture + individual fused
kernels). These are subsumed by the megakernel approach but documented as fallback.

ADRs created:
- docs/adr/025-purego-cuda-bindings.md: Replace CGo with dlopen via x/sys/unix.
- docs/adr/026-megakernel-decode.md: Single-kernel decode via code generation.

### Change Summary -- 2026-03-06 (v1)

Created Phase 34 plan. Trimmed Phase 33 completed epics (E75-E79) to
docs/design.md section 15.15. Created ADR 024 for CUDA graph and fused kernel
strategy.

---

## 9. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, and troubleshooting. Design decisions in docs/adr/ (001-026).
- **Phases 1-33:** All documented in docs/design.md sections 15.1-15.15.
- **Phase 34:** This plan is the source of truth. Two tracks: purego then megakernel.
- **Quality:** See docs/QUALITY.md for test coverage report.
- **How to build:**
  - Currently: `go build -tags cuda ./...` (CGo, needs CUDA headers)
  - After Track A: `go build ./...` (no tags, runtime GPU detection)
  - On DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key Starting Points

1. **Track A (purego):** Start with `internal/cuda/runtime.go`. This file has
   8 CGo functions (Malloc, Free, Memcpy, etc.). Replace with dlopen/dlsym
   from `golang.org/x/sys/unix`. Then do the same for the 7 kernel wrapper
   files in `internal/cuda/kernels/`.

2. **Track B (megakernel):** Start with `graph/compile.go`. Add exported
   methods (Instructions(), SlotShapes(), FrozenSlots()) to expose the
   instruction tape. Then build `internal/codegen/optable.go` (maps each
   OpName to a CUDA device function emitter) and `internal/codegen/emit.go`
   (walks the instruction tape and emits a complete .cu megakernel).

### Architecture-Agnostic Design

Both tracks are architecture-agnostic:

- **purego (Track A):** Replaces the CGo calling convention for ALL CUDA
  operations. Benefits transformers, RNNs, CNNs, S4, HRM, or any future
  architecture that uses GPU.

- **megakernel (Track B):** Walks the ExecutionPlan instruction tape, which
  is a flat list of primitive ops (Add, MatMul, RMSNorm, Softmax, etc.).
  Because Zerfoo is compositional (all complex layers decompose into these
  primitives), ANY model that compiles into an ExecutionPlan gets a megakernel
  automatically. No architecture-specific templates are needed.

  The code generator maps OpName -> CUDA device function. To support a new
  primitive op, add one entry to the op table in `internal/codegen/optable.go`.
  If any op in a model's instruction tape lacks an emitter, the fallback path
  (ExecutionPlan.Run()) is used for the entire model, with a warning log
  listing which ops need emitters.

### Performance Baselines

| Model | Quant | Device | tok/s | Source |
|-------|-------|--------|-------|--------|
| Gemma 3 2B | Q4_0 | CPU ARM64 | 6.86 | Phase 30 |
| Gemma 3 2B | Q4_0 | GPU (cuda) | 10.32 peak / 7.78 median | Phase 33 |
| Gemma 3 1B | Q4 | Ollama GB10 | 205 | Ollama benchmark |
| Gemma 3 4B | Q4 | Ollama GB10 | 77.7 | Ollama benchmark |
| Gemma 3 2B | Q4 | Ollama GB10 | ~100 (est.) | Interpolated |
| Theoretical max | Q4_0 1.5GB | GB10 273GB/s | ~182 | Bandwidth / model size |

### External References

| Source | Key Insight |
|--------|------------|
| golang.org/x/sys/unix | Dlopen, Dlsym, Dlclose for loading shared libraries |
| Stanford "No Bubbles" | Single-kernel transformer execution, cooperative launch |
| NVIDIA nvrtc docs | Runtime CUDA compilation API |
| NVIDIA cooperative groups | Grid-level synchronization for megakernels |
| llama.cpp CUDA backend | Fused dequant+GEMV, SwiGLU, memory-bandwidth optimization |
| DGX Spark specs | 273 GB/s LPDDR5x, ~182 tok/s theoretical for 1.5GB model |

---

## 10. Appendix

### Existing File Reference

| File | Purpose | Track A Change | Track B Change |
|------|---------|---------------|---------------|
| `internal/cuda/runtime.go` | CUDA runtime CGo bindings | Replace CGo with dlopen | none |
| `internal/cuda/kernels/elementwise.go` | 20+ kernel CGo wrappers | Replace CGo with dlopen | none |
| `internal/cuda/kernels/transpose.go` | Transpose kernel CGo | Replace CGo with dlopen | none |
| `internal/cuda/kernels/rmsnorm.go` | RMSNorm kernel CGo | Replace CGo with dlopen | none |
| `internal/cuda/kernels/gather.go` | Gather kernel CGo | Replace CGo with dlopen | none |
| `internal/cuda/kernels/gemm_q4.go` | Q4 GEMM kernel CGo | Replace CGo with dlopen | none |
| `internal/cuda/kernels/gemm_quantized.go` | Quantized GEMM kernel CGo | Replace CGo with dlopen | none |
| `internal/cuda/kernels/flash_attention.go` | Flash attention kernel CGo | Replace CGo with dlopen | none |
| `internal/cuda/mempool.go` | GPU memory pool | Remove build tag | none |
| `internal/gpuapi/cuda_runtime.go` | CUDA gpuapi Runtime impl | Remove build tag | none |
| `internal/gpuapi/cuda_kernels.go` | CUDA gpuapi KernelRunner impl | Remove build tag | none |
| `compute/gpu_engine.go` | GPUEngine | Remove build tag | none |
| `compute/gpu_kernels.go` | GPU kernel dispatch | Remove build tag | none |
| `graph/compile.go` | ExecutionPlan | none | IR extraction source |
| `generate/stream.go` | Token generation loop | none | Megakernel integration |
| `internal/codegen/` | (new) Code generator | n/a | New package |

### Estimated Effort Summary

| Epic | Track | Area | Tasks | Estimated Hours |
|------|-------|------|-------|----------------|
| E96: Composition Fixes | 0 | layers/*, compute/ | 14 tasks + 9 subtests | ~28h |
| E87: CUDA Runtime dlopen | A | internal/cuda/ | 4 tasks + 2 subtests | 8.25h |
| E88: Kernel dlopen Wrappers | A | internal/cuda/kernels/ | 5 tasks + 2 subtests | 11.25h |
| E89: Build Tag Removal | A | gpuapi/, compute/ | 4 tasks + 1 subtest | 6.75h |
| E90: Track A Benchmark | A | DGX Spark | 3 tasks + 1 subtest | 3.75h |
| **Track A Total** | **A** | **purego** | **16 tasks + 6 subtests** | **~30h** |
| E91: Instruction-Tape Code Generator | B | graph/, internal/codegen/ | 5 tasks + 4 subtests | 15.75h |
| E92: Register Device Functions | B | internal/cuda/kernels/ | 9 tasks + 4 subtests | 19.25h |
| E93: Integration + Compilation | B | generate/, internal/ | 5 tasks + 3 subtests | 13.75h |
| E94: Performance Tuning | B | DGX Spark | 4 tasks | 7.25h |
| E95: End-to-End Benchmark | B | DGX Spark | 4 tasks + 3 subtests | 5.25h |
| **Track B Total** | **B** | **megakernel** | **24 tasks + 13 subtests** | **~62h** |
| **Track 0 Total** | **0** | **composition** | **14 tasks + 9 subtests** | **~28h** |
| **Grand Total** | **0+A+B** | **all tracks** | **54 tasks + 28 subtests** | **~120h** |
