# Zerfoo Development Plan -- Phase 33: GPU Scalar Ops and D2H Elimination

## 1. Context

### Problem Statement

Phase 32 brought GPU inference from 5.12 to 6.84 tok/s (+33.6%) on DGX Spark
GB10 for Gemma 3 2B Q4. GPU is now faster than CPU (6.61 tok/s), but the 10
tok/s target was not met. Profiling shows three remaining bottlenecks that
cause CPU fallbacks and D2H round-trips:

1. **Pow CPU fallback (8.9%)**: `engine.Pow(base, exponent)` where exponent is
   a scalar tensor `[1]` with value 2.0 (x^2 in normalization). `gpuPow`
   requires `sameShape(base, exponent)` which fails for scalar broadcast,
   falling back to CPU. The CPU then reads GPU-resident base via
   `GPUStorage.Slice()`, triggering a full D2H copy.

2. **Binary op CPU fallback (10.4%)**: `gpuBroadcastOp` supports row, column,
   and same-shape 2D patterns. Scalar-vs-tensor (`[1]` op `[M,D]`) is not
   handled, falling back to CPU with D2H copies.

3. **GPUStorage.Slice D2H (24%)**: all CPU fallback ops that read GPU-resident
   tensor data trigger `GPUStorage.Slice()` which copies the entire buffer
   D2H. This is the root-cause multiplier for (1) and (2).

See docs/design.md for full architecture context and Phases 1-32 history.
Decision rationale: docs/adr/023-gpu-scalar-ops-d2h-elimination.md.

### What Was Delivered (Phase 32)

| Area | Key Result |
|------|------------|
| GPU tensor residency | Intermediate tensors stay on GPU; Q4 weights pre-uploaded |
| GPU Transpose | 2D tiled + N-D stride-based kernel |
| GPU Broadcasting | Stride-based 2D for Add/Sub/Mul/Div |
| GPU Gather | Embedding lookup on GPU |
| Fused GPU RMSNorm | Single-pass kernel with shared-memory reduction |
| Benchmark | 6.84 tok/s GPU, 6.61 tok/s CPU on DGX Spark GB10 |

### GPU Profile Breakdown (Phase 32, DGX Spark GB10)

| Component | % Time | Root Cause |
|-----------|--------|------------|
| runtime.cgocall | 58% | Activation H2D/D2H from CPU fallback ops |
| Pow CPU fallback | 8.9% | No GPU scalar-broadcast Pow kernel |
| binaryOp CPU fallback | 10.4% | Scalar-vs-tensor not in gpuBroadcastOp |
| GPUStorage.Slice D2H | 24% | CPU fallback ops reading GPU tensor data |
| MatMul (GPU) | ~30% | Core compute, already on GPU |

### Objectives

- O75: Add PowScalar GPU kernel to eliminate the 8.9% Pow CPU fallback.
- O76: Extend GPU binary ops with scalar-broadcast to eliminate the 10.4%
  binary op CPU fallback.
- O77: Add GPU Slice/Split/Concat kernels to eliminate the 24% D2H overhead
  from GPUStorage.Slice.
- O78: Achieve >= 10 tok/s for Gemma 3 2B Q4 on DGX Spark GB10.

### Non-Goals

- Multi-GPU inference or tensor parallelism.
- FP4 kernels (blocked on upstream CUTLASS SM121 fixes).
- Vulkan, SYCL, or ROCm kernel ports (stubs only).
- Training pipeline changes.
- Flash attention kernel improvements.
- Speculative decoding (separate phase).

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
- CPU baseline: 6.61 tok/s. GPU baseline: 6.84 tok/s.
- GPUEngine already has: Add/Sub/Mul/Div (same-shape + 2D broadcast),
  AddScalar/MulScalar/DivScalar, Pow (same-shape only), Exp, Log, Sqrt, Rsqrt,
  Tanh, Softmax, ReduceSum, ReduceMean, Fill, Sum, Transpose, Gather,
  FusedRMSNorm.
- GPUEngine CPU fallbacks: UnaryOp, Pow (broadcast), Split, Concat, Repeat,
  Copy, Zero, ScatterAdd, RandomUniform, OneHot.

### Success Metrics

| Metric | Current | Target | How Measured |
|--------|---------|--------|-------------|
| GPU tok/s (Gemma 3 2B Q4) | 6.84 | >= 10 | bench_tps -device cuda on DGX Spark |
| cgocall % of wall time | 58% | < 20% | pprof CPU profile |
| CPU fallback ops in hot loop | Pow, some binary, Slice | 0 | pprof + log grep |
| Pow CPU fallback % | 8.9% | 0% | pprof |
| GPUStorage.Slice D2H % | 24% | < 5% | pprof |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D407 | PowScalar GPU kernel | Eliminate 8.9% Pow CPU fallback |
| D408 | SubScalar GPU kernel | Complete scalar-op coverage for broadcast path |
| D409 | Scalar-broadcast in gpuBroadcastOp | Eliminate 10.4% binary op CPU fallback |
| D410 | GPU Slice kernel | Eliminate 24% D2H from GPUStorage.Slice |
| D411 | GPU Split kernel | Eliminate D2H in Split (calls Slice internally) |
| D412 | GPU Concat kernel | Eliminate D2H in Concat |
| D413 | End-to-end GPU benchmark >= 10 tok/s | Validate all improvements |

### Out of Scope

- GPU Copy (not in hot path; CPU fallback acceptable).
- GPU ScatterAdd (backward pass only).
- GPU RandomUniform (not in inference hot path).
- GPU OneHot (not in inference hot path).
- GPU Repeat (not observed in Gemma 3 profile).
- UnaryOp GPU kernel (closures cannot run on GPU; specific patterns replaced
  by dedicated kernels like PowScalar).

---

## 3. Checkable Work Breakdown

### E75: PowScalar GPU Kernel (O75)

Pow in Gemma 3 inference is always `x^2` (scalar exponent). The current
`gpuPow` requires same-shape tensors. Add a PowScalar kernel and detect
the scalar-exponent pattern.

Existing code:
- `compute/gpu_kernels.go` line 363 -- `gpuPow` with sameShape guard.
- `internal/cuda/kernels/elementwise.cu` -- existing CUDA kernels.
- `internal/gpuapi/kernels.go` -- KernelRunner interface.
- `layers/core/pow.go` -- Pow layer calling `engine.Pow(ctx, base, exponent)`.

- [ ] T75.1 Write CUDA PowScalar kernel  Owner: TBD  Est: 1.5h
  - Add to `internal/cuda/kernels/elementwise.cu`:
    `pow_scalar_kernel(float *x, float p, float *out, int n)` computing
    `out[i] = powf(x[i], p)`.
  - Acceptance: Kernel compiles for sm_121. Correct output for x^2 (n=2048)
    and x^0.5 (same as sqrt).
  - Dependencies: none.

- [ ] S75.1.1 PowScalar kernel unit tests  Owner: TBD  Est: 1h
  - Test: x^2 for 2048 elements matches CPU Pow within 1e-5.
  - Test: x^0.5 matches Sqrt within 1e-5.
  - Test: x^1 is identity within 1e-5.
  - Test: x^0 is all 1s.

- [ ] T75.2 Add PowScalar to KernelRunner interface and Go wrapper  Owner: TBD  Est: 1h
  - Add `PowScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error`
    to `internal/gpuapi/kernels.go` KernelRunner.
  - Add Go CGo wrapper in `internal/cuda/kernels/elementwise.go`.
  - Add stub implementations in `internal/gpuapi/cuda_kernels.go`,
    `internal/gpuapi/rocm_kernels.go`, `internal/gpuapi/opencl_kernels.go`.
  - Acceptance: Compiles on all backends. CUDA backend calls the kernel.
  - Dependencies: T75.1.

- [ ] T75.3 Wire PowScalar into GPUEngine.Pow  Owner: TBD  Est: 1.5h
  - Modify `gpuPow` in `compute/gpu_kernels.go`:
    - If `!sameShape(base, exponent)` and exponent has 1 element total,
      extract the scalar value and use `gpuScalarOp` with `kernels.PowScalar`.
    - If `!sameShape(base, exponent)` and base has 1 element total,
      create a Fill tensor and use same-shape Pow.
    - Otherwise keep CPU fallback.
  - Acceptance: `engine.Pow(x_gpu, scalar_2_tensor)` runs on GPU.
    Output has GPUStorage. Parity with CPU within 1e-5.
  - Dependencies: T75.2.

- [ ] S75.3.1 PowScalar parity test  Owner: TBD  Est: 1h
  - Compare GPUEngine.Pow(x, scalar_tensor) vs CPUEngine.Pow for:
    - Shape [1, 2048] ^ [1] (RMSNorm pattern).
    - Shape [8, 128, 64] ^ [1].
  - Verify output has GPUStorage.

- [ ] T75.4 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Packages: internal/cuda/kernels/, internal/gpuapi/, compute/.
  - Dependencies: T75.3.

### E76: Scalar-Broadcast for All Binary Ops (O76)

Extend `gpuBroadcastOp` to detect scalar-vs-tensor patterns ([1] op [M,D])
and dispatch to existing scalar kernel variants. Also add SubScalar which
is currently missing.

Existing code:
- `compute/gpu_kernels.go` line 77 -- `gpuBroadcastOp` switch statement.
- `compute/gpu_kernels.go` line 258 -- `gpuScalarOp` helper.
- KernelRunner has AddScalar, MulScalar, DivScalar but no SubScalar.

- [ ] T76.1 Write CUDA SubScalar kernel  Owner: TBD  Est: 1h
  - Add to `internal/cuda/kernels/elementwise.cu`:
    `sub_scalar_kernel(float *a, float scalar, float *c, int n)` computing
    `c[i] = a[i] - scalar`.
  - Add `SubScalar` to KernelRunner interface and all backend stubs.
  - Add Go CGo wrapper.
  - Acceptance: Kernel compiles. Correct output for a - 0.5 (n=1024).
  - Dependencies: none.

- [ ] S76.1.1 SubScalar kernel unit tests  Owner: TBD  Est: 45m
  - Test: a - 0.0 is identity.
  - Test: a - scalar matches CPU Sub within 1e-5.
  - Test: edge case a - large_value.

- [ ] T76.2 Add scalar-broadcast detection to gpuBroadcastOp  Owner: TBD  Est: 2h
  - In `gpuBroadcastOp`, before the existing switch statement:
    - If `b` has exactly 1 element total (product of shape == 1),
      extract `b.Data()[0]` as scalar and dispatch to the matching
      scalar kernel (AddScalar, SubScalar, MulScalar, DivScalar).
    - If `a` has exactly 1 element total, extract scalar and dispatch
      with operands swapped (for non-commutative ops like Sub/Div,
      use a reverse-scalar kernel or the broadcast kernel with stride 0).
  - For the `a`-is-scalar case on Sub and Div (non-commutative):
    - Sub: compute `scalar - b[i]` as `-(b[i] - scalar)` using SubScalar
      + MulScalar(-1) or add a `SubScalarRev` kernel. Simpler: use the
      existing broadcast kernel with strides (saRow=0, saCol=0).
  - Acceptance: All 4 binary ops (Add, Sub, Mul, Div) accept scalar-vs-tensor
    on GPU. No CPU fallback for these patterns.
  - Dependencies: T76.1.

- [ ] S76.2.1 Scalar-broadcast parity tests  Owner: TBD  Est: 1.5h
  - Test: [1] + [128, 2048] matches CPU Add within 1e-5.
  - Test: [128, 2048] - [1] matches CPU Sub within 1e-5.
  - Test: [1] * [128, 2048] matches CPU Mul within 1e-5.
  - Test: [128, 2048] / [1] matches CPU Div within 1e-5.
  - Test: [1] - [128, 2048] (scalar on left, non-commutative).
  - Test: [1] / [128, 2048] (scalar on left, non-commutative).
  - Verify all outputs have GPUStorage.

- [ ] T76.3 Add gpuSubScalar method to GPUEngine  Owner: TBD  Est: 30m
  - Wire SubScalar kernel through `gpuScalarOp` like AddScalar/MulScalar.
  - Acceptance: GPUEngine.Sub(tensor, scalar_tensor) uses GPU kernel.
  - Dependencies: T76.1.

- [ ] T76.4 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Packages: internal/cuda/kernels/, internal/gpuapi/, compute/.
  - Dependencies: T76.3.

### E77: GPU Slice, Split, and Concat Kernels (O77)

`GPUStorage.Slice()` is the #1 D2H source (24%). When a CPU fallback op
reads data from a GPU-resident tensor, `Slice()` copies the entire buffer
D2H. The fix is to keep sliced/split/concatenated tensors on GPU.

Existing code:
- `tensor/gpu_storage.go` line 188 -- `GPUStorage.Slice()` with D2H copy.
- `compute/gpu_engine.go` line 667 -- `Split` delegates to CPU.
- `compute/gpu_engine.go` line 671 -- `Concat` delegates to CPU.
- `compute/cpu_engine.go` -- CPU Split/Concat implementations.

- [ ] T77.1 Write CUDA Slice kernel  Owner: TBD  Est: 2h
  - Add to `internal/cuda/kernels/slice.cu`:
    `slice_strided_kernel(float *in, float *out, int *srcStrides, int *starts,
     int *outShape, int ndim, int total)` -- copies a contiguous sub-region.
  - For the common case (contiguous last-dim slice), a memcpy-based fast path.
  - Add `Slice` to KernelRunner interface:
    `Slice(in, out unsafe.Pointer, srcStrides, starts, outShape []int32, ndim, total int, stream Stream) error`
  - Add Go CGo wrapper and backend stubs.
  - Acceptance: Kernel compiles for sm_121. Correct output for 2D and 3D slices.
  - Dependencies: none.

- [ ] S77.1.1 GPU Slice kernel unit tests  Owner: TBD  Est: 1.5h
  - Test: 2D slice [4,8] -> rows [1:3] gives [2,8].
  - Test: 2D slice [4,8] -> cols [2:6] gives [4,4].
  - Test: 3D slice [2,4,8] -> [0:1, 1:3, :] gives [1,2,8].
  - Test: full slice (no-op) returns same data.
  - Compare with CPU Slice within 0 tolerance (exact).

- [ ] T77.2 Wire GPU Slice into GPUEngine  Owner: TBD  Est: 2h
  - Modify tensor Slice to detect GPUStorage and use the GPU kernel.
  - Option A: Add `Slice` method to Engine interface.
  - Option B: Override in `tensor/gpu_storage.go` to use a kernel-based
    sub-tensor extraction when a KernelRunner is available.
  - The simpler approach: add a `SliceGPU` method on GPUEngine that layers
    can call, similar to FusedRMSNormer. Layers that call tensor.Slice()
    in the hot path (attention mask creation, Split) type-assert the engine.
  - Acceptance: GPU-resident tensor.Slice() returns GPU-resident output.
    No D2H copy in pprof.
  - Dependencies: T77.1.

- [ ] S77.2.1 GPU Slice parity test  Owner: TBD  Est: 1h
  - Compare GPU Slice vs CPU Slice for shapes used in Gemma 3 inference.
  - Verify output has GPUStorage.

- [ ] T77.3 Write GPU Split using Slice kernel  Owner: TBD  Est: 1.5h
  - Modify `GPUEngine.Split` to call the GPU Slice kernel for each split chunk
    instead of delegating to CPU.
  - Split along axis: compute start/end for each chunk, call Slice kernel.
  - Acceptance: GPUEngine.Split returns GPU-resident tensors. Parity with CPU.
  - Dependencies: T77.2.

- [ ] S77.3.1 GPU Split parity test  Owner: TBD  Est: 1h
  - Split [8, 2048] into 8 chunks of [1, 2048] along axis 0.
  - Split [1, 2048] into 2 chunks of [1, 1024] along axis 1 (head split).
  - Verify all output tensors have GPUStorage.

- [ ] T77.4 Write CUDA Concat kernel  Owner: TBD  Est: 2h
  - Add to `internal/cuda/kernels/concat.cu`:
    `concat_kernel(float **inputs, int *inputSizes, float *output, int numInputs,
     int outerStride, int innerSize, int axis)`.
  - Add `Concat` to KernelRunner interface.
  - Wire into GPUEngine.Concat.
  - Acceptance: GPUEngine.Concat uses GPU kernel for GPU-resident tensors.
    Output has GPUStorage. Parity with CPU.
  - Dependencies: none.

- [ ] S77.4.1 GPU Concat parity test  Owner: TBD  Est: 1h
  - Concat 8 tensors of [1, 2048] along axis 0 gives [8, 2048].
  - Concat 2 tensors of [1, 1024] along axis 1 gives [1, 2048].
  - Verify output has GPUStorage.

- [ ] T77.5 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Packages: internal/cuda/kernels/, internal/gpuapi/, compute/, tensor/.
  - Dependencies: T77.4.

### E78: Float32 Weight Upload (O78)

With Pow and binary ops now on GPU, uploading float32 weights to GPU is
beneficial (previously it caused D2H from CPU fallback ops). This removes
the remaining H2D copies for float32 normalization weights and biases.

- [ ] T78.1 Enable float32 weight upload in GPUEngine.UploadWeights  Owner: TBD  Est: 1h
  - Remove the skip for non-Q4 tensors in `GPUEngine.UploadWeights`.
  - Upload all float32 weight tensors to GPU using NewGPUStorageFromSlice.
  - Acceptance: Both Q4 and float32 weights are GPU-resident after load.
  - Dependencies: E75, E76 (Pow and binary ops must be on GPU first).

- [ ] S78.1.1 Float32 weight upload test  Owner: TBD  Est: 45m
  - Load model with -device cuda.
  - Verify all Parameter tensors (including float32 norm weights) have GPUStorage.
  - Run forward pass. Verify correct output.

- [ ] T78.2 Run golangci-lint on compute/ and inference/  Owner: TBD  Est: 15m
  - Dependencies: T78.1.

### E79: End-to-End GPU Benchmark (O78)

After all optimizations, measure tok/s on DGX Spark and compare with baselines.

- [ ] T79.1 Profile GPU inference after all optimizations  Owner: TBD  Est: 2h
  - Build zerfoo with CUDA tags on DGX Spark including all E75-E78 changes.
  - Run `bench_tps -model ~/models/gemma3-q4 -device cuda -tokens 100`.
  - Capture pprof profile.
  - Acceptance: Profile captured showing reduced cgocall % and eliminated
    CPU fallbacks.
  - Dependencies: T78.1, T77.5.

- [ ] S79.1.1 GPU profile report  Owner: TBD  Est: 30m
  - Document: tok/s, cgocall %, remaining CPU fallbacks, GPU utilization.

- [ ] T79.2 Compare GPU vs CPU tok/s  Owner: TBD  Est: 1h
  - Run same prompt with -device cpu and -device cuda.
  - Measure tok/s for both. 3 runs each, report median.
  - Acceptance: GPU tok/s >= 10 (target). If not met, identify remaining
    bottleneck and document what would be needed.
  - Dependencies: T79.1.

- [ ] S79.2.1 Benchmark comparison report  Owner: TBD  Est: 30m

- [ ] T79.3 Verify output correctness  Owner: TBD  Est: 1h
  - Generate 50 tokens with same prompt on CPU and GPU.
  - Compare output text. Both should produce coherent English text.
  - Acceptance: No NaN or Inf. Coherent output on both devices.
  - Dependencies: T79.1.

- [ ] S79.3.1 Output correctness test  Owner: TBD  Est: 30m

- [ ] T79.4 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m
  - Dependencies: T79.3.

---

## 4. Parallel Work

Five epics fall into 3 tracks. E75/E76 (scalar ops) and E77 (D2H elimination)
are independent kernel work. E78 (weight upload) depends on E75+E76. E79
(benchmark) depends on all.

| Track | Epics | Description | Sync Points |
|-------|-------|-------------|-------------|
| A: Scalar Kernels | E75, E76 | PowScalar + SubScalar + scalar-broadcast | Converge before E78 |
| B: D2H Elimination | E77 | GPU Slice/Split/Concat kernels | Converge before E79 |
| C: Weight Upload + Benchmark | E78, E79 | Float32 upload + final measurement | After A + B |

### Within-Track Parallelism

| Track | Parallel Tasks | Notes |
|-------|----------------|-------|
| A | T75.1 and T76.1 | PowScalar and SubScalar kernels are independent |
| B | T77.1 and T77.4 | Slice and Concat kernels are independent |
| A+B | T75.1, T76.1, T77.1, T77.4 | All kernel development in parallel |

### Execution Order

Wave 1 (parallel):
- T75.1 (PowScalar kernel), T76.1 (SubScalar kernel), T77.1 (Slice kernel),
  T77.4 (Concat kernel) -- all independent CUDA kernel work.

Wave 2 (after Wave 1):
- T75.2 (PowScalar Go wrapper), T76.2 (scalar-broadcast detection),
  T76.3 (SubScalar wiring), T77.2 (GPU Slice wiring), T77.3 (GPU Split).

Wave 3 (after Wave 2):
- T75.3 (wire PowScalar into GPUEngine.Pow).
- All lint tasks (T75.4, T76.4, T77.5).

Wave 4 (after E75 + E76):
- T78.1 (float32 weight upload).

Wave 5 (after all):
- E79 (benchmark).

---

## 5. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M41: Scalar GPU ops | E75, E76 | none | PowScalar + SubScalar + scalar-broadcast on GPU. Zero Pow/binary CPU fallbacks. |
| M42: D2H eliminated | E77 | none | GPU Slice/Split/Concat. GPUStorage.Slice D2H < 5% in pprof. |
| M43: Full GPU coverage | E78 | M41 | All float32 weights on GPU. Zero H2D in forward pass. |
| M44: 10 tok/s GPU | E79 | M41, M42, M43 | bench_tps reports >= 10 tok/s with -device cuda on DGX Spark. |

Critical path: T75.1 -> T75.2 -> T75.3 -> T78.1 -> T79.1

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R76 | PowScalar kernel numerical precision differs from CPU powf | Output quality regression | Low | Test within 1e-5 tolerance. Use CUDA `powf` which matches glibc. |
| R77 | GPU Slice kernel for non-contiguous strides is complex | Implementation takes longer than estimated | Medium | Start with contiguous fast path (memcpy). Add strided path only if profiling shows it in hot path. |
| R78 | Float32 weight upload increases GPU memory usage | OOM on small GPUs | Low | DGX Spark GB10 has 128GB. Gemma 3 2B float32 weights are ~400MB. Not a concern. |
| R79 | 10 tok/s target still not achieved after all changes | Unknown bottleneck remains | Medium | If target not met, re-profile and document. Possible next steps: batched inference, BF16 compute, or CUDA graph capture. |
| R80 | Scalar-broadcast for non-commutative ops (Sub, Div) when scalar is on the left | Wrong results if operand order is not handled correctly | Medium | Add explicit tests for `[1] - [M,D]` and `[1] / [M,D]` cases. Use broadcast kernel with stride 0 for left-scalar. |
| R81 | DGX Spark network connectivity | Cannot deploy and test on GPU | Low | Keep local CPU tests passing. Batch DGX work into focused sessions. |

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
- Use Conventional Commits: `feat(cuda): add PowScalar kernel`.
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

### Change Summary -- 2026-03-06

Created Phase 33 plan. Trimmed Phase 32 completed epics (E69-E74) to
docs/design.md section 15.14. Created ADR 023 for scalar ops strategy.

New epics:
- E75: PowScalar GPU Kernel (8.9% Pow CPU fallback).
- E76: Scalar-Broadcast for All Binary Ops (10.4% CPU fallback).
- E77: GPU Slice/Split/Concat (24% D2H elimination).
- E78: Float32 Weight Upload (enable now that Pow/binary ops are on GPU).
- E79: End-to-End GPU Benchmark (validate >= 10 tok/s target).

ADRs created:
- docs/adr/023-gpu-scalar-ops-d2h-elimination.md: Scalar ops strategy decision.

---

## 9. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, and troubleshooting. Design decisions in docs/adr/ (001-023).
- **Phases 1-32:** All documented in docs/design.md sections 15.1-15.14.
- **Phase 33:** This plan is the source of truth.
- **Quality:** See docs/QUALITY.md for test coverage report.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - On DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build -tags cuda ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key Starting Points

1. **E75 (PowScalar):** `compute/gpu_kernels.go` line 363 -- `gpuPow` with
   `sameShape` guard. When exponent has 1 element, extract scalar and use
   `gpuScalarOp` with new `kernels.PowScalar`.

2. **E76 (Scalar-Broadcast):** `compute/gpu_kernels.go` line 77 --
   `gpuBroadcastOp` switch. Add case before existing switch: if either operand
   has 1 total element, extract scalar and dispatch to scalar kernel.

3. **E77 (GPU Slice):** `tensor/gpu_storage.go` line 188 -- `GPUStorage.Slice()`
   is the D2H bottleneck. Add CUDA kernel in `internal/cuda/kernels/slice.cu`.
   Wire via a `SliceGPU` method on GPUEngine.

4. **E78 (Weight Upload):** `compute/gpu_engine.go` line 154 -- remove the
   skip for non-Q4 tensors. Upload float32 weights to GPU.

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
| `compute/engine.go` | Engine[T] interface (30+ methods) |
| `compute/gpu_engine.go` | GPUEngine with CUDA acceleration + CPU fallback |
| `compute/gpu_kernels.go` | gpuBinaryOp, gpuUnaryOp, gpuScalarOp, gpuBroadcastOp |
| `compute/broadcast.go` | broadcastShape() for NumPy-style output shapes |
| `internal/cuda/kernels/elementwise.cu` | CUDA element-wise kernels |
| `internal/cuda/kernels/elementwise.go` | Go CGo wrappers for CUDA kernels |
| `internal/cuda/kernels/slice.cu` | (NEW) GPU strided slice kernel |
| `internal/cuda/kernels/concat.cu` | (NEW) GPU concat kernel |
| `internal/gpuapi/kernels.go` | KernelRunner interface |
| `internal/gpuapi/cuda_kernels.go` | CUDA KernelRunner implementation |
| `tensor/gpu_storage.go` | GPUStorage[T] with Slice() D2H copy |
| `layers/core/pow.go` | Pow layer calling engine.Pow() |
| `inference/inference.go` | Load(), WithDevice(), engine creation |
| `cmd/bench_tps/main.go` | tok/s benchmark binary |

### Estimated Effort Summary

| Epic | Area | Tasks | Estimated Hours |
|------|------|-------|----------------|
| E75: PowScalar Kernel | internal/cuda/, compute/ | 4 tasks + 2 subtests | 5.25h |
| E76: Scalar-Broadcast | internal/cuda/, compute/ | 4 tasks + 2 subtests | 5.75h |
| E77: GPU Slice/Split/Concat | internal/cuda/, compute/, tensor/ | 5 tasks + 4 subtests | 11.5h |
| E78: Float32 Weight Upload | compute/, inference/ | 2 tasks + 1 subtest | 2.0h |
| E79: GPU Benchmark | DGX Spark | 4 tasks + 3 subtests | 5.25h |
| **Total** | **GPU scalar ops + D2H** | **19 tasks + 12 subtests** | **~30h** |
