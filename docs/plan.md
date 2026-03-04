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

Architecture, design, GPU details, operations, and troubleshooting are
documented in docs/design.md (the single reference document). Stable design
decisions are extracted into docs/adr/ (see [ADR index](design.md#14-architectural-decision-records)).

### Objectives

O12-O28: COMPLETE (Phases 10-19). Device affinity, multi-GPU inference, NCCL,
cuDNN forward/backward, TensorRT with dynamic shapes, CUTLASS flash attention
and INT4/INT8 GEMM, GRAL, ROCm backend, OpenCL backend.

O29-O31: COMPLETE (Phase 20). ARM64 build compatibility, Blackwell GPU
validation, feature gap assessment.

- O32: Run all model parity tests on GPU by downloading, converting, and
  deploying ZMF model files for all 7 model families on DGX Spark. **(IN PROGRESS)**
- O33: Document multi-GPU test coverage gap and define the conditions under
  which multi-GPU tests can be validated. **(COMPLETE)**

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
- FP4 kernel implementation (assessment only in Phase 20; implementation deferred).
- BF16 training loop (assessment only; implementation deferred).
- ConnectX-7 multi-node inference (assessment only; implementation deferred).

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

### Success Metrics

All metrics for Phases 10-20 ACHIEVED. See ADRs 007-017 and design.md Section 15.

Phase 21 metrics:

| Metric | Target | Status |
|--------|--------|--------|
| Model parity on GPU | 7 model families tested | 17 PASS, 5 SKIP (see ADR-018) |
| Multi-GPU gap documented | Prerequisites listed | ACHIEVED (ADR-017 updated, script created) |

---

## 2. Scope and Deliverables

D11-D53: COMPLETE. See ADRs 007-017.

| ID | Description | Status |
|----|-------------|--------|
| D54 | Model parity on GPU | All 7 model families' parity tests pass on DGX Spark with ZMF models |
| D55 | Multi-GPU gap doc | Multi-GPU test coverage gap documented with prerequisites for validation |

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

### Phase 14: GPU Runtime Abstraction Layer (GRAL) — COMPLETE 2026-03-03

GRAL interfaces (`internal/gpuapi/`), CUDA adapters, GPUEngine/GPUStorage
refactor. Zero direct cuda/cublas/cudnn imports in compute/ and tensor/.
See [ADR-011](adr/011-gpu-runtime-abstraction-layer.md).

### Phase 15: AMD ROCm Backend — COMPLETE 2026-03-03

HIP runtime, rocBLAS, MIOpen bindings, HIP kernels (elementwise + flash
attention), ROCmEngine, device registration, inference routing.
See [ADR-012](adr/012-amd-rocm-backend.md).

### Phase 16: OpenCL Backend — COMPLETE 2026-03-03

OpenCL runtime, CLBlast BLAS, 17 elementwise kernels (runtime compiled),
OpenCLEngine, DNN stub (CPU fallback).
See [ADR-013](adr/013-opencl-backend.md).

### Phase 17: cuDNN Backward Pass — COMPLETE 2026-03-03

8 backward CGo bindings, CUDA DNN adapter, GPUEngine backward methods for
training (Conv2d, BatchNorm, activation, pooling).
See [ADR-014](adr/014-cudnn-backward-pass.md).

### Phase 18: CUTLASS INT4/INT8 GEMM — COMPLETE 2026-03-03

INT8 tiled kernel, INT4 packed kernel with left/right-multiply, CGo bindings,
MatMulNBits GPU dispatch via build tags.
See [ADR-015](adr/015-cutlass-quantized-gemm.md).

### Phase 19: TensorRT Dynamic Shapes — COMPLETE 2026-03-03

Optimization profiles (min/opt/max), SetInputShape, dynamic cache keys,
DynamicShapeConfig in converter.
See [ADR-016](adr/016-tensorrt-dynamic-shapes.md).

### Phase 20: DGX Spark Hardware Validation — COMPLETE 2026-03-03

ARM64 build (10 fixes), GPU tests (66 pkgs pass), benchmarks (MatMul 46x,
flash 147us), feature gaps (FP4 blocked, BF16 3-5d). Model parity: 17 PASS
(Llama3, Qwen25, Gemma3, Mistral, Phi3, FlashAttentionGQA), 5 SKIP.
See [ADR-017](adr/017-dgx-spark-hardware-validation.md),
[ADR-018](adr/018-model-parity-testing.md), design.md Section 15.

### Phase 21: Skipped Test Coverage (Model Parity on GPU + Multi-GPU Gap)

#### Phase 21 Context

Phase 20 validated all GPU code on DGX Spark with 66 packages passing. However,
25+ tests were skipped due to two gaps:

1. **Model parity tests (20 tests):** The parity test framework in
   `tests/parity/helpers_test.go` uses `envOrSkip(t, key)` to check environment
   variables like `GEMMA3_ZMF_PATH`, `LLAMA3_ZMF_PATH`, etc. Without ZMF model
   files on the DGX Spark, all model parity tests skip. This covers 7 model
   families: Gemma 3, Llama 3, Mistral, Phi-4, Qwen 2.5, DeepSeek V3, and
   SigLIP (+ Kimi-VL connector). Each family has ~3 tests (forward pass, greedy
   decode, generation).

2. **Multi-GPU tests (5 tests):** The DGX Spark GB10 has a single GPU.
   Tests that check `GetDeviceCount() >= 2` skip: TestMemPoolNoCrossDeviceReuse,
   TestMemPoolMultiDeviceStats, TestTwoGPUAllReduce, TestTwoGPUBroadcast,
   TestMultiGPU_DualDeviceInference, TestNcclStrategy_TwoGPUAllReduce.
   These require a second DGX Spark unit connected via ConnectX-7.

The model parity gap is addressable now. The `zonnx` CLI tool (in
`../zonnx/`) downloads ONNX models from HuggingFace and converts them to ZMF
format. The DGX Spark has 390 GB free disk and 115 GB free RAM -- sufficient
for all 7 model families.

**ZMF conversion workflow:**
1. Install `zonnx` on DGX Spark (`go install` or copy binary)
2. `zonnx download <hf-repo>` -- downloads ONNX model from HuggingFace
3. `zonnx convert <onnx-dir> <output.zmf>` -- converts ONNX to ZMF format
4. Set environment variables (e.g., `GEMMA3_ZMF_PATH=/path/to/gemma3.zmf`)
5. Run parity tests with GPU tags

**Model families and HuggingFace repos:**
| Family | HuggingFace Repo | Env Vars |
|--------|-----------------|----------|
| Gemma 3 | google/gemma-3-1b-it (ONNX variant) | GEMMA3_ZMF_PATH, GEMMA3_MODEL_DIR |
| Llama 3 | meta-llama/Llama-3.2-1B (ONNX variant) | LLAMA3_ZMF_PATH, LLAMA3_MODEL_DIR |
| Mistral | mistralai/Mistral-7B-v0.1 (ONNX variant) | MISTRAL_ZMF_PATH, MISTRAL_MODEL_DIR |
| Phi-4 | microsoft/phi-4 (ONNX variant) | PHI4_ZMF_PATH, PHI4_MODEL_DIR |
| Qwen 2.5 | Qwen/Qwen2.5-0.5B (ONNX variant) | QWEN_ZMF_PATH, QWEN_MODEL_DIR |
| DeepSeek V3 | deepseek-ai/DeepSeek-V3 (ONNX variant) | DEEPSEEK_ZMF_PATH, DEEPSEEK_MODEL_DIR |
| SigLIP | google/siglip-so400m-patch14-384 (ONNX variant) | SIGLIP_ZMF_PATH, SIGLIP_MODEL_DIR |

Note: Exact HuggingFace repo names and ONNX availability must be verified at
runtime. Some models may require gated access (HF_TOKEN). Smaller model
variants are preferred to fit within DGX Spark memory.

#### E114: Model Parity Test Coverage on GPU

- [ ] T114.1 Install zonnx CLI on DGX Spark  Owner: TBD  Est: 30m
  - Dependencies: None
  - Steps on DGX Spark (ndungu@192.168.86.250):
    1. Clone zonnx repo: `git clone` to ~/zonnx
    2. Build: `cd ~/zonnx && go build -o ~/bin/zonnx .`
    3. Verify: `~/bin/zonnx --help`
  - Acceptance: `zonnx download --help` and `zonnx convert --help` succeed.
  - [ ] S114.1.1 Clone zonnx repo on DGX Spark  Est: 5m
  - [ ] S114.1.2 Build zonnx binary  Est: 15m
  - [ ] S114.1.3 Verify CLI works  Est: 5m

- [x] T114.2 Download and convert Gemma 3 model  2026-03-04
  - Downloaded google/gemma-3-1b-it via optimum-cli ONNX export on DGX Spark.
  - Converted with zonnx (after root-cause fix for integer initializer promotion).
  - Fixed: tokenizer array-of-arrays merges format, Gather embedded-indices,
    Slice hybrid mode and range clamping, zonnx integer initializer promotion.
  - Acceptance: ~/models/gemma3/model.zmf exists (3.8 GB, 381 params).
    TestGemma3ForwardPass PASS, TestGemma3GreedyDecode PASS, TestGemma3Generation PASS.

- [x] T114.3 Download and convert Llama 3 model  2026-03-04
  - Used onnx-community/Llama-3.2-1B from HuggingFace (previously downloaded).
  - Re-converted with updated zonnx (external data support).
  - Fixed: zonnx importer now loads ONNX external data files (model.onnx_data).
  - Fixed: Cos and Sin ONNX ops added for Llama RoPE position encoding.
  - Acceptance: ~/models/llama3/model.zmf exists (4.7 GB). TestLlama3ForwardPass PASS.

- [x] T114.4 Download and convert remaining models  2026-03-04
  - [x] S114.4.1 Download and convert Mistral  2026-03-04
    - Exported mistralai/Mistral-7B-Instruct-v0.3 via optimum-cli (torch 2.4.1).
    - Converted to ZMF: ~/models/mistral/model.zmf (27GB).
    - Added LessOrEqual and Or ops for Mistral attention mask.
    - All 3 parity tests PASS (ForwardPass, GreedyDecode, Generation).
  - [x] S114.4.2 Download and convert Phi-3  2026-03-04
    - Phi-4-mini tokenizer incompatible with optimum 1.22.0; used Phi-3-mini instead.
    - Exported microsoft/Phi-3-mini-4k-instruct via optimum-cli.
    - Converted to ZMF: ~/models/phi4/model.zmf (15GB).
    - All 3 parity tests PASS (ForwardPass, GreedyDecode, Generation).
  - [x] S114.4.3 Download and convert Qwen 2.5  2026-03-04
    - Previously downloaded. Re-converted with latest zonnx (proto field promotion).
    - Fixed: Reshape rebuild in builder Pass 2, batched MatMul, Where broadcasting.
    - TestQwen25ForwardPass PASS on DGX Spark. Output: [1 8 151936].
  - [x] S114.4.4 Download and convert DeepSeek V3  2026-03-04
    - SKIPPED: 671B MoE model exceeds 128GB DGX Spark memory.
  - [x] S114.4.5 Download and convert SigLIP  2026-03-04
    - Downloaded Xenova/siglip-base-patch16-224 pre-built ONNX.
    - Converted vision_model.onnx to ZMF: ~/models/siglip/model.zmf (355MB).
    - Added Squeeze, Tile, Mod, Gemm ops.
    - SKIP: Concat shape mismatch in vision graph (needs further investigation).

- [x] T114.5 Run model parity tests on GPU  2026-03-04
  - Fixed 18 issues across zerfoo and zonnx repos. See ADR-018.
  - 17 PASS: FlashAttentionGQA, Llama3 (FP/GD/Gen), Qwen25 (FP/GD/Gen),
    Gemma3 (FP/GD/Gen), Mistral (FP/GD/Gen), Phi3 (FP/GD/Gen)
  - 5 SKIP: DeepSeek (too large), SigLIP (graph issue), MultiGPU (1 device)

- [x] T114.6 Create test automation script  2026-03-04
  - File: scripts/dgx-spark-parity.sh (new)
  - Purpose: Shell script that sets all ZMF env vars and runs the full parity
    test suite. Makes it easy to re-run after code changes.
  - Acceptance: `./scripts/dgx-spark-parity.sh` runs all model parity tests.

#### E115: Multi-GPU Test Coverage Assessment

- [x] T115.1 Document multi-GPU test coverage gap  2026-03-04
  - File: docs/adr/017-dgx-spark-hardware-validation.md (updated)
  - Added Multi-GPU Test Coverage Gap section listing all 6 tests that require
    >= 2 CUDA devices, their file locations, skip conditions, and
    hardware/software prerequisites for validation.
  - Commit: fb74ccd

- [x] T115.2 Add multi-GPU test runner script  2026-03-04
  - File: scripts/dgx-spark-multigpu.sh (new)
  - Shell script runs all 6 multi-GPU tests across 4 packages. Sets CUDA/CGo
    env vars, NCCL ConnectX-7 configuration, filters tests by name.
  - Commit: b3b0861

#### E116: Phase 21 Final Verification

- [ ] T116.1 Update documentation  Owner: TBD  Est: 30m
  - Dependencies: E114, E115
  - Files: docs/plan.md, docs/design.md, docs/adr/017-dgx-spark-hardware-validation.md
  - Steps:
    1. Mark all Phase 21 tasks complete with results
    2. Update design.md Section 15 with model parity results
    3. Update ADR-017 with model parity results and multi-GPU gap inventory
  - Acceptance: All docs reflect actual test results.

---

## 4. Timeline and Milestones

M72-M93: All ACHIEVED (Phases 14-20). See ADRs 011-017.

| ID | Milestone | Dependencies | Exit Criteria |
|----|-----------|--------------|---------------|
| M94 | Model parity on GPU | E114 | All 7 model families' parity tests pass (or skip with documented reason) on DGX Spark |
| M95 | Multi-GPU gap documented | E115 | Multi-GPU test inventory and prerequisites documented in ADR-017 |
| M96 | Phase 21 complete | E116 | Skipped test coverage addressed; docs updated |

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
| R32 | HuggingFace model download requires gated access | Model download blocked without HF_TOKEN | Medium | Set HF_TOKEN env var on DGX Spark. Accept HF terms of use for gated models. |
| R33 | Large models exceed DGX Spark memory (128 GB) | Cannot run parity tests for large models | Medium | Use smallest available model variants. Skip models that exceed memory limits and document. |
| R34 | ONNX models not available for all families | zonnx conversion not possible | Medium | Check HuggingFace for ONNX variants. If unavailable, export using optimum or skip with doc. |
| R35 | Model parity tests reveal GPU-vs-CPU numerical differences | Parity failures on GPU | Medium | Adjust tolerances for GPU execution. FP32 GPU math may differ from CPU in last bits. |

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
- **Phases 1-20:** Complete. See ADRs 001-017.
- **Phase 21 (Skipped test coverage):** IN PROGRESS. Install zonnx on DGX
  Spark, download remaining model families from HuggingFace, convert to ZMF,
  run model parity tests with GPU. Document multi-GPU gap.
  SSH: `ndungu@192.168.86.250`.
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

### Production Readiness Scorecard (After Phase 20)

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
