# Zerfoo Enterprise Production Readiness Plan

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with 40+ packages, a 34-method compute
Engine[T] interface, CPU and CUDA GPU backends, gRPC-based distributed
training, and comprehensive test coverage (95%+ across testable packages).

Phases 1-24 are complete. See docs/adr/ for design decisions and docs/design.md
for the consolidated reference document.

### Phase Summary

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

O1-O36: COMPLETE (Phases 1-22). See ADRs 001-019.
O37: Test coverage push. COMPLETE (Phase 23). See docs/QUALITY.md.
O38: Fix TODOs + code quality. COMPLETE (Phase 24).

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

---

## 2. Scope and Deliverables

D1-D65: All COMPLETE. See ADRs 001-019 and docs/QUALITY.md.

---

## 3. Work Breakdown

All phases (1-24) are complete. No active tasks.

### Remaining Hardware-Blocked Items

These require hardware not currently available:

1. **Multi-GPU parity test** -- requires >= 2 CUDA devices (second DGX Spark
   via ConnectX-7). 6 tests skip on single-GPU. See ADR-017 Section
   "Multi-GPU Test Coverage Gap".
2. **DeepSeek V3 parity** -- 671B MoE exceeds 128GB DGX Spark memory.

---

## 4. Risk Register

Active risks only. Resolved/mitigated risks removed.

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R17 | ROCm tests cannot run in CI | No AMD GPU in CI | High | Tests skip gracefully. Validate on AMD hardware manually. |
| R24 | Three GPU backends increase maintenance burden | Bug surface area grows | High | GRAL abstraction minimizes duplication. Only vendor-specific code is in internal/ packages. |
| R31 | Single-GPU DGX Spark cannot validate multi-GPU code | NCCL and multi-GPU tests remain unvalidated | High | Tests skip gracefully. Second DGX Spark unit needed. |
| R39 | Some code paths unreachable without GPU hardware | Cannot achieve 100% on GPU-tagged files locally | Medium | Accept 95% floor for packages with GPU build-tag code. Validate on DGX Spark. |

---

## 5. Operating Procedure

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
- Use Conventional Commits: `feat(cublas): add BF16 GemmEx binding`.
- Always run linters and formatters before committing.

---

## 6. Progress Log

| Date | Phase | Summary |
|------|-------|---------|
| 2026-03-05 | 22-24 | Phases 22-24 COMPLETE. BF16 GEMM, unified memory, SigLIP fix, coverage push, TODO fixes. ADR-019 written. Plan trimmed. |
| 2026-03-04 | 21 | Phase 21 COMPLETE. 18 ONNX fixes, 18 PASS across 6 model families. ADR-018 written. |
| 2026-03-03 | 20 | Phase 20 COMPLETE. ARM64 build (10 fixes), GPU tests (66 pkgs), benchmarks. ADR-017 written. |
| 2026-03-03 | 14-19 | Phases 14-19 COMPLETE. GRAL, ROCm, OpenCL, cuDNN backward, INT4/INT8, TRT dynamic. ADRs 011-016. |
| 2026-03-03 | 10-13 | Phases 10-13 COMPLETE. Multi-GPU, cuDNN, TensorRT, CUTLASS. ADRs 007-010. |

---

## 7. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. It is the single reference
  document. Design decisions are in docs/adr/ (ADR-001 through ADR-019).
- **Phases 1-24:** All complete. No active development tasks.
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

### External Dependencies

- **DGX Spark (ndungu@192.168.86.250, aitopatom-bfc8):**
  - Go 1.26.0 for linux/arm64, cuDNN 9.19.1, TensorRT 10.15.1,
    NCCL 2.29.7, CUTLASS 4.2, CUDA 13.0.2 / driver 580.126.09.
- HIP SDK (>= 5.0) for AMD ROCm backend.
- OpenCL 2.0+ headers + CLBlast for OpenCL backend.
- Second DGX Spark unit (optional) for multi-GPU validation via ConnectX-7.

### Known Untestable Gaps

- health: EngineCheck takes concrete *CPUEngine type, preventing mock testing
- layers/attention: dupl linter blocks MLA Forward engine error test
- Most remaining gaps are tensor.New unreachable error paths
- cmd tools: main() and os.Exit paths

---

## 8. Appendix

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
