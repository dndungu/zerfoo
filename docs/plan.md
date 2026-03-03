# Zerfoo Enterprise Production Readiness Plan

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with 40+ packages, a 34-method compute
Engine[T] interface, CPU and CUDA GPU backends, gRPC-based distributed
training, and comprehensive test coverage (95%+ across testable packages).
This plan tracked the gaps in observability, security, reliability,
configuration, CI/CD, model import, inference, and multi-architecture support
needed to reach production grade.

Architecture, design, GPU details, operations, and troubleshooting are
documented in docs/design.md (the single reference document). Stable design
decisions are extracted into docs/adr/ (see [ADR index](design.md#13-architectural-decision-records)).

### Non-Goals

- Multi-GPU or distributed GPU support.
- cuDNN, TensorRT, or other NVIDIA library integration.
- AMD ROCm or OpenCL backends.
- Mixed precision training.
- Breaking changes to the Engine[T] or Node[T] interfaces.
- Replacing gRPC with a different RPC framework.
- Adding third-party test frameworks (testify, etc.).
- SSM/Mamba architectures (Falcon Mamba, RWKV, Jamba).

### Constraints and Assumptions

- Use Go standard library only where possible. Minimize new dependencies.
- All CUDA code behind `//go:build cuda` build tags.
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- Table-driven tests using the standard testing package.

---

## 2. Scope and Deliverables

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

### Phase 1: Test Coverage (Complete)

30 of 33 packages at >= 95% statement coverage. Details in docs/design.md
Section 7.

### Phase 2: GPU Engine (Complete)

CUDA float32 GPUEngine with memory pool, cuBLAS, OOM fallback. 20 GPU-
accelerated operations. See [ADR-006](adr/006-gpu-engine-architecture.md).

### Phase 3: GPU Production Readiness (Complete)

Device-resident pipeline, stream pipelining, parity tests. Hardware validation
blocked (see E29 below).

### Phase 4: Enterprise Production Readiness (Complete)

Logging (E21), metrics (E22), TLS (E23), config (E24), shutdown (E25), health
(E26), CI hardening (E27), resource limits (E28), documentation (E30),
verification (E31). See [ADR-001](adr/001-enterprise-production-readiness.md).

### Phase 5: Distributed Training Protocol (Complete)

Worker service (E32), gRPC strategy (E33), multi-worker integration tests (E34),
worker lifecycle + CLI (E35), verification (E36). distributed/ at 96% coverage.
See [ADR-002](adr/002-distributed-training-protocol.md).

### Phase 6: Open Weights Model Import (Complete)

Gemma 3 ONNX import (E37), core operators (E38), vision encoder operators (E39),
MoE (E40), Gemma 3 validation (E41), Kimi-VL validation (E42), verification
(E43). 13 new operators added. See [ADR-003](adr/003-open-weights-model-import.md).

### Phase 7: Architecture Cleanup (Complete)

Dead code removal (E44), registration consolidation (E45), graph thread safety
(E46), documentation (E47), verification (E48).
See [ADR-001](adr/001-enterprise-production-readiness.md).

### Phase 8: Embeddable Inference Library (Complete)

BPE tokenizer (E49), KV cache (E50), generation loop (E51), streaming (E52),
model registry (E53), high-level API (E54), CLI commands (E55), end-to-end
validation (E56). See [ADR-004](adr/004-embeddable-inference-library.md).

### Phase 9: Multi-Architecture Support (Complete)

Config parsing (E57), param resolver (E58), Llama/Mistral validation (E59),
QKV bias (E60), YaRN RoPE (E61), Qwen validation (E62), partial RoPE (E63),
tied embeddings (E64), Phi-4 validation (E65), MLA (E66), shared MoE (E67),
DeepSeek validation (E68), verification (E69).
See [ADR-005](adr/005-multi-architecture-support.md).

### Blocked Items

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

## 4. Timeline Summary

All 9 phases complete (2026-02-24 through 2026-03-03). 69 epics (E1-E69),
~200 tasks. Only E29 (GPU hardware validation) remains blocked on external
GCP GPU quota.

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
7. Non-CUDA build (`go build ./...` without cuda tag) compiles.
8. Changes are committed in a small commit touching one directory only.

### Review and QA Steps

1. Read existing implementation before writing code.
2. Write tests first or alongside implementation. Use table-driven tests.
3. After implementation, run `go test -cover ./package/` to verify coverage.
4. Run `golangci-lint run --fix ./package/` to fix lint issues.
5. Run `gofmt -w .` to ensure formatting.
6. Run `go test ./... -count=1` to verify no regressions.
7. Run `go build ./...` (without cuda tag) to verify non-CUDA build.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(log): add structured logger`, `fix(distributed): add TLS config`.
- Never allow changes to pile up. Commit after each completed subtask.
- Always run linters and formatters before committing.

---

## 6. Progress Log

| Date | Phase | Summary |
|------|-------|---------|
| 2026-02-24 | 1 | Initial plan created |
| 2026-02-25 | 1 | Test coverage complete (30/33 packages >= 95%) |
| 2026-03-01 | 2-3 | GPU engine + production readiness complete |
| 2026-03-01 | 4 | Enterprise readiness complete (except E29 blocked) |
| 2026-03-01 | 5 | Phase 5 planned (distributed protocol) |
| 2026-03-02 | 5 | Distributed protocol complete (96% coverage) |
| 2026-03-02 | 6 | Open weights model import complete (13 new operators) |
| 2026-03-02 | 7 | Architecture cleanup complete |
| 2026-03-02 | 8 | Embeddable inference library complete |
| 2026-03-02 | 9 | Phase 9 planned (multi-architecture support) |
| 2026-03-03 | 9 | Multi-architecture support complete (6 model families) |

---

## 7. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. It is the single reference
  document. Design decisions are in docs/adr/.
- **Phase 1-3:** Complete. Test coverage, GPU engine, GPU production readiness.
- **Phase 4:** Complete (except E29 GPU validation, blocked on GCP quota).
- **Phase 5:** Complete. Distributed protocol, worker lifecycle, CLI. 96% coverage.
- **Phase 6:** Complete. Open weights import (Gemma 3, SigLIP, Kimi-VL). 13 operators.
- **Phase 7:** Complete. Dead code removed, registration consolidated, graph thread-safe.
- **Phase 8:** Complete. Inference library: tokenizer, KV cache, generation,
  streaming, registry, high-level API, CLI (pull/run/serve), OpenAI HTTP server.
- **Phase 9:** Complete. Multi-architecture support: Gemma 3, Llama 3, Mistral,
  Qwen 2.5, Phi-4, DeepSeek V3. Config registry, param resolver, QKV bias,
  YaRN RoPE, partial RoPE, tied embeddings, MLA, shared MoE.
- **GPU hardware validation:** Blocked on GCP GPU quota (E29).
- **Key files to read first:**
  - inference/inference.go -- High-level API: Load, Generate, Chat, GenerateStream
  - generate/generator.go -- Autoregressive generation loop
  - serve/server.go -- OpenAI-compatible HTTP server
  - compute/engine.go -- Engine[T] interface (34 methods)
  - graph/node.go -- Node[T] interface
  - distributed/interfaces.go -- Distributed training interfaces
- **How to run tests:** `go test ./... -cover` for full suite. `go test -tags cuda ./...` for GPU.
- **How to build:** `go build ./...` (CPU). `go build -tags cuda ./...` (GPU).
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.
- **Testing pattern for gRPC:** Use google.golang.org/grpc/test/bufconn for in-process tests.

### External Dependencies

- GCP GPU quota increase for hardware validation (preference ID: zerfoo-gpu-test,
  project: numerai-488804).

---

## 8. Appendix

### Production Readiness Scorecard (After Phase 9)

| Category | Score | How Achieved |
|----------|-------|-------------|
| Architecture | 10/10 | Multi-architecture config parsing (E57); MLA attention variant (E66) |
| Core Functionality | 10/10 | 6 model families supported: Gemma, Llama, Mistral, Qwen, Phi, DeepSeek |
| Testing | 10/10 | Parity tests for all supported architectures (E59, E62, E65, E68) |
| Error Handling | 9/10 | Structured logging, RPC validation, context deadlines |
| Security | 8/10 | TLS/mTLS for gRPC; HF_TOKEN for gated models |
| Observability | 8/10 | Logging, metrics, pprof endpoints |
| Configuration | 10/10 | Architecture-aware config parsing with HuggingFace field mapping (E57) |
| Operations | 10/10 | CLI pull/run/serve, OpenAI-compatible HTTP API |
| Documentation | 10/10 | Consolidated design.md + ADRs; supported architectures table |
| CI/CD | 9/10 | Blocking tests, coverage gate, benchmark gate |
| Model Coverage | 10/10 | Covers >90% of open-weight model downloads on HuggingFace |

### New Packages and Files Created

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
