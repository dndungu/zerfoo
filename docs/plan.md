# Zerfoo Development Plan -- Phase 29: 4x CPU Throughput

## 1. Context

### Problem Statement

Phase 28 DGX Spark benchmarks measured 3.80 tok/s for Gemma 3 2B Q4_0 on
ARM64 CPU (GB10, 20 cores). The target is >= 15 tok/s (4x improvement).
GPU inference is out of scope -- this phase is CPU-only optimization.

See docs/design.md for full architecture context and Phases 1-28 history.

### Profile Breakdown (Phase 28, DGX Spark GB10, Gemma 3 2B Q4_0)

| Component | % CPU | Notes |
|-----------|-------|-------|
| MatMul (NEON GEMM) | 19.6% | sgemmAccRowNeon |
| Runtime Transpose | 8.8% | Dynamic Q/K transposes in attention |
| GC + memclr | 6.9% | 79,537 allocs/token, 39.4 GB/op |
| Element-wise ops | 3.5% | Add, Mul, Pow (binaryOp) |
| Q4 dequantize | 3.1% | decodeQ4Blocks at load time |
| Other | ~58% | Parallelization overhead, scheduling |

The three highest-impact opportunities are:
1. **Q4 NEON dot product** -- `GemmQ4F32Fused` in `internal/xblas/gemm_quant.go`
   already does per-block dequant into a stack buffer, then calls `sgemmAccRow`.
   A true NEON kernel would extract nibbles and FMA directly in SIMD registers,
   avoiding even the per-block scalar dequant. (See docs/adr/020-q4-quantized-dot-product.md)
2. **TensorPool wiring in generate loop** -- `graph.Forward` already has pool
   support (`WithPool()`, ref-counting, `Release`). But `generate/generator.go`
   never creates or attaches a `compute.TensorPool`. Wiring it eliminates most
   of the 79K allocs/token.
3. **KV cache optimization** -- The generate loop already passes single tokens
   in decode mode (`gen.graph.Forward(genCtx, tokenTensor)` with `[]int{nextToken}`).
   The graph builders may still project Q/K/V for the full sequence. Optimize
   to only project the new token.

### Existing Code Inventory

| File | What Exists | What Is Missing |
|------|-------------|-----------------|
| `internal/xblas/gemm_quant.go` | `GemmQ4F32Fused` -- per-block dequant to stack buf, then `sgemmAccRow` | NEON kernel that does nibble extract + FMA in registers |
| `internal/xblas/gemm_simd_arm64.s` | `sgemmAccRowNeon` -- NEON float32 row accumulate | Q4-specific NEON inner loop |
| `compute/pool.go` | `TensorPool[T]` with `Acquire(shape)`, `Release(t)`, mutex-safe | Nothing -- pool itself is complete |
| `graph/graph.go:44` | `WithPool(pool TensorReleaser[T])` -- wires pool into Forward with ref-counting | Nothing -- graph integration is complete |
| `generate/generator.go:156,186` | Calls `gen.graph.Forward()` for prefill and decode | Never calls `gen.graph.WithPool()` |
| `generate/kv_cache.go` | `KVCache` and `PagedKVCache` with `CacheProvider` interface | May recompute full-sequence projections |
| `compute/fused_rmsnorm.go` | Scalar fused RMSNorm | NEON SIMD version |
| `compute/fused_silugate.go` | Scalar fused SiLU-gate | NEON SIMD version |

### Objectives

- O71: NEON Q4 dot-product kernel that performs nibble extraction and FMA
  in SIMD registers, replacing the scalar per-block dequant in GemmQ4F32Fused.
- O72: TensorPool wired from `generate/generator.go` into `graph.WithPool()`
  so intermediate tensors are recycled between decode steps. Target: < 1,000
  allocs/token.
- O73: KV cache used efficiently during autoregressive decode. Each step
  computes only the new token's Q/K/V projections and reuses cached K/V.
- O74: >= 15 tok/s on DGX Spark GB10 for Gemma 3 2B Q4_0 CPU ARM64.

### Non-Goals

- GPU inference pipeline (separate phase).
- Vision/multimodal models.
- Training performance or features.
- New model architectures.
- x86 AVX2/AVX-512 SIMD kernels (ARM64 NEON only for now).
- Breaking changes to Engine[T] or Node[T] interfaces.

### Constraints and Assumptions

- Go standard library only where possible.
- All CUDA code behind `//go:build cuda` build tags.
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- Table-driven tests using the standard testing package.
- DGX Spark GB10 at ssh ndungu@192.168.86.250 for benchmarks.
- ARM64 NEON assembly via Go asm (`internal/xblas/` pattern).
- Existing Q4_0 block format: 32 values per block, 18 bytes (2B float16 scale
  + 16B packed nibbles). `q4Block` struct in `tensor/quantized.go`.
- `GemmQ4F32Fused` already uses `dequantQ4Block` + `sgemmAccRow` pattern.
  The NEON kernel replaces `dequantQ4Block` with in-register nibble extraction.

### Success Metrics

| Metric | Current | Phase 29 Target |
|--------|---------|-----------------|
| tok/s (Gemma 3 2B Q4_0, CPU ARM64) | 3.80 | >= 15 |
| allocs/token | 79,537 | < 1,000 |
| GC % CPU | 6.9% | < 1% |
| MatMul+dequantize % CPU | 22.7% | < 15% |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D101 | NEON Q4 dot-product kernel | Replace scalar dequant in GemmQ4F32Fused with NEON nibble-extract + FMA |
| D102 | TensorPool wiring | Create pool in generator, pass to graph.WithPool() |
| D103 | KV cache decode optimization | Single-token projection with cached K/V reuse |
| D104 | NEON fused ops | RMSNorm and SiLU-gate NEON kernels |
| D105 | DGX Spark benchmark | >= 15 tok/s validated on GB10 |

### Out of Scope

- GPU inference path.
- Q4_K / Q5_K / Q6_K fused dot products (follow-up phase).
- x86 SIMD kernels.
- Prompt caching / prefix sharing.
- Multi-GPU.
- Rewriting TensorPool or graph.Forward pool logic (already works).

---

## 3. Checkable Work Breakdown

### E49: NEON Q4 Dot Product Kernel (O71, O74)

Decision rationale: docs/adr/020-q4-quantized-dot-product.md

The existing `GemmQ4F32Fused` in `internal/xblas/gemm_quant.go` dequantizes
each Q4 block (32 values) into a `[32]float32` stack buffer via `dequantQ4Block`,
then calls `sgemmAccRow`. The NEON kernel replaces `dequantQ4Block` + the
inner multiply with a single NEON routine that extracts nibbles and FMAs
directly in vector registers.

- [x] T49.1 NEON Q4 block dot-product function  Owner: TBD  Est: 4h
  - Add `q4DotBlockNeon(packed *byte, scale float32, x *float32) float32`
    in `internal/xblas/q4dot_arm64.s`.
  - Takes one Q4 block (16 packed bytes = 32 nibbles) and 32 float32
    activation values. Returns dot product of dequantized Q4 * activation.
  - NEON implementation:
    1. VLD1 16 packed bytes into V0.
    2. VAND V0, #0x0F -> V1 (low nibbles), USHR V0, #4 -> V2 (high nibbles).
    3. Widen uint4 to int16 via UXTL, subtract 8 (zero-point).
    4. Widen int16 to int32 via SXTL, then SCVTF to float32.
    5. VLD1 activation float32x4 into V-regs, FMLA accumulate.
    6. Multiply accumulated sum by scale, return.
  - Go declaration in `q4dot_arm64.go` with `//go:build arm64`.
  - Scalar fallback in `q4dot_generic.go` with `//go:build !arm64`.
  - Acceptance: Output matches `dequantQ4Block` + scalar dot within 1e-5.
  - Dependencies: none.

- [x] S49.1.1 NEON Q4 block dot-product tests  Owner: TBD  Est: 1h
  - Table-driven: zero block, all-same-nibble, random, max-scale, min-scale.
  - Compare NEON result against scalar `dequantQ4Block` + dot product.

- [ ] T49.2 Replace dequantQ4Block in GemmQ4F32Fused  Owner: TBD  Est: 3h
  - Modify `GemmQ4F32Fused` in `internal/xblas/gemm_quant.go`:
    - Instead of `dequantQ4Block(data, scale, &buf)` then accumulating via
      `sgemmAccRow`, call `q4DotBlockNeon` for decode (M=1, single output row)
      path directly.
    - For M>1 (batch), keep the existing `dequantQ4Block` + `sgemmAccRow`
      path (NEON dot is only faster for M=1 where memory bandwidth dominates).
  - Acceptance: `go test ./internal/xblas/ -run TestGemmQ4` passes.
    Benchmark at M=1, K=2048, N=8192: >= 1.5x faster than current.
  - Dependencies: T49.1.

- [ ] S49.2.1 GemmQ4F32Fused benchmark before/after  Owner: TBD  Est: 1h
  - Benchmark M=1 (decode) and M=32 (batch) at K=2048, N=8192.
  - Compare against current `GemmQ4F32Fused` on DGX Spark.

- [ ] T49.3 NEON Q4 row dot-product for M>1  Owner: TBD  Est: 4h
  - For batch (M>1), the per-block dot is not sufficient because each block
    contributes to all N output columns. Implement a hybrid:
    - Still use `q4DotBlockNeon` for the inner nibble extraction.
    - Restructure the outer loop to process multiple output columns per block.
  - Alternative: for M>1 (prefill only), keep dequant+sgemmAccRow (prefill
    is not the bottleneck -- decode is).
  - Acceptance: M=32 path produces correct output. No regression vs current.
  - Dependencies: T49.2.

- [ ] S49.3.1 Batch path correctness tests  Owner: TBD  Est: 30m

- [ ] T49.4 Run golangci-lint on internal/xblas/  Owner: TBD  Est: 15m

### E50: TensorPool Wiring in Generate Loop (O72, O74)

The pool infrastructure is complete:
- `compute.TensorPool[T]` in `compute/pool.go` (Acquire, Release, mutex-safe).
- `graph.Graph[T].WithPool(pool)` in `graph/graph.go` (ref-counting, release).

The gap: `generate/generator.go` never calls `graph.WithPool()`. Each
`graph.Forward` call allocates fresh tensors for every intermediate result,
producing 79,537 allocs/token.

- [x] T50.1 Create TensorPool in Generator constructor  Owner: TBD  Est: 2h
  - In `generate/generator.go`, add a `pool *compute.TensorPool[T]` field
    to `Generator[T]`.
  - Create the pool in `NewGenerator()` or `NewGeneratorWithEngine()`.
  - Call `gen.graph.WithPool(gen.pool)` before the first `Forward` call.
  - The pool persists across decode steps so buffers are reused.
  - Do NOT create the pool if the graph is nil (for testing).
  - Acceptance: `go test ./generate/ -race` passes. No behavior change.
  - Dependencies: none.

- [x] S50.1.1 Generator pool wiring tests  Owner: TBD  Est: 1h
  - Verify pool is attached: after Generate(), pool.Len() > 0 (buffers
    returned to pool).
  - Verify correctness: Generate() output unchanged with pool enabled.

- [x] T50.2 Measure allocation reduction  Owner: TBD  Est: 1h
  - Add a benchmark test in `generate/` that measures allocs/token with
    pool enabled vs disabled.
  - Use `testing.B.ReportAllocs()` and `testing.AllocsPerRun`.
  - Acceptance: allocs/token < 1,000 (from 79,537).
  - Dependencies: T50.1.

- [x] S50.2.1 Allocation benchmark test  Owner: TBD  Est: 30m
  - Table-driven: with pool, without pool. Report allocs/op and bytes/op.

- [x] T50.3 Run golangci-lint on generate/  Owner: TBD  Est: 15m

### E51: KV Cache Decode Optimization (O73, O74)

- [x] T51.1 Audit current decode-step graph execution  Owner: TBD  Est: 2h
  - Trace what happens when `gen.graph.Forward(genCtx, tokenTensor)` is
    called with a single-token input tensor during decode.
  - The graph builders in `inference/arch_common.go` build the full
    transformer graph at load time. During decode, the graph executes
    with a [1, hidden_dim] input. Verify:
    a) Does each attention layer's Q/K/V projection only process the new
       token (input is 1 token), or does it process the full cached sequence?
    b) Does the KV cache append the new K/V and return the full sequence
       K/V for attention, or does it recompute?
    c) How does causal masking work for the single-token case?
  - Document findings in docs/updates.md.
  - Acceptance: Written analysis with line numbers.
  - Dependencies: none.

- [x] T51.2 Optimize decode graph path  Owner: TBD  Est: 4h
  - Already optimized: Q/K/V projections process only new token [1,d],
    KV cache appends+returns full sequence, causal mask skipped for decode.
    See docs/updates.md T51.1 audit findings.
  - Dependencies: T51.1.

- [x] S51.2.1 Decode parity test  Owner: TBD  Est: 1h
  - Generate 10 tokens. Verify token-for-token identical output to baseline.

- [x] T51.3 Incremental RoPE for decode  Owner: TBD  Est: 2h
  - During decode, apply RoPE only to the new token's Q/K at the correct
    position index (seq_len so far), not recompute for full sequence.
  - The fused RoPE in `compute/fused_rope.go` takes a position offset.
    Verify the graph builder passes the correct offset during decode.
  - Acceptance: RoPE output matches full-sequence RoPE for the new token
    position.
  - Dependencies: T51.2.

- [x] S51.3.1 Incremental RoPE correctness test  Owner: TBD  Est: 30m

- [x] T51.4 Run golangci-lint on inference/ and generate/  Owner: TBD  Est: 15m

### E52: NEON Fused Ops (O74)

- [ ] T52.1 Fuse runtime attention transpose  Owner: TBD  Est: 3h
  - The 8.8% runtime transpose is Q/K transposes in attention.
  - Option A: Store K in transposed layout in the KV cache, so Q @ K^T
    becomes Q @ K_stored (no transpose needed).
  - Option B: Use a transposed-B GEMM variant (sgemmABT) that iterates K
    columns directly without materializing the transpose.
  - Choose the option that requires fewer graph builder changes.
  - Acceptance: Runtime transpose drops to < 2% CPU in profile.
  - Dependencies: E51 (KV cache changes may affect K layout).

- [ ] S52.1.1 Attention transpose elimination benchmark  Owner: TBD  Est: 30m

- [ ] T52.2 NEON RMSNorm  Owner: TBD  Est: 2h
  - `compute/fused_rmsnorm.go` uses scalar Go loops.
  - Add ARM64 assembly in `compute/fused_rmsnorm_arm64.s`:
    - FMLA for sum-of-squares across float32x4 lanes.
    - FRSQRTE + FRECPS for reciprocal sqrt (2 Newton iterations).
    - FMUL for element-wise weight scaling.
  - Go declaration in `fused_rmsnorm_arm64.go` with `//go:build arm64`.
  - Scalar fallback unchanged in `fused_rmsnorm.go`.
  - Acceptance: NEON RMSNorm matches scalar within 1e-5. >= 2x faster in
    benchmark.
  - Dependencies: none.

- [ ] S52.2.1 NEON RMSNorm parity and benchmark  Owner: TBD  Est: 30m

- [ ] T52.3 NEON SiLU-gate  Owner: TBD  Est: 2h
  - `compute/fused_silugate.go` uses scalar Go loops.
  - Add ARM64 assembly in `compute/fused_silugate_arm64.s`:
    - FNEG + polynomial FEXP approximation (minimax degree-4) for sigmoid.
    - FMUL for gate * up.
  - Acceptance: NEON SiLU-gate matches scalar within 1e-4. >= 2x faster.
  - Dependencies: none.

- [ ] S52.3.1 NEON SiLU-gate parity and benchmark  Owner: TBD  Est: 30m

- [ ] T52.4 Run golangci-lint on compute/  Owner: TBD  Est: 15m

### E53: DGX Spark Benchmark Validation (O74)

- [ ] T53.1 End-to-end benchmark with all optimizations  Owner: TBD  Est: 2h
  - Run Gemma 3 2B Q4_0 benchmark on DGX Spark GB10 with all Phase 29
    optimizations enabled (NEON Q4 dot, pool wired, KV decode opt, NEON ops).
  - Capture: tok/s, allocs/token, GC %, CPU profile breakdown.
  - Compare against Phase 28 baseline (3.80 tok/s, 79,537 allocs/token).
  - Acceptance: >= 15 tok/s OR clear documentation of remaining bottlenecks
    and next steps.
  - Dependencies: E49, E50, E51, E52.

- [ ] S53.1.1 Before/after profile comparison  Owner: TBD  Est: 30m

- [ ] T53.2 Regression test for correctness  Owner: TBD  Est: 1h
  - Run existing parity tests on DGX Spark with all optimizations.
  - Verify no output quality degradation from fused kernels or pooling.
  - Dependencies: T53.1.

- [ ] T53.3 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m

---

## 4. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M21: NEON Q4 dot product | E49 | none | NEON kernel passes correctness tests, >= 1.5x faster than current GemmQ4F32Fused at M=1 |
| M22: Zero-alloc forward | E50 | none | allocs/token < 1,000 via pool wiring in generator |
| M23: Efficient decode | E51 | none | Decode step only projects new token, reuses cached K/V |
| M24: NEON fusions | E52 | E51 | Runtime transpose < 2%, NEON RMSNorm and SiLU-gate |
| M25: 15 tok/s | E53 | E49-E52 | >= 15 tok/s on DGX Spark GB10 validated |

Recommended execution order:
1. **[E49, E50, E51]** -- All three are independent. E49 (NEON kernel) and
   E50 (pool wiring) are the highest-impact changes. E51 (KV decode audit)
   starts with analysis.
2. **E52** -- NEON fused ops. T52.1 (transpose) depends on E51 findings.
   T52.2-T52.3 (RMSNorm, SiLU-gate) are independent.
3. **E53** -- Final benchmark. Depends on all.

Critical path: E49 + E50 -> E53 (NEON GEMM + pooling are the largest gains).

---

## 5. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R71 | NEON assembly bugs cause silent numerical errors | Wrong model output | Medium | Compare NEON vs scalar for 10K random vectors. Use Go test fuzzing. |
| R72 | Q4 NEON speedup less than expected due to memory bandwidth saturation | < 1.5x improvement | Medium | Profile L1/L2 cache utilization. The decode M=1 case is bandwidth-bound; NEON reduces compute but bandwidth may dominate. |
| R73 | TensorPool wiring causes use-after-release bugs | Data corruption | High | ref-counting in graph.Forward already handles this. Run all tests with -race. Add stress test with 1000-token generate. |
| R74 | KV cache decode already optimized (single-token input already works) | E51 has no impact | Medium | Audit first (T51.1). If already optimal, mark as done and move on. |
| R75 | 15 tok/s not achievable on CPU alone | Target not met | Medium | Document achieved speedup and remaining bottlenecks. GPU inference as Phase 30 fallback. |
| R76 | Go assembler limitations prevent optimal NEON codegen | Suboptimal SIMD utilization | Low | Use Go asm TEXT directives (proven pattern in `gemm_simd_arm64.s`). |

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
8. Changes are committed in a small commit touching one directory only.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `perf(xblas): add NEON Q4 block dot product`.
- Always run linters and formatters before committing.

### Validation Strategy

- NEON kernels: compare against scalar fallback for 10K random inputs.
- TensorPool: measure allocs/token with testing.AllocsPerRun.
- KV cache: token-for-token parity against baseline (no optimization).
- All benchmarks on DGX Spark GB10 (ssh ndungu@192.168.86.250).

---

## 7. Progress Log

### Change Summary -- 2026-03-06 (plan refinement)

Refined Phase 29 plan based on codebase audit. Key corrections:

- E49: Updated to reflect that `GemmQ4F32Fused` already exists with per-block
  dequant + `sgemmAccRow`. The NEON kernel replaces `dequantQ4Block` with
  in-register nibble extraction, not a full rewrite. Added code inventory
  table to Context.
- E50: Corrected to reflect that `graph.Forward` pool wiring (ref-counting,
  `WithPool()`, `Release`) already works. The only gap is that
  `generate/generator.go` never calls `graph.WithPool()`. Reduced from 4
  tasks to 3 tasks. Removed T50.1 (audit) and T50.2 (wire into graph.Forward)
  since both are already done.
- E51: Added note that the generate loop already sends single-token tensors
  in decode mode. T51.1 audit will determine if further optimization is needed.
- Added "Existing Code Inventory" table to Context section.
- ADR 020 already exists, no changes needed.

### Change Summary -- 2026-03-06

Created Phase 29 plan targeting 4x CPU throughput (3.80 -> 15 tok/s).
Five epics: E49 (NEON Q4 dot product), E50 (TensorPool wiring), E51
(KV cache optimization), E52 (NEON fused ops), E53 (DGX benchmark
validation). Trimmed completed Phase 28 epics (E44-E48) from plan.
Merged Phase 28 stable knowledge into docs/design.md section 15.10.
Created ADR: docs/adr/020-q4-quantized-dot-product.md.

### Previous Entries

| Date | Phase | Summary |
|------|-------|---------|
| 2026-03-06 | 28 | Phase 28 ALL COMPLETE. E44-E48 done. GGUF inference, auto-download, chat templates, K-quants, DGX benchmarks (3.80 tok/s). |
| 2026-03-05 | 27 | Phase 27 kernel work complete. Transpose folding, TensorPool, fused ops. |
| 2026-03-05 | 26 | Phase 26 ALL COMPLETE. E34-E38 done. Gemma 3 2B Q4: 3.60 tok/s. |
| 2026-03-05 | 25 | Phase 25 ALL COMPLETE. All epics E25-E33 done. |
| 2026-03-05 | 22-24 | Phases 22-24 COMPLETE. BF16 GEMM, unified memory, SigLIP fix, coverage push. |
| 2026-03-04 | 21 | Phase 21 COMPLETE. 18 ONNX fixes, 18 PASS across 6 model families. |
| 2026-03-03 | 20 | Phase 20 COMPLETE. ARM64 build, GPU tests, benchmarks. |

---

## 8. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. Design decisions are in
  docs/adr/ (ADR-001 through ADR-020).
- **Phases 1-28:** All complete. See docs/design.md sections 15.1-15.10.
- **Phase 29:** This plan is the source of truth.
- **Quality:** See docs/QUALITY.md for test coverage report.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - On DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build -tags cuda,cutlass ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key Phase 29 Starting Points

1. **E49 (NEON Q4 dot):** Start in `internal/xblas/gemm_quant.go` --
   `GemmQ4F32Fused` and `dequantQ4Block` are the functions to optimize.
   The ARM64 NEON GEMM pattern is in `gemm_simd_arm64.s` (sgemmAccRowNeon).
   Q4 block format is in `tensor/quantized.go` (q4Block struct).
2. **E50 (pool wiring):** One-file change in `generate/generator.go`. Create
   `compute.NewTensorPool[T]()`, store on Generator, call
   `gen.graph.WithPool(gen.pool)`. The graph-level ref-counting is already
   implemented in `graph/graph.go:70-134`.
3. **E51 (KV cache):** Start by reading `generate/generator.go:175-199`
   (decode loop) and `generate/kv_cache.go` (cache interface). Trace how
   the graph builders in `inference/arch_common.go` handle single-token input.
4. **E52 (NEON fused ops):** `compute/fused_rmsnorm.go` and
   `compute/fused_silugate.go` are pure scalar Go. Add ARM64 asm files
   following the pattern in `internal/xblas/gemm_simd_arm64.s`.

### External Dependencies

- **DGX Spark (ndungu@192.168.86.250):**
  - Go 1.25.0 linux/arm64, CUDA 13.0, sm_121 (Blackwell).
  - Models: ~/models/gemma3-q4/ (Q4 ZMF), ~/models/gemma3/ (F32 ZMF).
  - Repos: ~/zerfoo/, ~/zonnx/, ~/zmf/.
- No external service dependencies for CPU-only work.

### Performance Baseline (Phase 28)

| Model | Params | Quant | tok/s | allocs/token | GC % |
|-------|--------|-------|-------|--------------|------|
| Gemma 3 2B | 2.6B | Q4_0 | 3.80 | 79,537 | 6.9% |
