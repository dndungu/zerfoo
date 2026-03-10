# Phase 35: Close the Performance Gap

## 1. Context

### Problem Statement

Zerfoo is a Go ML framework at Phase 34 (code complete). Current inference
performance on Gemma 3 2B Q4 (DGX Spark GB10, ARM64+Blackwell):

| Config | tok/s | Notes |
|--------|-------|-------|
| CPU ARM64 (post NEON) | 8.15 median | Phase 34 Track D |
| CPU ARM64 (pre NEON) | 6.86 | Phase 30 baseline |
| GPU CUDA (peak) | 10.32 | Phase 33, CUDA graph |
| GPU CUDA (median) | 7.78 | Phase 33 |
| llama.cpp (GB10) | ~100 (est.) | Reference target |
| Theoretical max | 182 | 273 GB/s / 1.5GB Q4 model |

The framework has excellent breadth (40+ layers, 6 model architectures,
training, distributed, serving) but inference throughput is 10-15x below
llama.cpp on the same hardware. The megakernel tracing compiler (Phase 34
Track C) is code complete but untested on hardware. NEON SIMD kernels
(Phase 34 Track D) are committed but unmerged to main.

Phase 35 focuses entirely on closing this performance gap through three
strategies: landing Phase 34 work, GPU megakernel tuning, and memory
bandwidth optimization.

See docs/design.md for full architecture and package layout.

### Objectives

- O1: Land all Phase 34 work (merge branch, validate on DGX Spark).
- O2: Achieve >= 10 tok/s CPU ARM64 inference (Gemma 3 2B Q4).
- O3: Achieve >= 20 tok/s GPU CUDA inference (Gemma 3 2B Q4).
- O4: Reduce kernel launch overhead via megakernel (from 650+ to 1 launch).
- O5: Maximize memory bandwidth utilization (currently ~50% of 273 GB/s).

### Non-Goals

- New model architectures or layer types.
- Training performance optimization.
- Multi-GPU or distributed inference.
- New quantization formats (Q8, FP8 inference).

### Constraints and Assumptions

- DGX Spark GB10: NVIDIA Blackwell sm_121, 273 GB/s memory bandwidth.
- SSH: ndungu@192.168.86.250, Go at /usr/local/go/bin/go.
- CUDA: /usr/local/cuda/bin/nvcc (13.0), CGO_CFLAGS/LDFLAGS required.
- Pre-commit hook rejects multi-directory commits.
- Q4 model: ~/models/gemma3-q4/model.zmf (1.5GB) on DGX Spark.
- F32 model: ~/models/gemma3/model.zmf (4GB) on DGX Spark.
- Megakernel tracing only valid for decode mode (seqLen=1), not prefill.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| CPU ARM64 tok/s | >= 10 | bench_tps on DGX Spark, Gemma 3 2B Q4, 64 tokens |
| GPU CUDA tok/s | >= 20 | bench_tps on DGX Spark, Gemma 3 2B Q4, 64 tokens |
| Megakernel fires | Yes | Log output: "megakernel: compiled and loaded" |
| Megakernel correctness | Match plan.Run() | 50-token generation, max delta < 1e-3 |
| All tests pass | Yes | go test ./... on DGX Spark with cuda build tag |
| feat/neon-softmax merged | Yes | PR merged to main on zerfoo/zerfoo |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D1 | Merged feat/neon-softmax branch | Land NEON kernels, tracing compiler, composition fixes |
| D2 | Validated megakernel on DGX Spark | Prove single-launch decode works end-to-end |
| D3 | GPU KV cache integration | Enable multi-token generation with megakernel |
| D4 | Profiled and tuned megakernel | Identify and fix bottlenecks with nsys |
| D5 | Fused attention kernel | Flash Attention v2 pattern for bandwidth efficiency |
| D6 | Operator fusion passes | RMSNorm+Linear, SiLU+Mul gate fusions |
| D7 | Performance benchmark report | Before/after comparison across all configs |

### Out of Scope

- Prefill optimization (batched tiled attention for long prompts).
- Multi-GPU tensor parallelism.
- New model format support.
- purego CUDA dlopen migration (deferred to Phase 36).
- Training loop performance.

---

## 3. Checkable Work Breakdown

### E1: Land Phase 34 (Priority 1)

- [x] T1.1 Create upstream PR for feat/neon-softmax  Owner: TBD  Est: 15m  2026-03-09
  - PR #45 created at https://github.com/zerfoo/zerfoo/pull/45

- [x] T1.2 SSH to DGX Spark and pull latest  Owner: TBD  Est: 10m  2026-03-09
  - Pulled, cleaned stale root files, built successfully (no cuda tag).

- [x] T1.3 Benchmark CPU with NEON kernels  Owner: TBD  Est: 30m  2026-03-09
  - Results: 7.80, 7.81, 7.44 tok/s. Median: 7.80 tok/s.
  - 13.7% improvement over 6.86 baseline. Below 10 tok/s target.

- [x] T1.4 Validate megakernel compilation on DGX Spark  Owner: TBD  Est: 30m  2026-03-09
  - GPU tok/s: 8.61 (25.9% over 6.84 baseline).
  - [Deviation: Blocker] Megakernel NOT firing. 16 unsupported ops:
    AutoPositionIds, AutoZeroKVCache, Shape, Unsqueeze, Cast, Equal,
    Where, ConstantOfShape, Expand, Range, Cos, Sin, Greater, Trilu,
    Max, ScatterND.
  - Root cause: codegen/optable.go missing emitters for these ops.
  - New epic E1b created to implement missing emitters.

- [ ] T1.5 Merge PR after validation  Owner: TBD  Est: 10m
  - Merge feat/neon-softmax to main after correctness validated.
  - Acceptance: PR merged. main branch updated. Output is coherent.
  - Dependencies: T1.1, T1.3, T1.4, E1b, E1c, E1d.

### E1b: Implement Missing Megakernel Op Emitters (Priority 1, Blocker)

All 16 missing op emitters were implemented and validated:
- [x] T1b.1-T1b.5 All emitters implemented  2026-03-09
- [x] T1b.6 Megakernel compiles and loads on DGX Spark  2026-03-09
  - "megakernel: compiled and loaded (2731 instructions, 376 frozen slots)"

### E1c: Fix Megakernel CUDA Memory Errors (Priority 1, Blocker)

Six bugs fixed to resolve CUDA error 700 (illegal memory access):

- [x] T1c.1 Workspace constant initialization  Owner: TBD  Est: done  2026-03-09
  - Added InitWorkspaceSlot method and detection logic (0a30af7)
- [x] T1c.2 Workspace layout size for constant slots  Owner: TBD  Est: done  2026-03-09
  - Patched slotShapes before emit so layout allocates enough space (c2d6e5a)
- [x] T1c.3 Fix GEMV dim/weight ordering  Owner: TBD  Est: done  2026-03-09
  - gemvDimsWithWeight replaces gemvDims, uses bShape for M/K, returns weightIdx (6b5b8d0)
- [x] T1c.4 Fix scalar broadcast in binary ops  Owner: TBD  Est: done  2026-03-09
  - indexExpr helper returns [0] for scalars, [tid] for vectors (6b5b8d0)
- [x] T1c.5 Allocate shared memory in kernel launch  Owner: TBD  Est: done  2026-03-09
  - Added smem arg to <<<grid, 1024, smem>>> (6b5b8d0)
- [x] T1c.6 Filter 1D Q4 frozen slots from dev_gemv_q4  Owner: TBD  Est: done  2026-03-09
  - Only dispatch 2D Q4 weight matrices to gemvQ4Op (6b5b8d0)
- [x] T1c.7 Error recovery and nil-fallback  Owner: TBD  Est: done  2026-03-09
  - Graceful fallback to per-instruction on launch failure (fad9935, 1aafa02)
- Status: Megakernel launches without memory errors. 3.78 tok/s on DGX Spark.

### E1d: Fix Model Output Correctness (Priority 0, Critical Blocker)

All execution paths (CPU, GPU per-instruction, GPU megakernel) produce
garbage output. The bug predates the megakernel work -- it exists at commits
as old as a9e6fb1 (the "Track A validated" commit from Phase 34).

Debug findings (2026-03-09):
- F32 model forward on "The capital of France is" produces top logit "The" (12.47)
  instead of "Paris". Token "Paris" has logit -1.61 (extremely negative).
- Q4 model forward produces top logit "disturbances" (19.85), "Paris" at -1.61.
- Greedy decode (temp=0) produces "notnotnot" (F32) or "decreases decreases" (Q4).
- Arena disabled: still garbage. Not arena-related.
- ZMF model file: 1.5GB Q4, md5 c23b670f023113f711606fabd3d23178.
- Config says "gemma-3-1b-it" (model_type: "gemma3_text").
- model_type "gemma3_text" NOT registered in ArchConfigRegistry (only "gemma3").
  Falls back to parseFallbackConfig -- parses fields correctly but Architecture
  field is "gemma3_text" instead of "gemma3".

Root cause investigation needed:
1. Check if ZMF model was exported correctly by zonnx (weight permutations,
   embedding scale, tied LM head handling).
2. Check if Q4 dequantization produces correct values by comparing a single
   layer's weight matrix against the ONNX reference.
3. Check if the graph builder (BuildFromZMF) wires attention correctly
   (Q/K/V projections, rotary embeddings, head splitting).
4. Check if inference.Load uses the correct architecture config parser.
5. Re-export the model from ONNX using latest zonnx and compare.

- [ ] T1d.1 Register "gemma3_text" model_type in ArchConfigRegistry  Owner: TBD  Est: 15m
  - Add r.Register("gemma3_text", parseGemmaConfig) in DefaultArchConfigRegistry.
  - Acceptance: inference.Load parses config as "gemma3_text" architecture.
  - Dependencies: none.
  - Note: This alone may not fix output -- the ZMF path does not use
    buildTransformerGraph (it uses BuildFromZMF from protobuf).

- [ ] T1d.2 Validate Q4 dequantization correctness  Owner: TBD  Est: 1h
  - Extract a single weight matrix from model.zmf using Go code.
  - Dequantize Q4 blocks and compare against reference values.
  - Load same tensor from ONNX model in Python and compare.
  - Acceptance: Q4 dequant values match ONNX within 0.5 (Q4 precision).
  - Dependencies: none.

- [ ] T1d.3 Verify ZMF graph wiring for Gemma 3  Owner: TBD  Est: 2h
  - Trace through BuildFromZMF to verify the computation graph.
  - Check: embedding lookup + scale, RMSNorm, attention Q/K/V/O projections,
    RoPE frequencies, GQA head splitting, FFN SwiGLU, final norm, LM head.
  - Compare graph structure against Gemma 3 HuggingFace reference.
  - Acceptance: Graph matches expected Gemma 3 architecture.
  - Dependencies: none.

- [ ] T1d.4 Re-export model from ONNX with latest zonnx  Owner: TBD  Est: 1h
  - Run zonnx quantizer on fresh ONNX export.
  - Compare outputs between old and new ZMF files.
  - Acceptance: New ZMF produces correct logits for "The capital of France is".
  - Dependencies: T1d.2.

- [ ] T1d.5 End-to-end correctness validation  Owner: TBD  Est: 30m
  - Generate 16 tokens with temp=0. Output must be coherent English.
  - Acceptance: Output is semantically correct for 3 test prompts.
  - Dependencies: T1d.1, T1d.3, T1d.4.

### E2: GPU KV Cache Integration (Priority 1)

- [ ] T2.1 Update tryCompileMegakernel for GPU KV cache  Owner: TBD  Est: 2h
  - Detect KVCache* ops in traced instruction tape.
  - Allocate GPUKVCache persistent memory buffer.
  - Pass device pointers to megakernel Launch().
  - Pass seq_pos from Go KV cache state.
  - Key files: generate/megakernel.go, graph/compile.go.
  - Acceptance: Megakernel accepts KV cache ops without panic.
  - Dependencies: T1.4 (megakernel must compile).

- [ ] S2.1.1 Unit test for GPU KV cache allocation  Owner: TBD  Est: 45m
  - Test GPUKVCache allocation, pointer passing, and cleanup.
  - Dependencies: T2.1.

- [ ] T2.2 End-to-end megakernel correctness test  Owner: TBD  Est: 2h
  - Load Gemma 3 2B Q4 on DGX Spark.
  - Generate 50 tokens with megakernel path.
  - Generate 50 tokens with plan.Run() path (reference).
  - Compare output token sequences. Max delta < 1e-3.
  - Acceptance: Outputs match. Test script committed.
  - Dependencies: T2.1.

- [ ] S2.2.1 KV cache 50-token generation test  Owner: TBD  Est: 1.5h
  - Generate 50 tokens using megakernel with KV cache active.
  - Verify KV cache state updates correctly across tokens.
  - Dependencies: T2.2.

- [ ] T2.3 Run golangci-lint on generate/ and graph/  Owner: TBD  Est: 15m
  - Dependencies: T2.2.

### E3: Megakernel Profiling and Tuning (Priority 2)

- [ ] T3.1 Profile megakernel with nsys on DGX Spark  Owner: TBD  Est: 2h
  - Run nsys profile on 64-token generation.
  - Identify: kernel duration, register usage, memory stalls, occupancy.
  - Export nsys report as text summary.
  - Key command: nsys profile --stats=true ./bench_tps ...
  - Acceptance: Profile report with top-10 bottleneck list.
  - Dependencies: E2 complete (megakernel fires with KV cache).

- [ ] T3.2 Fix register spilling if detected  Owner: TBD  Est: 3h
  - If nvcc reports register spilling: tile hidden_dim operations.
  - Use shared memory for intermediate results instead of registers.
  - Approach: split 2048-wide ops into 4x512 tiles with syncthreads.
  - Acceptance: No register spilling in nvcc --ptxas-options=-v output.
  - Dependencies: T3.1.
  - Risk: R92 -- register pressure with hidden_dim=2048. Mitigation: tile.

- [ ] T3.3 Implement persistent thread decode loop  Owner: TBD  Est: 3h
  - Replace per-token kernel re-launch with persistent threads.
  - Thread block stays resident across tokens, reducing launch overhead.
  - Key file: generate/megakernel.go (CUDA emitter).
  - Acceptance: Single kernel launch covers N-token generation.
  - Dependencies: T3.1.

- [ ] S3.3.1 Benchmark persistent thread vs re-launch  Owner: TBD  Est: 1h
  - Compare tok/s for both approaches on 64-token generation.
  - Dependencies: T3.3.

- [ ] T3.4 Add async prefetch for KV cache blocks  Owner: TBD  Est: 2h
  - Use CUDA async memcpy to prefetch next KV block while computing current.
  - Key files: internal/cuda/runtime.go (async copy), generate/megakernel.go.
  - Acceptance: KV cache reads overlap with compute in nsys timeline.
  - Dependencies: T3.1.

- [ ] S3.4.1 Measure KV prefetch bandwidth improvement  Owner: TBD  Est: 30m
  - Compare memory stall time before and after prefetch.
  - Dependencies: T3.4.

- [ ] T3.5 Tune thread block dimensions per-op  Owner: TBD  Est: 2h
  - Profile each emitted op (MatMul, RMSNorm, Softmax, etc.) separately.
  - Adjust block size (32/64/128/256) per op type for optimal occupancy.
  - Key file: generate/megakernel.go (grid/block config).
  - Acceptance: Each op achieves >= 50% theoretical occupancy.
  - Dependencies: T3.1.

- [ ] S3.5.1 Occupancy report per op type  Owner: TBD  Est: 30m
  - Document achieved occupancy for each op in megakernel.
  - Dependencies: T3.5.

- [ ] T3.6 Benchmark megakernel after tuning  Owner: TBD  Est: 1h
  - Run bench_tps 3x, record median tok/s.
  - Compare against pre-tuning baseline.
  - Acceptance: >= 20 tok/s or documented reason why not.
  - Dependencies: T3.2, T3.3, T3.4, T3.5.

- [ ] T3.7 Run golangci-lint on generate/, internal/cuda/  Owner: TBD  Est: 15m
  - Dependencies: T3.6.

### E4: Operator Fusion (Priority 2)

- [ ] T4.1 Fuse RMSNorm + Linear  Owner: TBD  Est: 3h
  - Emit single kernel: normalize then multiply by weight matrix.
  - Eliminates intermediate tensor write between RMSNorm and Linear.
  - Key insight: RMSNorm output is consumed once by the next Linear.
  - Detect pattern in traced tape: RMSNorm -> MatMul with weight constant.
  - Key files: generate/megakernel.go, internal/cuda/kernels/.
  - Acceptance: Fused kernel produces same output as separate ops (max delta 1e-5).
  - Dependencies: E2 complete.

- [ ] S4.1.1 Parity test: fused RMSNorm+Linear vs separate  Owner: TBD  Est: 1h
  - Random input tensor, compare outputs.
  - Dependencies: T4.1.

- [ ] T4.2 Fuse SiLU + Mul gate  Owner: TBD  Est: 2h
  - SwiGLU pattern: silu(x) * gate(x). Fuse into single elementwise kernel.
  - Detect pattern in traced tape: Sigmoid -> Mul -> Mul (gate).
  - Key files: generate/megakernel.go.
  - Acceptance: Fused kernel matches separate ops. One fewer kernel launch.
  - Dependencies: E2 complete.

- [ ] S4.2.1 Parity test: fused SiLU+Mul vs separate  Owner: TBD  Est: 45m
  - Dependencies: T4.2.

- [ ] T4.3 Fuse Softmax + Scale in attention  Owner: TBD  Est: 2h
  - Attention pattern: softmax(QK / sqrt(d_k)). Fuse scale into softmax kernel.
  - Key files: internal/cuda/kernels/flash_attention.go.
  - Acceptance: Fused kernel output matches within 1e-5.
  - Dependencies: E2 complete.

- [ ] S4.3.1 Parity test: fused attention softmax vs separate  Owner: TBD  Est: 45m
  - Dependencies: T4.3.

- [ ] T4.4 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Dependencies: T4.1, T4.2, T4.3.

### E5: Fused Attention Kernel (Priority 2)

- [ ] T5.1 Implement Flash Attention v2 decode kernel  Owner: TBD  Est: 4h
  - Single kernel for decode attention: Q*K^T / sqrt(d) -> softmax -> *V.
  - Input: Q [1, heads, 1, d_k], K_cache [1, heads, seq, d_k], V_cache [1, heads, seq, d_k].
  - Use online softmax (running max + sum) to avoid materializing full QK matrix.
  - Tile over seq dimension in blocks of 64.
  - Key files: internal/cuda/kernels/flash_attention.go (extend existing).
  - Acceptance: Output matches reference attention within 1e-4. No QK materialization.
  - Dependencies: E2 complete (KV cache working).
  - Decision rationale: docs/adr/010-cutlass-flash-attention.md

- [ ] S5.1.1 Flash Attention v2 correctness test  Owner: TBD  Est: 1.5h
  - Test with seq_len=1, 64, 256, 512. Compare with naive attention.
  - Dependencies: T5.1.

- [ ] T5.2 Wire fused attention into megakernel emitter  Owner: TBD  Est: 2h
  - Detect attention pattern in traced tape.
  - Replace Q*K^T, softmax, *V sequence with fused attention call.
  - Key files: generate/megakernel.go.
  - Acceptance: Megakernel uses fused attention. Fewer kernel launches in nsys.
  - Dependencies: T5.1, E3 complete.

- [ ] S5.2.1 Benchmark fused vs unfused attention  Owner: TBD  Est: 1h
  - Compare tok/s and memory bandwidth utilization.
  - Dependencies: T5.2.

- [ ] T5.3 Run golangci-lint  Owner: TBD  Est: 15m
  - Dependencies: T5.2.

### E6: Weight Prefetching (Priority 3)

- [ ] T6.1 Implement double-buffered weight loading  Owner: TBD  Est: 3h
  - While layer N computes, async-load layer N+1 weights from device memory.
  - Use two CUDA streams: compute stream + memory stream.
  - Key insight: Q4 weights are 1.5GB total, 26 layers ~58MB each.
  - Key files: compute/gpu_engine.go, internal/cuda/runtime.go.
  - Acceptance: nsys shows overlapped weight load and compute.
  - Dependencies: E3 complete (profiling done).

- [ ] S6.1.1 Measure weight prefetch overlap  Owner: TBD  Est: 30m
  - Verify in nsys that memory and compute streams overlap.
  - Dependencies: T6.1.

- [ ] T6.2 Run golangci-lint  Owner: TBD  Est: 15m
  - Dependencies: T6.1.

### E7: FP16 Intermediates (Priority 3)

- [ ] T7.1 Add FP16 accumulation mode for GPU intermediates  Owner: TBD  Est: 3h
  - Store intermediate activations as float16 instead of float32.
  - Reduces memory traffic by 2x for all elementwise and reduction ops.
  - Keep MatMul accumulation in float32 for numerical stability.
  - Key files: compute/gpu_engine.go, internal/cuda/kernels/.
  - Acceptance: Inference output within 1e-3 of float32 baseline.
  - Dependencies: E3 complete.
  - Risk: Numerical instability in long sequences. Mitigation: keep MatMul in F32.

- [ ] S7.1.1 Numerical parity test: FP16 vs FP32 intermediates  Owner: TBD  Est: 1.5h
  - 50-token generation, compare outputs.
  - Dependencies: T7.1.

- [ ] T7.2 Run golangci-lint  Owner: TBD  Est: 15m
  - Dependencies: T7.1.

### E8: Final Benchmark and Report (Priority 1)

- [ ] T8.1 Run comprehensive benchmark suite  Owner: TBD  Est: 2h
  - Configurations: CPU-only, GPU baseline, GPU megakernel, GPU fused.
  - Token counts: 16, 64, 256.
  - 3 runs each, report median and p95.
  - Acceptance: All configs benchmarked. Results in markdown table.
  - Dependencies: E3, E4, E5 complete.

- [ ] T8.2 Compare against Phase 34 baselines  Owner: TBD  Est: 30m
  - Document speedup ratios for each config.
  - Dependencies: T8.1.

- [ ] T8.3 Write benchmark report  Owner: TBD  Est: 1h
  - Summary of all optimizations and their individual impact.
  - File: docs/phase35-benchmarks.md (temporary, will be trimmed later).
  - Dependencies: T8.2.

---

## 4. Parallel Work

| Track | Epics | Tasks | Notes |
|-------|-------|-------|-------|
| Track A: Land Phase 34 | E1 | T1.1-T1.5 | Must complete first |
| Track B: KV Cache | E2 | T2.1-T2.3 | Depends on T1.4 (megakernel compiles) |
| Track C: Megakernel Tuning | E3 | T3.1-T3.7 | Depends on E2 (KV cache working) |
| Track D: Operator Fusion | E4 | T4.1-T4.4 | Can run in parallel with E3 after E2 |
| Track E: Fused Attention | E5 | T5.1-T5.3 | Can run in parallel with E3/E4 after E2 |
| Track F: Weight Prefetch | E6 | T6.1-T6.2 | Depends on E3 (profiling done) |
| Track G: FP16 Intermediates | E7 | T7.1-T7.2 | Depends on E3 (profiling done) |
| Track H: Final Benchmark | E8 | T8.1-T8.3 | After E3, E4, E5 complete |

### Sync Points

1. **After E1:** Branch merged, DGX Spark validated. Unlocks E2.
2. **After E2:** Megakernel fires with KV cache. Unlocks E3, E4, E5 in parallel.
3. **After E3:** Profiling done. Unlocks E6, E7 in parallel.
4. **After E3+E4+E5:** Core optimizations complete. Unlocks E8 (final benchmark).

### Parallelism Diagram

```
E1 (Land) --> E2 (KV Cache) --> E3 (Tuning) --> E6 (Prefetch) --> E8 (Bench)
                            |               --> E7 (FP16)     /
                            +--> E4 (Fusion) ----------------/
                            +--> E5 (Flash Attn) -----------/
```

---

## 5. Timeline and Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|----|-----------|-------------|---------------|
| M1 | Phase 34 landed | E1 | Branch merged, CPU >= 8 tok/s validated |
| M2 | Megakernel fires | E2 | 50-token generation matches plan.Run() |
| M3 | 20 tok/s GPU | E3, E4, E5 | bench_tps median >= 20 tok/s on DGX Spark |
| M4 | Bandwidth optimized | E6, E7 | >= 60% memory bandwidth utilization |
| M5 | Phase 35 complete | E8 | Benchmark report published, all tests pass |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Register pressure: hidden_dim=2048 prevents megakernel compilation | Must tile ops, complexity increases | High | Profile with nvcc --ptxas-options=-v. Tile into 4x512 blocks. |
| R2 | KV cache reads dominate bandwidth for long contexts | Cannot reach 60% utilization | High | Focus on short contexts (< 512). Add KV prefetch. |
| R3 | 20 tok/s not achievable with current architecture | Need architectural change | Medium | If 15+, profile. If < 15, consider separate kernel-per-layer fallback. |
| R4 | Tracing captures wrong execution path | Wrong megakernel output | Medium | Only trace decode mode (seqLen=1). Validate with plan.Run() reference. |
| R5 | Tensor identity via pointer breaks during pooling | Wrong slot wiring in megakernel | Medium | Disable TensorArena pooling during tracing phase. |
| R6 | GPU KV cache OOM for long contexts | Cannot generate > 512 tokens | Low | Default 512 tokens (~104MB). Document limit. |
| R7 | FP16 intermediates cause numerical drift | Output diverges from reference | Low | Keep MatMul in F32. Test with 50+ token sequences. |
| R8 | NEON kernels cause test failures on non-ARM64 | CI breaks on x86 runners | Low | Generic fallbacks exist for all NEON functions. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. Code compiles and passes go test ./... (with cuda tag on DGX Spark).
2. Acceptance criteria met and documented.
3. golangci-lint passes on modified packages.
4. Changes committed with Conventional Commits format.
5. Single directory per commit (pre-commit hook enforced).

### Review and QA

- Parity tests required for all fused kernels (max delta specified per task).
- Benchmark results recorded before and after each optimization.
- nsys profiles saved for megakernel tuning tasks.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits: feat(), fix(), perf(), test(), docs().
- Run golangci-lint before committing.
- Make many small logical commits, not large monolithic ones.

---

## 8. Progress Log

### Change Summary -- 2026-03-09 (update 2)

Megakernel CUDA memory errors (E1c) fully resolved. Six bugs fixed across
optable.go, emit.go, megakernel.go, compile.go, generator.go. Commits:
6b5b8d0, fad9935, 1aafa02. Megakernel now launches without crashes on DGX
Spark at 3.78 tok/s.

Critical finding: ALL execution paths produce garbage output. Root cause is
NOT the megakernel -- the bug exists at commits as far back as a9e6fb1
(Phase 34 Track A validation). F32 model also produces garbage. Created new
critical epic E1d to investigate and fix model output correctness.

Key debug data:
- "The capital of France is" -> top logit "disturbances" (Q4), "The" (F32)
- "Paris" token logit = -1.61 (should be top-1 or top-3)
- Arena disabled: still garbage (rules out TensorArena)
- model_type "gemma3_text" not registered (falls to fallback parser)

Updated E1b (all emitters done), added E1c (all memory fixes done), added
E1d (correctness investigation, new priority-0 blocker).

### Change Summary -- 2026-03-09

New plan created for Phase 35: Close the Performance Gap. Replaced Phase 34
documentation cleanup plan (E1-E6, all completed 2026-03-09). Previous plan
tasks T1.1-T6.2 were all completed and committed (see git log 5eabd34..6326107).

Preserved pending Phase 34 work as E1 (land branch) and E2 (KV cache integration,
previously T100.2-T100.3). Added new epics E3-E8 for megakernel tuning, operator
fusion, fused attention, weight prefetching, FP16 intermediates, and final
benchmarking.

Trimmed completed epics: E1-E6 (doc cleanup and code commits) removed entirely.
Stable knowledge preserved in docs/design.md.

---

## 9. Hand-off Notes

### For a New Contributor

- **Branch:** feat/neon-softmax contains all Phase 34 work (tracing compiler,
  NEON SIMD kernels, composition fixes). Must be merged to main first (E1).
- **DGX Spark:** SSH ndungu@192.168.86.250. Go at /usr/local/go/bin/go.
  CUDA at /usr/local/cuda. Build: make CUDA_ARCH=sm_121.
- **Models:** Q4 at ~/models/gemma3-q4/model.zmf, F32 at ~/models/gemma3/model.zmf.
- **PR workflow:** PRs go to zerfoo/zerfoo (upstream), not dndungu/zerfoo (fork).
  Use: gh pr create --repo zerfoo/zerfoo --head dndungu:<branch>
- **Key docs:** docs/design.md (architecture), docs/adr/ (29 ADRs covering all
  major decisions), docs/gpu.md (CUDA implementation details).
- **Megakernel:** generate/megakernel.go emits a single CUDA kernel from traced
  instruction tape. See docs/adr/026-megakernel-decode.md and
  docs/adr/028-tracing-compiler.md.
- **Critical bug:** All execution paths produce garbage output. Predates
  megakernel work. Root cause unknown -- see E1d for investigation plan.
  Both Q4 and F32 models affected. Not arena-related.
- **model_type mismatch:** Config says "gemma3_text" but only "gemma3" is
  registered. May affect inference.Load config parsing.
- **NEON kernels:** internal/xblas/ contains ARM64 plan9 assembly for softmax,
  RMSNorm, RoPE, SiLU, exp, elementwise, scalar ops.
  See docs/adr/029-neon-simd-cpu-acceleration.md.

### Deferred to Phase 36

- Prefill optimization (batched tiled attention for prompt ingestion).
- purego CUDA dlopen migration (replace CGo runtime functions).
- Multi-GPU tensor parallelism.
- Q8/FP8 inference quantization formats.

---

## 10. Appendix

### Key File Paths

| Path | Purpose |
|------|---------|
| generate/megakernel.go | Megakernel CUDA codegen and launch |
| graph/compile.go | Tracing compiler, CompileTraced |
| compute/cpu_engine.go | CPU engine (1886 lines, all ops) |
| compute/gpu_engine.go | GPU engine with CUDA dispatch |
| internal/cuda/kernels/ | CUDA kernel sources and Go wrappers |
| internal/xblas/ | ARM64 NEON assembly kernels |
| internal/cuda/runtime.go | CUDA runtime bindings |
| layers/attention/ | Attention layer implementations |

### Existing ADRs (relevant to Phase 35)

| ADR | Topic |
|-----|-------|
| docs/adr/010-cutlass-flash-attention.md | Flash Attention design |
| docs/adr/022-gpu-first-inference-pipeline.md | GPU-first inference |
| docs/adr/024-cuda-graph-fused-kernels.md | CUDA graph capture |
| docs/adr/026-megakernel-decode.md | Megakernel design |
| docs/adr/028-tracing-compiler.md | Tracing compiler |
| docs/adr/029-neon-simd-cpu-acceleration.md | NEON SIMD kernels |
