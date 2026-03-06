# Phase 27 -- Inference Throughput

## 1. Context

### Problem Statement

Phase 26 validated end-to-end Q4 inference on Gemma 3 2B: 3.60 tok/s on
DGX Spark ARM64 CPU. The target is >= 15 tok/s CPU, >= 60 tok/s GPU.
There is a 4.2x gap on CPU and GPU inference has not been validated at all.

CPU profile breakdown (after Phase 26 blocked 2D transpose optimization):

| Component | % CPU | Notes |
|-----------|-------|-------|
| Transpose (3D/4D) | 62% | Attention permute patterns, weight transposes on every forward pass |
| GEMM (sgemmAccRowNeon) | 16% | Actual compute -- the useful work |
| binaryOp | 3% | Element-wise add/mul |
| GC / malloc | 5% | 2.5M allocs per generation, 39GB total |
| Other | 14% | Model loading, tokenization, decoding |

Key bottlenecks in priority order:

1. **Redundant weight transposes.** Weight matrices are transposed on every
   forward pass even though weights never change. The ONNX-to-ZMF conversion
   emits explicit Transpose nodes for weight layout. Folding these at load
   time eliminates 62% of CPU time.

2. **Excessive allocation.** Every forward pass creates new tensors for all
   intermediate results (2.5M allocs for 32 tokens). A tensor arena that
   reuses buffers by shape eliminates GC pressure and cache thrashing.

3. **No GPU inference path.** The CUDA backend has cuBLAS SGEMM and a custom
   Q4 GEMM kernel (2383 GFLOPS) but has never been wired into end-to-end
   model inference. GPU should yield 10-50x over CPU.

4. **No GGUF end-to-end.** The GGUF parser/loader (E37) can read weights and
   metadata but cannot build a computation graph. Architecture-specific
   graph template builders are needed to run inference from GGUF files.

5. **No operator fusion.** Common multi-op patterns (RMSNorm, RoPE,
   SiLU-gate) execute as separate ops with intermediate tensor
   materialization.

### Phase 26 Summary (COMPLETE)

| Epic | Result |
|------|--------|
| E34: PagedAttention | Block pool + PagedKVCache + Generator integration |
| E35: Speculative Decoding | SpeculativeGenerator with adaptive draft length |
| E36: End-to-End Q4 Pipeline | Gemma 3 2B Q4: 1.96 -> 3.60 tok/s (1.84x via blocked transpose) |
| E37: GGUF Model Import | Parser + loader + arch mapping (llama/gemma) |
| E38: Performance CI Dashboard | bench.sh + GH Actions workflow + DGX GPU job |

### Objectives

- O51: Gemma 3 2B Q4 >= 15 tok/s on DGX Spark CPU (ARM64).
- O52: Gemma 3 2B Q4 >= 60 tok/s on DGX Spark GPU (GB10).
- O53: Load and run inference from a GGUF file without any external conversion.
- O54: Decode loop allocation < 100 allocs/token.
- O55: Fused RMSNorm, RoPE, and SiLU-gate kernels.

### Non-Goals

- Training performance (out of scope for this phase).
- New quantization formats (Q4_K, Q5_K, Q6_K). Phase 28 candidate.
- Multi-GPU inference. Requires second DGX Spark unit.
- Vision/multimodal models. Text-only focus.

### Success Metrics

| Metric | Phase 26 Baseline | Phase 27 Target |
|--------|-------------------|-----------------|
| CPU tok/s (Gemma 3 2B Q4) | 3.60 | >= 15 |
| GPU tok/s (Gemma 3 2B Q4) | untested | >= 60 |
| Allocs per token | ~80,000 | < 100 |
| Transpose % of CPU | 62% | < 5% |
| GGUF end-to-end | parser only | full inference |

---

## 2. Scope and Deliverables

### In Scope

- Weight transpose folding at model load time.
- 3D/4D blocked transpose fast paths for attention patterns.
- Tensor arena / buffer pool for zero-alloc forward passes.
- GPU inference pipeline validation and optimization.
- GGUF architecture-specific graph template builders (Llama, Gemma).
- GGUF tokenizer loading.
- Operator fusion: RMSNorm, RoPE, SiLU-gate.

### Out of Scope

- New quantization formats beyond Q4_0/Q8_0.
- Multi-GPU inference orchestration.
- Vision encoder / multimodal pipelines.
- Training loop optimizations.
- Mobile / WebAssembly targets.

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|---------------------|
| D81 | Weight transpose elimination | Transpose < 5% of CPU profile |
| D82 | Tensor arena | < 100 allocs/token in decode loop |
| D83 | GPU inference pipeline | >= 60 tok/s on Gemma 3 2B Q4 (GB10) |
| D84 | GGUF end-to-end inference | Load GGUF, generate text, no external tools |
| D85 | Fused operators | RMSNorm, RoPE, SiLU-gate as single-pass kernels |

---

## 3. Checkable Work Breakdown

### E39: Eliminate Redundant Transposes (O51)

- [ ] T39.1 Fold weight transposes at model load time  Owner: TBD  Est: 4h
  - In `model.BuildFromZMF`, detect patterns where a Transpose node has a
    constant weight parameter as input and its only consumer is MatMul.
  - Pre-transpose the weight data in-place and remove the Transpose node
    from the graph. Update the MatMul input to point directly to the
    transposed weight.
  - Handle both `axes=[1,0]` (2D) and `axes=[0,2,1,3]` style permutations.
  - Acceptance: Transpose nodes with constant inputs are eliminated from
    the built graph. Forward pass produces identical output (within fp
    tolerance) before and after folding.
  - Dependencies: none.

- [ ] S39.1.1 Unit tests for transpose folding  Owner: TBD  Est: 1h
  - Test: graph with Transpose(constant) -> MatMul folds correctly.
  - Test: graph with Transpose(dynamic_input) is NOT folded.
  - Test: output matches unfolded graph within 1e-6 tolerance.

- [ ] T39.2 Fast path for 3D/4D attention transposes  Owner: TBD  Est: 3h
  - Add blocked transpose for the common attention patterns:
    - `[B,S,H,D] -> [B,H,S,D]` (axes=[0,2,1,3]) -- head-first permute.
    - `[B,H,S,D] -> [B,S,H,D]` (axes=[0,2,1,3]) -- reverse.
  - Use tiled loops with cache-friendly access (extend the 2D block=64
    approach to the inner two dims).
  - Acceptance: 3D/4D transpose benchmark shows >= 3x improvement over
    generic path.
  - Dependencies: none.

- [ ] S39.2.1 Benchmark: 3D/4D transpose speedup  Owner: TBD  Est: 30m

- [ ] T39.3 End-to-end benchmark after transpose elimination  Owner: TBD  Est: 1h
  - Run Gemma 3 2B Q4 with transpose folding enabled.
  - Profile with pprof to verify Transpose is < 5% of CPU.
  - Measure tok/s improvement.
  - Acceptance: measurable speedup, Transpose no longer dominant in profile.
  - Dependencies: T39.1, T39.2.

- [ ] S39.3.1 Before/after profile comparison  Owner: TBD  Est: 30m

- [ ] T39.4 Run golangci-lint on model/ and compute/  Owner: TBD  Est: 15m

### E40: Tensor Arena / Zero-Alloc Forward (O54)

- [ ] T40.1 Design and implement TensorPool  Owner: TBD  Est: 4h
  - Create `compute/pool.go` with `TensorPool[T]` type.
  - Pool strategy: map from shape hash to free list of `*TensorNumeric[T]`.
  - Methods: `Acquire(shape []int) *TensorNumeric[T]` and
    `Release(t *TensorNumeric[T])`.
  - Thread-safe (sync.Pool or mutex-guarded free lists).
  - Acceptance: unit tests show acquire-release-acquire reuses same buffer.
  - Dependencies: none.

- [ ] S40.1.1 Unit tests for TensorPool  Owner: TBD  Est: 1h

- [ ] T40.2 Wire pool into graph executor  Owner: TBD  Est: 4h
  - Modify `graph.Graph.Forward()` to accept an optional `TensorPool`.
  - After a node's output is consumed by all dependents, release its tensor
    back to the pool.
  - Use reference counting: each node output has refcount = number of
    downstream consumers. Decrement on consume, release at zero.
  - Acceptance: decode loop allocs drop by >= 100x. Existing tests pass.
  - Dependencies: T40.1.

- [ ] S40.2.1 Allocation benchmark: before/after pool  Owner: TBD  Est: 30m

- [ ] T40.3 Benchmark tok/s with pool enabled  Owner: TBD  Est: 1h
  - Run Gemma 3 2B Q4 with tensor pool.
  - Verify tok/s improvement from reduced GC pressure.
  - Acceptance: < 100 allocs/token. Measurable tok/s improvement.
  - Dependencies: T40.2.

- [ ] S40.3.1 Allocation profile comparison  Owner: TBD  Est: 30m

- [ ] T40.4 Run golangci-lint on compute/ and graph/  Owner: TBD  Est: 15m

### E41: GPU Inference Pipeline (O52)

- [ ] T41.1 Validate GPU forward pass on Gemma 3 2B  Owner: TBD  Est: 4h
  - Load Gemma 3 2B Q4 ZMF with `inference.Load(..., WithDevice("cuda"))`.
  - Run single forward pass. Verify output shape and non-NaN values.
  - Identify and fix any ops that fall back to CPU (log warnings for
    unsupported GPU ops).
  - Acceptance: forward pass completes on GPU without CPU fallback for
    core ops (MatMul, Add, Mul, Softmax, RMSNorm, RoPE, Gather).
  - Dependencies: none.
  - Risk: some ops may not have GPU implementations yet.

- [ ] S41.1.1 GPU forward pass smoke test  Owner: TBD  Est: 1h

- [ ] T41.2 Profile and fix GPU bottlenecks  Owner: TBD  Est: 4h
  - Profile with CUDA events or nsys.
  - Typical issues: excessive CPU-GPU synchronization, small kernel launches,
    data transfers for unsupported ops, sub-optimal memory layout.
  - Fix top 3 bottlenecks.
  - Acceptance: GPU profile shows > 80% time in compute kernels.
  - Dependencies: T41.1.

- [ ] S41.2.1 GPU profile analysis  Owner: TBD  Est: 1h

- [ ] T41.3 GPU generation benchmark  Owner: TBD  Est: 2h
  - Run full generation (32 tokens) on GPU.
  - Measure tok/s. Target >= 60 tok/s.
  - Compare with CPU baseline.
  - Acceptance: GPU tok/s >= 10x CPU tok/s.
  - Dependencies: T41.2.

- [ ] S41.3.1 CPU vs GPU benchmark report  Owner: TBD  Est: 30m

- [ ] T41.4 Run golangci-lint on compute/ and inference/  Owner: TBD  Est: 15m

### E42: GGUF End-to-End Inference (O53)

- [ ] T42.1 Graph template builder for Llama architecture  Owner: TBD  Est: 6h
  - Create `inference/gguf_builder.go` with
    `BuildLlamaGraph(model *GGUFModel, engine Engine) (*graph.Graph, error)`.
  - Build the standard Llama graph: Embed -> [RMSNorm -> Attn -> RMSNorm ->
    MLP] x N -> RMSNorm -> LMHead.
  - Map GGUF tensor names (already canonical from E37 name mapping) to
    graph node parameters.
  - Support GQA (num_kv_heads < num_heads).
  - Acceptance: Llama 3.2 1B GGUF loads and produces non-NaN logits.
  - Dependencies: E37.

- [ ] S42.1.1 Llama GGUF forward pass test  Owner: TBD  Est: 1h

- [ ] T42.2 Graph template builder for Gemma architecture  Owner: TBD  Est: 3h
  - Extend for Gemma: shared embed/lm_head weight, GeGLU activation
    (GeLU instead of SiLU), embedding scaling by sqrt(hidden_size).
  - Acceptance: Gemma 3 2B GGUF loads and produces non-NaN logits.
  - Dependencies: T42.1.

- [ ] S42.2.1 Gemma GGUF forward pass test  Owner: TBD  Est: 1h

- [ ] T42.3 Tokenizer loading for GGUF models  Owner: TBD  Est: 3h
  - Extract tokenizer vocabulary from GGUF metadata (tokenizer.ggml.tokens,
    tokenizer.ggml.scores, tokenizer.ggml.merges).
  - Build a BPETokenizer from GGUF metadata without needing tokenizer.json.
  - Fallback: if GGUF tokenizer data is absent, look for tokenizer.json in
    the same directory as the GGUF file.
  - Acceptance: tokenizer encodes/decodes correctly for Llama and Gemma.
  - Dependencies: none.

- [ ] S42.3.1 GGUF tokenizer encode/decode tests  Owner: TBD  Est: 1h

- [ ] T42.4 Unified GGUF load function  Owner: TBD  Est: 2h
  - Create `inference.LoadGGUFModel(path string, opts ...Option) (*Model, error)`.
  - Dispatches to architecture-specific builder based on GGUF metadata
    `general.architecture`.
  - Wires tokenizer, graph, engine, and generator into a ready-to-use Model.
  - Acceptance: `LoadGGUFModel("model.gguf")` -> `model.Generate(ctx, prompt)`.
  - Dependencies: T42.1, T42.2, T42.3.

- [ ] S42.4.1 End-to-end GGUF generation test  Owner: TBD  Est: 1h

- [ ] T42.5 Run golangci-lint on inference/ and model/gguf/  Owner: TBD  Est: 15m

### E43: Operator Fusion (O55)

- [ ] T43.1 Fused RMSNorm kernel  Owner: TBD  Est: 3h
  - Create `compute/fused_rmsnorm.go` that computes
    `x * rsqrt(mean(x^2) + eps) * weight` in a single pass over the data.
  - Avoids materializing the squared, mean, and rsqrt intermediate tensors.
  - Wire into the RMSNorm layer as an optimized path when engine supports it.
  - Acceptance: output matches unfused RMSNorm within 1e-5. Benchmark shows
    >= 2x improvement for typical hidden sizes (1152, 2048, 4096).
  - Dependencies: none.

- [ ] S43.1.1 Fused RMSNorm correctness and benchmark tests  Owner: TBD  Est: 1h

- [ ] T43.2 Fused RoPE kernel  Owner: TBD  Est: 3h
  - Create `compute/fused_rope.go` that applies rotary position embeddings
    in a single pass: interleave cos/sin multiply and rotate in one loop.
  - Avoid creating separate cos, sin, split, and rotate intermediate tensors.
  - Acceptance: output matches unfused RoPE within 1e-5. Benchmark shows
    >= 2x improvement.
  - Dependencies: none.

- [ ] S43.2.1 Fused RoPE correctness and benchmark tests  Owner: TBD  Est: 1h

- [ ] T43.3 Fused SiLU-gate kernel  Owner: TBD  Est: 2h
  - Create `compute/fused_silugate.go` that computes
    `silu(gate_proj(x)) * up_proj(x)` as a single element-wise pass after
    the two MatMuls.
  - Fuses: silu activation + element-wise multiply into one kernel.
  - Acceptance: output matches unfused path within 1e-5. Benchmark shows
    improvement for MLP forward.
  - Dependencies: none.

- [ ] S43.3.1 Fused SiLU-gate correctness and benchmark tests  Owner: TBD  Est: 1h

- [ ] T43.4 End-to-end benchmark with all fusions enabled  Owner: TBD  Est: 1h
  - Run Gemma 3 2B Q4 with all fusions + transpose folding + tensor pool.
  - Measure final tok/s. Compare against Phase 26 baseline (3.60 tok/s).
  - Acceptance: >= 15 tok/s on DGX Spark CPU.
  - Dependencies: T43.1, T43.2, T43.3, E39, E40.

- [ ] S43.4.1 Final performance report  Owner: TBD  Est: 30m

- [ ] T43.5 Run golangci-lint on compute/  Owner: TBD  Est: 15m

---

## 4. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M11: Zero-transpose inference | E39 | none | Transpose < 5% of CPU, >= 8 tok/s |
| M12: Zero-alloc decode | E40 | E39 | < 100 allocs/token, >= 10 tok/s |
| M13: GPU inference | E41 | none | >= 60 tok/s on GB10 |
| M14: GGUF ecosystem | E42 | E37 | GGUF load -> generate, no external tools |
| M15: Fused ops | E43 | E39, E40 | >= 15 tok/s CPU with all optimizations |

Recommended execution order:

1. **E39** -- transpose elimination (biggest single-item speedup, purely mechanical)
2. **E40** -- tensor arena (allocation reduction, multiplicative with E39)
3. **[E41, E42]** -- GPU pipeline + GGUF builders (independent, parallelizable)
4. **E43** -- operator fusion (polish, benefits from profiling after E39+E40)

---

## 5. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R51 | Transpose folding changes graph semantics for edge cases | Incorrect output | Medium | Conservative pattern matching: only fold when Transpose input is a graph parameter with no other consumers. Full parity test suite validates. |
| R52 | Tensor pool introduces use-after-release bugs | Data corruption, flaky tests | Medium | Reference counting with debug mode that poisons released buffers. Race detector in CI. |
| R53 | GPU ops missing for some ONNX-derived nodes | Fallback to CPU kills throughput | High | Audit all node types in Gemma 3 graph before starting. Implement missing GPU ops first. |
| R54 | GGUF tokenizer metadata varies across model families | Tokenizer fails for some models | Medium | Start with Llama (well-documented). Test against HuggingFace tokenizer.json as ground truth. |
| R55 | Fused kernels diverge numerically from unfused path | Parity test failures | Low | Use Kahan summation for RMSNorm. Accept 1e-5 tolerance for fused vs unfused. |
| R56 | 3D/4D transpose patterns vary by architecture | Fast path misses common cases | Medium | Profile both Llama and Gemma to catalog all transpose axis patterns before implementing. |

---

## 6. Operating Procedure

Same as Phase 26 (see docs/plan.md section 6). Key reminders:

- TDD: write tests first, then implement.
- Single-directory commits. Pre-commit hook enforces this.
- `golangci-lint run` must report 0 issues before marking a task complete.
- Benchmarks with `-count=3` minimum for statistical validity.
- Profile before and after every optimization with pprof.

---

## 7. Progress Log

### 2026-03-05

Phase 27 plan created. Five epics (E39-E43) targeting 4.2x CPU throughput
improvement and GPU inference enablement. Baseline: 3.60 tok/s CPU (Gemma 3
2B Q4, DGX Spark ARM64).

---

## 8. Hand-off Notes

### For a New Contributor

- **Read first:** docs/design.md for architecture, docs/plan.md for Phase 25-26
  history, this file for Phase 27 scope.
- **Baseline numbers:** 3.60 tok/s CPU, untested GPU. Profile at
  `/tmp/q4_cpu2.prof` on DGX Spark.
- **DGX Spark:** `ssh ndungu@192.168.86.250`, Go at `/usr/local/go/bin/go`.
  Models at `~/models/gemma3-q4/` (Q4) and `~/models/gemma3/` (F32).
  zonnx at `~/zonnx/`, zmf at `~/zmf/` (fix/attribute-tensor branch).
- **Key files for E39:** `model/builder.go` (BuildFromZMF),
  `compute/cpu_engine.go` (Transpose), `graph/graph.go` (Forward).
- **Key files for E40:** `compute/engine.go` (Engine interface),
  `graph/graph.go` (node execution loop).
- **Key files for E41:** `compute/gpu_engine.go`, `internal/cuda/`,
  `inference/inference.go` (createEngine).
- **Key files for E42:** `inference/gguf.go` (LoadGGUF, GGUFModel),
  `model/gguf/arch.go` (MapTensorName, ModelConfig).

### Performance Reference

| Model | Params | Quant | Ollama CPU | Ollama GPU | Zerfoo CPU (current) | Zerfoo CPU (target) | Zerfoo GPU (target) |
|-------|--------|-------|------------|------------|---------------------|--------------------|--------------------|
| Gemma 3 2B | 2.6B | Q4_0 | ~20 | ~80 | 3.60 | >= 15 | >= 60 |
| Llama 3.2 1B | 1.2B | Q4_0 | ~40 | ~120 | untested | >= 30 | >= 100 |
