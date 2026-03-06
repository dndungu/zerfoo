# Zerfoo Development Plan -- Phase 31: Inference Validation and Training Improvements

## 1. Context

### Problem Statement

Two independent areas need work:

**Inference gaps:** Phase 26 delivered PagedAttention, speculative decoding,
GGUF import, and performance CI as scaffolding. None were validated with real
models or benchmarked end-to-end. Speculative decoding has no measured speedup.
GGUF loader was tested with synthetic files only. PagedAttention's GQA
integration uses gather-copy (Option A) instead of direct block-table reads
(Option B). GPU Q4 inference has never been measured end-to-end.

**Training gaps:** After 19 Numerai experiments on DGX Spark, the best training
val_corr is 0.010551 (Exp 15, 4-layer model). Full out-of-time validation shows
Mean CORR of only 0.002 (Sharpe 0.171). Top 20 Numerai models have mmcRep of
0.010-0.014. The 5x gap between training val_corr and full validation stems
from two structural issues: (1) val_corr oscillation -- every experiment shows
extreme oscillation between positive and negative val_corr across epochs, and
early stopping picks the unstable peak; (2) noisy targets -- the model overfits
to noise in the 40-era validation split.

See docs/design.md for full architecture context and Phases 1-30 history.

### What Was Delivered (Phases 26-30)

| Phase | Key Result |
|-------|------------|
| Phase 26 | PagedAttention (46% memory), speculative decoding scaffolding, GGUF parser/loader, performance CI workflow |
| Phase 27-28 | Transpose elimination, TensorPool, fused ops, GGUF end-to-end, model hub, chat templates, K-quant |
| Phase 29 | NEON Q4 dot product, 6.5 tok/s Gemma 3 2B Q4 |
| Phase 30 | Worker pool, graph compiler, 6.86 tok/s (5% over Phase 29). 15 tok/s NOT achieved -- GEMV kernels dominate at 74%, framework overhead is only ~5% |

### Objectives

**Inference:**

- O59: Validate speculative decoding on real models with measured tok/s speedup.
- O60: Wire PagedAttention into GQA for non-contiguous block reads (Option B)
  and benchmark multi-sequence serving.
- O61: End-to-end GGUF inference with real HuggingFace models (Llama 3.2 1B,
  Gemma 3 2B) and measured tok/s.
- O62: DGX Spark GPU inference pipeline for Q4 models via CUDA kernels.
- O63: Performance CI with regression alerting and DGX self-hosted runner.

**Training:**

- O64: Smooth val_corr oscillation via EMA of model weights.
- O65: Prevent premature early stopping via smoothed val_corr tracking.
- O66: Better loss landscape exploration via cosine warm restarts.
- O67: Flatter minima via stochastic weight averaging (SWA).
- O68: Regime-robust predictions via feature dropout.

### Non-Goals

- Training performance optimization beyond the listed improvements.
- Multi-node or multi-GPU inference.
- Pipeline or tensor parallelism.
- FP4 kernels (blocked on upstream CUTLASS SM121 fixes).
- Vulkan, SYCL, or ROCm backends.
- Q4_K_M, Q5_K, or other advanced quantization formats beyond existing K-quant.
- Prompt caching / prefix sharing.
- Vision model inference (text-only LLMs).
- Per-parameter learning rates (P3 priority, deferred).
- Gradient accumulation (P3 priority, deferred).
- GEMV kernel optimization (identified as bottleneck in Phase 30, separate phase).

### Constraints and Assumptions

- Go standard library only where possible. No cobra, viper, testify.
- Build tags for GPU code (`//go:build cuda`).
- Pre-commit hook rejects multi-directory commits.
- golangci-lint, go vet, gofmt required for all changes.
- Tests must pass with `-race` flag.
- Table-driven tests using the standard `testing` package.
- DGX Spark GB10 at `ssh ndungu@192.168.86.250` for GPU validation.
- Go 1.25.0, CUDA 13.0, sm_121 (Blackwell) on DGX Spark.
- Target models: Gemma 3 2B (ZMF + GGUF), Llama 3.2 1B (GGUF).
- Phase 30 is complete. Baseline: 6.86 tok/s Gemma 3 2B Q4 CPU ARM64.
- Training improvements target the Audacity training pipeline
  (`audacity/internal/training/`).

### Success Metrics

| Metric | Current | Target | How Measured |
|--------|---------|--------|-------------|
| Speculative decode speedup | unmeasured | >= 1.5x | E60 benchmark: speculative vs baseline tok/s |
| Multi-sequence memory (8 seqs) | unmeasured | <= 50% of pre-alloc | E59 benchmark with PagedAttention |
| Concurrent requests p99 latency | untested | < 2x of single-request | E59 load test on serve/ |
| GGUF Llama 3.2 1B tok/s (CPU) | N/A | >= 30 | E61 benchmark on Apple M-series |
| GGUF Gemma 3 2B tok/s (CPU) | N/A | >= 15 | E61 benchmark on Apple M-series |
| GPU Q4 tok/s (2B) | unmeasured | >= 60 | E62 benchmark on DGX Spark |
| CI regression detection | manual | automated PR comment | E63 workflow |
| Numerai full val CORR | 0.002 | >= 0.006 | Out-of-time validation after E64-E68 |
| Val_corr oscillation amplitude | +/-0.008 | +/-0.003 | Epoch-level val_corr std dev |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D301 | PagedAttention GQA integration + load test | Close Option B gap from E34 |
| D302 | Speculative decoding real-model validation | Close unmeasured speedup gap from E35 |
| D303 | GGUF end-to-end inference with real models | Close synthetic-only gap from E37 |
| D304 | GPU Q4 inference pipeline | First GPU end-to-end tok/s measurement |
| D305 | CI regression alerting with DGX runner | Close manual-only gap from E38 |
| D306 | EMA optimizer wrapper | Smooth val_corr oscillation |
| D307 | Smoothed early stopping | Prevent premature stopping on noise |
| D308 | Cosine warm restarts scheduler | Better loss landscape exploration |
| D309 | SWA optimizer wrapper | Flatter minima for generalization |
| D310 | Feature dropout layer | Regime-robust feature selection |

### Out of Scope

- Advanced quantization formats (Q4_K_M, Q5_K, Q6_K beyond existing).
- Prompt caching / prefix sharing.
- Multi-GPU KV cache partitioning.
- Vision encoder inference.
- Mixup data augmentation (P2 priority, deferred).
- Huber loss / multi-loss scheduling (P2 priority, deferred).
- Per-parameter learning rates (P3 priority, deferred).
- Gradient accumulation (P3 priority, deferred).

---

## 3. Checkable Work Breakdown

### E59: PagedAttention GQA Integration and Load Test (O60)

Wire PagedAttention block-table reads directly into GroupQueryAttention,
eliminating the gather-to-contiguous copy. Then validate with a multi-sequence
load test.

Existing code:
- `generate/paged_kv.go` -- PagedKVCache with Append/GetKV/Free/Truncate.
- `generate/block_pool.go` -- BlockPool with Alloc/Free/Cap.
- `generate/generator.go:58` -- WithPagedKV option.
- `layers/attention/grouped_query_attention.go` -- GQA layer.

- [x] T59.1 Add block-table KV reader to GQA  Owner: TBD  Est: 4h  (2026-03-06)
  - Modify `layers/attention/grouped_query_attention.go` to accept a
    `BlockTableReader` interface:
    ```go
    type BlockTableReader interface {
        ReadKV(layer int, seqLen int) (k, v *tensor.TensorNumeric[T])
    }
    ```
  - When `BlockTableReader` is set, GQA reads K/V directly from paged blocks
    instead of a contiguous cache tensor. Uses block table to iterate blocks
    and compute attention scores per-block.
  - Acceptance: GQA produces correct attention output with block-table reads.
    Output matches contiguous-cache path within 1e-6.
  - Dependencies: none (builds on existing PagedKVCache).

- [x] S59.1.1 GQA block-table correctness tests  Owner: TBD  Est: 2h  (2026-03-06)
  - Test: 16 tokens (1 block), compare to contiguous path.
  - Test: 33 tokens (3 blocks), verify cross-block attention is correct.
  - Test: 128 tokens (8 blocks), verify full sequence attention.
  - All tests compare block-table vs contiguous output within 1e-6.

- [ ] T59.2 Benchmark block-table vs gather-copy  Owner: TBD  Est: 2h
  - Benchmark GQA forward pass with:
    A. Gather blocks to contiguous buffer, then standard attention.
    B. Block-table direct reads (T59.1).
  - Measure: ns/op, allocs/op for sequence lengths 128, 512, 1024.
  - Acceptance: Block-table path has fewer allocs/op. ns/op comparison
    documented (may not be faster due to non-contiguous memory access).
  - Dependencies: T59.1.

- [ ] S59.2.1 Benchmark report  Owner: TBD  Est: 30m

- [ ] T59.3 Multi-sequence load test  Owner: TBD  Est: 3h
  - Create `serve/loadtest_test.go` with a test that:
    1. Starts the HTTP server with `WithPagedKV`.
    2. Sends 8 concurrent chat completion requests via HTTP.
    3. Measures p50, p95, p99 latency and peak memory.
  - Compare metrics with and without PagedAttention.
  - Acceptance: PagedAttention uses <= 50% memory of pre-allocated cache for
    8 sequences of mixed length (128-1024 tokens). p99 latency < 2x single.
  - Dependencies: T59.1.

- [ ] S59.3.1 Load test validation  Owner: TBD  Est: 1h
  - Verify all 8 requests complete successfully.
  - Verify memory measurements are consistent across 3 runs.

- [ ] T59.4 Run golangci-lint on layers/attention/ and serve/  Owner: TBD  Est: 15m
  - Dependencies: T59.3.

### E60: Speculative Decoding Real-Model Validation (O59)

The speculative decoding implementation exists (`generate/speculative.go`,
`generate/adaptive.go`) but has never been benchmarked on real models.
This epic validates the implementation and measures speedup.

Existing code:
- `generate/speculative.go` -- SpeculativeGenerator with draft/target loop.
- `generate/adaptive.go` -- adaptiveDraftLen with rolling acceptance rate.
- `serve/server.go:27-30` -- WithDraftModel server option.

- [ ] T60.1 Identify or create a draft model  Owner: TBD  Est: 3h
  - Option A: Use a smaller Gemma 3 model (if available in GGUF/ZMF).
  - Option B: Create a "tiny" model by truncating Gemma 3 2B to 2-4 layers.
    Save as ZMF. The truncated model is fast but low quality -- acceptable
    for speculative drafting.
  - Option C: Use Llama 3.2 1B as draft for a larger target (if available).
  - Acceptance: A draft model exists that runs >= 10x faster than the target.
  - Dependencies: none.

- [ ] S60.1.1 Draft model smoke test  Owner: TBD  Est: 1h
  - Load draft model, generate 20 tokens, verify no errors.
  - Measure draft-only tok/s as baseline.

- [ ] T60.2 Benchmark speculative vs baseline decode  Owner: TBD  Est: 2h
  - Run identical prompts (10 prompts, 100 tokens each) with:
    A. Standard Generator (baseline).
    B. SpeculativeGenerator with draft model from T60.1.
  - Measure: tok/s, acceptance rate, average accepted tokens per step.
  - Acceptance: Results documented. Speedup >= 1.5x if acceptance rate > 50%.
  - Dependencies: T60.1.

- [ ] S60.2.1 Benchmark report with speedup analysis  Owner: TBD  Est: 30m

- [ ] T60.3 Profile and optimize speculative overhead  Owner: TBD  Est: 3h
  - If speedup < 1.5x, profile the speculative loop:
    - Draft model forward pass cost relative to target.
    - KV cache rollback overhead.
    - Token verification loop overhead.
  - Fix top bottleneck (e.g., avoid rebuilding draft KV cache from scratch
    if only rolling back a few positions).
  - Acceptance: Measurable improvement over T60.2 baseline, or documented
    explanation if speculative approach is not beneficial for this model pair.
  - Dependencies: T60.2.

- [ ] S60.3.1 Before/after profile comparison  Owner: TBD  Est: 30m

- [ ] T60.4 Validate streaming with speculative decode  Owner: TBD  Est: 2h
  - Use `serve/` HTTP server with `WithDraftModel`.
  - Send a streaming chat completion request via SSE.
  - Verify tokens stream incrementally as they are verified (not batched).
  - Acceptance: SSE events arrive for each accepted token batch, not delayed
    until full generation completes.
  - Dependencies: T60.2.

- [ ] S60.4.1 Streaming speculative decode test  Owner: TBD  Est: 1h
  - Verify SSE event count matches number of verification rounds.
  - Verify final output matches non-streaming speculative generation.

- [ ] T60.5 Run golangci-lint on generate/ and serve/  Owner: TBD  Est: 15m
  - Dependencies: T60.4.

### E61: GGUF End-to-End Inference Validation (O61)

The GGUF parser, loader, and architecture mapping exist (`model/gguf/`), and
`inference.Load()` detects `.gguf` files. This epic validates the full pipeline
with real HuggingFace GGUF models.

Existing code:
- `model/gguf/parser.go:72` -- `Parse(r io.ReadSeeker) (*File, error)`.
- `model/gguf/loader.go` -- Tensor loading with Q4_0/Q8_0/F32/F16 support.
- `model/gguf/arch.go` -- Architecture mapping for llama/gemma.
- `model/gguf/tokenizer.go` -- GGUF tokenizer extraction.
- `inference/gguf.go` -- `LoadGGUF()` and `ToModelMetadata()`.
- `inference/inference.go:67-72` -- HuggingFace model aliases.
- `inference/inference.go:197-199` -- GGUF detection in `Load()`.

- [ ] T61.1 Download and validate Llama 3.2 1B Q4_0 GGUF  Owner: TBD  Est: 2h
  - Use `inference.Pull("llama-3-1b-q4")` to download from HuggingFace.
  - Call `inference.Load("llama-3-1b-q4")` and verify:
    - Model metadata (num_layers, hidden_size, num_heads) is correct.
    - Tokenizer loads and encodes/decodes correctly.
    - A single forward pass produces valid logits (not NaN/Inf).
  - Acceptance: Llama 3.2 1B loads from GGUF and produces valid logits.
  - Dependencies: none.

- [ ] S61.1.1 Llama 3.2 1B GGUF load test  Owner: TBD  Est: 1h
  - Verify metadata matches expected Llama 3.2 1B config.
  - Verify tokenizer roundtrip for 5 test strings.

- [ ] T61.2 Benchmark Llama 3.2 1B GGUF inference  Owner: TBD  Est: 2h
  - Generate 100 tokens with standard prompts.
  - Measure tok/s on Apple M-series CPU.
  - Compare with Gemma 3 2B ZMF performance (expect faster due to fewer params).
  - Acceptance: tok/s measured and documented. Target >= 30 tok/s on M-series.
  - Dependencies: T61.1.

- [ ] S61.2.1 Benchmark report  Owner: TBD  Est: 30m

- [ ] T61.3 Validate Gemma 3 2B GGUF inference  Owner: TBD  Est: 2h
  - Use `inference.Pull("gemma-3-2b-q4")` to download GGUF variant.
  - Load and generate tokens. Compare output quality with ZMF-loaded Gemma 3.
  - Acceptance: GGUF-loaded Gemma 3 2B produces coherent text. tok/s within
    10% of ZMF-loaded variant (mmap overhead should be similar).
  - Dependencies: T61.1.

- [ ] S61.3.1 GGUF vs ZMF output comparison  Owner: TBD  Est: 1h
  - Generate 50 tokens from same prompt with both loaders.
  - Verify logits match within tolerance (quantization rounding may differ).

- [ ] T61.4 Fix GGUF loader issues found during validation  Owner: TBD  Est: 4h
  - Buffer for issues found during T61.1-T61.3:
    - Architecture mapping mismatches.
    - Tensor name translation errors.
    - Quantization format differences between GGUF Q4_0 and Zerfoo Q4.
    - Tokenizer vocabulary mismatches.
  - Acceptance: All validation tests from T61.1-T61.3 pass after fixes.
  - Dependencies: T61.1, T61.2, T61.3.

- [ ] S61.4.1 Regression tests for fixed issues  Owner: TBD  Est: 1h

- [ ] T61.5 Run golangci-lint on model/gguf/ and inference/  Owner: TBD  Est: 15m
  - Dependencies: T61.4.

### E62: GPU Q4 Inference Pipeline (O62)

Run end-to-end Q4 inference on DGX Spark GB10 using CUDA kernels.
Phase 25 delivered the CUDA Q4 dequant-GEMM kernel (2383 GFLOPS). This epic
wires it into the full inference pipeline.

Existing code:
- `internal/cuda/kernels/` -- CUDA Q4 dequant-GEMM kernel.
- `compute/gpu_engine.go` -- GPUEngine with CUDA acceleration.
- `generate/generator.go` -- Autoregressive generation loop.
- DGX Spark models: `~/models/gemma3-q4/model.zmf`.

- [ ] T62.1 Profile GPU inference pipeline  Owner: TBD  Est: 3h
  - Run `cmd/zerfoo-predict` on DGX Spark with CUDA build tags.
  - Enable pprof and CUDA profiling (nsys or ncu).
  - Identify: which ops run on GPU vs CPU fallback, data transfer overhead,
    kernel launch overhead.
  - Acceptance: Profile captured showing GPU utilization breakdown.
  - Dependencies: none (requires DGX Spark access).

- [ ] S62.1.1 GPU profile report  Owner: TBD  Est: 30m

- [ ] T62.2 Fix GPU fallback bottlenecks  Owner: TBD  Est: 4h
  - From T62.1 profile, identify ops that fall back to CPU unnecessarily.
  - Common issues:
    - Host-to-device copies for each op instead of keeping tensors on GPU.
    - Missing GPU implementation for element-wise ops used in attention.
    - KV cache on CPU requiring copies for each attention step.
  - Fix top 3 bottlenecks.
  - Acceptance: Fewer CPU fallbacks in profile. Measurable GPU utilization
    improvement.
  - Dependencies: T62.1.

- [ ] S62.2.1 Before/after GPU utilization comparison  Owner: TBD  Est: 30m

- [ ] T62.3 Benchmark GPU Q4 inference tok/s  Owner: TBD  Est: 2h
  - Run Gemma 3 2B Q4 on DGX Spark with all fixes from T62.2.
  - Measure: tok/s (100-token average), GPU memory usage, GPU utilization %.
  - Compare with CPU-only baseline on the same machine.
  - Acceptance: GPU tok/s documented. Target >= 60 tok/s for Gemma 3 2B Q4.
  - Dependencies: T62.2.

- [ ] S62.3.1 GPU benchmark report  Owner: TBD  Est: 30m

- [ ] T62.4 GPU PagedAttention integration  Owner: TBD  Est: 4h
  - Ensure PagedKVCache works with GPU tensors:
    - Block pool allocates GPU memory.
    - Block-table reads work with GPU-resident KV data.
    - Attention computation stays on GPU end-to-end.
  - Acceptance: GPU inference with PagedAttention produces correct output.
    No host-device copies in the KV cache hot path.
  - Dependencies: T62.2, T59.1.

- [ ] S62.4.1 GPU paged attention correctness test  Owner: TBD  Est: 1h

- [ ] T62.5 Run golangci-lint on compute/ and internal/cuda/  Owner: TBD  Est: 15m
  - Dependencies: T62.4.

### E63: Performance CI Regression Alerting (O63)

The benchmark script and GitHub Actions workflow exist. This epic adds
regression detection with PR comments and validates the DGX self-hosted runner.

Existing code:
- `scripts/bench.sh` -- Benchmark runner producing JSON output.
- `.github/workflows/benchmark.yml` -- GH Actions workflow.

- [x] T63.1 Add regression comparison to benchmark workflow  Owner: TBD  Est: 3h  (2026-03-06)
  - Modify `.github/workflows/benchmark.yml` to:
    1. Download previous benchmark results from GitHub Actions artifacts.
    2. Compare current results with previous.
    3. If any metric regresses by > 5%, post a PR comment with details.
  - Use `gh api` or a lightweight Go script (`cmd/bench-compare/`) for
    comparison logic.
  - Acceptance: PR with a deliberate regression (e.g., sleep in hot path)
    triggers a warning comment.
  - Dependencies: none.

- [x] S63.1.1 Regression detection test  Owner: TBD  Est: 1h  (2026-03-06)
  - Create test benchmark data with a 10% regression.
  - Verify comparison script detects and reports it.

- [ ] T63.2 Set up DGX Spark as self-hosted GitHub Actions runner  Owner: TBD  Est: 2h
  - Install and configure `actions-runner` on DGX Spark.
  - Add `self-hosted` and `dgx-spark` labels.
  - Verify runner connects to GitHub and can pick up jobs.
  - Acceptance: A test workflow with `runs-on: [self-hosted, dgx-spark]`
    completes successfully.
  - Dependencies: none (requires admin access to DGX Spark).
  - Risk: Network/firewall issues may block runner registration.

- [ ] S63.2.1 Self-hosted runner validation  Owner: TBD  Est: 30m

- [ ] T63.3 Add GPU benchmark job to workflow  Owner: TBD  Est: 2h
  - Add a separate job in `.github/workflows/benchmark.yml` that:
    1. Runs on `[self-hosted, dgx-spark]`.
    2. Builds with CUDA tags (`go build -tags cuda ./...`).
    3. Runs GPU benchmarks (CUDA Q4 GEMM, end-to-end GPU tok/s).
    4. Stores results alongside CPU benchmarks.
  - Acceptance: GPU benchmarks appear in CI artifacts.
  - Dependencies: T63.1, T63.2.

- [ ] S63.3.1 GPU CI benchmark validation  Owner: TBD  Est: 30m

- [ ] T63.4 Historical benchmark tracking  Owner: TBD  Est: 2h
  - Store benchmark results in a JSON file committed to the repo
    (e.g., `bench_results/history.json`) or use GitHub Pages for a
    simple dashboard.
  - Each CI run appends results with commit SHA and timestamp.
  - Acceptance: Historical data queryable for the last 20 runs.
  - Dependencies: T63.1.

- [ ] S63.4.1 History query test  Owner: TBD  Est: 30m

- [ ] T63.5 Run golangci-lint on scripts/ and cmd/bench-compare/  Owner: TBD  Est: 15m
  - Dependencies: T63.4.

### E64: Smoothed Early Stopping (O65)

Current early stopping uses raw val_corr, which oscillates wildly (+0.010 one
epoch, -0.006 the next). A smoothed tracking metric prevents premature stopping
and picks more stable checkpoints.

Existing code:
- `audacity/internal/training/early_stopping.go` -- current implementation.

- [x] T64.1 Add exponential smoothing to early stopping  Owner: TBD  Est: 2h  (pre-existing)
  - Modify `audacity/internal/training/early_stopping.go`:
    - Add `smoothedBest float64` and `alpha float64` fields.
    - Track `smoothed = alpha * raw_val + (1-alpha) * smoothed`.
    - Base early stopping patience on smoothed val_corr, not raw.
    - Default alpha = 0.3 (responds to real trends, filters noise).
  - Save checkpoint when smoothed val_corr is best, not raw.
  - Acceptance: With synthetic oscillating val_corr data, smoothed early
    stopping continues training longer than raw early stopping.
  - Dependencies: none.

- [x] S64.1.1 Smoothed early stopping unit tests  Owner: TBD  Est: 1h  (pre-existing)
  - Test: oscillating val_corr [0.01, -0.006, 0.008, -0.004, 0.009, -0.003]
    does not trigger early stopping with patience=5, alpha=0.3.
  - Test: steadily declining val_corr does trigger early stopping.
  - Test: alpha=1.0 matches raw behavior (backwards compatibility).
  - Test: both raw and smoothed metrics are logged.

- [x] T64.2 Log both raw and smoothed val_corr  Owner: TBD  Est: 30m  (2026-03-06)
  - Add logging to show both raw and smoothed values each epoch.
  - Acceptance: Training output shows both metrics.
  - Dependencies: T64.1.

- [ ] T64.3 Run golangci-lint on audacity/internal/training/  Owner: TBD  Est: 15m
  - Dependencies: T64.2.

### E65: EMA of Model Weights (O64)

Exponential Moving Average maintains a smoothed copy of model weights.
Instead of using the "lucky peak" checkpoint that oscillates, EMA captures
the stable signal across all training steps.

Existing code:
- `training/optimizer/adamw.go` -- current AdamW optimizer.
- `training/optimizer/ema.go` -- already exists (check contents).

- [x] T65.1 Implement EMA optimizer wrapper  Owner: TBD  Est: 4h  (pre-existing)
  - Create or complete `training/optimizer/ema.go` with:
    ```go
    type EMA[T tensor.Float] struct {
        inner   optimizer.Optimizer[T]
        decay   float64           // typical: 0.999
        shadow  []*tensor.TensorNumeric[T]
        engine  compute.Engine[T]
    }
    func NewEMA[T tensor.Float](inner Optimizer[T], decay float64, engine Engine[T]) *EMA[T]
    func (e *EMA[T]) Step(params, grads []*tensor.TensorNumeric[T]) error
    func (e *EMA[T]) SwapWeights(params []*tensor.TensorNumeric[T])
    ```
  - `Step()` calls `inner.Step(params, grads)`, then updates shadow:
    `shadow[i] = decay * shadow[i] + (1-decay) * params[i]`
  - `SwapWeights()` swaps live and shadow params for validation/inference.
  - Uses `engine.Copy()`, `engine.MulScalar()`, `engine.Add()` which exist.
  - Acceptance: After 100 optimizer steps on noisy gradients, shadow params
    are smoother than live params (lower variance across steps).
  - Dependencies: none.

- [x] S65.1.1 EMA unit tests  Owner: TBD  Est: 2h  (pre-existing)
  - Test: shadow params converge to running average of live params.
  - Test: SwapWeights correctly exchanges live and shadow.
  - Test: SwapWeights twice restores original state.
  - Test: decay=0.0 makes shadow equal to current params.
  - Test: decay=1.0 makes shadow never change from initial.
  - Benchmark: EMA overhead per Step (expect < 10% of optimizer cost).

- [ ] T65.2 Wire EMA into Audacity training loop  Owner: TBD  Est: 2h
  - Modify training loop to:
    1. Wrap optimizer with EMA when `--ema-decay` flag is set.
    2. Before validation: call `SwapWeights()` to use shadow params.
    3. After validation: call `SwapWeights()` to restore live params.
    4. At end of training: final `SwapWeights()` so saved model uses EMA.
  - Acceptance: Training with `--ema-decay=0.999` produces a model.
    Validation uses EMA weights.
  - Dependencies: T65.1.

- [ ] S65.2.1 EMA training integration test  Owner: TBD  Est: 1h
  - Train 5 epochs with EMA on synthetic data.
  - Verify final saved model contains EMA weights, not live weights.

- [ ] T65.3 Run golangci-lint on training/optimizer/  Owner: TBD  Est: 15m
  - Dependencies: T65.2.

### E66: Cosine Annealing with Warm Restarts (O66)

The current cosine schedule decays LR from max to 0. Warm restarts reset LR
periodically, allowing the model to escape local minima and explore multiple
basins. Combined with EMA/SWA, each restart explores a new basin.

Existing code:
- `audacity/internal/training/lr_schedule.go` -- current LR schedulers.

- [x] T66.1 Implement CosineAnnealingWarmRestarts  Owner: TBD  Est: 3h  (2026-03-06)
  - Add to `audacity/internal/training/lr_schedule.go`:
    ```go
    type CosineWarmRestarts struct {
        T0     int     // initial cycle length in epochs
        TMult  int     // cycle length multiplier (1 = fixed, 2 = doubling)
        EtaMin float64 // minimum learning rate
        EtaMax float64 // maximum learning rate
    }
    func (c *CosineWarmRestarts) LR(epoch int) float64
    ```
  - With TMult=2 and T0=5: cycles are 5, 10, 20 epochs.
  - At each cycle start, LR resets to EtaMax.
  - Within cycle: LR = EtaMin + 0.5*(EtaMax-EtaMin)*(1 + cos(pi*t/T)).
  - Acceptance: LR values follow expected cosine warm restart pattern.
  - Dependencies: none.

- [x] S66.1.1 Cosine warm restarts unit tests  Owner: TBD  Est: 1h  (2026-03-06)
  - Test: T0=5, TMult=1: LR resets every 5 epochs.
  - Test: T0=5, TMult=2: cycles are 5, 10, 20.
  - Test: LR at cycle start equals EtaMax.
  - Test: LR at cycle end equals EtaMin.
  - Test: LR at cycle midpoint equals (EtaMax+EtaMin)/2.

- [ ] T66.2 Wire warm restarts into training config  Owner: TBD  Est: 1h
  - Add `--lr-schedule=cosine-warm-restarts`, `--lr-t0`, `--lr-tmult` flags.
  - Acceptance: Training with `--lr-schedule=cosine-warm-restarts --lr-t0=5`
    shows LR resets in training log.
  - Dependencies: T66.1.

- [ ] T66.3 Run golangci-lint on audacity/internal/training/  Owner: TBD  Est: 15m
  - Dependencies: T66.2.

### E67: Stochastic Weight Averaging (O67)

SWA averages model weights from multiple training checkpoints. It finds wider
optima that generalize better than sharp optima found by standard Adam/SGD.
Different from EMA: SWA averages at epoch boundaries, EMA averages every step.

- [x] T67.1 Implement SWA optimizer wrapper  Owner: TBD  Est: 3h  (2026-03-06)
  - Create `training/optimizer/swa.go` with:
    ```go
    type SWA[T tensor.Float] struct {
        inner       optimizer.Optimizer[T]
        avgParams   []*tensor.TensorNumeric[T]
        nAveraged   int
        startEpoch  int
        engine      compute.Engine[T]
    }
    func NewSWA[T tensor.Float](inner Optimizer[T], startEpoch int, engine Engine[T]) *SWA[T]
    func (s *SWA[T]) Step(params, grads []*tensor.TensorNumeric[T]) error
    func (s *SWA[T]) UpdateAverage(params []*tensor.TensorNumeric[T], epoch int)
    func (s *SWA[T]) SwapWeights(params []*tensor.TensorNumeric[T])
    ```
  - `UpdateAverage()` called at epoch end: `avg = (avg*n + params) / (n+1)`.
  - Only averages after `startEpoch`.
  - `SwapWeights()` swaps live and SWA params.
  - Uses `engine.Add()`, `engine.MulScalar()`, `engine.Copy()`.
  - Acceptance: After averaging 10 checkpoints, SWA params are the arithmetic
    mean of those checkpoints.
  - Dependencies: none.

- [x] S67.1.1 SWA unit tests  Owner: TBD  Est: 1.5h  (2026-03-06)
  - Test: UpdateAverage computes correct running mean.
  - Test: epochs before startEpoch do not contribute to average.
  - Test: SwapWeights exchanges live and SWA params.
  - Test: nAveraged increments correctly.

- [ ] T67.2 Wire SWA into Audacity training loop  Owner: TBD  Est: 2h
  - Modify training loop:
    1. When `--swa-start` flag is set, wrap optimizer with SWA.
    2. At each epoch end (after startEpoch), call `UpdateAverage()`.
    3. Before validation: `SwapWeights()` for SWA evaluation.
    4. At end of training: final `SwapWeights()` so saved model uses SWA.
  - Acceptance: Training with `--swa-start=10` averages checkpoints from
    epoch 10 onwards. Saved model uses SWA weights.
  - Dependencies: T67.1.

- [ ] S67.2.1 SWA training integration test  Owner: TBD  Est: 1h
  - Train 20 epochs with SWA starting at epoch 10.
  - Verify final model is average of checkpoints 10-19.

- [ ] T67.3 Run golangci-lint on training/optimizer/  Owner: TBD  Est: 15m
  - Dependencies: T67.2.

### E68: Feature Dropout (O68)

Standard dropout drops neurons in hidden layers. Feature dropout drops entire
input features (columns) during training, forcing the model to not rely on any
single feature. Important for Numerai where feature importance shifts across
market regimes.

- [x] T68.1 Implement FeatureDropout layer  Owner: TBD  Est: 3h  (2026-03-06)
  - Create `layers/regularization/feature_dropout.go`:
    ```go
    type FeatureDropout[T tensor.Float] struct {
        Rate    float64  // fraction of features to drop (0.05-0.10)
        engine  compute.Engine[T]
    }
    func (f *FeatureDropout[T]) Forward(ctx context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
    ```
  - During training: randomly zero out entire columns (feature dimension,
    axis=1) of the input tensor. Apply 1/(1-rate) scaling.
  - During inference: pass through unchanged.
  - Uses context to detect training vs inference mode.
  - Acceptance: With rate=0.1 and 100 features, ~10 features are zeroed
    per forward pass. Different features are zeroed each call.
  - Dependencies: none.

- [x] S68.1.1 FeatureDropout unit tests  Owner: TBD  Est: 1.5h  (2026-03-06)
  - Test: training mode zeros approximately rate*n_features columns.
  - Test: inference mode passes input through unchanged.
  - Test: scaling factor 1/(1-rate) is applied to non-dropped features.
  - Test: different random features are dropped each call.
  - Test: rate=0.0 passes through unchanged.
  - Test: rate=1.0 zeros all features.

- [ ] T68.2 Register FeatureDropout in layer registry  Owner: TBD  Est: 30m
  - Add to `layers/registry/register.go`.
  - Acceptance: Layer can be loaded from ZMF model file.
  - Dependencies: T68.1.

- [ ] T68.3 Wire into Audacity model graph  Owner: TBD  Est: 1h
  - Add `--feature-dropout` flag to control rate.
  - Insert FeatureDropout before the first linear layer in the model.
  - Acceptance: Training with `--feature-dropout=0.05` logs that feature
    dropout is active. Model trains without errors.
  - Dependencies: T68.1.

- [ ] S68.3.1 Feature dropout training test  Owner: TBD  Est: 1h
  - Train 3 epochs with feature dropout on synthetic data.
  - Verify loss decreases (model still learns with dropped features).

- [ ] T68.4 Run golangci-lint on layers/regularization/  Owner: TBD  Est: 15m
  - Dependencies: T68.3.

---

## 4. Parallel Work

All 10 epics fall into 6 independent tracks. Tracks share no dependencies
except at documented sync points, so they can execute concurrently.

| Track | Epics | Description | Sync Points |
|-------|-------|-------------|-------------|
| A: PagedAttention + GPU | E59 then E62 | GQA block-table reads, then GPU inference pipeline | E62.T62.4 waits on E59.T59.1 |
| B: Speculative Decode | E60 | Real-model speculative validation and benchmarking | none |
| C: GGUF Validation | E61 | End-to-end GGUF with real HuggingFace models | none |
| D: CI Regression | E63 | Benchmark regression alerting and DGX runner | none |
| E: Training Smoothing | E64, E65 | Smoothed early stopping and EMA of weights | none (E64 and E65 are independent of each other) |
| F: Training Exploration | E66, E67, E68 | Warm restarts, SWA, and feature dropout | M37 milestone waits on E65 from Track E |

### Within-Epic Parallelism

These tasks within their epics can execute concurrently:

| Epic | Parallel Tasks | Shared Prerequisite |
|------|----------------|---------------------|
| E60 | T60.3, T60.4 | Both depend on T60.2 only |
| E61 | T61.2, T61.3 | Both depend on T61.1 only |
| E63 | T63.1, T63.2 | Both have no dependencies |
| E68 | T68.2, T68.3 | Both depend on T68.1 only |

### Maximum Parallelism Summary

With unlimited workers, all 6 tracks start immediately. The only ordering
constraint is:

1. **Track A sequencing:** E59 must complete T59.1 before E62.T62.4 begins.
   E62.T62.1-T62.3 can start immediately (no dependency on E59).
2. **Track F milestone gate:** M37 exit criteria require E65 complete, but
   E66, E67, E68 task-level work has no dependency on E65 and can start
   immediately. Only the milestone sign-off waits.

With a single worker, recommended serial order (maximizes early value):

1. E64, E65 (training smoothing -- smallest, fastest value)
2. E66, E67, E68 (training exploration -- builds on E65 validation)
3. E61 (GGUF -- standalone, high visibility)
4. E63 (CI -- standalone infrastructure)
5. E59 (PagedAttention -- unblocks E62)
6. E60 (Speculative -- standalone)
7. E62 (GPU inference -- depends on E59, highest effort)

---

## 5. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M31: PagedAttention v2 | E59 | none | GQA block-table reads, 8-seq load test passes |
| M32: Speculative validated | E60 | none | Real-model speedup measured, >= 1.5x or documented explanation |
| M33: GGUF real models | E61 | none | Llama 3.2 1B and Gemma 3 2B GGUF load + generate |
| M34: GPU inference | E62 | E59 | GPU Q4 tok/s measured on DGX Spark |
| M35: CI regression | E63 | none | PR regression comments, DGX runner operational |
| M36: Training smoothing | E64, E65 | none | EMA + smoothed early stopping in Audacity |
| M37: Training exploration | E66, E67, E68 | E65 | Warm restarts + SWA + feature dropout in Audacity |

Critical path: E59.T59.1 -> E62.T62.4 -> E62.T62.5 (GPU PagedAttention is
the longest dependency chain).

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R59 | No suitable draft model exists for Gemma 3 speculative decoding | Speculative speedup unmeasurable | Medium | Create truncated model (T60.1 Option B). If acceptance rate too low, document as model-pair limitation. |
| R60 | GGUF real models have format quirks not covered by synthetic tests | Loader fails on real files | High | Start with well-tested Llama format. Allocate buffer time (T61.4). |
| R61 | DGX Spark self-hosted runner has network/security issues | GPU CI blocked | Medium | Fall back to manual GPU benchmarks documented in PR. |
| R62 | GPU Q4 inference limited by host-device transfer, not compute | GPU tok/s below target | Medium | Profile first (T62.1). Consider persistent GPU tensor allocation. |
| R63 | Block-table non-contiguous reads are slower than gather-copy | Option B worse than Option A | Medium | Benchmark both (T59.2). Keep Option A as default if Option B regresses. |
| R64 | HuggingFace model downloads fail in CI due to rate limits or auth | GGUF tests flaky | Medium | Cache model files on CI runner. Use smallest available model. Gate behind env var. |
| R65 | EMA memory doubles parameter footprint | OOM on DGX Spark | Low | Shadow params use same dtype. Gemma 3 2B Q4 is 1.5GB; doubling to 3GB is well within 128GB. |
| R66 | SWA + EMA combined overhead slows training significantly | Training wall time 2x | Low | EMA adds < 10% per step. SWA only averages at epoch boundaries. Profile if concerned. |
| R67 | Smoothed early stopping delays convergence detection | Wasted training time | Low | Use alpha=0.3 which still responds to real trends. Add max-epochs hard limit. |

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
8. Changes are committed in a small commit touching one directory only.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(training): add EMA optimizer wrapper`.
- Always run linters and formatters before committing.

### Benchmark Protocol

- benchtime=3s, count=5, report median and p99.
- All inference benchmarks on DGX Spark GB10 (ssh ndungu@192.168.86.250).
- Training benchmarks on DGX Spark or local machine.

### Quality Gate

- `go test -race ./package/`
- `golangci-lint run ./package/`
- `go vet ./package/`
- `go build ./...` (non-CUDA)

---

## 8. Progress Log

### Change Summary -- 2026-03-06 (parallel analysis)

Added section 4 "Parallel Work" identifying 6 independent execution tracks:
- Track A (E59 -> E62): PagedAttention then GPU inference (only cross-epic dependency).
- Track B (E60): Speculative decoding, fully independent.
- Track C (E61): GGUF validation, fully independent.
- Track D (E63): CI regression, fully independent.
- Track E (E64, E65): Training smoothing, both epics independent of each other.
- Track F (E66, E67, E68): Training exploration, M37 milestone gates on E65.

Identified within-epic parallelism in E60, E61, E63, E68. Renumbered
sections 5-9 to 6-10 to accommodate new section 4.

### Change Summary -- 2026-03-06

Created Phase 31 plan merging two sources:
- `docs/zerfoo-suggestions.md` -- Training improvements for Numerai performance
  (10 suggestions, 5 selected as P0/P1 for this phase).
- `docs/suggestion-plan.md` -- Inference validation gaps from Phase 26
  (5 epics: E59-E63).

Added training epics E64-E68:
- E64: Smoothed early stopping (from suggestion #4, P0).
- E65: EMA of model weights (from suggestion #1, P0).
- E66: Cosine annealing warm restarts (from suggestion #2, P1).
- E67: Stochastic weight averaging (from suggestion #3, P1).
- E68: Feature dropout (from suggestion #5, P1).

Deferred to future phase (P2/P3):
- Huber loss, mixup, multi-loss scheduling, gradient accumulation, per-param LR.

Trimmed completed Phase 30 (E54-E58) from plan. Merged Phase 30 stable
knowledge into docs/design.md section 15.12.

---

## 9. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. Design decisions are in
  docs/adr/ (ADR-001 through ADR-021).
- **Phases 1-30:** All complete. See docs/design.md sections 15.1-15.12.
- **Phase 31:** This plan is the source of truth.
- **Quality:** See docs/QUALITY.md for test coverage report.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - On DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build -tags cuda,cutlass ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key Starting Points

**Inference epics:**
1. **E59 (PagedAttention v2):** Modify `layers/attention/grouped_query_attention.go`
   for block-table reads. Load test in `serve/loadtest_test.go`.
2. **E60 (Speculative):** Benchmark `generate/speculative.go` with real models.
3. **E61 (GGUF):** Validate `inference.Load()` with real HF models.
4. **E62 (GPU):** Profile and fix GPU fallbacks on DGX Spark.
5. **E63 (CI):** Enhance `.github/workflows/benchmark.yml`.

**Training epics:**
6. **E64 (Smoothed early stopping):** Modify `audacity/internal/training/early_stopping.go`.
7. **E65 (EMA):** Complete `training/optimizer/ema.go`.
8. **E66 (Warm restarts):** Add to `audacity/internal/training/lr_schedule.go`.
9. **E67 (SWA):** Create `training/optimizer/swa.go`.
10. **E68 (Feature dropout):** Create `layers/regularization/feature_dropout.go`.

### External Dependencies

- **DGX Spark (ndungu@192.168.86.250):**
  - Go 1.25.0 linux/arm64, CUDA 13.0, sm_121 (Blackwell).
  - Models: ~/models/gemma3-q4/ (Q4 ZMF), ~/models/gemma3/ (F32 ZMF).
  - Repos: ~/zerfoo/, ~/zonnx/, ~/zmf/.
- **HuggingFace:** Model downloads for GGUF validation (T61.1-T61.3).

### Performance Baselines

| Model | Params | Quant | tok/s | Phase |
|-------|--------|-------|-------|-------|
| Gemma 3 2B | 2.6B | Q4_0 | 6.86 | 30 (CPU ARM64) |
| Gemma 3 2B | 2.6B | Q4_0 | 6.5 | 29 (CPU ARM64) |
| Gemma 3 2B | 2.6B | Q4_0 | 3.60 | 26 (CPU ARM64) |

### Numerai Training Baselines

| Metric | Current | Target |
|--------|---------|--------|
| Training val_corr (best) | 0.010551 | maintain |
| Full validation CORR | 0.002 | >= 0.006 |
| Full validation Sharpe | 0.171 | >= 0.5 |
| Val_corr oscillation | +/-0.008 | +/-0.003 |

---

## 10. Appendix

### Existing File Reference

| File | Purpose |
|------|---------|
| `generate/paged_kv.go` | PagedKVCache type with block management |
| `generate/block_pool.go` | BlockPool with pre-allocated blocks |
| `generate/speculative.go` | SpeculativeGenerator with draft-verify loop |
| `generate/adaptive.go` | Adaptive draft length based on acceptance rate |
| `generate/generator.go` | Core autoregressive generation loop |
| `layers/attention/grouped_query_attention.go` | GQA attention layer |
| `model/gguf/parser.go` | GGUF file format parser |
| `model/gguf/loader.go` | GGUF tensor loader (Q4_0, Q8_0, F32, F16) |
| `model/gguf/arch.go` | GGUF architecture mapping (llama, gemma) |
| `model/gguf/tokenizer.go` | GGUF tokenizer extraction |
| `inference/gguf.go` | LoadGGUF and model metadata conversion |
| `inference/inference.go` | High-level Load/Pull API with GGUF detection |
| `serve/server.go` | HTTP server with WithDraftModel option |
| `scripts/bench.sh` | Benchmark runner script |
| `.github/workflows/benchmark.yml` | GitHub Actions benchmark workflow |
| `compute/gpu_engine.go` | GPUEngine with CUDA acceleration |
| `internal/cuda/kernels/` | CUDA kernel source and Go wrappers |
| `internal/workerpool/pool.go` | Persistent worker pool (Phase 30) |
| `graph/compile.go` | Graph compiler and ExecutionPlan (Phase 30) |
| `training/optimizer/adamw.go` | AdamW optimizer |
| `training/optimizer/ema.go` | EMA optimizer (existing, may need completion) |
| `audacity/internal/training/early_stopping.go` | Early stopping logic |
| `audacity/internal/training/lr_schedule.go` | Learning rate schedulers |
| `layers/regularization/dropout.go` | Standard dropout layer |

### HuggingFace Model Aliases (from inference/inference.go)

```
gemma-3-1b-q4  -> google/gemma-3-1b-it-qat-q4_0-gguf
gemma-3-2b-q4  -> google/gemma-3-2b-it-qat-q4_0-gguf
llama-3-1b-q4  -> meta-llama/Llama-3.2-1B-Instruct-GGUF
llama-3-8b-q4  -> meta-llama/Llama-3.1-8B-Instruct-GGUF
mistral-7b-q4  -> mistralai/Mistral-7B-Instruct-v0.3-GGUF
qwen-2.5-7b-q4 -> Qwen/Qwen2.5-7B-Instruct-GGUF
```

### Estimated Effort Summary

| Epic | Area | Tasks | Estimated Hours |
|------|------|-------|----------------|
| E59: PagedAttention v2 | Inference | 4 tasks + 4 subtests | 12.75h |
| E60: Speculative validation | Inference | 5 tasks + 5 subtests | 13.0h |
| E61: GGUF real models | Inference | 5 tasks + 5 subtests | 13.75h |
| E62: GPU inference | Inference | 5 tasks + 4 subtests | 15.0h |
| E63: CI regression | Inference | 5 tasks + 5 subtests | 11.75h |
| E64: Smoothed early stopping | Training | 3 tasks + 1 subtest | 3.75h |
| E65: EMA of weights | Training | 3 tasks + 2 subtests | 9.25h |
| E66: Cosine warm restarts | Training | 3 tasks + 1 subtest | 4.25h |
| E67: SWA | Training | 3 tasks + 2 subtests | 7.75h |
| E68: Feature dropout | Training | 4 tasks + 2 subtests | 7.25h |
| **Total** | **Both** | **40 tasks + 31 subtests** | **~98h** |
