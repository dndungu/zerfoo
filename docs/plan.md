# Zerfoo Development Plan -- Phase 28: Make It Actually Work

## 1. Context

### Problem Statement

The README promises a 3-line experience:

```go
model, _ := inference.Load("gemma-3-1b-q4")
reply, _ := model.Generate(ctx, "What is the capital of France?")
fmt.Println(reply)
```

This does not work today. A developer who follows the README will hit failures
immediately. The gap between promise and reality is the single biggest barrier
to adoption.

See docs/design.md for full architecture context and Phases 1-27 history.

### Audit of README Promises vs Implementation

| Promise | Status | Blocker |
|---------|--------|---------|
| `inference.Load("gemma-3-1b-q4")` auto-downloads | Fails -- default registry has no pull function | E45 |
| GGUF models load and run | Parser only -- no graph builder | E44 |
| Chat works with all listed models | Only Gemma template, others get broken fallback | E46 |
| Q4 quantization | Only Q4_0/Q8_0 -- most HuggingFace GGUFs use Q4_K_M | E47 |
| `zerfoo run` / `zerfoo serve` | Code exists but blocked by all of the above | All |

### Phase 27 Summary

| Epic | Status | Result |
|------|--------|--------|
| E39: Transpose Elimination | Kernel complete, benchmark pending | FoldConstantTransposes + blocked 4D transpose |
| E40: Tensor Arena | Kernel complete, benchmark pending | TensorPool + ref-counted release in graph.Forward |
| E41: GPU Inference | Not started | Deferred to Phase 29 (requires DGX Spark) |
| E42: GGUF End-to-End | Not started | Superseded by E44 with refined scope |
| E43: Operator Fusion | Complete | Fused RMSNorm, RoPE, SiLU-gate |

Remaining Phase 27 benchmark tasks (T39.3, T40.3, T43.4) require DGX Spark
and are carried forward to E48.

### Objectives

- O61: `inference.Load("model-name")` downloads, loads, and returns a
  ready-to-generate Model in one call, for any supported architecture.
- O62: GGUF files load end-to-end without external conversion tools.
- O63: Chat completions produce correct output for all six listed
  architectures (Gemma, LLaMA, Mistral, Qwen, DeepSeek, Phi-4).
- O64: Q4_K_M and Q6_K quantized GGUFs load and run (covers >80% of
  HuggingFace GGUF models).
- O65: `zerfoo run` and `zerfoo serve` work end-to-end from a fresh install.

### Non-Goals

- GPU inference pipeline (deferred to Phase 29, needs DGX Spark).
- Vision/multimodal models (text-only focus).
- Training performance or new training features.
- Embeddings API (`model.Embed()` remains stubbed).
- Multi-GPU inference.
- Breaking changes to the Engine[T] or Node[T] interfaces.

### Constraints and Assumptions

- Use Go standard library only where possible. Minimize new dependencies.
- All CUDA code behind `//go:build cuda` build tags.
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- Table-driven tests using the standard testing package.
- DGX Spark GB10 at ssh ndungu@192.168.86.250 for GPU validation and benchmarks.
- GGUF parser is pure Go (no CGo dependency on llama.cpp).
- Reference implementation for K-quant dequantization: llama.cpp `ggml-quants.c`.

### Success Metrics

| Metric | Current | Phase 28 Target |
|--------|---------|-----------------|
| `inference.Load()` works out-of-box | No (missing pull func) | Yes |
| GGUF to inference | No (parser only) | Yes (Llama + Gemma) |
| Chat template coverage | 1/6 (Gemma only) | 6/6 |
| Quantization format coverage | Q4_0, Q8_0 | + Q4_K_M, Q5_K_M, Q6_K |
| `zerfoo run model.gguf` works | No | Yes |
| End-to-end integration tests | Gemma ZMF only | Gemma + Llama, GGUF + ZMF |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D91 | GGUF end-to-end inference | `inference.Load("path/to/model.gguf")` to `model.Generate()` for Llama and Gemma |
| D92 | Auto-download | `inference.Load("gemma-3-1b-q4")` downloads from HF, no manual setup |
| D93 | Chat templates | `model.Chat()` produces correctly formatted output for all 6 architectures |
| D94 | K-quant support | Q4_K_M, Q5_K_M, Q6_K GGUFs dequantize correctly |
| D95 | Working CLI | `zerfoo run model.gguf` starts interactive chat from a fresh install |
| D96 | Phase 27 benchmarks | Transpose folding + tensor pool + fused ops validated on DGX Spark |

### Out of Scope

- GPU inference (Phase 29).
- New model architectures beyond the six already listed.
- Embeddings / hidden state extraction.
- GGUF write / model export.
- Quantization (converting F32 to Q4_K) -- only dequantization for inference.
- Prompt caching / prefix sharing.
- Multi-GPU KV cache partitioning.

---

## 3. Checkable Work Breakdown

### E44: GGUF End-to-End Inference (O61, O62)

Carried forward from Phase 27 E42 with refined scope.

- [x] T44.1 Graph template builder for Llama architecture  Owner: TBD  Est: 6h  2026-03-06
  - Create `inference/arch_llama.go` with
    `buildLlamaGraph(tensors map[string]*tensor.TensorNumeric[float32], cfg ModelMetadata, engine Engine) (*graph.Graph[float32], error)`.
  - Build the standard Llama graph: Embed -> [RMSNorm -> GQA -> RMSNorm ->
    FFN(SiLU-gate)] x N -> RMSNorm -> LMHead.
  - Map canonical tensor names (from E37 arch mapping) to graph parameters.
  - Support GQA (num_kv_heads < num_heads).
  - Acceptance: Llama 3.2 1B GGUF loads and produces non-NaN logits.
    Greedy decode of "The capital of France is" produces "Paris".
  - Dependencies: none.

- [x] S44.1.1 Llama GGUF forward pass + decode test  Owner: TBD  Est: 1h  2026-03-06

- [x] T44.2 Graph template builder for Gemma architecture  Owner: TBD  Est: 4h  2026-03-06
  - Extend for Gemma differences: shared embed/lm_head weight tying,
    GeGLU activation (GELU instead of SiLU in MLP), embedding scaling
    by sqrt(hidden_size), pre/post normalization differences.
  - Refactored: extracted shared transformer loop into arch_common.go
    to eliminate duplication with Llama builder.
  - Acceptance: Gemma 3 1B GGUF loads and produces non-NaN logits.
    Greedy decode produces coherent text.
  - Dependencies: T44.1.

- [x] S44.2.1 Gemma GGUF forward pass + decode test  Owner: TBD  Est: 1h  2026-03-06

- [x] T44.3 GGUF tokenizer extraction  Owner: TBD  Est: 3h  2026-03-06
  - Extract tokenizer vocabulary from GGUF metadata keys:
    `tokenizer.ggml.tokens`, `tokenizer.ggml.scores`,
    `tokenizer.ggml.merges`, `tokenizer.ggml.token_type`.
  - Build a `tokenizer.Tokenizer` from GGUF metadata without needing a
    separate `tokenizer.json` file.
  - Fallback: if GGUF tokenizer data is absent, look for `tokenizer.json`
    in the same directory.
  - Acceptance: tokenizer encodes "Hello, world!" and decodes back to
    same string for both Llama and Gemma GGUFs.
  - Dependencies: none.

- [x] S44.3.1 GGUF tokenizer encode/decode tests  Owner: TBD  Est: 1h  2026-03-06

- [x] T44.4 Unified load function for GGUF  Owner: TBD  Est: 3h  2026-03-06
  - Extend `inference.Load()` to detect GGUF files (by extension or magic
    bytes) and dispatch to the GGUF loading path.
  - Add `inference.LoadFile(path string, opts ...Option) (*Model, error)`
    for loading from a local file path directly.
  - Architecture dispatch based on GGUF `general.architecture` metadata.
  - Wire tokenizer, graph, engine, and generator into a ready-to-use Model.
  - Acceptance: `inference.LoadFile("model.gguf")` to `model.Generate(ctx, prompt)`
    works end-to-end.
  - Dependencies: T44.1, T44.2, T44.3.

- [x] S44.4.1 End-to-end GGUF load + generate test  Owner: TBD  Est: 1h  2026-03-06

- [x] T44.5 Run golangci-lint on inference/ and model/gguf/  Owner: TBD  Est: 15m  2026-03-06

### E45: Model Hub & Auto-Download (O61, O65)

- [x] T45.1 Wire HuggingFace pull into default registry  Owner: TBD  Est: 2h  2026-03-06
  - Load() now calls NewHFPullFunc() on default LocalRegistry.
  - shouldDownload() includes .gguf files.
  - findGGUF() auto-detects GGUF files and routes to LoadFile().
  - Dependencies: none.

- [x] S45.1.1 Auto-download integration test (with mock HTTP)  Owner: TBD  Est: 1h  2026-03-06

- [x] T45.2 Model alias registry  Owner: TBD  Est: 2h  2026-03-06
  - Built-in alias map: gemma-3-{1b,2b}-q4, llama-3-{1b,8b}-q4, mistral-7b-q4, qwen-2.5-7b-q4.
  - ResolveAlias() and RegisterAlias() exported.
  - Dependencies: T45.1.

- [x] S45.2.1 Alias resolution tests  Owner: TBD  Est: 30m  2026-03-06

- [x] T45.3 Download progress and error UX  Owner: TBD  Est: 1h  2026-03-06
  - ProgressFunc already supported in HFPullOptions.OnProgress.
  - HF_TOKEN already supported via env var.
  - Dependencies: T45.1.

- [x] S45.3.1 Error message tests  Owner: TBD  Est: 30m  2026-03-06

- [x] T45.4 Run golangci-lint on registry/ and inference/  Owner: TBD  Est: 15m  2026-03-06

### E46: Chat Template Engine (O63)

- [x] T46.1 Template renderer  Owner: TBD  Est: 4h  2026-03-06
  - Pragmatic approach: hardcoded per-architecture formatters in
    inference/inference.go (formatGemma, formatLlama, formatMistral,
    formatQwen, formatDeepSeek, formatPhi, formatGeneric).
  - Dependencies: none.

- [x] S46.1.1 Template rendering tests for all 6 architectures  Owner: TBD  Est: 1h  2026-03-06

- [x] T46.2 Architecture-specific chat templates  Owner: TBD  Est: 2h  2026-03-06
  - All 6 templates implemented. Auto-detect via chatTemplateForArch()
    in inference/gguf.go, mapping GGUF general.architecture to template name.
  - Dependencies: T46.1.

- [x] S46.2.1 Chat format parity tests vs HuggingFace reference  Owner: TBD  Est: 1h  2026-03-06

- [x] T46.3 Run golangci-lint on inference/  Owner: TBD  Est: 15m  2026-03-06

### E47: K-Quant Dequantization (O64)

- [x] T47.1 Q4_K dequantization  Owner: TBD  Est: 4h  2026-03-06
  - Implemented Q4_K_M block dequantization in `tensor/quantized_kquant.go`.
  - Q4_K layout: super-blocks of 256 values (144 bytes each).
  - Wired into GGUF loader via `decodeQ4KTensor`.
  - Dependencies: none.

- [x] S47.1.1 Q4_K dequantization correctness tests  Owner: TBD  Est: 1h  2026-03-06

- [x] T47.2 Q6_K dequantization  Owner: TBD  Est: 3h  2026-03-06
  - Implemented Q6_K block dequantization (210-byte super-blocks, 6-bit values).
  - Dependencies: none.

- [x] T47.3 Q5_K dequantization  Owner: TBD  Est: 2h  2026-03-06
  - Implemented Q5_K_M block dequantization (176-byte super-blocks, 5-bit values).
  - Dependencies: none.

- [x] S47.3.1 Q5_K and Q6_K correctness tests  Owner: TBD  Est: 1h  2026-03-06

- [x] T47.4 Wire K-quants into GGUF loader  Owner: TBD  Est: 2h  2026-03-06
  - Extended `model/gguf/loader.go` with Q5_K and Q6_K tensorByteSize
    and decodeTensor cases.
  - Dependencies: T47.1, T47.2, T47.3.

- [x] S47.4.1 K-quant GGUF load test  Owner: TBD  Est: 30m  2026-03-06

- [x] T47.5 Run golangci-lint on tensor/ and model/gguf/  Owner: TBD  Est: 15m  2026-03-06

### E48: Phase 27 Deferred Benchmarks (O65)

These tasks were blocked on DGX Spark access during Phase 27.

- [ ] T48.1 End-to-end benchmark: transpose elimination  Owner: TBD  Est: 1h
  - Run Gemma 3 2B Q4 with FoldConstantTransposes enabled.
  - Profile with pprof. Verify Transpose < 5% of CPU.
  - Measure tok/s improvement vs Phase 26 baseline (3.60 tok/s).
  - Acceptance: measurable speedup. Transpose no longer dominant.
  - Dependencies: none. **Requires DGX Spark.**

- [ ] S48.1.1 Before/after profile comparison  Owner: TBD  Est: 30m

- [ ] T48.2 End-to-end benchmark: tensor pool  Owner: TBD  Est: 1h
  - Run Gemma 3 2B Q4 with TensorPool enabled.
  - Measure allocs/token and tok/s.
  - Acceptance: < 100 allocs/token. Measurable tok/s improvement.
  - Dependencies: T48.1. **Requires DGX Spark.**

- [ ] S48.2.1 Allocation profile comparison  Owner: TBD  Est: 30m

- [ ] T48.3 End-to-end benchmark: all fusions  Owner: TBD  Est: 1h
  - Run with transpose folding + tensor pool + fused RMSNorm/RoPE/SiLU-gate.
  - Measure final tok/s. Target >= 15 tok/s.
  - Compare against Phase 26 baseline (3.60 tok/s).
  - Acceptance: >= 15 tok/s on DGX Spark CPU.
  - Dependencies: T48.1, T48.2. **Requires DGX Spark.**

- [ ] S48.3.1 Final Phase 27+28 performance report  Owner: TBD  Est: 30m

---

## 4. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M16: GGUF inference | E44 | E37 | `LoadFile("model.gguf")` to generate text, Llama + Gemma |
| M17: One-command install | E45 | E44 | `inference.Load("gemma-3-1b-q4")` auto-downloads and runs |
| M18: Universal chat | E46 | none | `model.Chat()` correct for all 6 architectures |
| M19: K-quant ecosystem | E47 | none | Q4_K_M GGUFs from HuggingFace load and run |
| M20: Validated throughput | E48 | E39-E43 | >= 15 tok/s on DGX Spark (Phase 27 target) |

Recommended execution order:

1. **[E44, E47]** -- GGUF graph builders + K-quant dequant (independent, both
   needed for GGUF models from HuggingFace)
2. **E46** -- Chat templates (independent, small scope)
3. **E45** -- Auto-download (depends on E44 for GGUF loading to actually work)
4. **E48** -- DGX benchmarks (independent, requires hardware access)

Critical path: E44 -> E45 -> working README.

---

## 5. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R61 | GGUF graph builder produces incorrect attention patterns | Wrong output, NaN | High | Validate against llama.cpp logits for same model+prompt. Use known test vectors. |
| R62 | K-quant dequantization has subtle bit-packing errors | Slightly wrong weights, degraded quality | Medium | Compare dequantized values against llama.cpp `ggml_dequantize_row_q4_K` output. |
| R63 | HuggingFace API rate limits during auto-download | Load fails for new users | Medium | Cache aggressively. Retry with exponential backoff. Document `HF_TOKEN` for higher limits. |
| R64 | Chat template variations between model versions | Garbled chat output | Medium | Pin templates to specific model families, not versions. Test against HF tokenizer configs. |
| R65 | GGUF tokenizer metadata missing or malformed | Tokenizer fails to build | Medium | Fallback to `tokenizer.json` in same directory. Log clear warning. |
| R66 | Q4_K_M models have mixed quantization (some layers Q6_K) | Loader fails on unexpected quant type | High | Handle all K-quant types that appear in practice. Start by auditing a real Q4_K_M GGUF. |

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
- Use Conventional Commits: `feat(inference): add Llama GGUF graph builder`.
- Always run linters and formatters before committing.

### Validation Strategy

- For K-quant work: validate against llama.cpp reference dequantization
  (`ggml-quants.c`).
- For GGUF builders: parity tests comparing logits against a known-good
  implementation (Python transformers or llama.cpp).
- For chat templates: test against HuggingFace `tokenizer_config.json`
  reference strings.
- End-to-end tests use small models: Llama 3.2 1B Q4_0 (~700 MB),
  Gemma 3 1B Q4_0 (~700 MB).

---

## 7. Progress Log

### Change Summary -- 2026-03-06 (E44+E45+E46+E47 complete)

E45 (Model Hub & Auto-Download) ALL COMPLETE: T45.1-T45.4, S45.1.1-S45.3.1.
HF pull wired into default registry, model aliases for 6 models, GGUF auto-
detection in downloaded model directories, shouldDownload includes .gguf.

E46 (Chat Template Engine) ALL COMPLETE: T46.1-T46.3, S46.1.1-S46.2.1.
Hardcoded per-architecture formatters for Gemma, LLaMA 3, Mistral, Qwen 2.5,
DeepSeek, Phi-4. Auto-detection from GGUF architecture metadata. Table-driven
tests with 15 cases covering all templates.

### Change Summary -- 2026-03-06 (E44+E47 complete)

E44 (GGUF End-to-End Inference) ALL COMPLETE: T44.1-T44.5, S44.1.1-S44.4.1.
Llama and Gemma graph builders with shared transformer loop in arch_common.go.
GGUF tokenizer extraction, unified LoadFile, end-to-end generate tests.

E47 (K-Quant Dequantization) ALL COMPLETE: T47.1-T47.5, S47.1.1-S47.4.1.
Q4_K (144B/256val), Q6_K (210B/256val), Q5_K (176B/256val) dequantization.
All three wired into GGUF loader. Round-trip and storage tests pass.

Next: E46 (Chat Template Engine), then E45 (Auto-Download).

### Change Summary -- 2026-03-06

Trimmed completed Phases 25-27 from plan.md into docs/design.md (sections
15.7-15.9). Removed all completed epics (E25-E38, E39-E40 kernel tasks,
E43 kernel tasks). Merged Phase 28 content from docs/phase28.md as the
active plan. Added epics E44-E48 with 30 tasks targeting adoption blockers:
GGUF inference, auto-download, chat templates, K-quant dequantization,
and deferred DGX benchmarks.

Superseded tasks:
- E42 (GGUF End-to-End) replaced by E44 with refined scope and file paths.
- T39.3, T40.3, T43.4 (DGX benchmarks) carried forward to E48.
- E41 (GPU Inference) deferred to Phase 29.

### Previous Entries

| Date | Phase | Summary |
|------|-------|---------|
| 2026-03-05 | 27 | Merged Phase 27 epics (E39-E43). T39.1, T39.2, T40.1, T40.2, T43.1-T43.3 complete. |
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
  docs/adr/ (ADR-001 through ADR-019).
- **Phases 1-27:** All complete (kernel-level). See docs/design.md sections 15.7-15.9.
- **Phase 28:** This plan is the source of truth. See also docs/phase28.md for
  the standalone Phase 28 design document.
- **Quality:** See docs/QUALITY.md for test coverage report. 9 packages at 100%,
  42 of 50 at >= 95%.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - On DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build -tags cuda,cutlass ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key Phase 28 Starting Points

1. **E44 (start here):** GGUF graph builders. Key files: `inference/gguf.go`
   (existing stub), `model/gguf/` (parser + loader + arch mapping),
   `inference/inference.go` (Load function).
2. **E45:** Auto-download. Key files: `registry/registry.go` (model registry),
   `registry/pull.go` (HF download logic -- already implemented, just not wired).
3. **E46:** Chat templates. Key file: `inference/inference.go` (formatMessages method).
4. **E47:** K-quant dequant. Key files: `tensor/quantized.go` (Q4_0/Q8_0 reference),
   `model/gguf/parser.go` (GGUF type constants), `model/gguf/loader.go` (tensor loading).
   Reference: llama.cpp `ggml-quants.c`.
5. **E48 (DGX Spark required):** Deferred Phase 27 benchmarks.

### External Dependencies

- **DGX Spark (ndungu@192.168.86.250):**
  - Go 1.25.0 linux/arm64, CUDA 13.0, sm_121 (Blackwell).
  - Models: ~/models/gemma3-q4/ (Q4 ZMF), ~/models/gemma3/ (F32 ZMF).
  - Repos: ~/zerfoo/, ~/zonnx/, ~/zmf/.
- HuggingFace API for model downloads (HF_TOKEN for gated models).

### Baseline Performance Numbers

| Model | Params | Quant | CPU tok/s (DGX) | CPU Target |
|-------|--------|-------|-----------------|------------|
| Gemma 3 2B | 2.6B | Q4_0 | 3.60 | >= 15 |
| Gemma 3 2B | 2.6B | F32 | 3.51 | -- |
