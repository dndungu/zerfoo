# Phase 28 -- Make It Actually Work

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

Audit of README promises vs implementation:

| Promise | Status | Blocker |
|---------|--------|---------|
| `inference.Load("gemma-3-1b-q4")` auto-downloads | Fails — default registry has no pull function | E45 |
| GGUF models load and run | Parser only — no graph builder | E44 |
| Chat works with all listed models | Only Gemma template, others get broken fallback | E46 |
| Q4 quantization | Only Q4_0/Q8_0 — most HuggingFace GGUFs use Q4_K_M | E47 |
| `zerfoo run` / `zerfoo serve` | Code exists but blocked by all of the above | All |

Phase 27 delivered significant internal improvements (transpose elimination,
tensor pool, fused kernels) but none of them are visible to a new user who
can't get past model loading. Phase 28 fixes the front door.

### Phase 27 Summary

| Epic | Status | Result |
|------|--------|--------|
| E39: Transpose Elimination | Kernel complete, benchmark pending | FoldConstantTransposes + blocked 4D transpose |
| E40: Tensor Arena | Kernel complete, benchmark pending | TensorPool + ref-counted release in graph.Forward |
| E41: GPU Inference | Not started | Blocked on DGX Spark access |
| E42: GGUF End-to-End | Not started | Carried forward to E44 |
| E43: Operator Fusion | Complete | Fused RMSNorm (28x), RoPE (9→1 calls), SiLU-gate |

Remaining Phase 27 benchmark tasks (T39.3, T40.3, T43.4) require DGX Spark
and are deferred to E48.

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

### Success Metrics

| Metric | Current | Phase 28 Target |
|--------|---------|-----------------|
| `inference.Load()` works out-of-box | No (missing pull func) | Yes |
| GGUF → inference | No (parser only) | Yes (Llama + Gemma) |
| Chat template coverage | 1/6 (Gemma only) | 6/6 |
| Quantization format coverage | Q4_0, Q8_0 | + Q4_K_M, Q5_K_M, Q6_K |
| `zerfoo run model.gguf` works | No | Yes |
| End-to-end integration tests | Gemma ZMF only | Gemma + Llama, GGUF + ZMF |

---

## 2. Scope and Deliverables

### In Scope

- GGUF architecture-specific graph builders (Llama, Gemma).
- GGUF tokenizer extraction from metadata.
- Unified `inference.Load()` that handles both ZMF and GGUF.
- HuggingFace model auto-download wired into default registry.
- Chat templates for all listed architectures.
- K-quant dequantization (Q4_K, Q5_K, Q6_K).
- End-to-end integration tests for the README examples.
- DGX Spark benchmarks deferred from Phase 27.

### Out of Scope

- GPU inference (Phase 29).
- New model architectures beyond the six already listed.
- Embeddings / hidden state extraction.
- GGUF write / model export.
- Quantization (converting F32 → Q4_K) — only dequantization for inference.

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|---------------------|
| D91 | GGUF end-to-end inference | `inference.Load("path/to/model.gguf")` → `model.Generate()` works for Llama and Gemma |
| D92 | Auto-download | `inference.Load("gemma-3-1b-q4")` downloads from HF, no manual setup |
| D93 | Chat templates | `model.Chat()` produces correctly formatted output for all 6 architectures |
| D94 | K-quant support | Q4_K_M, Q5_K_M, Q6_K GGUFs dequantize correctly |
| D95 | Working CLI | `zerfoo run model.gguf` starts interactive chat from a fresh install |

---

## 3. Checkable Work Breakdown

### E44: GGUF End-to-End Inference (O61, O62)

Carried forward from Phase 27 E42 with refined scope.

- [ ] T44.1 Graph template builder for Llama architecture  Est: 6h
  - Create `inference/arch_llama.go` with
    `buildLlamaGraph(tensors map[string]*tensor.TensorNumeric[float32], cfg ModelMetadata, engine Engine) (*graph.Graph[float32], error)`.
  - Build the standard Llama graph: Embed → [RMSNorm → GQA → RMSNorm →
    FFN(SiLU-gate)] × N → RMSNorm → LMHead.
  - Map canonical tensor names (from E37 arch mapping) to graph parameters.
  - Support GQA (num_kv_heads < num_heads).
  - Acceptance: Llama 3.2 1B GGUF loads and produces non-NaN logits.
    Greedy decode of "The capital of France is" produces "Paris".
  - Dependencies: none.

- [ ] S44.1.1 Llama GGUF forward pass + decode test  Est: 1h

- [ ] T44.2 Graph template builder for Gemma architecture  Est: 4h
  - Extend for Gemma differences: shared embed/lm_head weight tying,
    GeGLU activation (GELU instead of SiLU in MLP), embedding scaling
    by sqrt(hidden_size), pre/post normalization differences.
  - Acceptance: Gemma 3 1B GGUF loads and produces non-NaN logits.
    Greedy decode produces coherent text.
  - Dependencies: T44.1.

- [ ] S44.2.1 Gemma GGUF forward pass + decode test  Est: 1h

- [ ] T44.3 GGUF tokenizer extraction  Est: 3h
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

- [ ] S44.3.1 GGUF tokenizer encode/decode tests  Est: 1h

- [ ] T44.4 Unified load function for GGUF  Est: 3h
  - Extend `inference.Load()` to detect GGUF files (by extension or magic
    bytes) and dispatch to the GGUF loading path.
  - Add `inference.LoadFile(path string, opts ...Option) (*Model, error)`
    for loading from a local file path directly.
  - Architecture dispatch based on GGUF `general.architecture` metadata.
  - Wire tokenizer, graph, engine, and generator into a ready-to-use Model.
  - Acceptance: `inference.LoadFile("model.gguf")` → `model.Generate(ctx, prompt)`
    works end-to-end.
  - Dependencies: T44.1, T44.2, T44.3.

- [ ] S44.4.1 End-to-end GGUF load + generate test  Est: 1h

- [ ] T44.5 Run golangci-lint on inference/ and model/gguf/  Est: 15m

### E45: Model Hub & Auto-Download (O61, O65)

- [ ] T45.1 Wire HuggingFace pull into default registry  Est: 2h
  - Modify `inference.Load()` to configure `NewHFPullFunc()` on the default
    registry when no custom registry is provided.
  - Support GGUF downloads: when model ID contains "gguf" or the HF repo
    has `.gguf` files, download the GGUF file instead of ONNX.
  - Support model ID formats: `"gemma-3-1b-q4"` (short alias),
    `"google/gemma-3-1b-it-qat-q4_0-gguf"` (full HF repo ID).
  - Acceptance: `inference.Load("gemma-3-1b-q4")` downloads the model on
    first call, caches it, and loads from cache on subsequent calls.
  - Dependencies: none.

- [ ] S45.1.1 Auto-download integration test (with mock HTTP)  Est: 1h

- [ ] T45.2 Model alias registry  Est: 2h
  - Create a built-in alias map from short names to HuggingFace repo IDs:
    ```
    "gemma-3-1b-q4" → "google/gemma-3-1b-it-qat-q4_0-gguf"
    "gemma-3-2b-q4" → "google/gemma-3-2b-it-qat-q4_0-gguf"
    "llama-3-1b-q4" → "meta-llama/Llama-3.2-1B-Instruct-GGUF"
    "llama-3-8b-q4" → "meta-llama/Llama-3.1-8B-Instruct-GGUF"
    "mistral-7b-q4" → "mistralai/Mistral-7B-Instruct-v0.3-GGUF"
    ```
  - Allow users to register custom aliases via `inference.RegisterAlias()`.
  - Acceptance: `inference.Load("llama-3-1b-q4")` resolves to the correct
    HuggingFace repo and downloads the right file.
  - Dependencies: T45.1.

- [ ] S45.2.1 Alias resolution tests  Est: 30m

- [ ] T45.3 Download progress and error UX  Est: 1h
  - Print download progress to stderr during `inference.Load()` when
    downloading (size, speed, ETA).
  - Clear error messages when: network unavailable, model not found,
    disk full, auth required for gated models.
  - Support `HF_TOKEN` environment variable for gated model access.
  - Acceptance: downloading a 700 MB model shows progress bar.
    `inference.Load("nonexistent-model")` returns a clear error.
  - Dependencies: T45.1.

- [ ] S45.3.1 Error message tests  Est: 30m

- [ ] T45.4 Run golangci-lint on registry/ and inference/  Est: 15m

### E46: Chat Template Engine (O63)

- [ ] T46.1 Template renderer  Est: 4h
  - Implement a minimal Jinja2-compatible template renderer in
    `inference/chat_template.go` that handles the subset used by LLM
    chat templates: `{% for %}`, `{% if %}`, `{{ variable }}`,
    string filters (`trim`, `strip`).
  - Or: hardcode the 6 known templates if Jinja2 parsing is too complex.
    Pragmatic approach: each architecture has a known, stable template.
  - Acceptance: renderer produces correct output for all 6 architectures
    given the same message list.
  - Dependencies: none.

- [ ] S46.1.1 Template rendering tests for all 6 architectures  Est: 1h

- [ ] T46.2 Architecture-specific chat templates  Est: 2h
  - Add templates for:
    - **Gemma 3**: `<start_of_turn>user\n...<end_of_turn>\n` (already done)
    - **LLaMA 3**: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>`
    - **Mistral**: `[INST] ... [/INST]`
    - **Qwen 2.5**: `<|im_start|>user\n...<|im_end|>\n`
    - **DeepSeek**: `<|begin▁of▁sentence|>User: ...\n\nAssistant:`
    - **Phi-4**: `<|user|>\n...<|end|>\n<|assistant|>\n`
  - Auto-detect template from model metadata (`chat_template` field in
    config.json, or `general.architecture` in GGUF).
  - Acceptance: `model.Chat()` produces the correct prompt format for
    each architecture. Verified by comparing formatted prompts against
    reference strings from HuggingFace tokenizer configs.
  - Dependencies: T46.1.

- [ ] S46.2.1 Chat format parity tests vs HuggingFace reference  Est: 1h

- [ ] T46.3 Run golangci-lint on inference/  Est: 15m

### E47: K-Quant Dequantization (O64)

- [ ] T47.1 Q4_K dequantization  Est: 4h
  - Implement Q4_K_M block dequantization in `tensor/quantized_kquant.go`.
  - Q4_K layout: super-blocks of 256 values. Each super-block has a
    fp16 d (scale), fp16 dmin (min), and 12 bytes of scales/mins for
    8 sub-blocks, plus 128 bytes of packed 4-bit quantized values.
  - Implement `DequantizeQ4K(block []byte) []float32`.
  - Wire into GGUF loader: when tensor type is `GGMLTypeQ4_K`,
    dequantize to float32 on load.
  - Acceptance: dequantized values match llama.cpp reference output
    within 1e-3 tolerance.
  - Dependencies: none.

- [ ] S47.1.1 Q4_K dequantization correctness tests  Est: 1h

- [ ] T47.2 Q6_K dequantization  Est: 3h
  - Implement Q6_K block dequantization.
  - Q6_K layout: super-blocks of 256 values. Higher precision than Q4_K
    (6 bits per value), used for important layers.
  - Acceptance: dequantized values match llama.cpp reference within 1e-3.
  - Dependencies: none.

- [ ] T47.3 Q5_K dequantization  Est: 2h
  - Implement Q5_K_M block dequantization. Layout is similar to Q4_K
    with 5-bit values.
  - Acceptance: dequantized values match llama.cpp reference within 1e-3.
  - Dependencies: none.

- [ ] S47.3.1 Q5_K and Q6_K correctness tests  Est: 1h

- [ ] T47.4 Wire K-quants into GGUF loader  Est: 2h
  - Extend `model/gguf/loader.go` to handle Q4_K, Q5_K, Q6_K tensor
    types via dequantize-on-load (convert to float32).
  - Update GGUF parser tensor type enum with K-quant constants.
  - Acceptance: a GGUF model quantized with `llama.cpp -Q4_K_M` loads
    without errors.
  - Dependencies: T47.1, T47.2, T47.3.

- [ ] S47.4.1 K-quant GGUF load test  Est: 30m

- [ ] T47.5 Run golangci-lint on tensor/ and model/gguf/  Est: 15m

### E48: Phase 27 Deferred Benchmarks (O65)

These tasks were blocked on DGX Spark access during Phase 27.

- [ ] T48.1 End-to-end benchmark: transpose elimination  Est: 1h
  - Run Gemma 3 2B Q4 with FoldConstantTransposes enabled.
  - Profile with pprof. Verify Transpose < 5% of CPU.
  - Measure tok/s improvement vs Phase 26 baseline (3.60 tok/s).
  - Acceptance: measurable speedup. Transpose no longer dominant.
  - Dependencies: none. **Requires DGX Spark.**

- [ ] S48.1.1 Before/after profile comparison  Est: 30m

- [ ] T48.2 End-to-end benchmark: tensor pool  Est: 1h
  - Run Gemma 3 2B Q4 with TensorPool enabled.
  - Measure allocs/token and tok/s.
  - Acceptance: < 100 allocs/token. Measurable tok/s improvement.
  - Dependencies: T48.1. **Requires DGX Spark.**

- [ ] S48.2.1 Allocation profile comparison  Est: 30m

- [ ] T48.3 End-to-end benchmark: all fusions  Est: 1h
  - Run with transpose folding + tensor pool + fused RMSNorm/RoPE/SiLU-gate.
  - Measure final tok/s. Target >= 15 tok/s.
  - Compare against Phase 26 baseline (3.60 tok/s).
  - Acceptance: >= 15 tok/s on DGX Spark CPU.
  - Dependencies: T48.1, T48.2. **Requires DGX Spark.**

- [ ] S48.3.1 Final Phase 27+28 performance report  Est: 30m

---

## 4. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M16: GGUF inference | E44 | E37 | `LoadFile("model.gguf")` → generate text, Llama + Gemma |
| M17: One-command install | E45 | E44 | `inference.Load("gemma-3-1b-q4")` auto-downloads and runs |
| M18: Universal chat | E46 | none | `model.Chat()` correct for all 6 architectures |
| M19: K-quant ecosystem | E47 | none | Q4_K_M GGUFs from HuggingFace load and run |
| M20: Validated throughput | E48 | E39-E43 | >= 15 tok/s on DGX Spark (Phase 27 target) |

Recommended execution order:

1. **[E44, E47]** — GGUF graph builders + K-quant dequant (independent, both
   needed for GGUF models from HuggingFace)
2. **E46** — Chat templates (independent, small scope)
3. **E45** — Auto-download (depends on E44 for GGUF loading to actually work)
4. **E48** — DGX benchmarks (independent, requires hardware access)

Critical path: E44 → E45 → working README.

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

Same as Phase 27 (see docs/phase27.md section 6):

- TDD: write tests first, then implement.
- Single-directory commits.
- `golangci-lint run` must report 0 issues before marking a task complete.
- For K-quant work: validate against llama.cpp reference dequantization.
- For GGUF builders: parity tests comparing logits against a known-good
  implementation (Python transformers or llama.cpp).
- For chat templates: test against HuggingFace `tokenizer_config.json`
  reference strings.

---

## 7. Hand-off Notes

### For a New Contributor

- **Read first**: This file for Phase 28 scope. `docs/phase27.md` for the
  optimization work that precedes this. `docs/design.md` for architecture.
- **The README gap**: The README at the repo root shows what the UX should
  look like. Phase 28's job is to make every example in the README work.
- **Key files for E44**: `inference/gguf.go` (existing stub), `model/gguf/`
  (parser + loader + arch mapping), `inference/inference.go` (Load function).
- **Key files for E45**: `registry/registry.go` (model registry),
  `registry/pull.go` (HF download logic — already implemented, just not wired).
- **Key files for E46**: `inference/inference.go` (formatMessages method).
- **Key files for E47**: `tensor/quantized.go` (Q4_0/Q8_0 reference),
  `model/gguf/parser.go` (GGUF type constants), `model/gguf/loader.go`
  (tensor loading).
- **Reference implementation**: `llama.cpp` source code is the definitive
  reference for GGUF format, K-quant dequantization, and chat templates.
  The `ggml-quants.c` file contains all dequantization routines.

### Testing Strategy

End-to-end integration tests should use small models:
- Llama 3.2 1B Q4_0 GGUF (~700 MB) for Llama path
- Gemma 3 1B Q4_0 GGUF (~700 MB) for Gemma path
- Test vectors: fixed prompt "The capital of France is" → greedy decode
  should produce "Paris" (or close) as first token.
- For K-quants: download a Q4_K_M model and verify forward pass produces
  non-NaN logits. Compare top-5 token probabilities against llama.cpp.
