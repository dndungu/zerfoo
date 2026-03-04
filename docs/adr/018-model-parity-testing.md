# ADR-018: Model Parity Testing on DGX Spark

## Status

Accepted (2026-03-04)

## Context

Phase 20 validated all 66 GPU packages on the DGX Spark GB10 (Blackwell sm_121,
CUDA 13.0.2, ARM64). However, 25+ parity tests were skipped because no ZMF
model files existed on the device. The parity test framework in
`tests/parity/helpers_test.go` uses `envOrSkip(t, key)` to gate on environment
variables like `LLAMA3_ZMF_PATH` and `QWEN_ZMF_PATH`.

Phase 21 addressed this by downloading ONNX models from HuggingFace, converting
them to ZMF via the `zonnx` CLI, deploying on DGX Spark, and running the full
parity suite. Ten ONNX-compatibility bugs surfaced and were fixed.

## Decision

Download, convert (ONNX -> ZMF via zonnx), and deploy model files for 7 model
families on DGX Spark. Fix ONNX compatibility issues as they surface. Use
`optimum-cli export onnx` when HuggingFace repos lack pre-built ONNX files.

## Bug Fixes

Ten issues were fixed across the zerfoo and zonnx repositories:

| # | Issue | Fix | Commit |
|---|-------|-----|--------|
| 1 | Reshape: ONNX dynamic 2-input mode unsupported | Support 2-input Reshape with shape tensor | 2a520c4 |
| 2 | Builder: dynamic shape input dropped for Reshape | Keep shape input wiring in builder | f757b29 |
| 3 | Builder: ZMF proto fields (Perm, Epsilon, Axis) not promoted | Promote proto fields during load | b12847a |
| 4 | Where: scalar broadcasting missing | Add scalar broadcasting to Where op | 7f8d73f |
| 5 | Builder: Reshape node not rebuilt after shape extraction (Pass 2) | Rebuild Reshape in Pass 2 | 4644fe3 |
| 6 | MatMul: batched (4D+) matmul unsupported | Support batched matmul dimensions | 5b94cbb |
| 7 | tensor_decoder: empty tensor data not guarded | Return error for empty data | da82e78 |
| 8 | Cos/Sin: elementwise layers missing for RoPE | Add Cos and Sin layers | e119caa |
| 9 | zonnx: external data files not loaded | Load ONNX external data (model.onnx_data) | 8830eb5 |
| 10 | Generator: 3D input shape incompatible with Gather | Use 2D [1, seqLen] input shape | 8f003ad |

## Results

| Test | Status | Notes |
|------|--------|-------|
| FlashAttentionGQA | PASS | |
| Llama3 ForwardPass | PASS | onnx-community/Llama-3.2-1B |
| Llama3 GreedyDecode | PASS | |
| Llama3 Generation | PASS | |
| Qwen25 ForwardPass | PASS | Qwen/Qwen2.5-0.5B |
| Qwen25 GreedyDecode | PASS | |
| Qwen25 Generation | PASS | |
| MultiGPU DualDevice | SKIP | Single GPU device |
| Mistral (3 tests) | SKIP | No ZMF: HF auth required |
| Phi4 (3 tests) | SKIP | No ZMF: HF auth required |
| Gemma3 (3 tests) | SKIP | No ZMF: HF auth required |
| DeepSeek (3 tests) | SKIP | No ZMF: model too large |
| SigLIP (1 test) | SKIP | No ZMF: HF auth required |

**Summary:** 8 PASS, 13 SKIP (5 families blocked on HF auth/download size,
1 test blocked on single-GPU hardware).

## Infrastructure

- **Test automation:** `scripts/dgx-spark-parity.sh` sets ZMF env vars and runs
  the full parity suite.
- **ONNX export:** Python venv with `optimum-cli export onnx` for models without
  pre-built ONNX files on HuggingFace.
- **ZMF conversion:** `zonnx convert <onnx-dir> <output.zmf>` on DGX Spark.

## Consequences

- Remaining 5 model families (Mistral, Phi4, Gemma3, DeepSeek, SigLIP) are
  blocked on HuggingFace authentication or download size constraints.
- Multi-GPU parity test requires a second DGX Spark unit connected via
  ConnectX-7.
- The 10 bug fixes improved ONNX compatibility for all models, not just the
  two tested families.
