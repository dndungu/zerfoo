# Phase 29 Updates

## 2026-03-06: E50 COMPLETE -- TensorPool wired into Generator

- `15bb955` T50.1+S50.1.1: Pool created in `NewGenerator`, attached via `graph.WithPool()`.
- `8e5678c` T50.2+S50.2.1: Allocation benchmark added. Trivial test graph shows no
  difference (no intermediate tensors to pool). Real model graph on DGX Spark needed
  to validate the 79K -> <1K allocs/token target.
- T50.3: golangci-lint already passing, marked complete.

## 2026-03-06: T51.1 -- KV Cache Decode Audit

**Findings (all good except RoPE):**

1. **Q/K/V projections: ALREADY OPTIMAL.** Single-token decode passes `[1,1]` input.
   Each attention layer projects only the new token (`layers/attention/grouped_query_attention.go:236-249`).
   No full-sequence reprocessing.

2. **KV cache: ALREADY OPTIMAL.** New K/V appended at cursor position, full cached
   sequence returned for attention (`generate/kvcache.go:99-143`). Single-token decode
   reuses all cached K/V from previous steps.

3. **Causal masking: ALREADY OPTIMAL.** No mask applied during decode because Q has
   length 1 and naturally attends to all cached positions
   (`layers/attention/grouped_query_attention.go:224-227`).

4. **RoPE: BUG FOUND.** During decode, `seqLen=1` causes RoPE to always use position
   index 0 (`cosAngles[0:1]`) instead of the actual sequence position. All decode-step
   keys get position-0 rotation. No position offset mechanism exists. This is exactly
   what T51.3 (Incremental RoPE for decode) should fix. The model still generates
   coherent text because prefill tokens have correct positions and the position error
   only affects decode tokens (short generations mask the impact).

**Decision:** T51.2 (optimize decode graph path) can be marked "already optimized" for
Q/K/V projection and KV cache. The only actionable optimization is T51.3 (RoPE fix).

## 2026-03-06: T51.3 + S51.3.1 -- RoPE position offset fix

**Bug fixed:** During decode, RoPE always used position 0 for all generated tokens.

**Fix:**
- `502986d` Added `SetPositionOffset(offset int)` to `RotaryPositionalEmbedding`.
  Both fused and unfused paths now slice cos/sin from `[offset:offset+seqLen]`.
  Test verifies single-position-with-offset matches full-sequence RoPE at that position.
- `559b7e4` In `GroupedQueryAttention.Forward`, before applying RoPE, set offset to
  `cache.SeqLen()` (the number of tokens already cached). During prefill, offset=0
  (no cache yet). During decode step N, offset=prefillLen+N-1.

**Impact:** This correctness fix should improve generation quality especially for
longer sequences where absolute position information matters. No performance cost
(offset is a simple integer stored on the RoPE struct).

## E51 COMPLETE -- KV cache decode already optimal + RoPE bug fixed

## 2026-03-06: T49.1 + S49.1.1 -- Q4 dot product scalar implementation

- `29a8f20` Added `q4DotBlock` in `internal/xblas/q4dot.go`. Scalar fallback that
  fuses nibble extraction and dot product in a single pass (no intermediate buffer).
  Tests verify parity against `dequantQ4Block`+manual dot for 6 test cases + real
  Q4 data. Benchmarks included.
- **Remaining E49 tasks (T49.2-T49.4):** Require NEON assembly on DGX Spark ARM64.
  The scalar `q4DotBlock` is the building block; `GemmQ4F32Fused` M=1 integration
  needs a different approach for N>1 (B columns are strided). The NEON version
  should process nibble extraction in SIMD registers and accumulate against B rows
  using the existing `sgemmAccRow` pattern but with in-register dequant.
- **Key insight:** For M=1 with large N (decode: 4096-8192), the bottleneck is
  the 32 sgemmAccRow calls per Q4 block. The NEON kernel should batch these by
  dequantizing nibbles into float32x4 registers and doing 4-wide FMLA directly.

## Session Summary -- Phase 29 Progress

| Epic | Status | Notes |
|------|--------|-------|
| E49 | T49.1 done | Scalar q4DotBlock + tests. NEON asm needs DGX Spark. |
| E50 | COMPLETE | TensorPool wired into Generator. Alloc benchmark ready. |
| E51 | COMPLETE | KV cache already optimal. RoPE position offset bug fixed. |
| E52 | Not started | NEON fused ops (RMSNorm, SiLU-gate, transpose). |
| E53 | Not started | DGX Spark benchmark validation. |

## What needs DGX Spark (ssh ndungu@192.168.86.250)

All remaining work requires ARM64 hardware:

1. **E49 T49.2-T49.4:** Write `q4dot_arm64.s` NEON assembly. The kernel should:
   - VLD1 16 packed bytes, VAND/USHR for nibble extraction
   - Widen uint4->int16->int32->float32, subtract 8 (zero-point)
   - FMLA against activation float32x4 registers
   - Integrate into GemmQ4F32Fused M=1 decode path

2. **E52 T52.1:** Profile to confirm 8.8% transpose is still a bottleneck after
   E50/E51 optimizations. Option A (K stored transposed in cache) is simplest.

3. **E52 T52.2-T52.3:** NEON RMSNorm and SiLU-gate assembly files.

4. **E53:** End-to-end benchmark with all optimizations enabled.

**RoPE fix (T51.3) may improve tok/s** — the position-0 bug meant decode tokens
lacked correct positional information. Rerun benchmark before and after to measure.
