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
