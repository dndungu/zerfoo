# Phase 29 Updates

## 2026-03-06: Q4 B-Operand NEON + Parallel GEMV (6.5 tok/s)

### Summary

Implemented full Q4 B-operand fast path with NEON assembly and multi-core
parallelism. Throughput: **3.80 → 6.5 tok/s** (1.7x improvement).

### Changes

1. **NEON q4DotBlockSIMD** (`045ad78`, `df0fcbb`): ARM64 assembly for fused
   Q4 nibble extraction + float32 dot product in NEON registers. Replaces
   per-block scalar dequant.

2. **q4DotRowSIMD** (`78674e4`, `b0290c4`): Row-level assembly that processes
   an entire row of Q4 blocks in a single call. Eliminates per-block Go
   function call overhead (BlockScaleF32→float16.ToFloat32, BlockData,
   q4DotBlock = 4 calls × 248K blocks). Uses `LDR H + FCVT S,H` for
   float16→float32 scale conversion directly in NEON.

3. **GemmF32Q4NT** (`78674e4`): Computes C = A × B^T where B is [N,K] in
   Q4 format. Reads Q4 blocks directly from weight memory without transpose
   or dequantization. The Transpose layer passes Q4 storage through with
   transposed shape; the engine detects Q4 on B operand and dispatches here.

4. **Parallel Q4 GEMV** (`5c06704`): For M=1 GEMV with N*K >= 64K, splits
   the N dimension across runtime.NumCPU() goroutines. Each computes
   independent output elements via q4DotRow.

5. **Decode transpose short-circuit** (`c702bf7`): When attention transpose
   swaps a dimension of size 1 (seq_len=1 during decode), skip the blocked
   copy and use plain `copy()`. Reduced memclr from 0.47s→0.15s.

### Benchmark Results (DGX Spark GB10, Gemma 3 2B Q4_0)

| Config | tok/s | CPU util |
|--------|-------|----------|
| Phase 28 baseline | 3.80 | ~230% |
| + Transpose cache | 5.55 | ~230% |
| + NEON q4DotBlockSIMD | 3.51 | ~230% |
| + q4DotRowSIMD (row-level) | 4.04 | ~230% |
| + Parallel Q4 GEMV | 5.73 | ~236% |
| + Decode transpose short-circuit | **6.5** | ~236% |

### Profile Breakdown (post-optimization, 30 tokens)

| Component | % CPU | Wall ms/token | Notes |
|-----------|-------|---------------|-------|
| sgemmAccRowNeon (SGEMM) | 35.2% | ~12 | Float32 lm_head/embedding |
| q4DotRowSIMD (Q4 GEMV) | 34.6% | ~12 | Q4 weight layers |
| binaryOp (Mul/Add) | 6.7% | ~2 | Element-wise ops |
| Transpose | 3.9% | ~3 | Attention reshapes |
| GC/malloc | ~5% | ~15 | Reduced by TensorPool |
| Other | ~15% | ~130 | Graph traversal, scheduling |

### Bottleneck Analysis

**Compute is fast, overhead dominates.** The two GEMM paths (Q4 + float32)
account for ~70% of CPU but only ~24ms wall time per token (parallelized).
The remaining ~150ms/token is overhead:

1. **Graph traversal**: ~780 node executions per token (30+ nodes × 26 layers).
   Each node: interface dispatch, shape validation, pool acquire/release.
2. **Memory management**: TensorPool reduces allocations but not to zero.
   GC still significant.
3. **Goroutine scheduling**: Per-MatMul goroutine launch/sync (~130 MatMul
   calls per token, each spawning/joining 20 goroutines).

### Why 15 tok/s requires architectural changes

To reach 15 tok/s (67ms/token), we need to cut overhead from ~150ms to ~43ms.
This requires:
- Fused operation graphs (batch multiple ops per kernel launch)
- Worker pool instead of per-call goroutine creation
- Zero-copy tensor views for reshape/transpose-of-1D
- Graph compilation to eliminate per-node dispatch overhead

These are beyond the scope of Phase 29's planned tasks.

### FCVT encoding bug fix (`b0290c4`)

The WORD encoding `0x1E22E0E7` was `FCVT H7,S7` (single→half), not
`FCVT S7,H7` (half→single). Caused SIGILL on NVIDIA Grace (Neoverse V2).
Correct encoding: `0x1EE240E7` (ftype=11, opc=000100).

---

## Previous Updates

### E50 COMPLETE -- TensorPool wired into Generator

- `15bb955` T50.1+S50.1.1: Pool created in `NewGenerator`, attached via `graph.WithPool()`.
- `8e5678c` T50.2+S50.2.1: Allocation benchmark added.
- T50.3: golangci-lint passing.

### E51 COMPLETE -- KV cache decode already optimal + RoPE bug fixed

- Q/K/V projections: already optimal (single-token decode).
- KV cache: already optimal (append + return full sequence).
- `502986d` RoPE position offset fix (was always using position 0 during decode).
- `559b7e4` GQA sets RoPE offset to cache.SeqLen().

### T52.1 COMPLETE -- Constant transpose elimination

- `a004055` Transpose layer data-pointer cache (3.53 → 5.42 tok/s).
- `9dc3272` MatMul B-operand cache + LMHead tied weight cache.

### Q4 Storage Fix

- ZMF loader now keeps Q4 weights in Q4Storage (4x less memory).
- Q4Storage.Slice() caches dequantized result.

### Known Issues

- Q4 ZMF model produces garbage output (pre-existing, not caused by Phase 29).
