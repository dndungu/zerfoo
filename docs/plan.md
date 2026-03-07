# Zerfoo Development Plan -- Phase 34: Close the Gap with llama.cpp

## 1. Context

### Problem Statement

Phase 33 achieved 10.32 tok/s peak (7.78 median) for Gemma 3 2B Q4 on DGX Spark
GB10. llama.cpp/Ollama on the same hardware achieves 24 tok/s for Gemma 3 12B Q4
and 38 tok/s for Llama 3.1 8B Q4. Zerfoo is at ~4.3% of the theoretical bandwidth
limit (273 GB/s / 1.5 GB model = ~182 tok/s) while llama.cpp reaches 13-21%.

The gap is caused by three categories of overhead:

1. **Per-op CGo kernel launch overhead**: Each forward pass dispatches 25+ GPU
   operations individually through CGo (~100ns per call) plus CUDA kernel launch
   latency. For single-token decode with small tensors, launch overhead dominates
   compute time. llama.cpp uses CUDA graph capture to eliminate this entirely.

2. **No kernel fusion**: Operations like Scale+Softmax, Gate*SiLU(Up), and
   elementwise chains are separate kernel launches with intermediate GPU memory
   round-trips. Each kernel reads and writes global memory. llama.cpp fuses these
   into single kernels that keep data in registers/shared memory.

3. **Per-op buffer allocation**: `ExecutionPlan.Run()` allocates intermediate GPU
   buffers from the memory pool on every op via `pool.Alloc/Free`. These shapes
   are fixed and known at compile time but allocated dynamically at runtime.

See docs/design.md for full architecture context and Phases 1-33 history.
Decision rationale: docs/adr/024-cuda-graph-fused-kernels.md.

### What Was Delivered (Phase 33)

| Area | Key Result |
|------|------------|
| PowScalar GPU kernel | Eliminated 8.9% Pow CPU fallback |
| SubScalar GPU kernel | Completed scalar-op coverage |
| Scalar-broadcast detection | Eliminated 10.4% binary op CPU fallback |
| GPU Split/Concat (D2D memcpy) | Eliminated 24% D2H from GPUStorage.Slice |
| Float32 weight upload | All weights GPU-resident at load time |
| Benchmark | 10.32 tok/s peak / 7.78 median GPU on DGX Spark GB10 |

### Current Architecture

The decode loop in `generate/stream.go` calls `ExecutionPlan.Run()` per token.
`Run()` iterates a flat `[]Instruction` array. Each instruction calls
`node.Forward(ctx, inputs)` which dispatches to `GPUEngine` methods. Each method
calls a CGo wrapper which launches a CUDA kernel. Buffer allocation happens inside
each GPUEngine method via `pool.Alloc`.

```
Token loop (Go)
  -> ExecutionPlan.Run() (Go, per-instruction loop)
    -> node.Forward() (Go)
      -> GPUEngine.Add() (Go)
        -> pool.Alloc() (Go)
        -> C.launch_add() (CGo -> CUDA kernel launch)
        -> pool.Free() for temporaries (Go)
```

Each of these layers adds latency. For a 26-layer transformer with ~25 ops per
layer, that is ~650 CGo round-trips per token.

### Objectives

- O80: Pre-allocate all intermediate GPU buffers at graph compile time.
- O81: Implement CUDA graph capture/replay for the decode forward pass.
- O82: Add fused CUDA kernels for SwiGLU, Scale+Softmax, and dequant+GEMV.
- O83: Achieve >= 20 tok/s median for Gemma 3 2B Q4 on DGX Spark GB10.

### Non-Goals

- Multi-GPU inference or tensor parallelism.
- BF16 elementwise kernels (considered but deferred; fused kernels provide more
  value for the same effort).
- Flash attention improvements (already integrated for the hot path).
- Speculative decoding (separate phase).
- Training pipeline changes.
- ROCm/OpenCL kernel implementations (stubs only).
- Vulkan, SYCL, or Metal backends.

### Constraints and Assumptions

- Go standard library only. No cobra, viper, testify.
- Build tags for GPU code (`//go:build cuda`).
- Pre-commit hook rejects multi-directory commits.
- golangci-lint, go vet, gofmt required for all changes.
- Tests must pass with `-race` flag.
- Table-driven tests using the standard `testing` package.
- DGX Spark GB10 at `ssh ndungu@192.168.86.250` for all GPU validation.
- Go 1.25.0, CUDA 13.0, sm_121 (Blackwell) on DGX Spark.
- Target model: Gemma 3 2B Q4 (ZMF), path: ~/models/gemma3-q4/model.zmf.
- GPU baseline from Phase 33: 10.32 tok/s peak, 7.78 tok/s median.
- CUDA graph capture requires fixed memory addresses across replays.
- ExecutionPlan already has the static graph structure needed for pre-allocation.
- Gemma 3 uses SwiGLU activation (gate * silu(up)) in every FFN block.
- Gemma 3 uses grouped-query attention with 8 heads, headDim=256.

### Success Metrics

| Metric | Current | Target | How Measured |
|--------|---------|--------|-------------|
| GPU tok/s median (Gemma 3 2B Q4) | 7.78 | >= 20 | bench_tps -device cuda, 7 runs, median |
| GPU tok/s peak | 10.32 | >= 25 | bench_tps -device cuda, best of 7 |
| Per-token decode latency | ~128ms | < 50ms | 1000/tok_s |
| Pool alloc calls per token | ~650 | 0 | pprof pool.Alloc count |
| CGo calls per token (graph mode) | ~650 | 1 | pprof runtime.cgocall count |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D414 | Pre-allocated buffer pool for ExecutionPlan | Eliminates per-op alloc; enables CUDA graph capture |
| D415 | CUDA graph CGo wrappers | Foundation for graph capture/replay |
| D416 | CUDA graph capture in ExecutionPlan | Eliminates all per-token CGo overhead |
| D417 | Fused SwiGLU CUDA kernel | Saves 2 launches + 1 intermediate per FFN block |
| D418 | Fused Scale+Softmax CUDA kernel | Saves 1 launch + 1 intermediate per attention head |
| D419 | Fused dequant+GEMV Q4 kernel | Eliminates separate dequant step in decode MatMul |
| D420 | End-to-end benchmark >= 20 tok/s | Validate all improvements |

### Out of Scope

- BF16 elementwise kernels (fused kernels subsume most of the benefit).
- Flash attention changes (already wired for non-causal inference).
- Prefill optimization (focus is on per-token decode latency).
- Quantized KV cache (separate phase, after decode speed is resolved).
- Multi-sequence batch decode.
- Megakernel / on-GPU interpreter approach (aspirational, not Phase 34).

---

## 3. Checkable Work Breakdown

### E80: Pre-Allocated Buffer Pool (O80)

At graph compile time, the shape of every intermediate tensor is known (from the
warmup Forward pass). Pre-allocate a fixed GPU buffer for each slot, eliminating
per-op pool.Alloc/Free calls. This is also a prerequisite for CUDA graph capture,
which requires fixed memory addresses.

Existing code:
- `graph/compile.go` -- ExecutionPlan with slots array and Compile() method.
- `compute/gpu_kernels.go` -- `makeGPUResult` allocates from pool per-op.
- `compute/gpu_engine.go` -- GPUEngine with pool (gpuapi.MemPool).
- `internal/gpuapi/mempool.go` -- MemPool interface (Alloc/Free).

- [ ] T80.1 Record intermediate tensor shapes during Compile  Owner: TBD  Est: 2h
  - In `Graph.Compile()`, after the warmup Forward pass, record the shape and
    byte size of each non-frozen slot's output tensor from `g.memo[n].Shape()`.
  - Store as `slotShapes [][]int` and `slotBytes []int` in ExecutionPlan.
  - Acceptance: After Compile(), `plan.slotShapes[i]` matches the shape that
    Forward() produces for instruction i.
  - Dependencies: none.

- [ ] S80.1.1 Test shape recording matches forward pass  Owner: TBD  Est: 1h
  - Run Compile(), then Run() with the same input. Verify every instruction's
    output shape matches the recorded slotShapes.

- [ ] T80.2 Pre-allocate GPU buffers for all intermediate slots  Owner: TBD  Est: 2h
  - Add `PreAllocate(pool gpuapi.MemPool, deviceID int) error` method to
    ExecutionPlan. Allocates one GPU buffer per non-frozen, non-input slot.
  - Store as `preAllocPtrs []unsafe.Pointer` in ExecutionPlan.
  - Acceptance: After PreAllocate(), every intermediate slot has a fixed GPU
    pointer. No pool.Alloc calls during Run().
  - Dependencies: T80.1.

- [ ] S80.2.1 Pre-allocation unit test  Owner: TBD  Est: 1h
  - PreAllocate() succeeds. Verify pointers are non-nil for compute slots.
  - Verify frozen slots and input slots have nil pre-alloc pointers.

- [ ] T80.3 Wire pre-allocated buffers into ExecutionPlan.Run  Owner: TBD  Est: 3h
  - Modify `Run()` to pass pre-allocated destination tensors to each instruction.
  - Each instruction's Forward call receives a pre-allocated output tensor via
    the `dst ...` parameter that GPUEngine methods already support.
  - The slot array stores pre-created tensors wrapping the pre-allocated GPU
    pointers (using `tensor.NewWithStorage` + `tensor.NewGPUStorageFromPtr`).
  - Acceptance: `Run()` produces correct output with zero pool.Alloc calls.
    Parity with non-pre-allocated Run() within 1e-5.
  - Dependencies: T80.2.
  - Risk: Some ops may not support the `dst` parameter or may allocate internally.
    Identify and fix these cases.

- [ ] S80.3.1 Pre-allocated Run parity test  Owner: TBD  Est: 1.5h
  - Run the same input through pre-allocated and non-pre-allocated plans.
  - Verify outputs match within 1e-5.
  - Verify zero pool.Alloc calls during pre-allocated Run (instrument pool).

- [ ] T80.4 Run golangci-lint on graph/ and compute/  Owner: TBD  Est: 15m
  - Dependencies: T80.3.

### E81: CUDA Graph CGo Wrappers (O81)

Add CGo wrappers for the CUDA graph API. These are used by the ExecutionPlan
to capture and replay the decode forward pass as a CUDA graph.

Existing code:
- `internal/cuda/` -- CUDA runtime wrappers (runtime.go, kernels/).
- `internal/gpuapi/runtime.go` -- Runtime interface with Stream.

- [ ] T81.1 Add CUDA graph CGo wrappers  Owner: TBD  Est: 2h
  - Create `internal/cuda/graph.go` with CGo wrappers for:
    - `cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)`
    - `cudaStreamEndCapture(stream, &graph)`
    - `cudaGraphInstantiate(&graphExec, graph, 0)` (CUDA 12+ simplified API)
    - `cudaGraphLaunch(graphExec, stream)`
    - `cudaGraphExecDestroy(graphExec)`
    - `cudaGraphDestroy(graph)`
  - Expose as Go functions: `BeginCapture(stream)`, `EndCapture(stream)`,
    `Instantiate(graph)`, `Launch(exec, stream)`, `DestroyExec(exec)`,
    `DestroyGraph(graph)`.
  - Acceptance: Compiles on CUDA build. Functions callable from Go.
  - Dependencies: none.

- [ ] S81.1.1 CUDA graph wrapper smoke test  Owner: TBD  Est: 1h
  - On DGX Spark: create stream, begin capture, launch a simple Add kernel,
    end capture, instantiate, launch graph exec, verify correct output.

- [ ] T81.2 Add CUDAGraph interface to gpuapi  Owner: TBD  Est: 1.5h
  - Add to `internal/gpuapi/runtime.go`:
    ```
    type GraphCapture interface {
        BeginCapture(stream Stream) error
        EndCapture(stream Stream) (unsafe.Pointer, error) // returns graph
        Instantiate(graph unsafe.Pointer) (unsafe.Pointer, error) // returns exec
        Launch(exec unsafe.Pointer, stream Stream) error
        DestroyExec(exec unsafe.Pointer)
        DestroyGraph(graph unsafe.Pointer)
    }
    ```
  - Implement for CUDA in `internal/gpuapi/cuda_runtime.go`.
  - Add stub implementations for ROCm and OpenCL (return "not implemented").
  - Acceptance: Compiles on all backends. Interface satisfied.
  - Dependencies: T81.1.

- [ ] S81.2.1 Interface compile-time verification  Owner: TBD  Est: 30m
  - Add `var _ gpuapi.GraphCapture = ...` assertions in gpuapi_test.go.

- [ ] T81.3 Run golangci-lint on internal/cuda/ and internal/gpuapi/  Owner: TBD  Est: 15m
  - Dependencies: T81.2.

### E82: CUDA Graph Capture in ExecutionPlan (O81)

Integrate CUDA graph capture into the ExecutionPlan decode loop. On the first
decode token, record the entire forward pass as a CUDA graph. On subsequent tokens,
replay the captured graph with a single `cudaGraphLaunch` call.

Existing code:
- `graph/compile.go` -- ExecutionPlan.Run() instruction loop.
- `generate/stream.go` -- token generation loop calling plan.Run().
- `generate/generator.go` -- Generator with plan compilation.

- [ ] T82.1 Add CaptureAndReplay mode to ExecutionPlan  Owner: TBD  Est: 3h
  - Add fields to ExecutionPlan:
    - `graphExec unsafe.Pointer` -- cached CUDA graph exec handle.
    - `graphCapture gpuapi.GraphCapture` -- graph API handle.
    - `captured bool` -- whether the graph has been captured.
  - Add method `SetGraphCapture(gc gpuapi.GraphCapture)` to enable graph mode.
  - Modify `Run()`:
    - If graph mode enabled and not captured: call BeginCapture, run all
      instructions normally (they record into the stream), call EndCapture
      + Instantiate, set captured=true.
    - If captured: update input slot GPU pointers (copy new input data into
      pre-allocated input buffer), then call Launch(graphExec, stream).
  - Acceptance: First Run() captures the graph. Subsequent Run() calls use
    graphExec with 1 CGo call instead of ~650.
  - Dependencies: E80 (pre-allocated buffers required), E81 (graph API).

- [ ] S82.1.1 CUDA graph capture correctness test  Owner: TBD  Est: 2h
  - Run 10 sequential tokens through graph-captured plan.
  - Compare each token's output with non-graph plan output within 1e-5.
  - Verify only 1 CGo call per token after first capture.

- [ ] T82.2 Wire graph capture into Generator  Owner: TBD  Est: 1.5h
  - In `generate/generator.go`, after plan compilation, call
    `plan.SetGraphCapture(engine.GraphCapture())` if the engine supports it.
  - Add `GraphCapture() gpuapi.GraphCapture` method to GPUEngine.
  - Acceptance: Generator uses CUDA graph replay for decode tokens.
  - Dependencies: T82.1.

- [ ] S82.2.1 Generator graph mode integration test  Owner: TBD  Est: 1h
  - Generate 20 tokens with graph mode. Verify coherent output.

- [ ] T82.3 Handle graph invalidation on sequence length change  Owner: TBD  Est: 1.5h
  - When the KV cache grows past a threshold (e.g., crosses a power-of-2 boundary),
    the graph topology may change (attention mask shape changes).
  - Detect this in Run(): if input shape differs from captured shape, re-capture.
  - Store `capturedInputShape []int` for comparison.
  - Acceptance: Graph re-captures when input shape changes. No stale output.
  - Dependencies: T82.1.

- [ ] S82.3.1 Re-capture correctness test  Owner: TBD  Est: 1h
  - Force input shape change mid-generation. Verify re-capture produces
    correct output.

- [ ] T82.4 Run golangci-lint on graph/ and generate/  Owner: TBD  Est: 15m
  - Dependencies: T82.3.

### E83: Fused SwiGLU Kernel (O82)

Gemma 3 FFN blocks compute `gate * silu(up)` where gate and up are separate MatMul
outputs. Currently this is 3 operations: Mul(gate, Silu(up)). Fusing into one
kernel saves 2 kernel launches and 1 intermediate buffer.

SwiGLU formula: `output[i] = gate[i] * (up[i] * sigmoid(up[i]))` which is
`output[i] = gate[i] * up[i] / (1 + expf(-up[i]))`.

- [ ] T83.1 Write CUDA fused SwiGLU kernel  Owner: TBD  Est: 2h
  - Add to `internal/cuda/kernels/fused_swiglu.cu`:
    ```c
    __global__ void kernel_fused_swiglu(const float* gate, const float* up,
                                         float* output, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float u = up[idx];
            output[idx] = gate[idx] * u / (1.0f + expf(-u));
        }
    }
    ```
  - Add launcher function `launch_fused_swiglu`.
  - Add Go CGo wrapper in `internal/cuda/kernels/fused_swiglu.go`.
  - Acceptance: Kernel compiles for sm_121. Correct output for n=2048.
  - Dependencies: none.

- [ ] S83.1.1 Fused SwiGLU kernel unit test  Owner: TBD  Est: 1h
  - Test: fused_swiglu(gate, up) matches gate * silu(up) within 1e-5.
  - Test: edge cases (up=0, up=large positive, up=large negative).

- [ ] T83.2 Add FusedSwiGLU to KernelRunner interface  Owner: TBD  Est: 1h
  - Add `FusedSwiGLU(gate, up, output unsafe.Pointer, n int, stream Stream) error`
    to KernelRunner.
  - Wire in cuda_kernels.go. Add stubs in rocm/opencl.
  - Acceptance: Compiles on all backends.
  - Dependencies: T83.1.

- [ ] T83.3 Wire fused SwiGLU into GPUEngine  Owner: TBD  Est: 2h
  - Add `FusedSwiGLU` interface to compute/engine.go (like FusedRMSNormer).
  - Implement in GPUEngine: takes gate and up tensors, returns fused output.
  - Modify `layers/activations/swiglu.go` to type-assert engine for FusedSwiGLU
    and use fused path when available.
  - Acceptance: SwiGLU layer uses fused kernel on GPU. Output matches unfused
    within 1e-5.
  - Dependencies: T83.2.

- [ ] S83.3.1 Fused SwiGLU parity test  Owner: TBD  Est: 1h
  - Compare fused vs unfused SwiGLU for shapes [1, 8192] and [1, 16384].
  - Verify output has GPUStorage.

- [ ] T83.4 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Dependencies: T83.3.

### E84: Fused Scale+Softmax Kernel (O82)

Attention scores are scaled by 1/sqrt(headDim) then softmaxed. Currently these
are separate MulScalar and Softmax kernel launches. Fusing them saves 1 launch
and 1 intermediate buffer per attention computation.

- [ ] T84.1 Write CUDA fused scale+softmax kernel  Owner: TBD  Est: 2.5h
  - Add to `internal/cuda/kernels/fused_scale_softmax.cu`:
    A kernel that reads input[row, col], multiplies by scale, computes row-wise
    softmax (find max, subtract max, exp, sum, divide) in shared memory.
  - Parameters: input, output pointers, scale (float), rows, cols.
  - Use shared memory for the max and sum reductions.
  - Acceptance: Kernel compiles for sm_121. Correct for rows=8, cols=256.
  - Dependencies: none.

- [ ] S84.1.1 Fused scale+softmax kernel unit test  Owner: TBD  Est: 1h
  - Test: fused(input, 1/sqrt(256)) matches MulScalar(input, scale) + Softmax
    within 1e-5.
  - Test: rows sum to 1.0 within 1e-6.

- [ ] T84.2 Add FusedScaleSoftmax to KernelRunner interface  Owner: TBD  Est: 1h
  - Add `FusedScaleSoftmax(input, output unsafe.Pointer, scale float32, rows, cols int, stream Stream) error`.
  - Wire and add stubs.
  - Acceptance: Compiles on all backends.
  - Dependencies: T84.1.

- [ ] T84.3 Wire fused scale+softmax into attention  Owner: TBD  Est: 2h
  - Add `FusedScaleSoftmaxer` interface to compute/engine.go.
  - Implement in GPUEngine.
  - Modify `layers/attention/scaled_dot_product_attention.go` to detect and use
    fused path when the engine supports it and flash attention is not used.
  - Acceptance: Attention layer uses fused scale+softmax on GPU.
  - Dependencies: T84.2.

- [ ] S84.3.1 Fused scale+softmax attention parity test  Owner: TBD  Est: 1h
  - Run attention forward pass with fused vs unfused paths.
  - Verify outputs match within 1e-5.

- [ ] T84.4 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Dependencies: T84.3.

### E85: Fused Dequant+GEMV Q4 Kernel (O82)

For single-token decode (batch=1), the Q4 MatMul is a GEMV: one activation
vector times a Q4 weight matrix. Currently, Q4 weights are dequantized in a
separate step before GEMM. A fused kernel reads Q4 blocks, dequantizes in
registers, and accumulates in F32 -- eliminating the dequantize buffer and
reducing memory bandwidth by ~4x (read Q4 instead of F32).

This is the single largest compute kernel in the decode loop. The existing
`gemm_q4.cu` kernel handles the GEMM case; this adds a GEMV fast path.

- [ ] T85.1 Write CUDA fused dequant+GEMV Q4 kernel  Owner: TBD  Est: 4h
  - Add to `internal/cuda/kernels/gemv_q4.cu`:
    - Each thread block computes one output element.
    - Read Q4_0 blocks (32 values per 18-byte block: 1 scale + 16 packed bytes).
    - Dequantize in registers: `val = (nibble - 8) * scale`.
    - Dot product with activation vector, accumulate in float32.
    - Block-level reduction via shared memory.
  - Parameters: q4_weights, activation, output, M (output rows), K (inner dim).
  - Acceptance: Kernel compiles for sm_121. Correct output for M=2048, K=2048.
  - Dependencies: none.
  - Risk: Q4_0 block layout must match `tensor/q4_storage.go` format exactly.

- [ ] S85.1.1 Fused dequant+GEMV Q4 unit test  Owner: TBD  Est: 1.5h
  - Test: GEMV output matches dequant + GEMM output within 1e-3.
  - Test: M=256, K=2048 (small) and M=8192, K=2048 (large).
  - Test: Edge case where K is not a multiple of 32.

- [ ] T85.2 Add GemvQ4 to KernelRunner interface  Owner: TBD  Est: 1h
  - Add `GemvQ4F32(aQ4, b, c unsafe.Pointer, m, k int, stream Stream) error`.
  - Wire in cuda_kernels.go. Add stubs.
  - Acceptance: Compiles on all backends.
  - Dependencies: T85.1.

- [ ] T85.3 Wire fused GEMV Q4 into GPUEngine.MatMul  Owner: TBD  Est: 2h
  - In `matMulQ4`, detect the GEMV case: when the activation tensor has 1 row
    (batch=1 decode), use `GemvQ4F32` instead of `GemmQ4F32`.
  - Acceptance: Single-token decode uses GEMV kernel. Output matches GEMM path
    within 1e-3.
  - Dependencies: T85.2.

- [ ] S85.3.1 Fused GEMV Q4 parity test  Owner: TBD  Est: 1h
  - Compare GEMV vs GEMM for shapes used in Gemma 3 decode (1x2048 * 2048x8192).
  - Verify correctness and that GEMV is used (log or counter).

- [ ] T85.4 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Dependencies: T85.3.

### E86: End-to-End Benchmark (O83)

After all optimizations, measure tok/s on DGX Spark and compare with baselines.

- [ ] T86.1 Profile GPU inference after all optimizations  Owner: TBD  Est: 2h
  - Build zerfoo with CUDA tags on DGX Spark including all E80-E85 changes.
  - Run `bench_tps -model ~/models/gemma3-q4 -device cuda -tokens 100`.
  - 7 runs, report median and peak.
  - Capture pprof profile.
  - Acceptance: Profile shows reduced cgocall overhead and fused kernel usage.
  - Dependencies: E80, E81, E82, E83, E84, E85.

- [ ] S86.1.1 GPU profile report  Owner: TBD  Est: 30m
  - Document: tok/s (median + peak), cgocall %, pool.Alloc count per token,
    fused kernel % of compute, remaining bottlenecks.

- [ ] T86.2 Compare GPU vs CPU and vs Phase 33  Owner: TBD  Est: 1h
  - Run bench_tps with -device cpu and -device cuda.
  - 7 runs each, report median.
  - Compare with Phase 33 baselines (7.78 median GPU, 6.75 CPU).
  - Acceptance: GPU tok/s >= 20 median. If not met, identify remaining
    bottleneck and document what would close the gap.
  - Dependencies: T86.1.

- [ ] S86.2.1 Benchmark comparison report  Owner: TBD  Est: 30m

- [ ] T86.3 Verify output correctness  Owner: TBD  Est: 1h
  - Generate 50 tokens with same prompt on CPU, GPU (no graph), GPU (graph).
  - Compare outputs. All should produce coherent text.
  - Acceptance: No NaN or Inf. Coherent output on all modes.
  - Dependencies: T86.1.

- [ ] S86.3.1 Output correctness report  Owner: TBD  Est: 30m

- [ ] T86.4 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m
  - Dependencies: T86.3.

---

## 4. Parallel Work

Seven epics fall into 4 tracks. E80 (buffer pre-alloc) and E81 (CUDA graph API)
are prerequisites for E82 (graph capture). E83/E84/E85 (fused kernels) are
independent of each other and of E80/E81. E86 (benchmark) depends on all.

| Track | Epics | Description | Sync Points |
|-------|-------|-------------|-------------|
| A: Buffer Pre-Alloc | E80 | Pre-allocate intermediate GPU buffers | Required before E82 |
| B: CUDA Graph API | E81 | CGo wrappers for CUDA graph capture | Required before E82 |
| C: CUDA Graph Capture | E82 | Wire graph capture into ExecutionPlan | After A + B |
| D: Fused Kernels | E83, E84, E85 | SwiGLU, Scale+Softmax, dequant+GEMV | Independent, converge at E86 |
| E: Benchmark | E86 | End-to-end validation | After C + D |

### Within-Track Parallelism

| Parallel Set | Tasks | Notes |
|-------------|-------|-------|
| Wave 1 | T80.1, T81.1, T83.1, T84.1, T85.1 | All independent: shape recording, graph wrappers, 3 fused kernels |
| Wave 2 | T80.2, T81.2, T83.2, T84.2, T85.2 | Buffer alloc, graph interface, kernel interfaces |
| Wave 3 | T80.3, T81.3, T83.3, T84.3, T85.3 | Buffer wiring, lint, layer integration |
| Wave 4 | T82.1, T82.2, T82.3 | CUDA graph capture (depends on E80 + E81) |
| Wave 5 | E86 | Benchmark (depends on all) |

### Execution Order

Wave 1 (parallel, all independent):
- T80.1 (record shapes), T81.1 (CUDA graph CGo), T83.1 (SwiGLU kernel),
  T84.1 (scale+softmax kernel), T85.1 (GEMV Q4 kernel).

Wave 2 (after Wave 1):
- T80.2 (pre-allocate buffers), T81.2 (graph interface), T83.2 (SwiGLU interface),
  T84.2 (scale+softmax interface), T85.2 (GEMV interface).

Wave 3 (after Wave 2):
- T80.3 (wire into Run), T83.3 (wire SwiGLU), T84.3 (wire scale+softmax),
  T85.3 (wire GEMV). Lint tasks: T80.4, T81.3, T83.4, T84.4, T85.4.

Wave 4 (after E80 + E81):
- T82.1 (capture mode), T82.2 (wire into Generator), T82.3 (re-capture).

Wave 5 (after all):
- E86 (benchmark + correctness).

---

## 5. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M45: Fixed buffers | E80 | none | ExecutionPlan.Run() uses pre-allocated buffers. Zero pool.Alloc during decode. |
| M46: Graph API ready | E81 | none | CUDA graph capture/replay works for simple kernel sequence. |
| M47: Graph decode | E82 | M45, M46 | Decode loop uses CUDA graph replay. 1 CGo call per token. |
| M48: Fused kernels | E83, E84, E85 | none | SwiGLU, Scale+Softmax, dequant+GEMV fused. Parity with unfused. |
| M49: 20 tok/s | E86 | M47, M48 | bench_tps reports >= 20 tok/s median on DGX Spark GB10. |

Critical path: T80.1 -> T80.2 -> T80.3 -> T82.1 -> T82.2 -> T86.1

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R82 | CUDA graph capture fails for ops with dynamic behavior (e.g., KV cache append, conditional branches) | Graph capture mode unusable | High | Isolate dynamic ops outside the captured region. Only capture the static compute graph; handle KV cache updates separately. Profile to identify which ops break capture. |
| R83 | Pre-allocated buffers waste GPU memory for variable-length sequences | OOM for long sequences | Medium | Allocate for max sequence length at compile time. DGX Spark has 128GB unified memory; Gemma 3 2B intermediates are ~100MB. Not a concern for this model. |
| R84 | Fused dequant+GEMV Q4 kernel has different numerical precision than separate dequant+GEMM | Output quality regression | Medium | Test within 1e-3 tolerance (Q4 already has quantization noise). Compare generated text quality. |
| R85 | CUDA graph re-capture overhead negates benefits if triggered too frequently | Throughput drops during re-captures | Low | Only re-capture on shape change. For autoregressive decode, shape is fixed (batch=1, seq=1). Re-capture only needed for prefill or context length change. |
| R86 | CGo overhead for the single graph launch is still significant | Diminishing returns from graph capture | Low | One CGo call per token is ~100ns, negligible vs 50ms decode. |
| R87 | 20 tok/s target still not achieved after all optimizations | Unknown bottleneck remains | Medium | Profile after each epic. If CUDA graphs alone reach 15+ tok/s, fused kernels may close the rest. If not, investigate memory bandwidth utilization and consider BF16 intermediates. |
| R88 | DGX Spark GB10 thermal throttling causes high variance | Unreliable benchmark results | Medium | Run benchmarks after 5-minute cooldown. Use median of 7+ runs. Monitor GPU temperature via nvidia-smi. |

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
8. CUDA build (`go build -tags cuda ./...`) compiles on DGX Spark.
9. Changes are committed in a small commit touching one directory only.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(cuda): add fused SwiGLU kernel`.
- Always run linters and formatters before committing.

### DGX Spark Protocol

- SSH: `ssh ndungu@192.168.86.250`
- Go: `/usr/local/go/bin/go`
- CUDA: `/usr/local/cuda/bin/nvcc`, `CGO_CFLAGS='-I/usr/local/cuda/include'`,
  `CGO_LDFLAGS='-L/usr/local/cuda/lib64'`
- GPU: NVIDIA GB10, sm_121, `make CUDA_ARCH=sm_121`
- Model: `~/models/gemma3-q4/model.zmf`
- Repo: `~/zerfoo/`

### Benchmark Protocol

- 7 runs minimum, report median and peak.
- All GPU benchmarks on DGX Spark GB10.
- Use `bench_tps -device cuda -tokens 100` for tok/s measurement.
- Use `bench_tps -device cpu -tokens 100` for CPU comparison.
- Capture pprof with `-cpuprofile` flag.
- 5-minute GPU cooldown between benchmark sessions.

### Quality Gate

- `go test -race ./package/`
- `golangci-lint run ./package/`
- `go vet ./package/`
- `go build ./...` (non-CUDA)
- `go build -tags cuda ./...` (CUDA, on DGX Spark)

---

## 8. Progress Log

### Change Summary -- 2026-03-06

Created Phase 34 plan. Trimmed Phase 33 completed epics (E75-E79) to
docs/design.md section 15.15. Created ADR 024 for CUDA graph and fused kernel
strategy.

New epics:
- E80: Pre-Allocated Buffer Pool (eliminate per-op pool.Alloc).
- E81: CUDA Graph CGo Wrappers (foundation for graph capture).
- E82: CUDA Graph Capture in ExecutionPlan (1 CGo call per token).
- E83: Fused SwiGLU Kernel (save 2 launches per FFN block).
- E84: Fused Scale+Softmax Kernel (save 1 launch per attention head).
- E85: Fused Dequant+GEMV Q4 Kernel (eliminate separate dequant step).
- E86: End-to-End Benchmark (validate >= 20 tok/s target).

ADRs created:
- docs/adr/024-cuda-graph-fused-kernels.md: CUDA graph capture and fused kernel
  strategy decision.

Trimmed from Phase 33:
- E75 (PowScalar), E76 (Scalar-Broadcast), E77 (GPU Split/Concat), E78 (Float32
  Weight Upload), E79 (Benchmark) -- all completed tasks moved to docs/design.md
  section 15.15. Remaining subtask tests (S75.1.1, S76.1.1, etc.) archived as
  they are validation-only and covered by the end-to-end benchmark.

---

## 9. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, and troubleshooting. Design decisions in docs/adr/ (001-024).
- **Phases 1-33:** All documented in docs/design.md sections 15.1-15.15.
- **Phase 34:** This plan is the source of truth.
- **Quality:** See docs/QUALITY.md for test coverage report.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - On DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build -tags cuda ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key Starting Points

1. **E80 (Buffer Pre-Alloc):** `graph/compile.go` -- ExecutionPlan.Compile() has
   access to tensor shapes from warmup Forward. Add slotShapes/slotBytes fields,
   PreAllocate() method, and wire into Run() via dst parameters.

2. **E81 (CUDA Graph API):** Create `internal/cuda/graph.go` with CGo wrappers for
   cudaStreamBeginCapture, cudaStreamEndCapture, cudaGraphInstantiate,
   cudaGraphLaunch. Add GraphCapture interface to internal/gpuapi/runtime.go.

3. **E82 (Graph Capture):** Modify ExecutionPlan.Run() to capture the instruction
   loop as a CUDA graph on first call, then replay on subsequent calls.

4. **E83 (Fused SwiGLU):** `layers/activations/swiglu.go` computes
   `gate * silu(up)`. Add a fused CUDA kernel and FusedSwiGLU engine interface.

5. **E84 (Fused Scale+Softmax):** `layers/attention/scaled_dot_product_attention.go`
   does MulScalar(scores, scale) then Softmax. Fuse into one kernel.

6. **E85 (Fused GEMV Q4):** `compute/gpu_engine.go matMulQ4()` calls GemmQ4F32.
   Add GemvQ4F32 for batch=1 case. Q4 block format is in tensor/q4_storage.go.

### Performance Baselines

| Model | Quant | Device | tok/s | Phase |
|-------|-------|--------|-------|-------|
| Gemma 3 2B | Q4_0 | CPU ARM64 | 6.86 | 30 |
| Gemma 3 2B | Q4_0 | CPU ARM64 | 5.94 | 31 (bench_tps) |
| Gemma 3 2B | Q4_0 | GPU (cuda) | 5.12 | 31 (bench_tps) |
| Gemma 3 2B | Q4_0 | CPU ARM64 | 6.61 | 32 (bench_tps) |
| Gemma 3 2B | Q4_0 | GPU (cuda) | 6.84 | 32 (bench_tps) |
| Gemma 3 2B | Q4_0 | CPU ARM64 | 6.75 | 33 (bench_tps) |
| Gemma 3 2B | Q4_0 | GPU (cuda) | 10.32 peak / 7.78 median | 33 (bench_tps, 7 runs) |

### External References

| Source | Key Insight |
|--------|------------|
| NVIDIA CUDA Graphs blog | cudaStreamBeginCapture + cudaGraphLaunch pattern |
| llama.cpp CUDA backend | Fused dequant+GEMV, SwiGLU, CUDA graph capture |
| DGX Spark specs | 273 GB/s LPDDR5x, ~182 tok/s theoretical for 1.5GB model |
| Ollama benchmarks | Gemma 3 12B Q4: 24 tok/s, Llama 3.1 8B Q4: 38 tok/s on GB10 |

---

## 10. Appendix

### Existing File Reference

| File | Purpose |
|------|---------|
| `graph/compile.go` | ExecutionPlan: Compile(), Run(), Instruction struct |
| `compute/gpu_engine.go` | GPUEngine with CUDA acceleration + CPU fallback |
| `compute/gpu_kernels.go` | gpuBinaryOp, gpuScalarOp, gpuBroadcastOp, gpuSplit, gpuConcat |
| `compute/engine.go` | Engine[T] interface, FusedRMSNormer interface |
| `internal/cuda/kernels/elementwise.cu` | CUDA element-wise kernels (25+ ops) |
| `internal/cuda/kernels/elementwise.go` | Go CGo wrappers for CUDA kernels |
| `internal/cuda/kernels/gemm_q4.cu` | Q4 dequant+GEMM CUDA kernel |
| `internal/cuda/kernels/rmsnorm.cu` | Fused RMSNorm CUDA kernel |
| `internal/gpuapi/kernels.go` | KernelRunner interface |
| `internal/gpuapi/runtime.go` | Runtime + Stream + MemcpyKind interfaces |
| `generate/stream.go` | Token generation loop (GenerateStream) |
| `generate/generator.go` | Generator with plan compilation |
| `layers/activations/swiglu.go` | SwiGLU activation (gate * silu(up)) |
| `layers/attention/scaled_dot_product_attention.go` | SDPA with flash attention path |
| `tensor/q4_storage.go` | Q4Storage with Q4_0 block format |
| `cmd/bench_tps/main.go` | tok/s benchmark binary |

### Estimated Effort Summary

| Epic | Area | Tasks | Estimated Hours |
|------|------|-------|----------------|
| E80: Buffer Pre-Alloc | graph/, compute/ | 4 tasks + 3 subtests | 9.75h |
| E81: CUDA Graph API | internal/cuda/, internal/gpuapi/ | 3 tasks + 2 subtests | 5.0h |
| E82: Graph Capture | graph/, generate/ | 4 tasks + 3 subtests | 9.0h |
| E83: Fused SwiGLU | internal/cuda/, compute/, layers/ | 4 tasks + 2 subtests | 6.25h |
| E84: Fused Scale+Softmax | internal/cuda/, compute/, layers/ | 4 tasks + 2 subtests | 6.75h |
| E85: Fused Dequant+GEMV | internal/cuda/, compute/ | 4 tasks + 2 subtests | 8.75h |
| E86: Benchmark | DGX Spark | 4 tasks + 3 subtests | 4.25h |
| **Total** | **CUDA graphs + fused kernels** | **27 tasks + 17 subtests** | **~50h** |
