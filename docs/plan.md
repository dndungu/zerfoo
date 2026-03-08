# Zerfoo Development Plan -- Phase 34: Close the Gap with llama.cpp

## 1. Context

See docs/design.md for full architecture context and Phases 1-33 history.
See docs/design.md section 15.16 for Phase 34 completed work (Tracks 0/A/B and Track D).

### Problem Statement

Phase 33 achieved 10.32 tok/s peak (7.78 median) for Gemma 3 2B Q4 on DGX Spark
GB10. Ollama/llama.cpp on the same hardware achieves ~100 tok/s. The theoretical
max is ~182 tok/s (273 GB/s / 1.5GB model).

Track A (purego) and Track B (megakernel infrastructure) are mostly complete.
The megakernel does NOT fire on the real model because graph.Compile() records
composite node OpTypes (GroupedQueryAttention, FFN, EmbeddingLookup, LMHead)
instead of primitive Engine ops (MatMul, Softmax, Add, etc.). The codegen
emitter table only knows primitives, so CheckSupport() fails and falls back.

Track C (tracing compiler) resolves this: an EngineProxy records every Engine
method call during compilation, automatically decomposing composite nodes into
primitive ops. See docs/adr/028-tracing-compiler.md.

Track D (NEON SIMD CPU acceleration) is complete. Achieved 8.15 tok/s median
(+18.8% over 6.86 baseline). Target was 10 tok/s; remaining gap requires GEMM
cache tiling. See docs/design.md section 15.16 and docs/adr/029-neon-simd-cpu-acceleration.md.

### Objectives

- O96: Refactor remaining layers with inline math to compose Engine primitives.
- O87: Replace remaining CGo CUDA bindings with dlopen-based pure Go bindings.
- O88: Eliminate remaining `//go:build cuda` tags.
- O97: Implement tracing compiler that decomposes composite nodes into primitives.
- O98: Implement GPU KV cache for megakernel attention.
- O89: Generate a single-kernel decode for Gemma 3 2B transformer.
- O90: Achieve >= 50 tok/s median for Gemma 3 2B Q4 on DGX Spark GB10.

### Non-Goals

- Multi-GPU inference or tensor parallelism.
- Megakernel for batch > 1.
- Training pipeline changes.
- ROCm/OpenCL/Vulkan/Metal backends.
- Prefill optimization (focus is per-token decode latency).
- Quantized KV cache.
- AMD64 AVX2 SIMD (future, same pattern as ARM64).

### Constraints and Assumptions

- Go standard library only. No third-party test/CLI/DI libraries.
- Pre-commit hook rejects multi-directory commits.
- golangci-lint, go vet, gofmt required for all changes.
- Tests must pass with `-race` flag.
- Table-driven tests using the standard `testing` package.
- DGX Spark GB10 at `ssh ndungu@192.168.86.250` for all GPU validation.
- Go 1.25.0, CUDA 13.0, sm_121 (Blackwell) on DGX Spark.
- Target model: Gemma 3 2B Q4 (ZMF), path: ~/models/gemma3-q4/model.zmf.

### Success Metrics

| Metric | Current | Target | How Measured |
|--------|---------|--------|-------------|
| GPU tok/s median | 7.78 | >= 50 | bench_tps -device cuda, 7 runs, median |
| GPU tok/s peak | 10.32 | >= 60 | bench_tps -device cuda, best of 7 |
| CPU tok/s ARM64 | 8.15 | >= 10 | bench_tps -device cpu, 7 runs, median |
| Kernel launches per token | ~650 | 1 | nsys profile |
| Global mem round-trips | ~650 | ~26 (weight reads only) | nsys profile |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Track | Rationale |
|----|-------------|-------|-----------|
| D430 | Remaining layers compose Engine primitives | 0 | Prerequisite for tracing |
| D431 | EngineProxy with tracing mode | C | Automatic primitive decomposition |
| D432 | GPU KV cache for megakernel | C | Attention in megakernel needs GPU-resident KV |
| D433 | CompileTraced() in graph/ | C | Traced instruction tape with primitive ops |
| D434 | Megakernel fires on real Gemma 3 model | C+B | End-to-end integration |
| D425 | Megakernel performance tuning | B | nsys profiling, memory optimization |
| D429 | End-to-end benchmark >= 50 tok/s | B | Final validation |

### Out of Scope

- Megakernel for non-transformer architectures (future, automatic via tracing).
- CUDA graph capture (subsumed by megakernel).
- Individual fused kernels (subsumed by megakernel).
- purego loaders for cublas/cudnn/nccl/tensorrt.
- AMD64 AVX2 SIMD implementations (future, same file pattern).

---

## 3. Checkable Work Breakdown

### Track 0: Remaining Composition Fixes

E96 Priority 1 (T96.1-T96.3) and Priority 2 (T96.4-T96.5) are complete.
See docs/design.md section 15.16. Remaining:

#### E96: Refactor Violated Layers (remaining)

##### Priority 2: Other Model Variants (continued)

- [x] T96.6 Refactor Conv2d to im2col + engine.MatMul  Owner: TBD  Est: 3h  2026 03 08
  - File: `layers/core/conv2d.go` lines 60-144.
  - Violation: 6-nested loop convolution with direct data access.
  - Fix: im2col transform then engine.MatMul(weight_matrix, col_matrix).
  - Acceptance: Zero nested compute loops in Forward(). Output within 1e-5.
  - Dependencies: none.

- [x] S96.6.1 Conv2d composition parity test  Owner: TBD  Est: 1h  2026 03 08

##### Priority 3: Specialized Layers (not on Gemma 3 path)

- [x] T96.7 Refactor MoEGate to compose engine primitives  Owner: TBD  Est: 2h  2026 03 08
  - File: `layers/core/moe.go` lines 43-100.
  - Fix: engine.Softmax for routing (already composed). TopK selection retains data access (no engine.TopK primitive).
  - Dependencies: none.

- [x] T96.8 Refactor MixtureOfExperts to compose engine primitives  Owner: TBD  Est: 2h  2026 03 08
  - File: `layers/core/moe.go` lines 217-282.
  - Fix: engine.MulScalar for expert weight scaling, engine.Add for accumulation, engine.Concat for token outputs.
  - Dependencies: T96.7.

- [x] S96.8.1 MoE composition parity test  Owner: TBD  Est: 1h  2026 03 08
  - All existing MoE tests pass with composed implementation.

- [x] T96.9 Refactor PolynomialExpansion to compose engine primitives  Owner: TBD  Est: 1.5h  2026 03 08
  - File: `layers/core/polynomial.go` lines 191-249.
  - Fix: engine.Pow for each term, engine.MulScalar, engine.Add.
  - Dependencies: none.

- [x] S96.9.1 Polynomial composition parity test  Owner: TBD  Est: 45m  2026 03 08

- [x] T96.10 Refactor SpectralFingerprint to compose engine primitives  Owner: TBD  Est: 2h  2026 03 08
  - File: `layers/core/spectral_fingerprint.go` lines 96-157.
  - Fix: engine.MatMul with precomputed Fourier basis matrix.
  - Dependencies: none.

- [ ] T96.11 Refactor S4 layer to compose engine primitives  Owner: TBD  Est: 3h
  - File: `layers/sequence/s4.go` lines 184-222.
  - Fix: Per-step engine.Exp, engine.Mul, engine.Add for scan.
  - Dependencies: none.

- [ ] S96.11.1 S4 composition parity test  Owner: TBD  Est: 1h

- [ ] T96.12 Refactor SpectralFeature to remove Gonum FFT  Owner: TBD  Est: 2h
  - File: `layers/features/spectral.go` lines 57-77.
  - Fix: engine.MatMul with precomputed Fourier basis.
  - Dependencies: none.

- [ ] S96.12.1 SpectralFeature composition parity test  Owner: TBD  Est: 45m

##### Quality Gate

- [ ] T96.13 Run golangci-lint on all modified layer packages  Owner: TBD  Est: 30m
  - Dependencies: T96.6-T96.12.

- [ ] T96.14 Run full test suite and verify no regressions  Owner: TBD  Est: 1h
  - Dependencies: T96.13.

- [ ] S96.14.1 Composition audit verification  Owner: TBD  Est: 30m
  - Grep all layers Forward() methods for tensor.Data() compute access.

---

### Track A: Remaining purego Cleanup

- [ ] T87.3 Replace runtime.go CGo functions with dlopen calls  Owner: TBD  Est: 3h
  - Rewrite `internal/cuda/runtime.go` to remove `import "C"`.
  - Dependencies: T87.2 (done).

- [ ] S87.3.1 Runtime function parity test  Owner: TBD  Est: 1.5h

- [ ] S88.2.1 Elementwise kernel parity test  Owner: TBD  Est: 1.5h

- [ ] S88.3.1 Full kernel test suite  Owner: TBD  Est: 2h

- [ ] T89.2 Remove build tags from compute/ GPU files  Owner: TBD  Est: 2h  BLOCKED
  - BLOCKED: compute/ references cublas/cudnn CGo. Needs purego loaders.

- [ ] S89.3.1 Cross-platform build verification  Owner: TBD  Est: 1h

---

### Track C: Tracing Compiler (Unblocks Megakernel)

Decision rationale: docs/adr/028-tracing-compiler.md.

The tracing compiler resolves the fundamental blocker: graph.Compile() records
composite node OpTypes but the megakernel emitter only knows primitives. The
tracing compiler wraps the Engine with a proxy that records every primitive
Engine call during Forward(), producing an instruction tape of primitive ops.

#### E97: EngineProxy and Tracer (O97)

##### T97.1 Create EngineProxy[T] implementing Engine[T]  Owner: TBD  Est: 4h  [x] 2026 03 08

Create `compute/engine_proxy.go`.

```go
type EngineProxy[T tensor.Numeric] struct {
    real   Engine[T]
    tracer *Tracer[T] // nil when not tracing
}
```

EngineProxy implements all ~25 Engine[T] methods. Each method:
1. Delegates to `p.real.Method(ctx, args...)`
2. If `p.tracer != nil`, calls `p.tracer.Record(opName, inputTensors, outputTensor)`
3. Returns the real result

Method-to-OpName mapping (must match codegen/optable.go emitter names):

| Engine Method | OpName | Category |
|---------------|--------|----------|
| Add | "Add" | binary |
| Sub | "Sub" | binary |
| Mul | "Mul" | binary |
| Div | "Div" | binary |
| Pow | "Pow" | binary |
| MatMul | "MatMul" | binary |
| Exp | "Exp" | unary |
| Log | "Log" | unary |
| Tanh | "Tanh" | unary |
| Sqrt | "Sqrt" | unary |
| Rsqrt | "Rsqrt" | unary |
| MulScalar | "MulScalar" | scalar |
| AddScalar | "AddScalar" | scalar |
| DivScalar | "DivScalar" | scalar |
| Softmax | "Softmax" | reduction |
| ReduceSum | "ReduceSum" | reduction |
| ReduceMean | "ReduceMean" | reduction |
| Reshape | "Reshape" | shape |
| Transpose | "Transpose" | shape |
| Concat | "Concat" | shape |
| Split | "Split" | shape |
| Repeat | "Repeat" | shape |
| Sum | "Sum" | reduction |

Methods NOT traced (side effects, not compute):
- Ops() -- returns arithmetic, no compute
- UnaryOp -- opaque closure, not emittable (see T97.6)
- Zero, Zeros, Copy, Fill -- mutations, not in compute graph
- RandomUniform -- non-deterministic
- Gather -- special: uses int indices (see T97.7)
- ScatterAdd -- gradient-only
- OneHot -- not in inference path
- TanhPrime -- gradient-only

Acceptance:
- All Engine[T] methods compile and delegate correctly.
- In non-tracing mode (tracer == nil), zero overhead beyond interface dispatch.
- Unit test: create EngineProxy wrapping CPUEngine, call Add/MatMul/Softmax,
  verify results match CPUEngine directly.
- Dependencies: none.

- [x] S97.1.1 EngineProxy unit tests  Owner: TBD  Est: 1.5h  2026 03 08
  - Test each traced method records correct OpName.
  - Test non-traced methods delegate without recording.
  - Test tracing off (tracer == nil) produces no trace.

##### T97.2 Create Tracer[T] with tensor identity tracking  Owner: TBD  Est: 3h  [x] 2026 03 08

Create `compute/tracer.go`.

```go
type TracedOp struct {
    OpName    string
    InputIDs  []int // slot indices for inputs
    OutputID  int   // slot index for output
    ExtraArgs map[string]any // axis, scalar value, shape, etc.
}

type Tracer[T tensor.Numeric] struct {
    ops       []TracedOp
    tensorMap map[uintptr]int // tensor pointer -> slot index
    nextSlot  int
    frozen    map[uintptr]bool // known frozen tensors (weights)
    shapes    map[int][]int    // slot -> shape
}
```

The Tracer tracks tensor identity by pointer (unsafe.Pointer of
*tensor.TensorNumeric[T]). When a tensor first appears as input, it gets a
slot index. When a tensor is produced as output, it gets a new slot index.
Frozen tensors (model weights) are pre-registered so they get frozen slot
indices.

Key methods:
- `NewTracer[T](frozenTensors []*tensor.TensorNumeric[T]) *Tracer[T]`
  Pre-registers frozen tensors with slot indices.
- `Record(opName string, inputs []*tensor.TensorNumeric[T], output *tensor.TensorNumeric[T], extra map[string]any)`
  Assigns slot IDs, appends TracedOp.
- `slotFor(t *tensor.TensorNumeric[T]) int`
  Returns existing slot or assigns new one.
- `TracedOps() []TracedOp` -- returns recorded ops.
- `SlotShapes() [][]int` -- returns shapes for each slot.
- `FrozenSlots() []int` -- returns frozen slot indices.

ExtraArgs captures method-specific parameters:
- Softmax: `{"axis": 1}`
- Transpose: `{"axes": [0,2,1,3]}`
- Reshape: `{"shape": [1,8,1,256]}`
- MulScalar: `{"scalar": 0.044715}`
- Concat: `{"axis": -1}`
- Split: `{"numSplits": 2, "axis": -1}`
- Repeat: `{"axis": 1, "repetitions": 4}`
- ReduceSum/ReduceMean: `{"axis": 2, "keepDims": false}`

Acceptance:
- Tracer correctly assigns unique slot indices to distinct tensors.
- Same tensor pointer reused as input maps to same slot index.
- Frozen tensors identified correctly.
- ExtraArgs captured for all parameterized ops.
- Dependencies: none.

- [x] S97.2.1 Tracer unit tests  Owner: TBD  Est: 1.5h  2026 03 08
  - Test tensor identity: same pointer = same slot.
  - Test frozen tensor registration.
  - Test ExtraArgs for Softmax, Transpose, Reshape, MulScalar.

##### T97.3 Wire EngineProxy into graph construction  Owner: TBD  Est: 2h  [x] 2026 03 08

Modify `inference/arch_common.go` (and arch_llama.go if needed):
- Instead of passing the raw engine to buildTransformerGraph(), wrap it in
  an EngineProxy first.
- Store the EngineProxy on the Graph so Compile can access it.

Add to `graph/graph.go`:
```go
func (g *Graph[T]) SetEngineProxy(proxy *compute.EngineProxy[T])
func (g *Graph[T]) EngineProxy() *compute.EngineProxy[T]
```

The EngineProxy is the same object that all layers received at construction
time. When tracing is activated on it, ALL layers' engine calls go through
the tracing path.

Acceptance:
- Existing tests pass unchanged (EngineProxy in non-tracing mode is transparent).
- Graph stores a reference to the EngineProxy.
- All nodes in the Gemma 3 graph use the same EngineProxy instance.
- Dependencies: T97.1.

- [x] S97.3.1 Graph EngineProxy integration test  Owner: TBD  Est: 1h  2026 03 08
  - All existing inference tests pass with EngineProxy wired into graph.

##### T97.4 Implement CompileTraced() in graph/compile.go  Owner: TBD  Est: 4h  [x] 2026 03 08

Add a new `CompileTraced()` method to `Graph[T]`:

```go
func (g *Graph[T]) CompileTraced(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*ExecutionPlan[T], error)
```

Steps:
1. Collect frozen tensors: iterate g.nodes, find constant/parameter nodes,
   get their tensor values from g.memo (populated by prior Forward()).
2. Create a Tracer, pre-register frozen tensors.
3. Call `g.EngineProxy().StartTracing(tracer)`.
4. Run Forward() on each node in topological order (same as current Compile
   warmup pass). The EngineProxy records every engine call.
5. Call `g.EngineProxy().StopTracing()`.
6. Convert TracedOps to Instructions:
   - Each TracedOp becomes an Instruction with OpName from the trace.
   - InputIdx from TracedOp.InputIDs.
   - OutputIdx from TracedOp.OutputID.
   - Forward function: a closure that calls the real engine method with the
     correct arguments (reconstructed from ExtraArgs).
7. Build ExecutionPlan with traced instructions, slot shapes from tracer,
   frozen slots from tracer, input/output slots from graph input/output
   tensor identity.

The existing Compile() method is preserved for backward compatibility. The
generate/ decode loop calls CompileTraced() instead of Compile() when the
EngineProxy is available.

Acceptance:
- CompileTraced() produces an ExecutionPlan where every instruction has a
  primitive OpName (Add, MatMul, Softmax, etc.) -- no composite names.
- The plan's Run() produces identical output to the non-traced plan's Run().
- For a simple Add(input, constant) graph, CompileTraced produces the same
  instructions as Compile (both see "Add").
- For a graph with FFN node, CompileTraced produces ~7 instructions (MatMul,
  MatMul, Concat, Split, Mul, Mul, MatMul) where Compile produces 1 ("FFN").
- Dependencies: T97.2, T97.3.

- [x] S97.4.1 CompileTraced unit tests  Owner: TBD  Est: 2h  2026 03 08
  - Primitive node graph: CompileTraced matches Compile.
  - FFN composite: CompileTraced produces primitive ops.
  - GQA composite: CompileTraced produces ~20+ primitive ops.
  - Output of plan.Run() matches for both compile paths.

##### T97.5 Handle Split (multi-output) and Concat (multi-input) in tracer  Owner: TBD  Est: 2h  [x] 2026 03 08

Split produces multiple output tensors. Concat takes multiple input tensors.
These need special handling in the Tracer:

- Split: Record one "Split" op with a single output slot. But Split returns
  []TensorNumeric. The tracer must assign slot indices to each output tensor.
  Options: (a) Record multiple ops ("SplitPart0", "SplitPart1"), or (b) record
  one "Split" op and let the emitter handle multi-output.
  Decision: Use approach (b). Add `OutputIDs []int` to TracedOp for multi-output
  ops. The emitter generates code for each output slice.

- Concat: Already handled naturally -- inputs are multiple tensors, each with
  their own slot index. TracedOp.InputIDs has multiple entries.

Acceptance:
- Split traced with correct number of OutputIDs.
- Concat traced with all input slot indices.
- Emitter table updated for Split multi-output.
- Dependencies: T97.2.

- [x] S97.5.1 Split/Concat tracing test  Owner: TBD  Est: 1h  2026 03 08

##### T97.6 Handle UnaryOp and FusedRoPE fallback  Owner: TBD  Est: 2h

Two engine calls cannot be automatically traced to primitive ops:

1. `UnaryOp(ctx, tensor, func(T) T)` -- the closure is opaque. The tracer
   cannot determine what math the closure performs.
2. `FusedRoPE()` (via FusedRMSNormer type assertion) -- a fused kernel that
   bypasses the individual engine calls.

Strategy:
- When the tracer encounters UnaryOp, mark the trace as "incomplete". Set a
  flag on the Tracer: `hasOpaqueOps = true`.
- When the tracer encounters FusedRoPE/FusedRMSNormGPU, record it as
  "FusedRMSNorm" (already in emitter table) or "FusedRoPE" (add to emitter).
- CompileTraced() checks `tracer.HasOpaqueOps()`. If true, fall back to the
  non-traced Compile() path (the megakernel will not fire, same as today).
- Track 0 composition fixes (T96.1-T96.3, done) already eliminated the main
  UnaryOp violations in the Gemma 3 path. The sigmoid in SwiGLU is the
  remaining concern.

For SwiGLU's sigmoid: check if SwiGLU/Sigmoid uses engine.UnaryOp or explicit
engine calls. If it uses UnaryOp, refactor to use engine.Exp + engine.AddScalar
+ engine.Div (1 / (1 + exp(-x))). This is a focused composition fix.

Acceptance:
- UnaryOp calls set hasOpaqueOps flag.
- CompileTraced falls back when opaque ops detected.
- Gemma 3 model produces a clean trace (no opaque ops) after sigmoid fix.
- Dependencies: T97.4.

- [ ] S97.6.1 Opaque op fallback test  Owner: TBD  Est: 1h

##### T97.7 Handle Gather with int indices  Owner: TBD  Est: 1.5h  [x] 2026 03 08

Engine.Gather has a different signature: it takes `*tensor.TensorNumeric[int]`
for indices, not `*tensor.TensorNumeric[T]`. The tracer's tensor identity
tracking uses `*tensor.TensorNumeric[T]` pointers.

Strategy:
- Record Gather as a special op with the params tensor (float) as a frozen
  slot and the indices tensor tracked by its pointer cast to uintptr.
- The Gather emitter already exists in optable.go.
- At megakernel runtime, the input token IDs are passed as the indices.

Acceptance:
- Gather traced correctly with frozen params slot and input indices slot.
- EmbeddingLookup node produces a traced Gather instruction.
- Dependencies: T97.2.

- [x] S97.7.1 Gather tracing test  Owner: TBD  Est: 45m  2026 03 08

##### T97.8 Run golangci-lint on compute/, graph/  Owner: TBD  Est: 30m  [x] 2026 03 08
  - Dependencies: T97.1-T97.7.

#### E98: GPU KV Cache for Megakernel Attention (O98)

The GroupedQueryAttention node reads and writes the KV cache during Forward().
The KV cache is currently Go-managed (CPU memory). For the megakernel to handle
attention, the KV data must be on GPU.

##### T98.1 TracingCacheProvider[T]  Owner: TBD  Est: 3h  [x] 2026 03 08

Create `generate/tracing_cache.go`:

```go
type TracingCacheProvider[T tensor.Numeric] struct {
    real   CacheProvider[T]
    tracer *compute.Tracer[T]
}
```

Wraps the real CacheProvider. During tracing:
- `Update(layerIdx, k, v)` records two ops: "KVCacheAppendK" and
  "KVCacheAppendV" with the K/V tensor slots and layer index as ExtraArgs.
- `Get(layerIdx)` records two ops: "KVCacheGetK" and "KVCacheGetV" with
  output tensor slots.
- `SeqLen()` records "KVCacheSeqLen" (returns a scalar).

The tracer captures the full attention dataflow including cache operations.

Acceptance:
- KV cache operations appear in the trace as named ops.
- The traced instruction tape for GQA includes KVCache* ops between the
  Q/K/V projections and the attention score computation.
- Dependencies: T97.2.

- [x] S98.1.1 TracingCacheProvider unit test  Owner: TBD  Est: 1h  2026 03 08

##### T98.2 GPU KV cache buffer management  Owner: TBD  Est: 4h

Create `internal/codegen/kv_cache.go`:

The megakernel needs persistent GPU memory for KV data that survives between
kernel launches. For Gemma 3 2B (26 layers, 4 KV heads, 256 head_dim,
max_seq_len=8192):

- Per-layer K buffer: [max_seq_len, num_kv_heads * head_dim] float32
  = 8192 * 1024 * 4 = 32MB per layer
- Total KV: 26 layers * 2 (K+V) * 32MB = ~1.6GB
- For shorter contexts (512 tokens): 26 * 2 * 2MB = ~104MB

Implement:
- `GPUKVCache` struct: allocates GPU buffers for K/V per layer.
- `Append(layerIdx int, k, v []float32, seqPos int)`: copies new K/V to
  the correct position in GPU buffer.
- `Pointers(layerIdx int) (kPtr, vPtr unsafe.Pointer, seqLen int)`: returns
  device pointers for the megakernel to read.
- Memory allocated once at model load, reused across generations.

Acceptance:
- GPU buffers allocated and freed correctly.
- Append writes to correct offset. Pointers return valid device addresses.
- Memory budget configurable (default: 512 tokens, ~104MB).
- Dependencies: none (uses internal/cuda for GPU memory).

- [ ] S98.2.1 GPU KV cache allocation test  Owner: TBD  Est: 1h
  - Test on DGX Spark: allocate, append, read back, verify data.

##### T98.3 KV cache op emitters in codegen  Owner: TBD  Est: 3h

Add emitters to `internal/codegen/optable.go` for the KV cache ops:

- "KVCacheAppendK": `dev_kv_append(kv_k[layer], slot_k, seq_pos, head_dim);`
- "KVCacheAppendV": `dev_kv_append(kv_v[layer], slot_v, seq_pos, head_dim);`
- "KVCacheGetK": `float* slot_k = kv_k[layer];` (pointer alias, no copy)
- "KVCacheGetV": `float* slot_v = kv_v[layer];`
- "KVCacheSeqLen": `int seq_len = kv_seq_len;` (passed as kernel arg)

Update `emit.go` EmitMegakernel() to:
- Accept KV cache device pointers as kernel arguments.
- Emit KV cache pointer declarations at kernel top.
- Pass seq_pos and kv_seq_len as kernel arguments.

Acceptance:
- Emitted CUDA compiles with nvcc.
- KV cache ops appear correctly in generated kernel.
- Dependencies: T98.2.

- [ ] S98.3.1 KV cache emitter test  Owner: TBD  Est: 1h

##### T98.4 Run golangci-lint on generate/, internal/codegen/  Owner: TBD  Est: 30m
  - Dependencies: T98.1-T98.3.

#### E99: New Primitive Op Emitters (O89)

The traced instruction tape may contain ops not yet in the emitter table.
Add emitters for ops that appear in the Gemma 3 trace.

##### T99.1 Add Slice emitter to optable.go  Owner: TBD  Est: 1.5h  [x] 2026 03 08

RoPE uses engine.Slice (or tensor slicing). The tracer records Slice ops.
Add emitter:
- "Slice": `dev_slice(slot_out, slot_in, start, end, axis, dim);`
- Implement `dev_slice` in megakernel_ops.cu.

Acceptance: Emitted code compiles. Slice op in trace gets emitted.
Dependencies: none.

- [x] S99.1.1 Slice emitter test  Owner: TBD  Est: 45m  2026 03 08

##### T99.2 Add Repeat emitter to optable.go  Owner: TBD  Est: 1.5h  [x] 2026 03 08

GQA uses engine.Repeat for K/V head replication.
- "Repeat": `dev_repeat(slot_out, slot_in, axis, repetitions, dims);`
- Implement `dev_repeat` in megakernel_ops.cu.

Acceptance: Emitted code compiles. Repeat op in trace gets emitted.
Dependencies: none.

- [x] S99.2.1 Repeat emitter test  Owner: TBD  Est: 45m  2026 03 08

##### T99.3 Add ReduceSum and ReduceMean emitters  Owner: TBD  Est: 2h  [x] 2026 03 08

Used in normalization and attention layers.
- "ReduceSum": `dev_reduce_sum(slot_out, slot_in, axis, dim);`
- "ReduceMean": `dev_reduce_mean(slot_out, slot_in, axis, dim);`
- Implement with shared memory reduction (similar to RMSNorm/Softmax).

Acceptance: Emitted code compiles.
Dependencies: none.

- [x] S99.3.1 Reduction emitter test  Owner: TBD  Est: 45m  2026 03 08

##### T99.4 Verify emitter coverage against real Gemma 3 trace  Owner: TBD  Est: 2h

After implementing CompileTraced (T97.4), run it on the Gemma 3 Q4 model on
DGX Spark. Print all unique OpNames in the traced instruction tape. Verify
every op has an emitter in optable.go. Add any missing emitters.

Acceptance:
- CheckSupport() returns empty list for the traced Gemma 3 instruction tape.
- All ops in the trace have emitters.
- Dependencies: T97.4, T99.1-T99.3.

- [ ] S99.4.1 Full trace coverage report  Owner: TBD  Est: 30m

##### T99.5 Run golangci-lint on internal/codegen/  Owner: TBD  Est: 15m
  - Dependencies: T99.1-T99.4.

#### E100: Tracing Compiler Integration (O89, O90)

Wire CompileTraced into the generate loop and verify the megakernel fires.

##### T100.1 Update generate/ to use CompileTraced  Owner: TBD  Est: 2h

In `generate/generator.go` and `generate/stream.go`, update the planOnce.Do
block:

```go
gen.planOnce.Do(func() {
    var compiled *graph.ExecutionPlan[T]
    var cErr error
    if proxy := gen.graph.EngineProxy(); proxy != nil {
        compiled, cErr = gen.graph.CompileTraced(genCtx, tokenTensor)
    } else {
        compiled, cErr = gen.graph.Compile(genCtx, tokenTensor)
    }
    if cErr == nil {
        gen.plan.Store(compiled)
        go tryCompileMegakernel(compiled, nil)
    }
})
```

Also wire the TracingCacheProvider: when creating the cache for the decode
context, wrap it with TracingCacheProvider if the graph has an EngineProxy.

Acceptance:
- CompileTraced is called when EngineProxy is available.
- tryCompileMegakernel receives a plan with primitive ops.
- Megakernel CheckSupport() passes (returns empty unsupported list).
- Dependencies: E97, E98, E99.

- [ ] S100.1.1 Integration test  Owner: TBD  Est: 1.5h
  - Run bench_tps on DGX Spark. Verify "megakernel: compiled and loaded" log.

##### T100.2 Update tryCompileMegakernel for GPU KV cache  Owner: TBD  Est: 2h

The existing tryCompileMegakernel in `generate/megakernel.go` creates a
MegakernelRunner. Update it to:

1. Detect KVCache* ops in the traced instruction tape.
2. Allocate GPUKVCache with the correct dimensions from slot shapes.
3. Pass KV cache device pointers to the runner's Launch().
4. On each Launch(), pass current seq_pos from the Go KV cache.

Acceptance:
- Megakernel launches with KV cache pointers.
- KV cache positions increment correctly per token.
- Dependencies: T98.2, T100.1.

- [ ] S100.2.1 KV cache integration test  Owner: TBD  Est: 1.5h
  - Generate 50 tokens with megakernel. Compare with plan.Run() output.

##### T100.3 End-to-end megakernel correctness test  Owner: TBD  Est: 2h

On DGX Spark:
1. Load Gemma 3 2B Q4 model.
2. Generate 50 tokens with prompt "The capital of France is".
3. Compare output: megakernel path vs ExecutionPlan.Run() path.
4. Verify no NaN/Inf. Verify output is coherent text.

This is the previously blocked S93.3.1.

Acceptance:
- Both paths produce identical output (or within float32 tolerance).
- No NaN or Inf in any intermediate.
- Dependencies: T100.2.

##### T100.4 Run golangci-lint on generate/, graph/  Owner: TBD  Est: 15m
  - Dependencies: T100.1-T100.3.

---

### Track B: Megakernel Performance Tuning (Existing)

#### E94: Megakernel Performance Tuning (O90)

- [ ] T94.1 Profile megakernel with nsys  Owner: TBD  Est: 2h
  - Identify bandwidth utilization, occupancy, register usage, stalls.
  - Dependencies: E100.

- [ ] T94.2 Optimize memory access patterns  Owner: TBD  Est: 3h
  - Coalesced weight reads, bank conflict avoidance, warp-level reduction.
  - Dependencies: T94.1.

- [ ] T94.3 Tune thread block and grid dimensions  Owner: TBD  Est: 2h
  - Experiment with block sizes (128, 256, 512).
  - Dependencies: T94.1.

- [ ] T94.4 Run golangci-lint on modified packages  Owner: TBD  Est: 15m
  - Dependencies: T94.3.

#### E95: End-to-End Benchmark (O90)

- [ ] T95.1 Profile GPU inference after all optimizations  Owner: TBD  Est: 2h
  - bench_tps -device cuda -tokens 100, 7 runs, median and peak.
  - Dependencies: E94.

- [ ] S95.1.1 GPU profile report  Owner: TBD  Est: 30m

- [ ] T95.2 Compare all configurations  Owner: TBD  Est: 1.5h
  - CPU, GPU+plan.Run(), GPU+megakernel. 7 runs each.
  - Dependencies: T95.1.

- [ ] S95.2.1 Benchmark comparison report  Owner: TBD  Est: 30m

- [ ] T95.3 Verify output correctness across all paths  Owner: TBD  Est: 1h
  - Dependencies: T95.1.

- [ ] S95.3.1 Output correctness report  Owner: TBD  Est: 30m

- [ ] T95.4 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m
  - Dependencies: T95.3.

---

## 4. Parallel Work

| Track | Epics | Description | Prerequisite |
|-------|-------|-------------|-------------|
| 0 remaining | E96 P2-P3 | Fix 7 remaining composition violations | none |
| A remaining | T87.3, S88/S89 | Remaining purego tests and cleanup | none |
| C: tracing | E97, E98, E99, E100 | Tracing compiler + GPU KV cache + emitters + integration | none (builds on completed Track B infrastructure) |
| B: tuning | E94, E95 | GPU performance tuning and benchmark | Track C complete |

### Track C Internal Parallelism

| Wave | Tasks | Notes |
|------|-------|-------|
| C1 | T97.1, T97.2, T98.2, T99.1, T99.2, T99.3 | All independent: EngineProxy, Tracer, GPU KV cache, new emitters |
| C2 | T97.3, T97.5, T97.6, T97.7, T98.1, T98.3, T99.4 | After C1: wire proxy, handle edge cases, KV emitters, verify coverage |
| C3 | T97.4, T97.8 | After C2: CompileTraced implementation + lint |
| C4 | T98.4, T99.5, T100.1, T100.2 | After C3: integration wiring |
| C5 | T100.3, T100.4 | After C4: end-to-end test |

Track 0 remaining and Track A remaining can run in parallel with Track C.
Track B (E94, E95) starts only after Track C is complete.

---

## 5. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M56: Tracing compiler works | E97 | none | CompileTraced produces primitive-op instruction tape for Gemma 3. All ops covered by emitters. |
| M57: GPU KV cache works | E98 | none | KV data on GPU, append/read correct, megakernel_ops.cu KV functions compile. |
| M58: Megakernel fires | E100 | M56, M57, E99 | bench_tps shows "megakernel: compiled and loaded". Output matches plan.Run(). |
| M59: 50 tok/s GPU | E94, E95 | M58 | bench_tps >= 50 tok/s median on DGX Spark GB10. |
| M60: 10 tok/s CPU ARM64 | -- | -- | PARTIAL: 8.15 tok/s achieved (+18.8%). Remaining gap requires GEMM cache tiling. See docs/design.md 15.16. |

Critical path (GPU): T97.1 -> T97.3 -> T97.4 -> T100.1 -> T100.2 -> T100.3 -> T94.1 -> T95.1

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R100 | Tracing captures wrong execution path: Forward() has conditional logic (e.g., KV cache presence check in GQA). The trace is only valid for the traced path (decode mode, seqLen=1, cache active). | Megakernel produces wrong output for different paths | Medium | Only use traced plan for decode (seqLen=1). Prefill continues to use non-traced Forward(). The generate loop already separates prefill from decode. |
| R101 | Tensor identity via pointer is fragile: if the engine reuses a tensor buffer (pool allocation), the same pointer may represent different logical tensors. | Wrong slot wiring in traced tape | Medium | During tracing, disable tensor pooling (set pool to nil on EngineProxy). Each engine call allocates fresh tensors. This is only during one Compile() call, not runtime. |
| R102 | GPU KV cache memory budget: 26 layers x 2 (K/V) x 8192 seq x 1024 dim x 4 bytes = 1.6GB for full context. DGX Spark unified memory is large but this is significant. | Out of memory for long contexts | Low | Default to 512-token budget (~104MB). Allow user-configurable max_seq_len for megakernel path. Fall back to plan.Run() for longer contexts. |
| R103 | EngineProxy adds interface dispatch overhead to normal (non-tracing) inference. | Small performance regression on non-megakernel path | Low | Interface dispatch is ~1-2ns. With 650 engine calls per token at 7.78 tok/s, total overhead is ~1us. Negligible vs 128ms per token. |
| R104 | UnaryOp in sigmoid (inside SwiGLU) blocks clean trace for Gemma 3. | Must refactor sigmoid before megakernel works | Medium | T97.6 addresses this. Refactor Sigmoid.Forward() to use engine.Exp + engine.AddScalar + engine.Div. Small, focused change. |
| R92 | Register pressure: hidden_dim=2048 does not fit in registers | Must use shared/global memory, slower | High | Tile the hidden dimension. Profile with nvcc --ptxas-options=-v. |
| R95 | KV cache reads limit bandwidth utilization for long contexts | Cannot reach theoretical max | High | Focus on short contexts (< 512). Document crossover point. |
| R98 | 50 tok/s target not achieved | Unknown bottleneck | Medium | If 30+ tok/s, profile with nsys. If < 30, fall back to fused kernels (archived E83-E85). |

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
7. `go build ./...` compiles on any machine.
8. On DGX Spark: GPU path activates and produces correct results.
9. Changes are committed in a small commit touching one directory only.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(graph): add CompileTraced tracing compiler`.
- Always run linters and formatters before committing.

### DGX Spark Protocol

- SSH: `ssh ndungu@192.168.86.250`
- Go: `/usr/local/go/bin/go`
- CUDA: `/usr/local/cuda/bin/nvcc`, sm_121
- Model: `~/models/gemma3-q4/model.zmf`
- Repo: `~/zerfoo/`

### Quality Gate

- `go test -race ./package/`
- `golangci-lint run ./package/`
- `go vet ./package/`
- `go build ./...`

---

## 8. Progress Log

### Change Summary -- 2026-03-08 (v14)

Wave C3 complete (3 tasks). CompileTraced tracing compiler implemented.

**Track C completed tasks (Wave C3):**
- T97.4 + S97.4.1: CompileTraced in graph/compile.go. Produces primitive-op
  ExecutionPlans by tracing Forward via EngineProxy. makeTracedForward dispatch
  covers all 22 traced ops. 5 table-driven tests. Commit 580eb11.
- T97.8: golangci-lint clean on compute/ and graph/. 0 issues.

**Also completed (Wave C3 prep):**
- EngineProxy: All ops now record ExtraArgs (scalar, axis, shape, keepDims,
  repetitions, numSplits). Tracer gains exported SlotFor/NextSlot. EngineProxy
  gains Real(). Commit 216f921.

**Next unblocked:** T97.6 (UnaryOp fallback), T99.4 (emitter coverage),
T98.2 (GPU KV cache -- requires DGX Spark).

### Change Summary -- 2026-03-08 (v13)

Wave C2 complete (6 tasks). All code committed to feat/neon-softmax. Pre-commit
hooks pass on all commits. Wave C3 unblocked.

**Track C completed tasks (Wave C2):**
- T97.3 + S97.3.1: Wire EngineProxy into buildTransformerGraph. All ~15 layer
  constructors receive proxy instead of raw engine. Commit e15a802.
- T97.5 + S97.5.1: Split multi-output tracing via RecordMultiOutput with
  OutputIDs[]. Concat already handled via multi-input. Commit 747398b.
- T97.7 + S97.7.1: Gather int indices tracing via RecordGather with
  slotForIntTensor(). Frozen params get slot 0. Commit 747398b.
- T98.1 + S98.1.1: TracingCacheProvider wraps CacheProvider, records
  KVCacheAppendK/V and KVCacheGetK/V ops. Commit 40df565.

**Track 0 completed tasks (Wave C2):**
- T96.7: MoEGate topK selection retains data access (no engine.TopK primitive).
  engine.Softmax already composed. Commit e076e78.
- T96.8 + S96.8.1: MixtureOfExperts refactored from ops.Mul/ops.Add loops to
  engine.MulScalar + engine.Add + engine.Concat. Per-token accumulators avoid
  broadcasting bug. Commit e076e78.

**Deviation:** [Deviation: Bug] Fixed MoE broadcasting -- initial refactor
accumulated [1,modelDim] expert output into [seqLen,modelDim] accumulator,
broadcasting to all rows. Fixed with per-token [1,modelDim] accumulators +
engine.Concat at end.

**Wave C3 unblocked:** T97.4 (CompileTraced), T97.8 (lint).
T97.6 (UnaryOp fallback) depends on T97.4.

### Change Summary -- 2026-03-08 (v12)

Wave C1 complete (5 tasks in parallel via worktrees). Track C foundation and
Track 0 composition fixes merged into feat/neon-softmax.

**Track C completed tasks:**
- T97.1 + S97.1.1: EngineProxy[T] with TraceRecorder interface. Commit 4f5518b.
- T97.2 + S97.2.1: Tracer[T] with pointer-based tensor identity tracking. Commit 4f5518b.
- T99.1 + S99.1.1: Slice emitter in optable.go + megakernel_ops.cu.
- T99.2 + S99.2.1: Repeat emitter in optable.go + megakernel_ops.cu.
- T99.3 + S99.3.1: ReduceSum and ReduceMean emitters with shared memory reduction.

**Track 0 completed tasks:**
- T96.6 + S96.6.1: Conv2d refactored to im2col + engine.MatMul. 8 parity tests.
- T96.9 + S96.9.1: PolynomialExpansion composed via engine.Fill/Pow/Mul/Concat.
- T96.10: SpectralFingerprint composed via precomputed Fourier basis + engine.MatMul.

**Deviation:** Fixed pre-existing SWA test helpers (noopOptimizer/setOptimizer
redeclaration between ema_test.go and helpers_test.go). Commit a79afed.

**Wave C2 unblocked:** T97.3, T97.5, T97.7, T98.1, T98.2, T98.3.

### Change Summary -- 2026-03-07 (v11)

Trimmed plan. Stable knowledge preserved in docs/design.md section 15.16 and
docs/adr/029-neon-simd-cpu-acceleration.md. Removed completed epics: E101,
E102, E103, E104 (Track D). Removed resolved risks: R105, R106, R107, R108.
Removed completed Track D parallel work section and wave entries. Removed
NEON Exp Polynomial Reference and ARM64 encoding cheat sheet from Appendix
(moved to docs/design.md 15.16). Updated ADR index in docs/design.md (added
ADRs 022-029). Updated milestone M60 to PARTIAL (8.15 tok/s achieved).
Updated CPU tok/s current from 6.86 to 8.15 in success metrics. Removed
Track D objectives (O101-O104) and deliverables (D435-D439). Removed
completed Track D from hand-off notes key file map. Kept only v10 progress
log entry.

### Change Summary -- 2026-03-07 (v10)

Wave D4 complete. All Track D tasks done. Benchmark: 8.15 tok/s median (+18.8%
over 6.86 baseline). Target was 10 tok/s, achieved 81.5%. GEMM dominates at 72%
of CPU, already NEON-accelerated. Remaining gap requires GEMM cache tiling.

**Completed tasks:**
- T104.1: CPU benchmark 8.15 tok/s median (7.72-8.45 range). Commit d2da5fe.
- T104.2: Profile: sgemmAccRowNeon 37%, q4DotRowSIMD 35%, Transpose 4.4%.
- T104.3: 100 tokens generated without crashes.
- T104.4: go vet + go build clean on arm64.
- 7 critical NEON assembly bugs fixed (RoPE IP0/IP1, FMLS encoding, exp clamping,
  q4dot callee-saved registers, RMSNorm lane zeroing, RMSNorm ABI return value).

**System issue:** DGX Spark Go 1.25.0 has intermittent segfaults (~10-40%) across
ALL packages including tensor/ (no assembly). Confirmed not caused by our code.

---

## 9. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for full context. ADRs in docs/adr/.
- **Phase 34 status:** Tracks 0/A/B infrastructure and Track D (NEON SIMD)
  complete. Track C (tracing compiler) is the primary active work.
  See docs/design.md section 15.16 for all completed work.
- **Track C core problem:** The megakernel is fully built (emit, compile, load,
  wire) but never fires because the instruction tape has composite ops. The
  tracing compiler (E97) fixes this by recording primitive Engine calls.
- **Key starting points:**
  - Track C: `compute/engine.go` (Engine interface) -> `compute/engine_proxy.go`
    (T97.1) -> `compute/tracer.go` (T97.2) -> `graph/compile.go` CompileTraced() (T97.4)
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key File Map

| File | Purpose | Track C Change |
|------|---------|---------------|
| `compute/engine.go` | Engine[T] interface (~25 methods) | Reference for EngineProxy |
| `compute/engine_proxy.go` | (new) EngineProxy[T] wrapping Engine | T97.1 |
| `compute/tracer.go` | (new) Tracer[T] records TracedOps | T97.2 |
| `compute/cpu_engine.go` | CPUEngine implementation | -- |
| `graph/compile.go` | Compile() + ExecutionPlan | Add CompileTraced() T97.4 |
| `graph/graph.go` | Graph[T] struct | Add EngineProxy accessors T97.3 |
| `generate/tracing_cache.go` | (new) TracingCacheProvider | T98.1 |
| `internal/codegen/kv_cache.go` | (new) GPUKVCache | T98.2 |
| `internal/codegen/optable.go` | Op emitter table (26 ops) | Add Slice, Repeat, Reduce T99 |
| `internal/codegen/emit.go` | EmitMegakernel() | Add KV cache kernel args T98.3 |
| `generate/generator.go` | Generate() decode loop | Use CompileTraced T100.1 |
| `generate/stream.go` | GenerateStream() decode loop | Use CompileTraced T100.1 |
| `generate/megakernel.go` | tryCompileMegakernel() | Wire GPU KV cache T100.2 |
| `inference/arch_common.go` | buildTransformerGraph() | Wrap engine with EngineProxy T97.3 |

### Performance Baselines

| Config | tok/s | Source |
|--------|-------|--------|
| CPU ARM64 (post Track D) | 8.15 median | Phase 34 Track D |
| CPU ARM64 (pre Track D) | 6.86 | Phase 30 |
| GPU (cuda) | 10.32 peak / 7.78 median | Phase 33 |
| GPU (purego) | 6.59 CPU | Phase 34 Track A |
| Ollama GB10 | ~100 (est.) | Interpolated |
| Theoretical | ~182 | 273 GB/s / 1.5GB |

---

## 10. Appendix

No appendix content. Technical references for NEON assembly (exp polynomial,
instruction encoding) moved to docs/design.md section 15.16.
