# Zerfoo Development Plan -- Phase 34: Close the Gap with llama.cpp

## 1. Context

See docs/design.md for full architecture context and Phases 1-33 history.
See docs/design.md section 15.16 for Phase 34 completed work (Tracks 0/A/B).

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

CPU inference is 6.86 tok/s. llama.cpp achieves ~100 tok/s on the same hardware
because it uses NEON SIMD assembly for all hot-path operations (softmax, rmsnorm,
silu, elementwise, rope, scalar ops), has zero-allocation compute graphs, and uses
quantized integer accumulation. Zerfoo currently has NEON assembly only for matmul
(q4dot, vdotf32, sgemmAccRow). Track D adds NEON SIMD for all remaining hot-path
ops, same-shape fast paths to eliminate broadcasting overhead, and a tensor arena.
See docs/adr/029-neon-simd-cpu-acceleration.md.

### Objectives

- O96: Refactor remaining layers with inline math to compose Engine primitives.
- O87: Replace remaining CGo CUDA bindings with dlopen-based pure Go bindings.
- O88: Eliminate remaining `//go:build cuda` tags.
- O97: Implement tracing compiler that decomposes composite nodes into primitives.
- O98: Implement GPU KV cache for megakernel attention.
- O89: Generate a single-kernel decode for Gemma 3 2B transformer.
- O90: Achieve >= 50 tok/s median for Gemma 3 2B Q4 on DGX Spark GB10.
- O101: NEON SIMD acceleration for all CPU hot-path operations.
- O102: Same-shape fast paths and Pow x^2 specialization in CPUEngine.
- O103: Tensor arena for buffer reuse across engine operations.
- O104: Achieve >= 10 tok/s CPU ARM64 for Gemma 3 2B Q4 on DGX Spark GB10.

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
- ARM64 NEON assembly uses Go plan9 syntax in _arm64.s files.
- Generic fallbacks in _generic.go files for non-ARM64 platforms.

### Success Metrics

| Metric | Current | Target | How Measured |
|--------|---------|--------|-------------|
| GPU tok/s median | 7.78 | >= 50 | bench_tps -device cuda, 7 runs, median |
| GPU tok/s peak | 10.32 | >= 60 | bench_tps -device cuda, best of 7 |
| CPU tok/s ARM64 | 6.86 | >= 10 | bench_tps -device cpu, 7 runs, median |
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
| D435 | Same-shape binaryOp fast path + Pow x^2 | D | Eliminate broadcasting overhead |
| D436 | NEON Softmax, RMSNorm, SiLU, RoPE assembly | D | Vectorize CPU hot-path ops |
| D437 | NEON vectorized elementwise + scalar ops | D | SIMD Add/Mul/Sub/MulScalar/etc. |
| D438 | Tensor arena for buffer reuse | D | Eliminate GC allocation overhead |
| D439 | CPU benchmark >= 10 tok/s ARM64 | D | Final CPU validation |

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

- [ ] T96.6 Refactor Conv2d to im2col + engine.MatMul  Owner: TBD  Est: 3h
  - File: `layers/core/conv2d.go` lines 60-144.
  - Violation: 6-nested loop convolution with direct data access.
  - Fix: im2col transform then engine.MatMul(weight_matrix, col_matrix).
  - Acceptance: Zero nested compute loops in Forward(). Output within 1e-5.
  - Dependencies: none.

- [ ] S96.6.1 Conv2d composition parity test  Owner: TBD  Est: 1h

##### Priority 3: Specialized Layers (not on Gemma 3 path)

- [ ] T96.7 Refactor MoEGate to compose engine primitives  Owner: TBD  Est: 2h
  - File: `layers/core/moe.go` lines 43-100.
  - Fix: engine.Softmax for routing, engine.TopK or engine.Sort + engine.Slice.
  - Dependencies: none.

- [ ] T96.8 Refactor MixtureOfExperts to compose engine primitives  Owner: TBD  Est: 2h
  - File: `layers/core/moe.go` lines 217-282.
  - Fix: engine.Gather for token extraction, engine.MulScalar + engine.Add.
  - Dependencies: T96.7.

- [ ] S96.8.1 MoE composition parity test  Owner: TBD  Est: 1h

- [ ] T96.9 Refactor PolynomialExpansion to compose engine primitives  Owner: TBD  Est: 1.5h
  - File: `layers/core/polynomial.go` lines 191-249.
  - Fix: engine.Pow for each term, engine.MulScalar, engine.Add.
  - Dependencies: none.

- [ ] S96.9.1 Polynomial composition parity test  Owner: TBD  Est: 45m

- [ ] T96.10 Refactor SpectralFingerprint to compose engine primitives  Owner: TBD  Est: 2h
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

##### T97.1 Create EngineProxy[T] implementing Engine[T]  Owner: TBD  Est: 4h

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

- [ ] S97.1.1 EngineProxy unit tests  Owner: TBD  Est: 1.5h
  - Test each traced method records correct OpName.
  - Test non-traced methods delegate without recording.
  - Test tracing off (tracer == nil) produces no trace.

##### T97.2 Create Tracer[T] with tensor identity tracking  Owner: TBD  Est: 3h

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

- [ ] S97.2.1 Tracer unit tests  Owner: TBD  Est: 1.5h
  - Test tensor identity: same pointer = same slot.
  - Test frozen tensor registration.
  - Test ExtraArgs for Softmax, Transpose, Reshape, MulScalar.

##### T97.3 Wire EngineProxy into graph construction  Owner: TBD  Est: 2h

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

- [ ] S97.3.1 Graph EngineProxy integration test  Owner: TBD  Est: 1h
  - Build a small graph with EngineProxy, run Forward(), verify output matches.

##### T97.4 Implement CompileTraced() in graph/compile.go  Owner: TBD  Est: 4h

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

- [ ] S97.4.1 CompileTraced unit tests  Owner: TBD  Est: 2h
  - Primitive node graph: CompileTraced matches Compile.
  - FFN composite: CompileTraced produces primitive ops.
  - GQA composite: CompileTraced produces ~20+ primitive ops.
  - Output of plan.Run() matches for both compile paths.

##### T97.5 Handle Split (multi-output) and Concat (multi-input) in tracer  Owner: TBD  Est: 2h

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

- [ ] S97.5.1 Split/Concat tracing test  Owner: TBD  Est: 1h

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

##### T97.7 Handle Gather with int indices  Owner: TBD  Est: 1.5h

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

- [ ] S97.7.1 Gather tracing test  Owner: TBD  Est: 45m

##### T97.8 Run golangci-lint on compute/, graph/  Owner: TBD  Est: 30m
  - Dependencies: T97.1-T97.7.

#### E98: GPU KV Cache for Megakernel Attention (O98)

The GroupedQueryAttention node reads and writes the KV cache during Forward().
The KV cache is currently Go-managed (CPU memory). For the megakernel to handle
attention, the KV data must be on GPU.

##### T98.1 TracingCacheProvider[T]  Owner: TBD  Est: 3h

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

- [ ] S98.1.1 TracingCacheProvider unit test  Owner: TBD  Est: 1h

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

##### T99.1 Add Slice emitter to optable.go  Owner: TBD  Est: 1.5h

RoPE uses engine.Slice (or tensor slicing). The tracer records Slice ops.
Add emitter:
- "Slice": `dev_slice(slot_out, slot_in, start, end, axis, dim);`
- Implement `dev_slice` in megakernel_ops.cu.

Acceptance: Emitted code compiles. Slice op in trace gets emitted.
Dependencies: none.

- [ ] S99.1.1 Slice emitter test  Owner: TBD  Est: 45m

##### T99.2 Add Repeat emitter to optable.go  Owner: TBD  Est: 1.5h

GQA uses engine.Repeat for K/V head replication.
- "Repeat": `dev_repeat(slot_out, slot_in, axis, repetitions, dims);`
- Implement `dev_repeat` in megakernel_ops.cu.

Acceptance: Emitted code compiles. Repeat op in trace gets emitted.
Dependencies: none.

- [ ] S99.2.1 Repeat emitter test  Owner: TBD  Est: 45m

##### T99.3 Add ReduceSum and ReduceMean emitters  Owner: TBD  Est: 2h

Used in normalization and attention layers.
- "ReduceSum": `dev_reduce_sum(slot_out, slot_in, axis, dim);`
- "ReduceMean": `dev_reduce_mean(slot_out, slot_in, axis, dim);`
- Implement with shared memory reduction (similar to RMSNorm/Softmax).

Acceptance: Emitted code compiles.
Dependencies: none.

- [ ] S99.3.1 Reduction emitter test  Owner: TBD  Est: 45m

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

### Track D: NEON SIMD CPU Acceleration (NEW)

Decision rationale: docs/adr/029-neon-simd-cpu-acceleration.md.

CPU inference is 6.86 tok/s. Profiling shows the hot path after matmul includes
Pow (8.9%), binaryOp broadcasting overhead (10.4%), Softmax, RMSNorm, SiLU, and
RoPE -- all running through generic Go per-element loops. llama.cpp vectorizes
all of these with NEON SIMD. Track D adds NEON assembly for these ops, same-shape
fast paths, and a tensor arena for buffer reuse.

All assembly files follow the established pattern:
- `internal/xblas/<name>_arm64.go` -- Go declarations with `//go:noescape`
- `internal/xblas/<name>_arm64.s` -- ARM64 NEON plan9 assembly
- `internal/xblas/<name>_generic.go` -- `//go:build !arm64` pure-Go fallback

#### E101: Same-Shape Fast Paths and Pow Specialization (O102)

These are pure Go changes in compute/cpu_engine.go. No assembly needed. Highest
ROI because they eliminate unnecessary work rather than doing work faster.

##### T101.1 Add same-shape fast path to binaryOp  Owner: TBD  Est: 2h  [x] 2026 03 07

Current `binaryOp()` at `compute/cpu_engine.go:546` computes broadcast shapes
and does per-element coordinate decoding (lines 576-594) even when both tensors
have identical shapes (no broadcast). This coordinate-decode loop is 10.4% of
CPU inference time.

Add a fast path at the top of binaryOp():
```go
if slicesEqual(a.Shape(), b.Shape()) {
    // Same shape: no broadcast needed. Direct element-wise loop.
    aData := a.Data()
    bData := b.Data()
    rData := result.Data()
    parallelFor(len(aData), func(start, end int) {
        for i := start; i < end; i++ {
            rData[i] = op(aData[i], bData[i])
        }
    })
    return result, nil
}
```

This fast path is taken for all same-shape binary ops (Add, Sub, Mul, Div, Pow)
which is the common case in transformer inference (residual connections, attention
scores, etc.).

Acceptance:
- binaryOp benchmark: >= 40% faster for same-shape [1, 2048] tensors.
- All existing binary op tests pass unchanged.
- Broadcasting still works for non-matching shapes.
- Dependencies: none.

- [x] S101.1.1 Same-shape fast path benchmark test  Owner: TBD  Est: 1h  2026 03 07
  - BenchmarkBinaryOpSameShape vs BenchmarkBinaryOpBroadcast.
  - Verify fast path taken for same-shape, slow path for broadcast.

##### T101.2 Add Pow x^2 specialization  Owner: TBD  Est: 1.5h  [x] 2026 03 07

Current `Pow()` at `compute/cpu_engine.go:1358` calls `e.ops.Pow(base, exponent)`
which calls `math.Pow()` for every element. Profiling shows Pow is 8.9% of
CPU inference time. In Gemma 3, Pow is called exclusively by RMSNorm with
exponent=2.0 (squaring).

Add specialization in Pow():
```go
func (e *CPUEngine[T]) Pow(ctx context.Context, base, exponent *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
    defer e.recordOp("Pow", time.Now())

    // Specialization: if exponent is a scalar broadcast tensor with value 2.0,
    // use x*x instead of math.Pow(x, 2).
    if isScalarBroadcast(exponent) {
        expVal := e.ops.ToFloat64(exponent.Data()[0])
        if expVal == 2.0 {
            return e.UnaryOp(ctx, base, func(x T) T { return e.ops.Mul(x, x) }, dst...)
        }
    }
    return e.binaryOp(ctx, base, exponent, e.ops.Pow, dst...)
}
```

`isScalarBroadcast()` checks if the tensor has exactly one unique value
(shape is [1] or all elements are identical).

Acceptance:
- Pow with exponent=2.0 is >= 5x faster than math.Pow path.
- Pow with other exponents still works via math.Pow.
- RMSNorm end-to-end benchmark shows improvement.
- All existing Pow tests pass.
- Dependencies: none.

- [x] S101.2.1 Pow specialization benchmark test  Owner: TBD  Est: 45m  2026 03 07
  - BenchmarkPowSquare vs BenchmarkPowGeneric.

##### T101.3 Add same-shape fast path to scalar ops  Owner: TBD  Est: 1h  [x] 2026 03 07

Current `MulScalar()`, `AddScalar()`, `DivScalar()` at `cpu_engine.go:359-394`
use `e.ops.Mul(aData[i], scalar)` per element. For float32 specialization,
this can directly use `aData[i] * scalar` when T is float32 (the common
inference type). But even without type specialization, the existing loop is
already simple. The main optimization here is to dispatch to NEON in T101.7.

For now, ensure the scalar ops use the same parallelFor pattern with the
compute pool (they already do). This task is primarily to verify the scalar
ops are ready for NEON dispatch in T101.7.

Acceptance:
- Scalar ops use parallelFor.
- Benchmarks established as baselines for NEON comparison.
- Dependencies: none.

- [x] S101.3.1 Scalar op baseline benchmarks  Owner: TBD  Est: 30m  2026 03 07

##### T101.4 Run golangci-lint on compute/  Owner: TBD  Est: 15m  [x] 2026 03 07
  - Dependencies: T101.1-T101.3.

#### E102: NEON Assembly for Hot-Path Operations (O101)

ARM64 NEON assembly for the operations that dominate non-matmul CPU time.
All functions go in `internal/xblas/` following the existing pattern.

##### T102.1 NEON vectorized Softmax  Owner: TBD  Est: 4h  [x] 2026 03 07

Create `internal/xblas/softmax_arm64.go` and `softmax_arm64.s`.

```go
// SoftmaxF32 computes softmax(x) in-place for a float32 vector of length n.
// Uses 3-pass NEON: (1) find max via FMAXP, (2) exp(x-max) via polynomial
// approximation + FMLA, (3) normalize by reciprocal sum.
//go:noescape
func SoftmaxF32(data *float32, n int)
```

NEON implementation:
1. Pass 1 (max): Load 4 floats at a time, FMAX across lanes, FMAXP to reduce.
2. Pass 2 (exp + sum): Subtract max, compute exp() using a degree-4 polynomial
   approximation of exp() (sufficient for float32 precision):
   `exp(x) ~ 1 + x + x^2/2 + x^3/6 + x^4/24` for x in [-max, 0].
   For wider range, use the identity `exp(x) = 2^(x/ln2)` with integer part
   extracted via FCVTZS and fractional part via polynomial.
   Accumulate sum with FADD.
3. Pass 3 (normalize): Compute 1/sum, FMUL each element.

Also create `internal/xblas/softmax_generic.go`:
```go
//go:build !arm64
func SoftmaxF32(data *float32, n int) { softmaxF32Scalar(data, n) }
```

Wire into CPUEngine.Softmax for the common case (float32, last-axis, contiguous).

Acceptance:
- SoftmaxF32 output matches math.Exp-based softmax within 1e-5 relative error.
- Benchmark: >= 3x faster than current per-element softmax for n=2048.
- Handles n not divisible by 4 (tail elements processed scalar).
- Dependencies: none.

- [x] S102.1.1 NEON Softmax correctness + benchmark tests  [x] 2026 03 07  Owner: TBD  Est: 1.5h
  - Test various lengths: 1, 4, 7, 128, 2048.
  - Compare output against reference math.Exp softmax.
  - BenchmarkSoftmaxNEON vs BenchmarkSoftmaxScalar.

##### T102.2 NEON vectorized RMSNorm  Owner: TBD  Est: 4h  [x] 2026 03 07

Create `internal/xblas/rmsnorm_arm64.go` and `rmsnorm_arm64.s`.

```go
// RMSNormF32 computes x * rsqrt(mean(x^2) + eps) * weight for one row.
// x is input [D], weight is [D], out is [D], eps is epsilon.
// Returns the scale factor rsqrt(mean(x^2) + eps).
//go:noescape
func RMSNormF32(out, x, weight *float32, D int, eps float32) float32
```

NEON implementation:
1. Sum of squares: Load 4 x[i] at a time, FMUL x*x, FADD to accumulator.
   Use dual accumulators V0/V1 to hide latency.
2. Compute mean: FADDP to reduce, FDIV by D.
3. Compute rsqrt: FADD eps, FRSQRTE + Newton-Raphson refinement (2 iterations
   for float32 precision). ARM NEON has FRSQRTE instruction.
4. Normalize: Load x[i] and weight[i], FMUL x * scale * weight, store to out.

Also create generic fallback.

Wire into `compute.FusedRMSNorm()` (currently at `compute/fused_rmsnorm.go`).
Call `RMSNormF32()` per row instead of the per-element Go loop.

Acceptance:
- Output matches current FusedRMSNorm within 1e-5 relative error.
- Benchmark: >= 3x faster for D=2048 (Gemma 3 hidden size).
- Dependencies: none.

- [x] S102.2.1 NEON RMSNorm correctness + benchmark tests  [x] 2026 03 07  Owner: TBD  Est: 1.5h

##### T102.3 NEON vectorized SiLU (x * sigmoid(x))  Owner: TBD  Est: 3h  [x] 2026 03 07

Create `internal/xblas/silu_arm64.go` and `silu_arm64.s`.

```go
// SiLUF32 computes silu(x) = x / (1 + exp(-x)) for n float32 values.
// Reads from x, writes to out (may alias x for in-place).
//go:noescape
func SiLUF32(out, x *float32, n int)

// SiLUGateF32 computes silu(gate) * up for n float32 values.
// This is the SwiGLU operation: result[i] = gate[i] * sigmoid(gate[i]) * up[i].
//go:noescape
func SiLUGateF32(out, gate, up *float32, n int)
```

NEON implementation:
- exp(-x) via the same polynomial approximation as Softmax T102.1.
  Factor out the exp polynomial into a shared macro or inline function
  in the assembly.
- sigmoid(x) = 1 / (1 + exp(-x)): FNEG, exp polynomial, FADD 1, FRECPE + NR.
- silu(x) = x * sigmoid(x): FMUL.
- SiLUGateF32 fuses the SwiGLU operation: silu(gate) * up in one pass.

Wire into `compute.FusedSiLUGate()` (at `compute/fused_silugate.go`).
Call `SiLUGateF32()` instead of per-element Go loop with `math.Exp`.

Acceptance:
- Output matches math.Exp-based SiLU within 1e-5 relative error.
- Benchmark: >= 3x faster for n=2048.
- SiLUGateF32 matches FusedSiLUGate output.
- Dependencies: none.

- [x] S102.3.1 NEON SiLU/SiLUGate correctness + benchmark tests  [x] 2026 03 07  Owner: TBD  Est: 1.5h

##### T102.4 NEON vectorized RoPE  Owner: TBD  Est: 3h  [x] 2026 03 07

Create `internal/xblas/rope_arm64.go` and `rope_arm64.s`.

```go
// RoPEF32 applies rotary position embeddings to one position.
// in is [head_dim], cos/sin are [half_dim], out is [head_dim].
// rotaryDim must be even and <= head_dim.
//go:noescape
func RoPEF32(out, in, cos, sin *float32, halfDim, headDim int)
```

NEON implementation:
1. Load 4 cos[i] and 4 sin[i].
2. Load 4 in[i] (first half) and 4 in[i+halfDim] (second half).
3. Compute: out[i] = in[i]*cos[i] - in[i+half]*sin[i]
           out[i+half] = in[i+half]*cos[i] + in[i]*sin[i]
   Using FMUL + FMLS (fused multiply-subtract) and FMLA (fused multiply-add).
4. Copy pass-through dimensions (rotaryDim < headDim) with LDP/STP.

Wire into `compute.FusedRoPE()` (at `compute/fused_rope.go`).
Call `RoPEF32()` per (batch, seq) position instead of the per-element loop.

Acceptance:
- Output matches current FusedRoPE within 1e-6 relative error.
- Benchmark: >= 2x faster for head_dim=256.
- Handles non-aligned halfDim (tail scalar).
- Dependencies: none.

- [x] S102.4.1 NEON RoPE correctness + benchmark tests  [x] 2026 03 07  Owner: TBD  Est: 1.5h

##### T102.5 NEON vectorized elementwise (Add, Mul, Sub for same-shape)  Owner: TBD  Est: 3h  [x] 2026 03 07

Create `internal/xblas/elementwise_arm64.go` and `elementwise_arm64.s`.

```go
// VaddF32 computes out[i] = a[i] + b[i] for n float32 values using NEON.
//go:noescape
func VaddF32(out, a, b *float32, n int)

// VmulF32 computes out[i] = a[i] * b[i] for n float32 values using NEON.
//go:noescape
func VmulF32(out, a, b *float32, n int)

// VsubF32 computes out[i] = a[i] - b[i] for n float32 values using NEON.
//go:noescape
func VsubF32(out, a, b *float32, n int)

// VdivF32 computes out[i] = a[i] / b[i] for n float32 values using NEON.
//go:noescape
func VdivF32(out, a, b *float32, n int)
```

NEON implementation: Load 8 elements (2x V registers), FADD/FMUL/FSUB/FDIV,
store. Same loop structure as existing vdotf32 with tail handling.

Wire into the same-shape fast path from T101.1. When T is float32 and shapes
match, dispatch to the NEON function instead of the per-element Go loop:
```go
if slicesEqual(a.Shape(), b.Shape()) {
    // Type-assert to float32 for NEON dispatch
    if af32, ok := any(a).(*tensor.TensorNumeric[float32]); ok {
        xblas.VaddF32(&rData[0], &af32.Data()[0], &bf32.Data()[0], len(rData))
        return result, nil
    }
    // Generic same-shape loop fallback
    ...
}
```

Acceptance:
- Output matches Go loop within float32 precision.
- Benchmark: >= 2x faster for n=2048.
- Tail elements handled correctly.
- Dependencies: T101.1 (same-shape fast path).

- [x] S102.5.1 NEON elementwise correctness + benchmark tests  [x] 2026 03 07  Owner: TBD  Est: 1h

##### T102.6 NEON vectorized scalar ops (MulScalar, AddScalar, DivScalar)  Owner: TBD  Est: 2h  [x] 2026 03 07

Create `internal/xblas/scalar_arm64.go` and `scalar_arm64.s`.

```go
// VmulScalarF32 computes out[i] = a[i] * scalar for n float32 values.
//go:noescape
func VmulScalarF32(out, a *float32, scalar float32, n int)

// VaddScalarF32 computes out[i] = a[i] + scalar for n float32 values.
//go:noescape
func VaddScalarF32(out, a *float32, scalar float32, n int)

// VdivScalarF32 computes out[i] = a[i] / scalar for n float32 values.
//go:noescape
func VdivScalarF32(out, a *float32, scalar float32, n int)
```

NEON implementation: VDUP scalar to all 4 lanes, then FMUL/FADD/FDIV with
loaded data, same loop structure as VmulF32 but with broadcast scalar.

Wire into CPUEngine.MulScalar/AddScalar/DivScalar via float32 type assertion.

Acceptance:
- Output matches Go loop.
- Benchmark: >= 2x faster for n=2048.
- Dependencies: none.

- [x] S102.6.1 NEON scalar ops correctness + benchmark tests  [x] 2026 03 07  Owner: TBD  Est: 1h

##### T102.7 Factor out shared NEON exp polynomial  Owner: TBD  Est: 1.5h  [x] 2026 03 07

T102.1 (Softmax), T102.3 (SiLU) both need a vectorized exp() approximation.
Factor the exp polynomial into a reusable assembly macro or a separate function:

```go
// VexpF32 computes out[i] = exp(x[i]) for n float32 values.
// Uses range-reduced polynomial: exp(x) = 2^(int(x/ln2)) * poly(frac).
//go:noescape
func VexpF32(out, x *float32, n int)
```

The range-reduction approach:
1. n = round(x / ln2) via FCVTNS (round to nearest int).
2. r = x - n * ln2 (reduced range in [-ln2/2, ln2/2]).
3. poly(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 (degree 5 for <1e-6 error).
4. result = ldexp(poly(r), n) via integer add to float32 exponent bits.

This function is used by:
- SoftmaxF32 (exp(x-max) in pass 2)
- SiLUF32 (exp(-x) for sigmoid)
- Exp engine op (standalone)

Acceptance:
- VexpF32 output matches math.Exp within 1e-6 relative error for [-88, 88].
- Handles -inf, +inf, NaN correctly.
- Shared between Softmax and SiLU implementations.
- Dependencies: none (but should be done before or alongside T102.1, T102.3).

- [x] S102.7.1 VexpF32 correctness test  Owner: TBD  Est: 1h  2026 03 07
  - Test edge cases: 0, -0, very negative (underflow), very positive (overflow), NaN.
  - Test accuracy: 10000 random values in [-88, 88], max relative error < 1e-6.

##### T102.8 Wire NEON functions into CPUEngine  Owner: TBD  Est: 2h  [x] 2026 03 07

Update `compute/cpu_engine.go` and `compute/fused_*.go` to dispatch to the
new NEON functions when T is float32 on ARM64:

1. `Softmax()`: For float32, last-axis, call `xblas.SoftmaxF32()` per row.
2. `FusedRMSNorm()`: Call `xblas.RMSNormF32()` per row.
3. `FusedSiLUGate()`: Call `xblas.SiLUGateF32()`.
4. `FusedRoPE()`: Call `xblas.RoPEF32()` per position.
5. `Exp()`: Call `xblas.VexpF32()` for float32.
6. `binaryOp()` same-shape fast path: Call `xblas.VaddF32` etc.
7. `MulScalar/AddScalar/DivScalar()`: Call `xblas.VmulScalarF32` etc.

Use runtime type assertion `any(a).(*tensor.TensorNumeric[float32])` to detect
float32 tensors. Non-float32 types fall through to the generic loop.

Acceptance:
- All existing tests pass (NEON path produces same results as Go path).
- bench_tps CPU shows measurable improvement.
- No regression for non-float32 types.
- Dependencies: T102.1-T102.7, T101.1.

- [x] S102.8.1 End-to-end wiring integration test  Owner: TBD  Est: 1h  [x] 2026 03 07
  - Run full Gemma 3 inference on ARM64 (DGX Spark). Verify output unchanged.

##### T102.9 Run golangci-lint on compute/, internal/xblas/  Owner: TBD  Est: 30m  [x] 2026 03 07
  - Dependencies: T102.1-T102.8.

#### E103: Tensor Arena for Buffer Reuse (O103)

A major source of CPU overhead is Go heap allocation of intermediate tensors.
Each engine op creates a new tensor (slice header + backing array), which the
GC must collect. A tensor arena pre-allocates a pool of buffers and reuses them.

##### T103.1 Design and implement TensorArena  Owner: TBD  Est: 4h  [x] 2026 03 07

Create `compute/arena.go`.

```go
type TensorArena struct {
    mu      sync.Mutex
    buffers map[int][]*[]float32 // size -> free list of backing arrays
    stats   ArenaStats
}

type ArenaStats struct {
    Hits   int64
    Misses int64
    Bytes  int64
}

func NewTensorArena() *TensorArena
func (a *TensorArena) Get(size int) []float32    // reuse or allocate
func (a *TensorArena) Put(buf []float32)          // return to pool
func (a *TensorArena) Stats() ArenaStats
func (a *TensorArena) Reset()                     // release all buffers
```

Strategy:
- Bucket sizes by power-of-2 rounding (e.g., request 2048 elements gets a
  2048-element buffer; request 2049 gets a 4096 buffer).
- Per-bucket free list (stack). Get pops, Put pushes.
- Arena.Reset() clears all free lists (called between generations).
- Thread-safe via sync.Mutex (low contention since parallelFor uses shared pool).

Acceptance:
- Get returns buffers of correct minimum size.
- Put + Get cycle reuses the same buffer.
- No data corruption from buffer reuse (caller zeroes if needed).
- Dependencies: none.

- [x] S103.1.1 TensorArena unit tests  Owner: TBD  Est: 1.5h  2026 03 07
  - Test Get/Put cycle, size bucketing, Reset.
  - Test concurrent Get/Put from multiple goroutines.

##### T103.2 Wire TensorArena into CPUEngine  Owner: TBD  Est: 3h  [x] 2026 03 07

Update `CPUEngine.getOrCreateDest()` to use the arena when a dst tensor is
not provided:

```go
func (e *CPUEngine[T]) getOrCreateDest(shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
    if len(dst) > 0 && dst[0] != nil {
        return dst[0], nil
    }
    size := 1
    for _, d := range shape {
        size *= d
    }
    // Get buffer from arena
    buf := e.arena.Get(size)
    return tensor.NewFromBuffer(shape, buf[:size])
}
```

Add `arena *TensorArena` field to CPUEngine. Initialize in `NewCPUEngine()`.
Arena.Reset() called at the start of each Forward() pass (or per generation).

For the `dst` pattern: when the engine is done with an intermediate tensor,
the arena reclaims it. This requires a tensor finalizer or explicit release.
Simplest approach: the ExecutionPlan.Run() loop can call arena.Put() for slot
buffers that are no longer referenced (their last consumer has run).

Acceptance:
- Arena hit rate > 80% during decode (most tensors have repeating shapes).
- No double-free or use-after-free (verified with race detector).
- Memory usage stable during 100-token generation (no growth).
- Dependencies: T103.1.

- [x] S103.2.1 Arena integration benchmarks  Owner: TBD  Est: 1.5h  [x] 2026 03 07
  - BenchmarkForwardWithArena vs BenchmarkForwardWithoutArena.
  - Memory allocation profile comparison.

##### T103.3 Run golangci-lint on compute/  Owner: TBD  Est: 15m  [x] 2026 03 07
  - Dependencies: T103.1-T103.2.

#### E104: CPU Benchmark Validation (O104)

##### T104.1 CPU ARM64 benchmark with all optimizations  Owner: TBD  Est: 2h

On DGX Spark:
1. Run `bench_tps -device cpu -tokens 100` with all Track D changes.
2. 7 runs, report median and peak tok/s.
3. Compare against 6.86 tok/s baseline.

Acceptance:
- Median >= 10 tok/s (46% improvement over 6.86).
- No NaN/Inf. Output identical to pre-optimization.
- Dependencies: E101, E102, E103.

- [ ] S104.1.1 CPU benchmark report  Owner: TBD  Est: 30m
  - Table: operation, before, after, speedup.

##### T104.2 Per-operation profiling  Owner: TBD  Est: 1.5h

Run CPU inference with operation timing enabled (CPUEngine.recordOp metrics).
Compare per-op times before and after Track D:

Expected improvements:
- Pow: 8.9% -> ~2% (x*x specialization)
- binaryOp: 10.4% -> ~4% (same-shape fast path + NEON)
- Softmax: ~5% -> ~1.5% (NEON)
- RMSNorm: ~4% -> ~1.5% (NEON)
- SiLU/FFN: ~4% -> ~1.5% (NEON SiLUGate)
- RoPE: ~3% -> ~1% (NEON)
- Scalar ops: ~3% -> ~1% (NEON)
- Allocations: ~8% -> ~3% (arena)

Acceptance:
- Per-op timing report generated.
- Each optimized op shows >= 2x speedup.
- Dependencies: T104.1.

- [ ] S104.2.1 Per-operation profiling report  Owner: TBD  Est: 30m

##### T104.3 Output correctness verification  Owner: TBD  Est: 1h

Generate 100 tokens with "The capital of France is" on DGX Spark CPU.
Compare token-by-token with pre-optimization output. Must be identical
(all optimizations are mathematically equivalent, not approximations,
except exp polynomial which is within 1e-6).

Acceptance:
- All 100 tokens match pre-optimization output.
- No NaN or Inf.
- Dependencies: T104.1.

##### T104.4 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m
  - Dependencies: T104.1-T104.3.

---

## 4. Parallel Work

| Track | Epics | Description | Prerequisite |
|-------|-------|-------------|-------------|
| 0 remaining | E96 P2-P3 | Fix 7 remaining composition violations | none |
| A remaining | T87.3, S88/S89 | Remaining purego tests and cleanup | none |
| C: tracing | E97, E98, E99, E100 | Tracing compiler + GPU KV cache + emitters + integration | none (builds on completed Track B infrastructure) |
| D: NEON SIMD | E101, E102, E103, E104 | CPU SIMD acceleration + arena + benchmark | none |
| B: tuning | E94, E95 | GPU performance tuning and benchmark | Track C complete |

### Track C Internal Parallelism

| Wave | Tasks | Notes |
|------|-------|-------|
| C1 | T97.1, T97.2, T98.2, T99.1, T99.2, T99.3 | All independent: EngineProxy, Tracer, GPU KV cache, new emitters |
| C2 | T97.3, T97.5, T97.6, T97.7, T98.1, T98.3, T99.4 | After C1: wire proxy, handle edge cases, KV emitters, verify coverage |
| C3 | T97.4, T97.8 | After C2: CompileTraced implementation + lint |
| C4 | T98.4, T99.5, T100.1, T100.2 | After C3: integration wiring |
| C5 | T100.3, T100.4 | After C4: end-to-end test |

### Track D Internal Parallelism

| Wave | Tasks | Notes |
|------|-------|-------|
| D1 | T101.1, T101.2, T101.3, T102.7, T103.1 | All independent: fast paths, exp polynomial, arena |
| D2 | T102.1, T102.2, T102.3, T102.4, T102.5, T102.6 | After D1 (T102.7 needed for exp): all 6 NEON kernels in parallel |
| D3 | T101.4, T102.8, T102.9, T103.2, T103.3 | After D2: wire everything into CPUEngine + arena + lint |
| D4 | T104.1, T104.2, T104.3, T104.4 | After D3: benchmark and validate |

Track C and Track D are fully independent and can run in parallel.
Track 0 remaining and Track A remaining can run in parallel with both.
Track B (E94, E95) starts only after Track C is complete.

---

## 5. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M56: Tracing compiler works | E97 | none | CompileTraced produces primitive-op instruction tape for Gemma 3. All ops covered by emitters. |
| M57: GPU KV cache works | E98 | none | KV data on GPU, append/read correct, megakernel_ops.cu KV functions compile. |
| M58: Megakernel fires | E100 | M56, M57, E99 | bench_tps shows "megakernel: compiled and loaded". Output matches plan.Run(). |
| M59: 50 tok/s GPU | E94, E95 | M58 | bench_tps >= 50 tok/s median on DGX Spark GB10. |
| M60: 10 tok/s CPU ARM64 | E104 | E101, E102, E103 | bench_tps -device cpu >= 10 tok/s median on DGX Spark GB10. |

Critical path (GPU): T97.1 -> T97.3 -> T97.4 -> T100.1 -> T100.2 -> T100.3 -> T94.1 -> T95.1

Critical path (CPU): T102.7 -> T102.1/T102.3 -> T102.8 -> T104.1

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R100 | Tracing captures wrong execution path: Forward() has conditional logic (e.g., KV cache presence check in GQA). The trace is only valid for the traced path (decode mode, seqLen=1, cache active). | Megakernel produces wrong output for different paths | Medium | Only use traced plan for decode (seqLen=1). Prefill continues to use non-traced Forward(). The generate loop already separates prefill from decode. |
| R101 | Tensor identity via pointer is fragile: if the engine reuses a tensor buffer (pool allocation), the same pointer may represent different logical tensors. | Wrong slot wiring in traced tape | Medium | During tracing, disable tensor pooling (set pool to nil on EngineProxy). Each engine call allocates fresh tensors. This is only during one Compile() call, not runtime. |
| R102 | GPU KV cache memory budget: 26 layers x 2 (K/V) x 8192 seq x 1024 dim x 4 bytes = 1.6GB for full context. DGX Spark unified memory is large but this is significant. | Out of memory for long contexts | Low | Default to 512-token budget (~104MB). Allow user-configurable max_seq_len for megakernel path. Fall back to plan.Run() for longer contexts. |
| R103 | EngineProxy adds interface dispatch overhead to normal (non-tracing) inference. | Small performance regression on non-megakernel path | Low | Interface dispatch is ~1-2ns. With 650 engine calls per token at 7.78 tok/s, total overhead is ~1us. Negligible vs 128ms per token. |
| R104 | UnaryOp in sigmoid (inside SwiGLU) blocks clean trace for Gemma 3. | Must refactor sigmoid before megakernel works | Medium | T97.6 addresses this. Refactor Sigmoid.Forward() to use engine.Exp + engine.AddScalar + engine.Div. Small, focused change. |
| R105 | NEON exp polynomial approximation may have insufficient precision for some models. | Softmax/SiLU output differs enough to change generated tokens | Low | Degree-5 polynomial with range reduction achieves <1e-6 relative error. Verify with full 100-token generation comparison test (T104.3). |
| R106 | Plan9 assembler lacks many NEON mnemonics; must use raw WORD encoding. | Assembly is harder to write and debug | High | Existing q4dot_arm64.s demonstrates the pattern. Comment each WORD with the mnemonic it represents. Use the same encoding patterns throughout. |
| R107 | Tensor arena may cause use-after-free if buffer returned to pool while still referenced. | Data corruption, intermittent test failures | Medium | Run all tests with -race flag. Arena.Put() only called by ExecutionPlan.Run() for slots that have been consumed by all downstream ops. Never Put() a buffer that is still a function argument. |
| R108 | float32 type assertion in CPUEngine NEON dispatch may not optimize with Go generics. | Dispatch overhead cancels NEON benefit for small tensors | Low | Only dispatch to NEON for tensors with >= 32 elements. Below that, the scalar loop is fast enough. Profile the type assertion overhead. |
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

### Change Summary -- 2026-03-07 (v9)

Wave D3 complete (sequential). NEON wired into CPUEngine, TensorArena integrated.

**Completed tasks:**
- T101.4: golangci-lint compute/ -- 0 issues.
- T102.8: Wire NEON into CPUEngine (Softmax, Exp, Add/Sub/Mul/Div, scalars, fused ops). Commits 7ac9a35, 0afe430.
- T102.9: golangci-lint compute/ + internal/xblas/ -- 0 issues.
- T103.1: TensorArena (re-implemented, original commit was empty). Commit dc97cd7.
- T103.2: Wire TensorArena into CPUEngine getOrCreateDest. Commit e3775a8.
- T103.3: golangci-lint compute/ -- 0 issues.

**Deviation:** T103.1 was marked complete in Wave D1 but the commit was empty (no files).
Re-implemented as part of Wave D3 with proper tests and committed.

**Wave D4 unblocked:** T104.1-T104.4 (benchmark validation on DGX Spark).

### Change Summary -- 2026-03-07 (v8)

Wave D2 complete (6 NEON assembly kernels in parallel via worktrees). All
committed to feat/neon-softmax. Tests pass, lint clean.

**Completed tasks:**
- T102.1: NEON Softmax (3-pass: max, exp+sum, normalize). Commit bc775d8.
- T102.2: NEON RMSNorm (FRSQRTE + 2 Newton-Raphson). Commit a40a2f7.
- T102.3: NEON SiLU + SiLUGate (inline exp polynomial + FRECPE). Commit d766923.
- T102.4: NEON RoPE (4-wide SIMD + scalar tail + passthrough). Commit ef099ef.
- T102.5: NEON elementwise VaddF32/VmulF32/VsubF32/VdivF32. Commit a40a2f7.
- T102.6: NEON scalar VmulScalarF32/VaddScalarF32/VdivScalarF32. Commit c751b5c.
- Lint fix: rmsnorm_generic.go D->dim, rope_test.go rand/v2. Commit 0faf769.

**Wave D3 unblocked:** T101.4 (lint compute/), T102.8 (wire NEON into CPUEngine),
T102.9 (lint xblas/), T103.2 (wire arena into CPUEngine), T103.3 (lint compute/).

### Change Summary -- 2026-03-07 (v7)

Wave D1 complete (5 tasks in parallel via worktrees). All merged into
feat/same-shape-fast-path. Tests pass, lint clean.

**Completed tasks:**
- T101.1: Same-shape fast path in binaryOp (7-8x speedup for same-shape ops). Commit f733d15.
- T101.2: Pow x^2 specialization using x*x (13-15x speedup). Commit c28a529.
- T101.3: Scalar op baseline benchmarks (MulScalar, AddScalar, DivScalar). Commit 3d8c3d7.
- T102.7: NEON VexpF32 vectorized exp polynomial (max error 8.98e-08). Commit 5931298.
- T103.1: TensorArena with power-of-2 bucketed pooling (6 tests + bench). Commit b4b5eb1.

**Wave D2 unblocked:** T102.1 (Softmax), T102.2 (RMSNorm), T102.3 (SiLU),
T102.4 (RoPE), T102.5 (elementwise), T102.6 (scalar ops) -- all 6 NEON kernels.

### Change Summary -- 2026-03-07 (v6)

Added Track D (NEON SIMD CPU Acceleration) with 4 new epics to close the CPU
performance gap with llama.cpp.

**New epics:**
- E101: Same-Shape Fast Paths and Pow Specialization (T101.1-T101.4)
- E102: NEON Assembly for Hot-Path Operations (T102.1-T102.9)
- E103: Tensor Arena for Buffer Reuse (T103.1-T103.3)
- E104: CPU Benchmark Validation (T104.1-T104.4)

**New objectives:** O101 (NEON SIMD), O102 (fast paths), O103 (arena), O104 (10 tok/s CPU).
**New milestone:** M60 (10 tok/s CPU ARM64).
**New risks:** R105 (exp precision), R106 (plan9 asm), R107 (arena use-after-free), R108 (dispatch overhead).

**ADRs created:**
- docs/adr/029-neon-simd-cpu-acceleration.md: NEON SIMD strategy for CPU parity
  with llama.cpp. Evaluates 3 approaches (CGo+intrinsics, plan9 assembly,
  compiler autovectorization), selects plan9 assembly.

**Key design decisions:**
- All NEON assembly follows existing internal/xblas/ pattern with _arm64.s +
  _arm64.go + _generic.go triple.
- exp() approximation shared between Softmax and SiLU via VexpF32 (T102.7).
- Same-shape fast path (T101.1) is pure Go, NEON dispatch (T102.5) layers on top.
- Tensor arena uses power-of-2 bucketing with per-bucket free lists.
- Track D is fully independent of Track C and can run in parallel.

### Change Summary -- 2026-03-07 (v5)

Added Track C (Tracing Compiler) to resolve megakernel blocker.

**Root cause identified**: graph.Compile() records composite node OpTypes
(GroupedQueryAttention, FFN, EmbeddingLookup, LMHead) instead of primitive
Engine ops. The megakernel emitter only knows primitives, so CheckSupport()
fails silently. The megakernel never fires on the real model.

**Solution**: Tracing compiler (approach C). An EngineProxy wraps the Engine
and records every primitive Engine method call during compilation. Forward()
calls proceed normally through all composite layers, but the proxy captures
every primitive call (MatMul, Softmax, etc.) as a separate instruction. This
is the JAX/PyTorch FX pattern -- layers remain high-level abstractions, the
compiler flattens automatically.

New epics:
- E97: EngineProxy and Tracer (T97.1-T97.8)
- E98: GPU KV Cache for Megakernel Attention (T98.1-T98.4)
- E99: New Primitive Op Emitters (T99.1-T99.5)
- E100: Tracing Compiler Integration (T100.1-T100.4)

New milestones: M56 (tracing works), M57 (GPU KV cache), M58 (megakernel fires),
M59 (50 tok/s).

ADRs created:
- docs/adr/028-tracing-compiler.md: Tracing compiler for automatic primitive
  op decomposition. Evaluates 3 approaches, selects approach C.

Trimmed: Completed epics E90, E91, E92, and completed tasks from E96 P1,
E87, E88, E89, E93 moved to docs/design.md section 15.16.

### Change Summary -- 2026-03-07 (v4)

Added Track 0 (E96: Composition Fixes). 5-agent audit found 12 violations.
Priority 1 (T96.1-T96.3) and Priority 2 partial (T96.4-T96.5) completed.
ADR 027 created. Milestone M49 added. Risk R99 added.

---

## 9. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for full context. ADRs in docs/adr/.
- **Phase 34 status:** Tracks 0/A/B infrastructure complete. Track C (tracing
  compiler) and Track D (NEON SIMD) are the active work. See docs/design.md
  section 15.16 for what was already delivered.
- **Track C core problem:** The megakernel is fully built (emit, compile, load,
  wire) but never fires because the instruction tape has composite ops. The
  tracing compiler (E97) fixes this by recording primitive Engine calls.
- **Track D core problem:** CPU inference is 6.86 tok/s, llama.cpp is ~100.
  The gap is NEON SIMD for all non-matmul ops, broadcasting overhead, and GC
  allocation pressure. E101 (fast paths), E102 (NEON assembly), E103 (arena).
- **Key starting points:**
  - Track C: `compute/engine.go` (Engine interface) -> `compute/engine_proxy.go`
    (T97.1) -> `compute/tracer.go` (T97.2) -> `graph/compile.go` CompileTraced() (T97.4)
  - Track D: `compute/cpu_engine.go:546` (binaryOp) -> T101.1 same-shape fast path.
    `internal/xblas/q4dot_arm64.s` (existing NEON pattern) -> T102.x new kernels.
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key File Map

| File | Purpose | Track C Change | Track D Change |
|------|---------|---------------|---------------|
| `compute/engine.go` | Engine[T] interface (~25 methods) | Reference for EngineProxy | Reference for NEON dispatch |
| `compute/engine_proxy.go` | (new) EngineProxy[T] wrapping Engine | T97.1 | -- |
| `compute/tracer.go` | (new) Tracer[T] records TracedOps | T97.2 | -- |
| `compute/cpu_engine.go` | CPUEngine implementation | -- | T101.1 fast path, T102.8 NEON wiring |
| `compute/fused_rmsnorm.go` | FusedRMSNorm | -- | T102.2 NEON dispatch |
| `compute/fused_rope.go` | FusedRoPE | -- | T102.4 NEON dispatch |
| `compute/fused_silugate.go` | FusedSiLUGate | -- | T102.3 NEON dispatch |
| `compute/arena.go` | (new) TensorArena | -- | T103.1 |
| `graph/compile.go` | Compile() + ExecutionPlan | Add CompileTraced() T97.4 | -- |
| `graph/graph.go` | Graph[T] struct | Add EngineProxy accessors T97.3 | -- |
| `generate/tracing_cache.go` | (new) TracingCacheProvider | T98.1 | -- |
| `internal/codegen/kv_cache.go` | (new) GPUKVCache | T98.2 | -- |
| `internal/codegen/optable.go` | Op emitter table (26 ops) | Add Slice, Repeat, Reduce T99 | -- |
| `internal/codegen/emit.go` | EmitMegakernel() | Add KV cache kernel args T98.3 | -- |
| `internal/xblas/softmax_arm64.s` | (new) NEON Softmax | -- | T102.1 |
| `internal/xblas/rmsnorm_arm64.s` | (new) NEON RMSNorm | -- | T102.2 |
| `internal/xblas/silu_arm64.s` | (new) NEON SiLU/SiLUGate | -- | T102.3 |
| `internal/xblas/rope_arm64.s` | (new) NEON RoPE | -- | T102.4 |
| `internal/xblas/elementwise_arm64.s` | (new) NEON Add/Mul/Sub/Div | -- | T102.5 |
| `internal/xblas/scalar_arm64.s` | (new) NEON MulScalar/etc. | -- | T102.6 |
| `internal/xblas/exp_arm64.s` | (new) NEON VexpF32 | -- | T102.7 |
| `generate/generator.go` | Generate() decode loop | Use CompileTraced T100.1 | -- |
| `generate/stream.go` | GenerateStream() decode loop | Use CompileTraced T100.1 | -- |
| `generate/megakernel.go` | tryCompileMegakernel() | Wire GPU KV cache T100.2 | -- |
| `inference/arch_common.go` | buildTransformerGraph() | Wrap engine with EngineProxy T97.3 | -- |

### Performance Baselines

| Config | tok/s | Source |
|--------|-------|--------|
| CPU ARM64 | 6.86 | Phase 30 |
| GPU (cuda) | 10.32 peak / 7.78 median | Phase 33 |
| GPU (purego) | 6.59 CPU | Phase 34 Track A |
| Ollama GB10 | ~100 (est.) | Interpolated |
| Theoretical | ~182 | 273 GB/s / 1.5GB |

---

## 10. Appendix

### NEON Exp Polynomial Reference

The vectorized exp() (T102.7) uses range reduction with degree-5 polynomial:

```
Input: x (float32)
1. n = round(x * (1/ln2))        // FMUL + FCVTNS
2. r = x - n * ln2               // FMSUB (fused)
3. p = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))  // Horner's method
   c0=1.0, c1=1.0, c2=0.5, c3=1/6, c4=1/24, c5=1/120
4. result = ldexp(p, n)           // add n to float32 exponent bits:
                                  //   FCVTZS Vn.4S, Vn.4S (float->int)
                                  //   SHL Vn.4S, Vn.4S, #23 (shift to exponent)
                                  //   ADD Vp.4S, Vp.4S, Vn.4S (add exponent)
```

Max relative error: < 2e-7 for x in [-87.3, 88.7] (float32 range).

### ARM64 NEON Instruction Encoding Cheat Sheet

Common instructions requiring WORD encoding in Go plan9 assembly:

| Instruction | Encoding | Description |
|-------------|----------|-------------|
| FMAXP Vd.4S, Vn.4S, Vm.4S | `6E20F400+...` | Pairwise max |
| FADDP Vd.4S, Vn.4S, Vm.4S | `6E20D400+...` | Pairwise add |
| FRSQRTE Vd.4S, Vn.4S | `6EA1D800+...` | Reciprocal sqrt estimate |
| FRSQRTS Vd.4S, Vn.4S, Vm.4S | `0EA0FC00+...` | RSqrt Newton step |
| FCVTNS Vd.4S, Vn.4S | `4E21A800+...` | Float to int nearest |
| FRINTX Vd.4S, Vn.4S | `6E219800+...` | Round to integral |
| FNEG Vd.4S, Vn.4S | `6EA0F800+...` | Negate |

Note: All register operand fields must be encoded correctly in the immediate.
Use `aarch64-linux-gnu-objdump -d` on a test .o to verify encodings.
