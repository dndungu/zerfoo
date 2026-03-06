# Zerfoo Development Plan -- Phase 30: Graph Compilation and Worker Pool

## 1. Context

### Problem Statement

Phase 29 achieved 6.5 tok/s for Gemma 3 2B Q4_0 on DGX Spark GB10 (ARM64,
20 cores). Compute kernels (SGEMM + Q4 GEMV) take only ~24ms/token wall time,
but total per-token latency is ~154ms. The ~130ms gap is framework overhead.

See docs/design.md for full architecture context and Phases 1-29 history.

### Overhead Breakdown (Phase 29, DGX Spark GB10, Gemma 3 2B Q4_0)

| Source | Wall ms/token | Root Cause |
|--------|---------------|------------|
| Graph traversal | ~50 | 780 node executions/token. Each: interface dispatch on Node[T].Forward(), shape validation, dependency lookup in map, memo read/write under mutex. |
| Goroutine scheduling | ~40 | ~130 MatMul calls/token, each spawning ~20 goroutines via sync.WaitGroup. 2600 goroutine create/join cycles per token. |
| Memory / GC | ~25 | TensorPool reduces allocs but pool itself has mutex contention. GC scans pooled memory. Memo map rebuilt every Forward(). |
| Other (scheduling, context) | ~15 | Channel ops in ParallelForward ready queue, context.Done() select in every Engine op. |

### Existing Code Inventory

| File | What Exists | What Changes |
|------|-------------|-------------|
| `graph/graph.go:54-138` | Sequential Forward(): lock, build memo, iterate nodes, call node.Forward(), refcount release | Add Compile() that produces ExecutionPlan; Forward() checks for compiled plan first |
| `graph/parallel.go:12-163` | ParallelForward(): spawn N goroutines, in-degree scheduling via channel, mutex-protected memo | Replace with worker pool dispatch in compiled path |
| `graph/node.go:11-29` | Node[T] interface with Forward() returning (*TensorNumeric[T], error) | No changes to interface; compiled path bypasses it |
| `graph/optimize.go:25-157` | FoldConstantTransposes pass | Add new passes: buffer aliasing, instruction fusion |
| `compute/cpu_engine.go:55-89` | parallelFor(): threshold 32768, spawns goroutines per call | Replace with worker pool Submit() |
| `compute/cpu_engine.go:790-984` | MatMul(): shape validation, type dispatch, batch loop | Compiled path skips validation; direct kernel call |
| `compute/pool.go:14-55` | TensorPool[T]: shape-keyed, mutex per Acquire/Release | Replace with pre-allocated BufferArena in compiled path |
| `internal/xblas/gemm_simd_arm64.go:45-54` | sgemmGemvParallel: spawns goroutines per call | Use worker pool instead |
| `internal/xblas/gemm_quant.go:99-106` | gemmF32Q4NTParallel: spawns goroutines per call | Use worker pool instead |
| `generate/generator.go:181-210` | Decode loop: repeated graph.Forward() with single-token input | Call plan.Run() instead of Forward() after Compile() |

### Objectives

- O81: Graph compiler that pre-compiles the computation graph into a flat
  instruction sequence, eliminating per-node interface dispatch, shape
  validation, and pool operations during decode.
- O82: Persistent worker pool that replaces per-call goroutine creation in
  MatMul, GEMV, and element-wise parallel operations.
- O83: Pre-allocated buffer arena that replaces per-Forward() memo map and
  TensorPool mutex contention.
- O84: >= 15 tok/s on DGX Spark GB10 for Gemma 3 2B Q4_0 CPU ARM64.

### Non-Goals

- GPU inference pipeline (separate phase).
- Training performance or backward pass compilation.
- New model architectures or operators.
- x86 AVX2/AVX-512 worker pool (ARM64 only for now).
- Changing the Node[T] or Engine[T] interfaces.
- Parallel graph execution (ParallelForward) -- compiled path replaces it.

### Constraints and Assumptions

- Go standard library only where possible.
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- Table-driven tests using the standard testing package.
- DGX Spark GB10 at ssh ndungu@192.168.86.250 for benchmarks.
- The interpreted Forward() path must remain functional as fallback.
- Compile() runs after the first forward pass (prefill), when shapes are known.
- The compiled plan is invalidated if input shape changes.

### Success Metrics

| Metric | Current (Phase 29) | Phase 30 Target |
|--------|---------------------|-----------------|
| tok/s (Gemma 3 2B Q4_0, CPU ARM64) | 6.5 | >= 15 |
| Framework overhead ms/token | ~130 | < 43 |
| Goroutine creates/token | ~2600 | 0 (worker pool) |
| Mutex acquires/token (pool) | ~780 | 0 (pre-allocated arena) |

### Decision Rationale

See docs/adr/021-graph-compilation-worker-pool.md

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D201 | Persistent worker pool | Eliminate 2600 goroutine create/join cycles per token |
| D202 | Graph compiler | Eliminate per-node interface dispatch, shape validation, memo map |
| D203 | Buffer arena | Pre-allocate all intermediate tensors, eliminate pool mutex |
| D204 | Instruction fusion | Batch consecutive element-wise ops into single kernel |
| D205 | DGX Spark benchmark | >= 15 tok/s validated on GB10 |

### Out of Scope

- GPU compiled execution path.
- Backward pass compilation.
- Dynamic batching or multi-sequence compilation.
- Prompt caching / prefix sharing.
- Changing public API of graph.Graph or compute.Engine.

---

## 3. Checkable Work Breakdown

### E54: Persistent Worker Pool (O82, O84)

Replace per-call goroutine spawning with a fixed-size worker pool. The pool
is created once at engine initialization and reused across all Forward() calls.

Currently, goroutines are spawned in three places:
1. `internal/xblas/gemm_simd_arm64.go:76-101` -- sgemmGemvParallel
2. `internal/xblas/gemm_quant.go:121-139` -- gemmF32Q4NTParallel
3. `compute/cpu_engine.go:55-89` -- parallelFor (element-wise ops)

Each spawns `runtime.NumCPU()` goroutines per call via sync.WaitGroup.

- [x] T54.1 Create WorkerPool type  Owner: TBD  Est: 2h
  - Add `internal/workerpool/pool.go` with:
    ```
    type Pool struct {
        workers int
        tasks   chan func()
        wg      sync.WaitGroup
    }
    func New(workers int) *Pool
    func (p *Pool) Submit(tasks []func(), done *sync.WaitGroup)
    func (p *Pool) Close()
    ```
  - `New(n)` spawns n goroutines that block on `tasks` channel.
  - `Submit(tasks, done)` sends each task func to the channel. Caller
    calls `done.Wait()` to synchronize.
  - `Close()` closes the channel; workers exit their loop.
  - Workers are long-lived: created once, reused across all calls.
  - Acceptance: Pool processes 10K batches of 20 tasks correctly.
    No goroutine leaks (runtime.NumGoroutine stable after Close).
  - Dependencies: none.

- [x] S54.1.1 WorkerPool unit tests  Owner: TBD  Est: 1h
  - Table-driven: 1 worker, 4 workers, 20 workers.
  - Verify: all tasks execute, correct results, no races (-race flag).
  - Verify: Close() is safe to call multiple times.
  - Verify: Submit with empty task slice is a no-op.
  - Benchmark: Submit+Wait latency vs goroutine spawn+WaitGroup.

- [x] T54.2 Wire worker pool into xblas GEMV  Owner: TBD  Est: 2h
  - Add a package-level `var defaultPool *workerpool.Pool` in
    `internal/xblas/pool.go`.
  - `InitPool(n int)` creates the pool. Called by CPUEngine constructor.
  - `ShutdownPool()` closes it. Called by engine Close() or test cleanup.
  - Modify `sgemmGemvParallel` (gemm_simd_arm64.go:76-101):
    - Replace `wg.Add(1); go func() { ... }` with pool.Submit().
    - Keep the same chunking logic (divide N across nCores).
  - Modify `gemmF32Q4NTParallel` (gemm_quant.go:121-139):
    - Same replacement pattern.
  - Acceptance: `go test ./internal/xblas/ -race` passes. Benchmark shows
    reduced latency for repeated GEMV calls (pool reuse vs goroutine spawn).
  - Dependencies: T54.1.

- [x] S54.2.1 GEMV worker pool benchmark  Owner: TBD  Est: 30m
  - BenchmarkGemvWorkerPool: 1000 iterations of M=1 GEMV at N=2048, K=2048.
  - Compare: goroutine-per-call vs pool. Report ns/op difference.

- [x] T54.3 Wire worker pool into CPUEngine parallelFor  Owner: TBD  Est: 1h
  - Modify `compute/cpu_engine.go:55-89` (parallelFor):
    - Accept optional `*workerpool.Pool` parameter (or use engine field).
    - If pool is available, use pool.Submit() instead of spawning goroutines.
    - If pool is nil, fall back to current goroutine-per-chunk behavior.
  - Add `pool *workerpool.Pool` field to CPUEngine struct.
  - Initialize pool in CPUEngine constructor (NewCPUEngine or similar).
  - Pass pool to xblas via `xblas.InitPool()` from the same constructor.
  - Acceptance: `go test ./compute/ -race` passes. Element-wise ops
    (Mul, Add) use pool when available.
  - Dependencies: T54.1.

- [x] S54.3.1 CPUEngine parallelFor pool tests  Owner: TBD  Est: 30m
  - Verify element-wise Mul of large tensors uses pool (no new goroutines).
  - Verify small tensors (< 32768 elements) skip parallelization as before.

- [x] T54.4 Run golangci-lint on internal/workerpool/ and internal/xblas/  Owner: TBD  Est: 15m
  - Dependencies: T54.2.

- [x] T54.5 Run golangci-lint on compute/  Owner: TBD  Est: 15m
  - Dependencies: T54.3.

### E55: Graph Compiler (O81, O83, O84)

Pre-compile the computation graph into a flat instruction sequence. The compiler
runs after the first Forward() call (prefill), when all tensor shapes are known.
The compiled ExecutionPlan replaces the interpreted node-by-node loop for decode.

Decision rationale: docs/adr/021-graph-compilation-worker-pool.md

- [x] T55.1 Define Instruction and ExecutionPlan types  Owner: TBD  Est: 2h
  - Add `graph/compile.go` with:
    ```
    type Instruction[T tensor.Numeric] struct {
        Kernel   func(ctx context.Context, inputs []*tensor.TensorNumeric[T], output *tensor.TensorNumeric[T]) error
        InputIdx []int   // indices into buffer arena
        OutputIdx int    // index into buffer arena
    }
    type ExecutionPlan[T tensor.Numeric] struct {
        instructions []Instruction[T]
        buffers      []*tensor.TensorNumeric[T]  // pre-allocated arena
        inputIdx     []int                        // which buffers are graph inputs
        outputIdx    int                          // which buffer is graph output
    }
    func (p *ExecutionPlan[T]) Run(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
    ```
  - `Run()` copies input data into pre-allocated input buffers, iterates
    the flat instruction slice calling each Kernel with pre-resolved buffer
    pointers. No interface dispatch, no shape validation, no memo map.
  - Acceptance: ExecutionPlan.Run() with hand-built instructions produces
    correct output for a simple 3-node graph (Input -> MatMul -> Output).
  - Dependencies: none.

- [x] S55.1.1 ExecutionPlan unit tests  Owner: TBD  Est: 1h
  - Table-driven: 1-instruction plan, 5-instruction chain, diamond dependency.
  - Verify output matches interpreted Forward() for same graph.

- [x] T55.2 Implement Graph.Compile()  Owner: TBD  Est: 4h
  - Add `Compile(ctx context.Context) (*ExecutionPlan[T], error)` method
    to `graph.Graph[T]`.
  - Algorithm:
    1. Execute one Forward() pass to populate memo with tensor shapes.
    2. Assign buffer index to each node (topological order).
    3. For each non-input node, create an Instruction:
       a. Resolve the node's Forward() into a direct kernel function.
          Use a type switch on the node's OpType() to map to the
          appropriate CPUEngine method (e.g., "MatMul" -> engine.MatMul,
          "Add" -> engine.Add, etc.).
       b. Record InputIdx from dependency buffer indices.
       c. Record OutputIdx from this node's buffer index.
    4. Pre-allocate output buffers for all non-input nodes using shapes
       from the memo pass.
    5. Return the ExecutionPlan.
  - The Compile() method stores the plan on the Graph struct. Subsequent
    Forward() calls check if a compiled plan exists and use it.
  - Acceptance: Compile() + Run() produces identical output to Forward()
    for a 10-node graph with MatMul, Add, and RMSNorm.
  - Dependencies: T55.1.

- [x] S55.2.1 Graph.Compile() correctness tests  Owner: TBD  Est: 2h
  - Build a small transformer-like graph: Embed -> MatMul -> Add -> RMSNorm
    -> MatMul -> Softmax -> Output.
  - Verify compiled Run() output matches interpreted Forward() within 1e-6.
  - Verify compiled plan buffer count equals number of unique intermediate
    tensors.

- [x] T55.3 Wire Compile() into Generator decode loop  Owner: TBD  Est: 2h
  - In `generate/generator.go`, after the prefill Forward() call (line ~175):
    1. Call `gen.graph.Compile(genCtx)` to produce the execution plan.
    2. Store plan on generator struct.
  - In the decode loop (line ~181-210):
    1. If plan exists, call `plan.Run(genCtx, tokenTensor)` instead of
       `gen.graph.Forward(genCtx, tokenTensor)`.
    2. Fall back to Forward() if Compile() returned an error.
  - Acceptance: Generate() produces identical tokens with compiled path.
    No behavior change for callers.
  - Dependencies: T55.2.

- [x] S55.3.1 Generator compiled decode test  Owner: TBD  Est: 1h
  - Generate 20 tokens with compiled path, compare to interpreted path.
  - Verify token-for-token identical output.

- [x] T55.4 OpType-to-kernel mapping for all used ops  Owner: TBD  Est: 3h
  - The Compile() type switch in T55.2 initially covers a minimal set.
    This task adds mappings for all operators used in Gemma 3 2B:
    - MatMul, Add, Mul, Sub, Pow, Sqrt, Div (arithmetic)
    - RMSNorm, LayerNorm (normalization)
    - Softmax, SiLU (activations)
    - Reshape, Transpose, Unsqueeze, Concat, Slice, Gather (tensor ops)
    - RotaryEmbedding (positional)
    - GroupQueryAttention (attention)
    - Linear, LMHead (projection)
  - For each: create a closure that calls the engine method with
    pre-validated shapes. Skip shape validation inside the closure.
  - Unsupported ops: fall back to calling node.Forward() (interpreted).
  - Acceptance: Compile() succeeds for the full Gemma 3 2B graph.
    No "unsupported op" fallbacks for the standard ops listed above.
  - Dependencies: T55.2.

- [x] S55.4.1 Full Gemma 3 graph compilation test  Owner: TBD  Est: 1h
  - Load Gemma 3 2B Q4_0 model, run Compile(), verify all instructions
    use direct kernel functions (no interpreted fallbacks).
  - Verify Run() output matches Forward() for a 5-token prompt.

- [x] T55.5 Run golangci-lint on graph/  Owner: TBD  Est: 15m
  - Dependencies: T55.4.

- [x] T55.6 Run golangci-lint on generate/  Owner: TBD  Est: 15m
  - Dependencies: T55.3.

### E56: Buffer Arena (O83, O84)

Replace the per-Forward() memo map and TensorPool with a pre-allocated buffer
arena. All intermediate tensor buffers are allocated once during Compile() and
reused across decode steps without mutex or map operations.

- [x] T56.1 Implement BufferArena  Owner: TBD  Est: 2h
  - Add `graph/arena.go` with:
    ```
    type BufferArena[T tensor.Numeric] struct {
        buffers []*tensor.TensorNumeric[T]
    }
    func NewBufferArena[T tensor.Numeric](shapes [][]int) *BufferArena[T]
    func (a *BufferArena[T]) Get(idx int) *tensor.TensorNumeric[T]
    func (a *BufferArena[T]) Reset()
    ```
  - `NewBufferArena(shapes)` pre-allocates one tensor per shape.
  - `Get(idx)` returns the pre-allocated buffer at index idx. No mutex.
  - `Reset()` zeros all buffers for the next decode step. Uses
    `runtime.memclrNoHeapPointers` pattern (zero the data slice).
  - The arena is created by Compile() and stored in ExecutionPlan.
  - Acceptance: Arena allocates correct shapes, Get returns valid tensors,
    Reset zeros all data.
  - Dependencies: none.

- [x] S56.1.1 BufferArena unit tests  Owner: TBD  Est: 30m
  - Verify: allocation sizes match requested shapes.
  - Verify: Get(idx) returns same pointer across calls (no re-alloc).
  - Verify: Reset() zeros data without re-allocating.

- [x] T56.2 Wire BufferArena into ExecutionPlan  Owner: TBD  Est: 1h
  - DEVIATION: Superseded by slot-based ExecutionPlan. Slots store
    node.Forward() results directly, eliminating the copy+shape-loss
    bug from the arena approach. Memo map elimination achieved via
    indexed slot array instead of pre-allocated arena buffers.
  - Dependencies: T55.1, T56.1.

- [x] S56.2.1 Zero-alloc Run() benchmark  Owner: TBD  Est: 30m
  - DEVIATION: Slot approach still allocates per-instruction (node.Forward
    creates tensors). Zero-alloc goal deferred to future optimization.
    Current win: eliminated memo map + dependency map lookups.

- [x] T56.3 Buffer aliasing optimization  Owner: TBD  Est: 2h
  - DEVIATION: N/A for slot-based approach. Slots hold actual tensors
    from Forward(), not pre-allocated buffers. Aliasing would require
    a different mechanism (slot reuse after last consumer).
  - Dependencies: T56.2.

- [x] S56.3.1 Buffer aliasing correctness test  Owner: TBD  Est: 30m
  - DEVIATION: Skipped per T56.3.

- [x] T56.4 Run golangci-lint on graph/  Owner: TBD  Est: 15m
  - 0 issues.
  - Dependencies: T56.3.

### E57: Instruction Fusion (O84)

Batch consecutive element-wise operations into a single fused kernel call.
The Gemma 3 2B graph has sequences like RMSNorm -> Linear -> SiLU -> Mul
where intermediate tensors are consumed once. Fusing these into a single
pass over memory reduces cache pressure and eliminates intermediate writes.

- [x] T57.1 Identify fusible instruction patterns  Owner: TBD  Est: 2h
  - DEVIATION: N/A. Gemma 3 graph uses high-level nodes (RMSNorm, Linear,
    GQA, SwiGLU, TokenEmbedding) that already fuse element-wise ops
    internally. No sequences of bare Pow/Sqrt/Mul/Add exist in the graph.
    Instruction fusion targets a lower-level IR that doesn't match our
    graph representation.
  - Dependencies: T55.2.

- [x] S57.1.1 Fusion pattern detection tests  Owner: TBD  Est: 1h
  - DEVIATION: Skipped per T57.1. No fusible patterns exist.

- [x] T57.2 Implement fused element-wise kernels  Owner: TBD  Est: 3h
  - DEVIATION: Skipped per T57.1.

- [x] S57.2.1 Fused kernel correctness and benchmark  Owner: TBD  Est: 1h
  - DEVIATION: Skipped per T57.1.

- [x] T57.3 Wire fusion into Compile()  Owner: TBD  Est: 2h
  - DEVIATION: Skipped per T57.1.

- [x] S57.3.1 End-to-end fusion integration test  Owner: TBD  Est: 30m
  - DEVIATION: Skipped per T57.1.

- [x] T57.4 Run golangci-lint on graph/ and compute/  Owner: TBD  Est: 15m
  - N/A -- no new code added.

### E58: DGX Spark Benchmark Validation (O84)

- [ ] T58.1 End-to-end benchmark with all optimizations  Owner: TBD  Est: 2h
  - Run Gemma 3 2B Q4_0 on DGX Spark GB10 with:
    - Persistent worker pool (E54)
    - Compiled graph execution (E55)
    - Buffer arena (E56)
    - Instruction fusion (E57)
  - Measure: tok/s (100-token average), framework overhead ms/token,
    allocs/token, goroutine count.
  - Acceptance: >= 15 tok/s.
  - Dependencies: E54, E55, E56, E57.

- [ ] S58.1.1 Before/after profile comparison  Owner: TBD  Est: 30m
  - CPU profile: compiled vs interpreted path.
  - Memory profile: arena vs pool.
  - Goroutine profile: worker pool vs per-call spawn.

- [ ] T58.2 Regression test for correctness  Owner: TBD  Est: 1h
  - `go test ./... -race` on DGX Spark.
  - Generate 50 tokens with compiled path, compare to interpreted.
  - Acceptance: Token-for-token identical output.
  - Dependencies: T58.1.

- [ ] T58.3 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m
  - Dependencies: T58.2.

- [ ] T58.4 Overhead analysis and next steps  Owner: TBD  Est: 1h
  - If 15 tok/s not achieved, produce detailed profile showing remaining
    bottleneck and estimate effort to close the gap.
  - Document findings in docs/updates.md.
  - Dependencies: T58.1.

---

## 4. Timeline and Milestones

| Milestone | ID | Dependencies | Exit Criteria |
|-----------|----|-------------|---------------|
| M26: Worker pool | E54 | none | Worker pool processes GEMV without goroutine spawning. Benchmark shows reduced overhead. |
| M27: Graph compiler | E55 | none | Compile() produces ExecutionPlan for Gemma 3 2B. Run() output matches Forward(). |
| M28: Buffer arena | E56 | E55 | Zero heap allocations during compiled Run(). Arena size reduced by buffer aliasing. |
| M29: Instruction fusion | E57 | E55, E56 | >= 5 fused instruction chains in Gemma 3 2B. Output matches unfused. |
| M30: 15 tok/s | E58 | E54-E57 | >= 15 tok/s on DGX Spark GB10 validated. |

Recommended execution order:
1. **[E54, E55]** -- Independent. Worker pool and graph compiler can be built
   in parallel. E54 is simpler and delivers immediate benefit.
2. **E56** -- Buffer arena depends on E55 (ExecutionPlan types).
3. **E57** -- Instruction fusion depends on E55 and E56.
4. **E58** -- Final benchmark. Depends on all.

Critical path: E55 -> E56 -> E57 -> E58 (graph compiler is the foundation).

---

## 5. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R81 | Worker pool channel overhead exceeds goroutine spawn cost for small tasks | No improvement or regression | Low | Benchmark both paths. Keep threshold: small tasks bypass pool. |
| R82 | OpType-to-kernel mapping incomplete for some Gemma 3 ops | Compile() falls back to interpreted for some nodes, reducing benefit | Medium | Audit full Gemma 3 graph node list before starting T55.4. Interpreted fallback ensures correctness. |
| R83 | Buffer aliasing causes correctness bugs due to incorrect liveness analysis | Wrong output | Medium | Extensive testing: aliased vs non-aliased comparison for every graph topology. |
| R84 | Instruction fusion introduces numerical drift beyond tolerance | Model output quality degrades | Low | Fused kernels must match sequential within 1e-6. Skip fusion for ops sensitive to ordering. |
| R85 | 15 tok/s still not achievable due to memory bandwidth saturation | Target not met | Medium | Profile memory bandwidth utilization. If bandwidth-bound, document and explore memory layout optimization in Phase 31. |
| R86 | Compiled plan invalidation on shape change causes decode failure | Runtime crash | Low | Defensive check: if input shape differs from compiled shapes, fall back to interpreted Forward(). Log warning. |

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
- Use Conventional Commits: `feat(graph): add ExecutionPlan compiler`.
- Always run linters and formatters before committing.

### Validation Strategy

- Worker pool: benchmark spawn overhead vs channel overhead for 10K calls.
- Graph compiler: output parity between compiled Run() and interpreted Forward().
- Buffer arena: zero allocs measured via testing.AllocsPerRun.
- Instruction fusion: numerical parity within 1e-6 for all fused patterns.
- All benchmarks on DGX Spark GB10 (ssh ndungu@192.168.86.250).

---

## 7. Progress Log

### Change Summary -- 2026-03-06

Created Phase 30 plan for architectural overhead reduction. Five epics:
E54 (persistent worker pool), E55 (graph compiler), E56 (buffer arena),
E57 (instruction fusion), E58 (DGX benchmark validation).

Trimmed completed Phase 29 epics (E49-E53) from plan. Merged Phase 29 stable
knowledge into docs/design.md section 15.11, including post-optimization
profile, bottleneck analysis, and key files.

Created ADR: docs/adr/021-graph-compilation-worker-pool.md.
Updated ADR table in docs/design.md to include ADR-020.

---

## 8. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. Design decisions are in
  docs/adr/ (ADR-001 through ADR-021).
- **Phases 1-29:** All complete. See docs/design.md sections 15.1-15.11.
- **Phase 30:** This plan is the source of truth.
- **Quality:** See docs/QUALITY.md for test coverage report.
- **How to build:**
  - CPU: `go build ./...`
  - CUDA: `go build -tags cuda ./...`
  - On DGX Spark: `make CUDA_ARCH=sm_121` in internal/cuda/kernels/,
    then `go build -tags cuda,cutlass ./...`
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.

### Key Phase 30 Starting Points

1. **E54 (worker pool):** Create `internal/workerpool/pool.go`. Wire into
   `internal/xblas/gemm_simd_arm64.go:76` (sgemmGemvParallel) and
   `internal/xblas/gemm_quant.go:121` (gemmF32Q4NTParallel) and
   `compute/cpu_engine.go:55` (parallelFor).
2. **E55 (graph compiler):** Add `graph/compile.go`. The entry point is
   `Graph.Compile()` called after prefill in `generate/generator.go:175`.
   The compiled plan replaces Forward() in the decode loop at line 181.
3. **E56 (buffer arena):** Add `graph/arena.go`. Used by ExecutionPlan.Run()
   to pre-allocate all intermediate tensors.
4. **E57 (instruction fusion):** Add `graph/fusion.go` and
   `compute/fused_elementwise.go`. Fusion pass runs after Compile() produces
   the initial instruction list.

### External Dependencies

- **DGX Spark (ndungu@192.168.86.250):**
  - Go 1.25.0 linux/arm64, CUDA 13.0, sm_121 (Blackwell).
  - Models: ~/models/gemma3-q4/ (Q4 ZMF), ~/models/gemma3/ (F32 ZMF).
  - Repos: ~/zerfoo/, ~/zonnx/, ~/zmf/.
- No external service dependencies for CPU-only work.

### Performance Baseline (Phase 29)

| Model | Params | Quant | tok/s | Overhead ms/token | Compute ms/token |
|-------|--------|-------|-------|-------------------|------------------|
| Gemma 3 2B | 2.6B | Q4_0 | 6.5 | ~130 | ~24 |
