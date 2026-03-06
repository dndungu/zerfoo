# Zerfoo

[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/zerfoo.svg)](https://pkg.go.dev/github.com/zerfoo/zerfoo)
[![Go Report Card](https://goreportcard.com/badge/github.com/zerfoo/zerfoo)](https://goreportcard.com/report/github.com/zerfoo/zerfoo)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Run LLMs locally in pure Go. No Python, no containers, no Cgo by default.

Zerfoo loads GGUF and ZMF models, quantizes them to 4-bit, and serves them through an OpenAI-compatible API — all from a single binary.

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest

# Chat with a model
zerfoo run gemma-3-2b-q4

# Serve with OpenAI-compatible API
zerfoo serve gemma-3-2b-q4 --port 8080
curl localhost:8080/v1/chat/completions \
  -d '{"model":"gemma-3-2b-q4","messages":[{"role":"user","content":"Hello!"}]}'
```

## Supported Models

| Model | Formats | Quantization |
|-------|---------|-------------|
| Gemma 3 | ZMF, GGUF | F32, Q4_0 |
| LLaMA 3 | ZMF, GGUF | F32, Q4_0 |
| Mistral | ZMF, GGUF | F32, Q4_0 |
| Qwen 2.5 | ZMF, GGUF | F32, Q4_0 |
| DeepSeek | ZMF, GGUF | F32, Q4_0 |
| Phi-4 | ZMF, GGUF | F32, Q4_0 |

## Performance

Measured on NVIDIA DGX Spark (ARM64 CPU + GB10 GPU):

| Metric | Value |
|--------|-------|
| Gemma 3 2B Q4 CPU inference | 3.60 tok/s |
| CUDA Q4 GEMM kernel | 2,383 GFLOPS |
| Q4 model size (2B params) | 1.5 GB (vs 4 GB F32) |
| PagedAttention memory | 46% of pre-allocated KV cache |

## Why Go for ML?

- **Single binary** — `go build` produces one static binary. No virtualenvs, no `pip install`, no CUDA toolkit on the host.
- **Go generics** — Type-safe tensors (`TensorNumeric[float32]`, `TensorNumeric[float64]`) catch shape and dtype errors at compile time.
- **Goroutine concurrency** — Parallel graph execution, batch scheduling, and streaming generation use goroutines and channels, not thread pools or async/await.
- **Low deploy friction** — Cross-compile for Linux/macOS/Windows ARM64/AMD64. Ship a single binary to edge devices, containers, or bare metal.

## Features

### Inference

- **GGUF and ZMF model loading** with mmap for zero-copy weight access
- **Q4_0 and Q8_0 quantization** — 4-bit and 8-bit integer quantization with fused dequant+multiply kernels
- **Speculative decoding** with adaptive draft length for faster autoregressive generation
- **PagedAttention** — block-level KV cache allocation, 46% memory reduction vs pre-allocated
- **Fused kernels** — single-pass RMSNorm, RoPE, and SiLU-gate to eliminate intermediate allocations
- **OpenAI-compatible HTTP API** — `/v1/chat/completions`, `/v1/completions`, `/v1/models` with streaming support
- **NEON/AVX2 SIMD GEMM** — platform-optimized matrix multiply

### Training

- **Static computation graph** with automatic differentiation (reverse-mode backprop)
- **Optimizers**: SGD, Adam, AdamW with weight decay
- **Layers**: Dense, Conv2D, GQA, RoPE, RMSNorm, LayerNorm, BatchNorm, Dropout, SwiGLU, GELU, and [40+ more](https://pkg.go.dev/github.com/zerfoo/zerfoo/layers)
- **Distributed training** via gRPC with All-Reduce and Parameter Server strategies

### GPU (CUDA)

- cuBLAS SGEMM for float32 matrix multiply
- Custom Q4 dequant-GEMM kernel (2,383 GFLOPS on GB10)
- Flash Attention kernels with shared-memory reduction
- Stream-based async execution with device memory pooling

## Quick Start

### Run Inference

```go
package main

import (
    "context"
    "fmt"

    "github.com/zerfoo/zerfoo/inference"
)

func main() {
    model, err := inference.Load("gemma-3-2b-q4",
        inference.WithDevice("cpu"),
        inference.WithMmap(true),
    )
    if err != nil {
        panic(err)
    }

    response, err := model.Generate(context.Background(), "Explain Go generics in one paragraph",
        inference.WithMaxTokens(256),
        inference.WithTemperature(0.7),
    )
    if err != nil {
        panic(err)
    }
    fmt.Println(response)
}
```

### Build and Train a Model

```go
package main

import (
    "context"
    "fmt"

    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/layers/activations"
    "github.com/zerfoo/zerfoo/layers/core"
    "github.com/zerfoo/zerfoo/numeric"
    "github.com/zerfoo/zerfoo/tensor"
    "github.com/zerfoo/zerfoo/training/optimizer"
    "github.com/zerfoo/zerfoo/types"
)

func main() {
    ctx := context.Background()
    ops := numeric.Float32Ops{}
    engine := compute.NewCPUEngine(ops)

    // Define model
    builder := graph.NewBuilder[float32](engine)
    input := builder.Input([]int{1, 10})

    dense1, _ := core.NewDense("dense1", engine, ops, 10, 32)
    node1 := builder.AddNode(dense1, input)

    act1 := activations.NewReLU[float32](engine, ops)
    node2 := builder.AddNode(act1, node1)

    dense2, _ := core.NewDense("dense2", engine, ops, 32, 1)
    output := builder.AddNode(dense2, node2)

    g, _ := builder.Build(output)

    // Train
    opt := optimizer.NewSGD[float32](engine, ops, 0.01)
    inputTensor, _ := tensor.New[float32]([]int{1, 10}, nil)
    targetTensor, _ := tensor.New[float32]([]int{1, 1}, nil)

    for i := 0; i < 100; i++ {
        pred, _ := g.Forward(ctx, inputTensor)
        loss := pred.Data()[0] - targetTensor.Data()[0]
        grad, _ := tensor.New[float32]([]int{1, 1}, []float32{2 * loss})
        g.Backward(ctx, types.FullBackprop, grad)
        opt.Step(ctx, g.Parameters())
    }
    fmt.Println("Training complete!")
}
```

### Serve an API

```go
package main

import (
    "log"

    "github.com/zerfoo/zerfoo/inference"
    "github.com/zerfoo/zerfoo/serve"
)

func main() {
    model, _ := inference.Load("gemma-3-2b-q4")
    server := serve.New(model)
    log.Fatal(server.ListenAndServe(":8080"))
}
```

### Custom Layers

Implement the `graph.Node[T]` interface to add custom operations:

```go
type SquareNode[T tensor.Numeric] struct {
    engine      compute.Engine[T]
    outputShape []int
}

func (n *SquareNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
    n.outputShape = inputs[0].Shape()
    return n.engine.Mul(ctx, inputs[0], inputs[0])
}

func (n *SquareNode[T]) Backward(ctx context.Context, mode types.BackwardMode, grad *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
    two := n.engine.Ops().FromFloat64(2.0)
    twoX, _ := n.engine.MulScalar(ctx, inputs[0], two)
    dInput, _ := n.engine.Mul(ctx, grad, twoX)
    return []*tensor.TensorNumeric[T]{dInput}, nil
}
```

## CLI Reference

```
zerfoo run <model>           Interactive chat
zerfoo serve <model>         Start OpenAI-compatible HTTP server
zerfoo predict               Batch inference from CSV/JSON
zerfoo tokenize              Tokenize text
zerfoo pull <model>          Download model
zerfoo worker                Start distributed training worker
```

## Benchmarks

```bash
# Run all benchmarks
go test ./compute -bench=. -benchmem

# Compare performance between commits
go run cmd/bench-compare/main.go \
  --baseline=benchmarks/baseline.txt \
  --current=benchmarks/current.txt \
  --threshold=10
```

## Project Structure

```
compute/         Engine interface (34 ops) and CPU/GPU implementations
graph/           Computation DAG, topological execution, graph optimizations
layers/          40+ layer types (attention, normalization, activations, core)
tensor/          Generic N-dimensional arrays with Q4/Q8 quantized storage
generate/        Token generation, sampling, speculative decoding, PagedKV cache
inference/       High-level model loading and generation API
serve/           OpenAI-compatible HTTP server
model/           ZMF and GGUF model format loaders
training/        Optimizers (SGD, Adam, AdamW), loss functions, training loops
internal/cuda/   CUDA kernels (Q4 GEMM, Flash Attention, elementwise ops)
internal/xblas/  NEON/AVX2 SIMD matrix multiply
distributed/     gRPC-based distributed training coordination
```

## Building with CUDA

```bash
# Standard CPU build (no Cgo required)
go build ./cmd/zerfoo

# CUDA build (requires CUDA toolkit)
CGO_CFLAGS='-I/usr/local/cuda/include' \
CGO_LDFLAGS='-L/usr/local/cuda/lib64' \
go build -tags cuda ./cmd/zerfoo
```

## Contributing

See **[docs/design.md](docs/design.md)** for architecture and design decisions.

```bash
# Run tests
go test ./... -race -timeout 120s

# Lint
golangci-lint run ./...
```

## License

Apache 2.0
