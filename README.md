# Zerfoo

[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/zerfoo.svg)](https://pkg.go.dev/github.com/zerfoo/zerfoo)
[![Go Report Card](https://goreportcard.com/badge/github.com/zerfoo/zerfoo)](https://goreportcard.com/report/github.com/zerfoo/zerfoo)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Embed LLMs directly in Go applications. No Python. No sidecar. No HTTP calls to localhost.

```go
model, _ := inference.Load("gemma-3-1b-q4")
reply, _ := model.Generate(ctx, "What is the capital of France?")
fmt.Println(reply) // Paris is the capital of France.
```

That's it. One binary, one import, one function call.

## Install

```bash
go get github.com/zerfoo/zerfoo@latest
```

Gemma 3 1B Q4 is ~700 MB and runs on any laptop with 2 GB free RAM. No GPU required.

## Examples

### Chat bot in 10 lines

```go
model, _ := inference.Load("gemma-3-1b-q4")
defer model.Close()

reply, _ := model.Chat(ctx, []inference.Message{
    {Role: "user", Content: "Write a haiku about Go"},
})
fmt.Println(reply.Content)
```

### Streaming tokens to a terminal

```go
model, _ := inference.Load("gemma-3-1b-q4")
defer model.Close()

model.GenerateStream(ctx, "Explain quantum computing", func(token string, done bool) error {
    fmt.Print(token)
    return nil
})
```

### Summarize text inside a CLI tool

```go
func summarize(text string) string {
    model, _ := inference.Load("gemma-3-1b-q4")
    defer model.Close()

    summary, _ := model.Generate(ctx,
        "Summarize this in one sentence:\n\n"+text,
        inference.WithMaxTokens(64),
        inference.WithTemperature(0.3),
    )
    return summary
}
```

### Add an AI endpoint to an existing HTTP server

```go
mux := http.NewServeMux()

// Your existing routes
mux.HandleFunc("GET /health", healthHandler)

// Add LLM-powered endpoint
model, _ := inference.Load("gemma-3-1b-q4")
mux.HandleFunc("POST /ask", func(w http.ResponseWriter, r *http.Request) {
    var req struct{ Question string }
    json.NewDecoder(r.Body).Decode(&req)

    answer, _ := model.Generate(r.Context(), req.Question,
        inference.WithMaxTokens(256),
    )
    json.NewEncoder(w).Encode(map[string]string{"answer": answer})
})

http.ListenAndServe(":8080", mux)
```

### Drop-in OpenAI-compatible server

```go
model, _ := inference.Load("gemma-3-1b-q4")
server := serve.NewServer(model)
http.ListenAndServe(":8080", server.Handler())
```

Works with any OpenAI client library — just point it at `localhost:8080`.

### Classify text with structured output

```go
model, _ := inference.Load("gemma-3-1b-q4")
defer model.Close()

prompt := `Classify this support ticket as "billing", "technical", or "general".
Reply with only the category name.

Ticket: I can't log in to my account after changing my password.

Category:`

category, _ := model.Generate(ctx, prompt,
    inference.WithMaxTokens(4),
    inference.WithTemperature(0),
)
fmt.Println(strings.TrimSpace(category)) // technical
```

### Generate code

```go
model, _ := inference.Load("gemma-3-2b-q4") // Larger model for code tasks
defer model.Close()

code, _ := model.Generate(ctx,
    "Write a Go function that reverses a string. Return only the code.",
    inference.WithMaxTokens(128),
    inference.WithTemperature(0.2),
)
fmt.Println(code)
```

### Process a batch of inputs

```go
model, _ := inference.Load("gemma-3-1b-q4")
defer model.Close()

questions := []string{
    "What is 2+2?",
    "Name the largest ocean.",
    "Who wrote Hamlet?",
}

for _, q := range questions {
    answer, _ := model.Generate(ctx, q, inference.WithMaxTokens(32))
    fmt.Printf("Q: %s\nA: %s\n\n", q, answer)
}
```

## Recommended Models

| Model | Size | RAM | Best for |
|-------|------|-----|----------|
| `gemma-3-1b-q4` | ~700 MB | 2 GB | Laptops, CI, edge devices, quick tasks |
| `gemma-3-2b-q4` | ~1.5 GB | 4 GB | Code generation, longer reasoning |
| `llama-3-8b-q4` | ~4.5 GB | 8 GB | Complex tasks, higher quality output |

All models run on CPU out of the box. Add `inference.WithDevice("cuda")` for GPU acceleration.

## Supported Architectures

Gemma 3, LLaMA 3, Mistral, Qwen 2.5, DeepSeek, Phi-4 — in GGUF or ZMF format, with F32 or Q4_0 quantization.

## CLI

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest

zerfoo run gemma-3-1b-q4             # Interactive chat
zerfoo serve gemma-3-1b-q4           # OpenAI-compatible API on :8080
zerfoo predict -model gemma-3-1b-q4  # Batch inference from CSV/JSON
```

## Performance

| Metric | Value |
|--------|-------|
| Gemma 3 2B Q4 CPU (ARM64) | 3.60 tok/s |
| CUDA Q4 GEMM (GB10) | 2,383 GFLOPS |
| Q4 model compression | 3.7x smaller than F32 |
| PagedAttention savings | 46% less memory |

## How It Works

Zerfoo is a full ML framework written in Go — tensors, computation graphs, automatic differentiation, SIMD kernels, and CUDA support. The `inference` package wraps all of that into the simple API shown above.

Under the hood, `inference.Load` does:
1. Downloads the model (or loads from cache)
2. Memory-maps the weights (zero-copy, no heap allocation)
3. Builds a static computation graph with optimized fused kernels
4. Returns a `*Model` ready for generation

```
inference.Load("gemma-3-1b-q4")
    │
    ├── model/gguf    → Parse GGUF file, load Q4 weights
    ├── graph/        → Build computation DAG, fold transposes
    ├── compute/      → CPU engine with NEON/AVX2 SIMD, fused RMSNorm/RoPE
    └── generate/     → Autoregressive decode with PagedKV cache
```

## Building with CUDA

```bash
# CPU only (default, no Cgo)
go build ./cmd/zerfoo

# With GPU support
CGO_CFLAGS='-I/usr/local/cuda/include' \
CGO_LDFLAGS='-L/usr/local/cuda/lib64' \
go build -tags cuda ./cmd/zerfoo
```

## Project Structure

```
inference/       Load models and generate text (start here)
serve/           OpenAI-compatible HTTP server
compute/         Engine interface (34 ops), CPU and CUDA backends
graph/           Computation DAG with automatic differentiation
layers/          40+ layer types (attention, normalization, activations)
tensor/          N-dimensional arrays with Q4/Q8 quantized storage
generate/        Token sampling, speculative decoding, PagedKV cache
model/           ZMF and GGUF model format loaders
training/        SGD, Adam, AdamW optimizers and training loops
internal/cuda/   CUDA kernels (Q4 GEMM, Flash Attention)
internal/xblas/  NEON/AVX2 SIMD matrix multiply
distributed/     gRPC-based distributed training
```

## Contributing

See [docs/design.md](docs/design.md) for architecture decisions.

```bash
go test ./... -race -timeout 120s
golangci-lint run ./...
```

## License

Apache 2.0
