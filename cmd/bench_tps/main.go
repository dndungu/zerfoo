// bench_tps measures tokens-per-second for a local ZMF model.
//
// Usage: bench_tps -model /path/to/model/dir [-prompt "text"] [-tokens 64]
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"sync/atomic"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/registry"
)

// dirRegistry is a minimal ModelRegistry that returns a fixed directory.
type dirRegistry struct {
	path string
}

func (r *dirRegistry) Get(modelID string) (*registry.ModelInfo, bool) {
	if _, err := os.Stat(r.path); err != nil {
		return nil, false
	}
	return &registry.ModelInfo{
		ID:   modelID,
		Path: r.path,
	}, true
}

func (r *dirRegistry) Pull(_ context.Context, _ string) (*registry.ModelInfo, error) {
	return nil, fmt.Errorf("pull not supported")
}

func (r *dirRegistry) List() []registry.ModelInfo { return nil }

func (r *dirRegistry) Delete(_ string) error { return nil }

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func run() error {
	layerreg.RegisterAll()

	modelDir := flag.String("model", "", "path to model directory (config.json, tokenizer.json, model.zmf)")
	prompt := flag.String("prompt", "The meaning of life is", "prompt text")
	maxTokens := flag.Int("tokens", 64, "max tokens to generate")
	useMmap := flag.Bool("mmap", false, "use memory-mapped loading")
	device := flag.String("device", "cpu", "compute device (cpu, cuda, cuda:0)")
	greedy := flag.Bool("greedy", false, "use greedy decoding (temperature=0)")
	cpuprofile := flag.String("cpuprofile", "", "write CPU profile to file")
	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			return fmt.Errorf("cpuprofile: %w", err)
		}
		defer func() { _ = f.Close() }()
		if err := pprof.StartCPUProfile(f); err != nil {
			return fmt.Errorf("cpuprofile start: %w", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *modelDir == "" {
		return fmt.Errorf("usage: bench_tps -model /path/to/model/dir")
	}

	reg := &dirRegistry{path: *modelDir}

	fmt.Printf("Loading model from %s (device=%s)...\n", *modelDir, *device)
	t0 := time.Now()
	mdl, err := inference.Load("bench", inference.WithRegistry(reg), inference.WithMmap(*useMmap), inference.WithDevice(*device))
	if err != nil {
		return fmt.Errorf("load error: %w", err)
	}
	fmt.Printf("Loaded in %.1fs\n", time.Since(t0).Seconds())

	fmt.Printf("Prompt: %q\n", *prompt)
	fmt.Printf("Max tokens: %d\n", *maxTokens)

	// Warm-up run (short).
	fmt.Println("Warm-up...")
	var genOpts []inference.GenerateOption
	genOpts = append(genOpts, inference.WithMaxTokens(4))
	if *greedy {
		genOpts = append(genOpts, inference.WithTemperature(0))
	}
	_, _ = mdl.Generate(context.Background(), *prompt, genOpts...)

	// Timed run with streaming to count tokens.
	fmt.Println("Generating...")
	var tokenCount atomic.Int64
	var output string
	handler := generate.TokenStreamFunc(func(token string, done bool) error {
		if !done {
			tokenCount.Add(1)
			output += token
		}
		return nil
	})

	t1 := time.Now()
	streamOpts := []inference.GenerateOption{inference.WithMaxTokens(*maxTokens)}
	if *greedy {
		streamOpts = append(streamOpts, inference.WithTemperature(0))
	}
	err = mdl.GenerateStream(context.Background(), *prompt, handler, streamOpts...)
	elapsed := time.Since(t1)
	if err != nil {
		return fmt.Errorf("generate error: %w", err)
	}

	genTokens := tokenCount.Load()
	tps := float64(genTokens) / elapsed.Seconds()

	fmt.Printf("\n--- Results ---\n")
	fmt.Printf("Output: %s\n", output)
	fmt.Printf("Generated tokens: %d\n", genTokens)
	fmt.Printf("Time: %.3fs\n", elapsed.Seconds())
	fmt.Printf("Throughput: %.2f tok/s\n", tps)
	return nil
}
