package parity_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/inference"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/registry"
)

// Q4 model tests use separate env vars:
//   GEMMA3_Q4_ZMF_PATH  = path to Q4 quantized model.zmf
//   GEMMA3_Q4_MODEL_DIR = directory containing config.json, tokenizer.json, model.zmf (Q4)

var gemma3Q4Config = modelParityConfig{
	Name:           "Gemma 3 Q4",
	ZMFEnvVar:      "GEMMA3_Q4_ZMF_PATH",
	ModelDirEnvVar: "GEMMA3_Q4_MODEL_DIR",
	ModelID:        "gemma-3-q4",
	MinVocabSize:   256000,
}

func TestGemma3Q4ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, gemma3Q4Config)
}

func TestGemma3Q4GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, gemma3Q4Config)
}

func TestGemma3Q4Generation(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, gemma3Q4Config)
}

func BenchmarkGemma3Q4TokPerSec(b *testing.B) {
	layerreg.RegisterAll()

	modelDir := os.Getenv("GEMMA3_Q4_MODEL_DIR")
	if modelDir == "" {
		b.Skip("GEMMA3_Q4_MODEL_DIR not set; skipping")
	}

	reg := &dirRegistry{
		models: map[string]*registry.ModelInfo{
			gemma3Q4Config.ModelID: {ID: gemma3Q4Config.ModelID, Path: modelDir},
		},
	}

	mdl, err := inference.Load(gemma3Q4Config.ModelID, inference.WithRegistry(reg))
	if err != nil {
		b.Fatalf("inference.Load failed: %v", err)
	}

	ctx := context.Background()
	prompt := "The meaning of life is"
	maxTokens := 32

	b.ResetTimer()
	for b.Loop() {
		start := time.Now()
		_, err := mdl.Generate(ctx, prompt,
			inference.WithTemperature(0),
			inference.WithMaxTokens(maxTokens),
		)
		elapsed := time.Since(start)
		if err != nil {
			b.Fatalf("Generate failed: %v", err)
		}
		b.ReportMetric(float64(maxTokens)/elapsed.Seconds(), "tok/s")
	}
}
