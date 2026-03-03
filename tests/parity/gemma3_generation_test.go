package parity_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/registry"
)

// dirRegistry is a mock ModelRegistry that maps model IDs to local directories.
type dirRegistry struct {
	models map[string]*registry.ModelInfo
}

func (r *dirRegistry) Get(modelID string) (*registry.ModelInfo, bool) {
	info, ok := r.models[modelID]
	return info, ok
}

func (r *dirRegistry) Pull(_ context.Context, _ string) (*registry.ModelInfo, error) {
	return nil, nil
}

func (r *dirRegistry) List() []registry.ModelInfo { return nil }
func (r *dirRegistry) Delete(_ string) error      { return nil }

// TestGemma3Generation loads a Gemma 3 model via inference.Load and verifies
// that text generation produces coherent, non-empty output.
//
// Requires GEMMA3_ZMF_PATH pointing to a .zmf file whose parent directory
// also contains config.json and tokenizer.json. Alternatively, set
// GEMMA3_MODEL_DIR to the directory directly.
func TestGemma3Generation(t *testing.T) {
	modelDir := os.Getenv("GEMMA3_MODEL_DIR")
	if modelDir == "" {
		zmfPath := os.Getenv("GEMMA3_ZMF_PATH")
		if zmfPath == "" {
			t.Skip("GEMMA3_MODEL_DIR and GEMMA3_ZMF_PATH not set; skipping")
		}
		modelDir = filepath.Dir(zmfPath)
	}

	layerreg.RegisterAll()

	reg := &dirRegistry{
		models: map[string]*registry.ModelInfo{
			"gemma-3": {ID: "gemma-3", Path: modelDir},
		},
	}

	mdl, err := inference.Load("gemma-3", inference.WithRegistry(reg))
	if err != nil {
		t.Fatalf("inference.Load failed: %v", err)
	}

	ctx := context.Background()

	// Test 1: Greedy generation is non-empty and deterministic.
	t.Run("greedy_deterministic", func(t *testing.T) {
		prompt := "The capital of France is"
		result1, err := mdl.Generate(ctx, prompt,
			inference.WithTemperature(0),
			inference.WithMaxTokens(20),
		)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}
		if result1 == "" {
			t.Fatal("greedy generation produced empty output")
		}
		t.Logf("greedy output: %q", result1)

		// Second greedy pass should be identical.
		result2, err := mdl.Generate(ctx, prompt,
			inference.WithTemperature(0),
			inference.WithMaxTokens(20),
		)
		if err != nil {
			t.Fatalf("Generate (second) failed: %v", err)
		}
		if result1 != result2 {
			t.Errorf("greedy outputs differ:\n  run1: %q\n  run2: %q", result1, result2)
		}
	})

	// Test 2: Streaming output matches non-streaming.
	t.Run("stream_parity", func(t *testing.T) {
		prompt := "Hello world"
		nonStream, err := mdl.Generate(ctx, prompt,
			inference.WithTemperature(0),
			inference.WithMaxTokens(10),
		)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}

		var sb strings.Builder
		err = mdl.GenerateStream(ctx, prompt,
			generate.TokenStreamFunc(func(token string, done bool) error {
				if !done {
					sb.WriteString(token)
				}
				return nil
			}),
			inference.WithTemperature(0),
			inference.WithMaxTokens(10),
		)
		if err != nil {
			t.Fatalf("GenerateStream failed: %v", err)
		}

		streamed := sb.String()
		if nonStream != streamed {
			t.Errorf("stream/non-stream mismatch:\n  non-stream: %q\n  stream:     %q",
				nonStream, streamed)
		}
	})

	// Test 3: Chat generates a response.
	t.Run("chat", func(t *testing.T) {
		resp, err := mdl.Chat(ctx, []inference.Message{
			{Role: "user", Content: "Say hello in French"},
		}, inference.WithMaxTokens(20))
		if err != nil {
			t.Fatalf("Chat failed: %v", err)
		}
		if resp.Content == "" {
			t.Error("Chat produced empty content")
		}
		if resp.TokensUsed <= 0 {
			t.Error("TokensUsed should be positive")
		}
		t.Logf("chat response: %q (tokens: %d)", resp.Content, resp.TokensUsed)
	})
}
