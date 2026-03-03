package parity_test

import (
	"os"
	"path/filepath"
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

// TestLlama3ForwardPass loads a ZMF-converted Llama 3 model and verifies
// that a single forward pass succeeds, produces a valid output shape, and
// contains no NaN or Inf values.
//
// Requires LLAMA3_ZMF_PATH pointing to a .zmf file. Skips otherwise.
func TestLlama3ForwardPass(t *testing.T) {
	zmfPath := os.Getenv("LLAMA3_ZMF_PATH")
	if zmfPath == "" {
		t.Skip("LLAMA3_ZMF_PATH not set; skipping Llama 3 forward pass test")
	}

	layerreg.RegisterAll()
	g := loadZMFGraph(t, zmfPath)

	runForwardPassTest(t, g, forwardPassConfig{
		Name:         "Llama 3",
		SeqLen:       8,
		MinVocabSize: 100000, // Llama 3 8B vocab: 128256
	})
}

// TestLlama3GreedyDecode runs 5 greedy decode steps starting from a short
// prompt and verifies tokens are in valid range.
//
// Requires LLAMA3_ZMF_PATH. Skips otherwise.
func TestLlama3GreedyDecode(t *testing.T) {
	zmfPath := os.Getenv("LLAMA3_ZMF_PATH")
	if zmfPath == "" {
		t.Skip("LLAMA3_ZMF_PATH not set; skipping Llama 3 greedy decode test")
	}

	layerreg.RegisterAll()
	g := loadZMFGraph(t, zmfPath)

	runGreedyDecodeTest(t, g, []float32{1, 2, 3}, 5)
}

// TestLlama3Generation loads a Llama 3 model via inference.Load and verifies
// that text generation produces coherent, non-empty output.
//
// Requires LLAMA3_MODEL_DIR or LLAMA3_ZMF_PATH (the parent directory must
// contain config.json and tokenizer.json). Skips otherwise.
func TestLlama3Generation(t *testing.T) {
	modelDir := os.Getenv("LLAMA3_MODEL_DIR")
	if modelDir == "" {
		zmfPath := os.Getenv("LLAMA3_ZMF_PATH")
		if zmfPath == "" {
			t.Skip("LLAMA3_MODEL_DIR and LLAMA3_ZMF_PATH not set; skipping")
		}
		modelDir = filepath.Dir(zmfPath)
	}

	layerreg.RegisterAll()

	runGenerationTests(t, generationTestConfig{
		ModelID:  "llama-3",
		ModelDir: modelDir,
	})
}
