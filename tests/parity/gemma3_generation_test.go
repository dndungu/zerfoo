package parity_test

import (
	"os"
	"path/filepath"
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

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

	runGenerationTests(t, generationTestConfig{
		ModelID:  "gemma-3",
		ModelDir: modelDir,
	})
}
