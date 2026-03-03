package parity_test

import (
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/layers/registry"
)

// TestGemma3ForwardPass loads a ZMF-converted Gemma 3 model and verifies
// that a single forward pass succeeds, produces a valid output shape, and
// contains no NaN or Inf values.
//
// The test is skipped when GEMMA3_ZMF_PATH is not set; it is intended for
// CI environments where the model file is present.
func TestGemma3ForwardPass(t *testing.T) {
	zmfPath := os.Getenv("GEMMA3_ZMF_PATH")
	if zmfPath == "" {
		t.Skip("GEMMA3_ZMF_PATH not set; skipping Gemma 3 forward pass test")
	}

	registry.RegisterAll()
	g := loadZMFGraph(t, zmfPath)

	runForwardPassTest(t, g, forwardPassConfig{
		Name:         "Gemma 3",
		SeqLen:       8,
		MinVocabSize: 256000, // Gemma 3 vocab: 262144
	})
}

// TestGemma3GreedyDecode runs 5 greedy decode steps starting from a short
// prompt. Each step picks the argmax over the vocab dimension and appends
// the token to the sequence for the next step.
//
// Assertions: no error, no panic, 5 output tokens in [0, vocabSize).
func TestGemma3GreedyDecode(t *testing.T) {
	zmfPath := os.Getenv("GEMMA3_ZMF_PATH")
	if zmfPath == "" {
		t.Skip("GEMMA3_ZMF_PATH not set; skipping Gemma 3 greedy decode test")
	}

	registry.RegisterAll()
	g := loadZMFGraph(t, zmfPath)

	runGreedyDecodeTest(t, g, []float32{1, 2, 3}, 5)
}
