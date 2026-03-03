package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var gemma3Config = modelParityConfig{
	Name:           "Gemma 3",
	ZMFEnvVar:      "GEMMA3_ZMF_PATH",
	ModelDirEnvVar: "GEMMA3_MODEL_DIR",
	ModelID:        "gemma-3",
	MinVocabSize:   256000, // Gemma 3 vocab: 262144
}

func TestGemma3ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, gemma3Config)
}

func TestGemma3GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, gemma3Config)
}

func TestGemma3Generation(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, gemma3Config)
}
