package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var mistralConfig = modelParityConfig{
	Name:           "Mistral",
	ZMFEnvVar:      "MISTRAL_ZMF_PATH",
	ModelDirEnvVar: "MISTRAL_MODEL_DIR",
	ModelID:        "mistral",
	MinVocabSize:   30000, // Mistral 7B vocab: 32000
}

func TestMistralForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, mistralConfig)
}

func TestMistralGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, mistralConfig)
}

func TestMistralGeneration(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, mistralConfig)
}
