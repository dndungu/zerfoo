package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var llama3Config = modelParityConfig{
	Name:           "Llama 3",
	ZMFEnvVar:      "LLAMA3_ZMF_PATH",
	ModelDirEnvVar: "LLAMA3_MODEL_DIR",
	ModelID:        "llama-3",
	MinVocabSize:   100000, // Llama 3 8B vocab: 128256
}

func TestLlama3ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, llama3Config)
}

func TestLlama3GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, llama3Config)
}

func TestLlama3Generation(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, llama3Config)
}
