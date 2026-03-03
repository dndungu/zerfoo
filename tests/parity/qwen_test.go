package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var qwenConfig = modelParityConfig{
	Name:           "Qwen 2.5",
	ZMFEnvVar:      "QWEN25_ZMF_PATH",
	ModelDirEnvVar: "QWEN25_MODEL_DIR",
	ModelID:        "qwen-2.5",
	MinVocabSize:   150000, // Qwen 2.5 vocab: 151936
}

func TestQwen25ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, qwenConfig)
}

func TestQwen25GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, qwenConfig)
}

func TestQwen25Generation(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, qwenConfig)
}
