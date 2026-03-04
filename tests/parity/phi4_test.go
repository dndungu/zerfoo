package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var phi4Config = modelParityConfig{
	Name:           "Phi-3",
	ZMFEnvVar:      "PHI4_ZMF_PATH",
	ModelDirEnvVar: "PHI4_MODEL_DIR",
	ModelID:        "phi-3",
	MinVocabSize:   30000, // Phi-3-mini vocab: 32064
}

func TestPhi4ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, phi4Config)
}

func TestPhi4GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, phi4Config)
}

func TestPhi4Generation(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, phi4Config)
}
