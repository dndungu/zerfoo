package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var deepseekV3Config = modelParityConfig{
	Name:           "DeepSeek-V3",
	ZMFEnvVar:      "DEEPSEEK_ZMF_PATH",
	ModelDirEnvVar: "DEEPSEEK_MODEL_DIR",
	ModelID:        "deepseek-v3",
	MinVocabSize:   100000, // DeepSeek V3 vocab: 129280
}

func TestDeepSeekV3ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, deepseekV3Config)
}

func TestDeepSeekV3GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, deepseekV3Config)
}

func TestDeepSeekV3Generation(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, deepseekV3Config)
}
