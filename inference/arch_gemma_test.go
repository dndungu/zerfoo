package inference

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func makeGemmaTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := makeLlamaTestTensors(cfg)
	// Gemma ties lm_head to embedding weights -- remove separate lm_head.weight.
	delete(tensors, "lm_head.weight")
	return tensors
}

func TestBuildGemmaGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGemmaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGemmaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemmaGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildGemmaGraph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGemmaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGemmaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemmaGraph: %v", err)
	}

	// Embed tokens and scale by sqrt(hidden_size) as Gemma does.
	tokenIDs := []int{1, 5, 10, 3}
	embTable := emb.Data()
	hiddenDim := cfg.HiddenSize
	seqLen := len(tokenIDs)
	scale := float32(math.Sqrt(float64(hiddenDim)))

	inputData := make([]float32, seqLen*hiddenDim)
	for i, id := range tokenIDs {
		for j := 0; j < hiddenDim; j++ {
			inputData[i*hiddenDim+j] = embTable[id*hiddenDim+j] * scale
		}
	}
	input, err := tensor.New([]int{1, seqLen, hiddenDim}, inputData)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != cfg.VocabSize {
		t.Fatalf("unexpected output shape: %v, want [1, %d, %d]", shape, seqLen, cfg.VocabSize)
	}

	data := output.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("Inf at index %d", i)
		}
	}
}

func TestBuildGemmaGraph_TiedEmbedding(t *testing.T) {
	// Gemma should work without lm_head.weight (tied to embedding).
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGemmaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemmaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemmaGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
}
