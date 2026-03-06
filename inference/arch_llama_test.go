package inference

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// makeLlamaTestTensors creates a minimal set of Llama-architecture tensors
// with canonical names for testing. All weights are initialized with small
// random-like values derived from position to produce non-NaN output.
func makeLlamaTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads

	fill := func(shape []int, scale float32) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			// Small deterministic values to avoid NaN.
			data[i] = scale * float32(math.Sin(float64(i)*0.01))
		}
		t, _ := tensor.New(shape, data)
		return t
	}
	ones := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 1.0
		}
		t, _ := tensor.New(shape, data)
		return t
	}

	// Global tensors.
	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)

	// Per-layer tensors.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
	}

	return tensors
}

func itoa(i int) string {
	return fmt.Sprintf("%d", i)
}

func TestBuildLlamaGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildLlamaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLlamaGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildLlamaGraph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildLlamaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLlamaGraph: %v", err)
	}

	// Create input: batch=1, seq_len=4 of token IDs as float32 embeddings.
	// Simulate what the generator does: embed token IDs, then pass through graph.
	tokenIDs := []int{1, 5, 10, 3}
	embTable := emb.Data()
	hiddenDim := cfg.HiddenSize
	seqLen := len(tokenIDs)

	inputData := make([]float32, seqLen*hiddenDim)
	for i, id := range tokenIDs {
		for j := 0; j < hiddenDim; j++ {
			inputData[i*hiddenDim+j] = embTable[id*hiddenDim+j]
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

	// Check output shape: [1, seqLen, vocabSize].
	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != cfg.VocabSize {
		t.Fatalf("unexpected output shape: %v, want [1, %d, %d]", shape, seqLen, cfg.VocabSize)
	}

	// Check no NaN values.
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

func TestBuildLlamaGraph_MissingTensor(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}

	// Empty tensors -- should fail with a clear error.
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildLlamaGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}
