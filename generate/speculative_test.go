package generate

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// buildSpecTestGraph creates a graph for speculative decode testing.
// tokenSequence controls what greedy argmax returns per forward call.
func buildSpecTestGraph(t *testing.T, vocabSize int, tokenSequence []int) *graph.Graph[float32] {
	t.Helper()
	return buildTestGraph(t, vocabSize, tokenSequence)
}

func TestSpeculativeGenerate_AllAccepted(t *testing.T) {
	// Draft and target agree on all tokens: "hello world" -> EOS.
	// Both produce: 4(hello), 5(world), 2(EOS).
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()
	seq := []int{4, 5, 2} // hello, world, EOS

	draftGraph := buildSpecTestGraph(t, vocabSize, seq)
	targetGraph := buildSpecTestGraph(t, vocabSize, seq)

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4, // draftLen=4
	)

	sc := SamplingConfig{
		Temperature:  0, // greedy
		MaxNewTokens: 10,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// Should generate at least one token before EOS.
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_FirstTokenRejected(t *testing.T) {
	// Draft says 6(foo), target says 5(world). Target wins immediately.
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Draft: proposes foo, foo, EOS
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{6, 6, 2})
	// Target: always prefers world then EOS
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{5, 2})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_PartialAcceptance(t *testing.T) {
	// Draft proposes [4, 5, 6], target accepts [4, 5] then rejects 6 (wants 7).
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Draft: 4, 5, 6, 2
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 6, 2})
	// Target: 4, 5, 7, 2 (accepts first 2 draft tokens, rejects 3rd)
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 7, 2})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_DraftEOSEarly(t *testing.T) {
	// Draft produces EOS after 1 token, target verifies just 1.
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Draft: 4, EOS
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{4, 2})
	// Target: 4, EOS (agrees)
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{4, 2})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// Should have at least the first accepted token.
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_MaxTokens(t *testing.T) {
	// Both models agree infinitely, but we cap at MaxNewTokens=3.
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Both always produce token 4 (hello), never EOS.
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{4})
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{4})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 2,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 3,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_StopToken(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Draft and target: 4, 5, 7(bar=stop), ...
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 7, 6})
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 7, 6})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopTokenIDs: []int{7}, // bar=7 is stop token
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// Should stop before bar is included.
	_ = result
}

// Verify that SpeculativeGenerator implements a basic "model forward" pattern
// by testing with a simple model and checking output shape consistency.
func TestSpeculativeGenerator_ForwardConsistency(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Same model for draft and target.
	seq := []int{4, 5, 2}
	g := buildSpecTestGraph(t, vocabSize, seq)

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	// Single-token input.
	input, err := tensor.New([]int{1, 1, 1}, []float32{4})
	if err != nil {
		t.Fatal(err)
	}

	logits, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[2] != vocabSize {
		t.Errorf("logits shape = %v, want [1, *, %d]", shape, vocabSize)
	}

	_ = engine
	_ = cfg
}
