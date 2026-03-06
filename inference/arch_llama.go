package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// buildLlamaGraph constructs a computation graph for the Llama architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// The Llama architecture is:
//
//	Embed -> [RMSNorm -> GQA -> Add -> RMSNorm -> FFN(SiLU-gate) -> Add] x N -> RMSNorm -> LMHead
func buildLlamaGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	embedWeight, ok := tensors["model.embed_tokens.weight"]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %q", "model.embed_tokens.weight")
	}

	// Llama can tie lm_head to embedding weights.
	lmHeadWeight, ok := tensors["lm_head.weight"]
	if !ok {
		lmHeadWeight = embedWeight
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, lmHeadWeight)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}

// lmHeadNode projects hidden states to vocabulary logits.
// weight shape: [vocabSize, hiddenDim].
// input shape: [batch, seqLen, hiddenDim].
// output shape: [batch, seqLen, vocabSize].
type lmHeadNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	weight *tensor.TensorNumeric[T]
}

func (h *lmHeadNode[T]) OpType() string                  { return "LMHead" }
func (h *lmHeadNode[T]) Attributes() map[string]any       { return nil }
func (h *lmHeadNode[T]) OutputShape() []int               { return nil }
func (h *lmHeadNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (h *lmHeadNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	batch, seqLen, hiddenDim := shape[0], shape[1], shape[2]

	flat, err := h.engine.Reshape(ctx, input, []int{batch * seqLen, hiddenDim})
	if err != nil {
		return nil, err
	}

	// weight is [vocabSize, hiddenDim], transpose to [hiddenDim, vocabSize].
	wT, err := h.engine.Transpose(ctx, h.weight, []int{1, 0})
	if err != nil {
		return nil, err
	}

	out, err := h.engine.MatMul(ctx, flat, wT)
	if err != nil {
		return nil, err
	}

	vocabSize := h.weight.Shape()[0]
	return h.engine.Reshape(ctx, out, []int{batch, seqLen, vocabSize})
}

func (h *lmHeadNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
