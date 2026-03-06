package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/zerfoo/numeric"
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
	ops := numeric.Float32Ops{}

	// Look up a required tensor by canonical name.
	lookup := func(name string) (*tensor.TensorNumeric[float32], error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", name)
		}
		return t, nil
	}

	// Helper to create a graph.Parameter from a loaded tensor.
	param := func(name string, t *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
		return &graph.Parameter[float32]{Name: name, Value: t}
	}

	// Transpose a 2D weight from GGUF layout [out, in] to Linear layout [in, out].
	transposeWeight := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		tr, err := engine.Transpose(context.Background(), t, []int{1, 0})
		if err != nil {
			return nil, fmt.Errorf("transpose %s: %w", name, err)
		}
		return tr, nil
	}

	// --- Global tensors ---
	embedWeight, err := lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}
	finalNormWeight, err := lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}

	// Build graph.
	builder := graph.NewBuilder[float32](engine)
	// Input shape: [batch, seqLen, hiddenSize] (already-embedded tokens).
	input := builder.Input([]int{1, 1, cfg.HiddenSize})

	hidden := input
	headDim := cfg.HiddenSize / cfg.NumHeads

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// --- Input LayerNorm ---
		inputNormW, err := lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		inputNorm, err := normalization.NewRMSNormFromParam[float32](
			engine, ops, 1e-5, param(prefix+"input_layernorm.weight", inputNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// --- Self Attention (GQA) ---
		qW, err := lookup(prefix + "self_attn.q_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		kW, err := lookup(prefix + "self_attn.k_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		vW, err := lookup(prefix + "self_attn.v_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		oW, err := lookup(prefix + "self_attn.o_proj.weight")
		if err != nil {
			return nil, nil, err
		}

		// GGUF stores weights as [out, in]; Linear expects [in, out].
		qWT, err := transposeWeight(prefix+"self_attn.q_proj.weight", qW)
		if err != nil {
			return nil, nil, err
		}
		kWT, err := transposeWeight(prefix+"self_attn.k_proj.weight", kW)
		if err != nil {
			return nil, nil, err
		}
		vWT, err := transposeWeight(prefix+"self_attn.v_proj.weight", vW)
		if err != nil {
			return nil, nil, err
		}
		oWT, err := transposeWeight(prefix+"self_attn.o_proj.weight", oW)
		if err != nil {
			return nil, nil, err
		}

		wq := core.NewDenseFromParams(
			core.NewLinearFromParam(engine, param(prefix+"self_attn.q_proj.weight", qWT)),
			nil,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(engine, param(prefix+"self_attn.k_proj.weight", kWT)),
			nil,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(engine, param(prefix+"self_attn.v_proj.weight", vWT)),
			nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(engine, param(prefix+"self_attn.o_proj.weight", oWT)),
			nil,
		)

		ropeOpts := []embeddings.RotaryPositionalEmbeddingOption{
			embeddings.WithRotaryBase(cfg.RopeTheta),
		}
		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			context.Background(), engine, headDim, cfg.MaxSeqLen, ropeOpts...,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d rope: %w", i, err)
		}

		gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
			engine, ops, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads,
			wq, wk, wv, wo, rope,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d gqa: %w", i, err)
		}
		gqa.LayerIndex = i

		attnOut := builder.AddNode(gqa, normed)

		// --- Residual Add ---
		add1 := core.NewAdd[float32](engine)
		residual1 := builder.AddNode(add1, attnOut, hidden)

		// --- Post-Attention LayerNorm ---
		postNormW, err := lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		postNorm, err := normalization.NewRMSNormFromParam[float32](
			engine, ops, 1e-5, param(prefix+"post_attention_layernorm.weight", postNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		normed2 := builder.AddNode(postNorm, residual1)

		// --- FFN (SiLU-gate / SwiGLU) ---
		// Llama uses gate_proj (w1/gate), up_proj (w3/up), down_proj (w2/down)
		// with SiLU activation: output = down_proj(silu(gate_proj(x)) * up_proj(x))
		gateW, err := lookup(prefix + "mlp.gate_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		upW, err := lookup(prefix + "mlp.up_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		downW, err := lookup(prefix + "mlp.down_proj.weight")
		if err != nil {
			return nil, nil, err
		}

		ffn, err := core.NewFFN[float32](
			prefix+"mlp", engine, ops,
			cfg.HiddenSize, cfg.IntermediateSize, cfg.HiddenSize,
			core.WithSwiGLU[float32](),
			core.WithFFNNoBias[float32](),
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d ffn: %w", i, err)
		}

		// Transpose FFN weights from GGUF [out, in] to Linear [in, out].
		gateWT, err := transposeWeight(prefix+"mlp.gate_proj.weight", gateW)
		if err != nil {
			return nil, nil, err
		}
		upWT, err := transposeWeight(prefix+"mlp.up_proj.weight", upW)
		if err != nil {
			return nil, nil, err
		}
		downWT, err := transposeWeight(prefix+"mlp.down_proj.weight", downW)
		if err != nil {
			return nil, nil, err
		}

		// Assign transposed weights to FFN's internal Dense layers.
		// FFN.Parameters() returns: [w1_weight, w2_weight, w3_weight] (no bias).
		// w1 = gate_proj, w2 = down_proj, w3 = up_proj
		ffnParams := ffn.Parameters()
		ffnParams[0].Value = gateWT // w1 = gate_proj
		ffnParams[1].Value = downWT // w2 = down_proj
		ffnParams[2].Value = upWT   // w3 = up_proj

		ffnOut := builder.AddNode(ffn, normed2)

		// --- Residual Add ---
		add2 := core.NewAdd[float32](engine)
		hidden = builder.AddNode(add2, ffnOut, residual1)
	}

	// --- Final RMSNorm ---
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		engine, ops, 1e-5, param("model.norm.weight", finalNormWeight),
	)
	if err != nil {
		return nil, nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head ---
	lmHeadWeight, err := lookup("lm_head.weight")
	if err != nil {
		// Llama can tie lm_head to embedding weights.
		lmHeadWeight = embedWeight
	}
	lmHead := &lmHeadNode[float32]{engine: engine, weight: lmHeadWeight}
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
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
