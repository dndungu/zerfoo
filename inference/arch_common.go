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
)

// buildTransformerGraph constructs a computation graph for a decoder-only
// transformer from pre-loaded GGUF tensors. Both Llama and Gemma share the
// same transformer body; they differ only in LM head weight tying.
//
// lmHeadWeight is the weight tensor to use for the final projection. Callers
// pass either a dedicated lm_head.weight or the embedding table for tied models.
func buildTransformerGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
	lmHeadWeight *tensor.TensorNumeric[float32],
) (*graph.Graph[float32], error) {
	ops := numeric.Float32Ops{}

	lookup := func(name string) (*tensor.TensorNumeric[float32], error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", name)
		}
		return t, nil
	}

	param := func(name string, t *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
		return &graph.Parameter[float32]{Name: name, Value: t}
	}

	transposeWeight := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		tr, err := engine.Transpose(context.Background(), t, []int{1, 0})
		if err != nil {
			return nil, fmt.Errorf("transpose %s: %w", name, err)
		}
		return tr, nil
	}

	finalNormWeight, err := lookup("model.norm.weight")
	if err != nil {
		return nil, err
	}

	builder := graph.NewBuilder[float32](engine)
	input := builder.Input([]int{1, 1, cfg.HiddenSize})

	hidden := input
	headDim := cfg.HiddenSize / cfg.NumHeads

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// --- Input LayerNorm ---
		inputNormW, err := lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		inputNorm, err := normalization.NewRMSNormFromParam[float32](
			engine, ops, 1e-5, param(prefix+"input_layernorm.weight", inputNormW),
		)
		if err != nil {
			return nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// --- Self Attention (GQA) ---
		qW, err := lookup(prefix + "self_attn.q_proj.weight")
		if err != nil {
			return nil, err
		}
		kW, err := lookup(prefix + "self_attn.k_proj.weight")
		if err != nil {
			return nil, err
		}
		vW, err := lookup(prefix + "self_attn.v_proj.weight")
		if err != nil {
			return nil, err
		}
		oW, err := lookup(prefix + "self_attn.o_proj.weight")
		if err != nil {
			return nil, err
		}

		qWT, err := transposeWeight(prefix+"self_attn.q_proj.weight", qW)
		if err != nil {
			return nil, err
		}
		kWT, err := transposeWeight(prefix+"self_attn.k_proj.weight", kW)
		if err != nil {
			return nil, err
		}
		vWT, err := transposeWeight(prefix+"self_attn.v_proj.weight", vW)
		if err != nil {
			return nil, err
		}
		oWT, err := transposeWeight(prefix+"self_attn.o_proj.weight", oW)
		if err != nil {
			return nil, err
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
			return nil, fmt.Errorf("layer %d rope: %w", i, err)
		}

		gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
			engine, ops, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads,
			wq, wk, wv, wo, rope,
		)
		if err != nil {
			return nil, fmt.Errorf("layer %d gqa: %w", i, err)
		}
		gqa.LayerIndex = i

		attnOut := builder.AddNode(gqa, normed)

		// --- Residual Add ---
		add1 := core.NewAdd[float32](engine)
		residual1 := builder.AddNode(add1, attnOut, hidden)

		// --- Post-Attention LayerNorm ---
		postNormW, err := lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := normalization.NewRMSNormFromParam[float32](
			engine, ops, 1e-5, param(prefix+"post_attention_layernorm.weight", postNormW),
		)
		if err != nil {
			return nil, err
		}
		normed2 := builder.AddNode(postNorm, residual1)

		// --- FFN (SwiGLU) ---
		gateW, err := lookup(prefix + "mlp.gate_proj.weight")
		if err != nil {
			return nil, err
		}
		upW, err := lookup(prefix + "mlp.up_proj.weight")
		if err != nil {
			return nil, err
		}
		downW, err := lookup(prefix + "mlp.down_proj.weight")
		if err != nil {
			return nil, err
		}

		ffn, err := core.NewFFN[float32](
			prefix+"mlp", engine, ops,
			cfg.HiddenSize, cfg.IntermediateSize, cfg.HiddenSize,
			core.WithSwiGLU[float32](),
			core.WithFFNNoBias[float32](),
		)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn: %w", i, err)
		}

		gateWT, err := transposeWeight(prefix+"mlp.gate_proj.weight", gateW)
		if err != nil {
			return nil, err
		}
		upWT, err := transposeWeight(prefix+"mlp.up_proj.weight", upW)
		if err != nil {
			return nil, err
		}
		downWT, err := transposeWeight(prefix+"mlp.down_proj.weight", downW)
		if err != nil {
			return nil, err
		}

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
		return nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head ---
	lmHead := &lmHeadNode[float32]{engine: engine, weight: lmHeadWeight}
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, fmt.Errorf("build graph: %w", err)
	}

	return g, nil
}
