package gather

import (
	"strings"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// BuildGather constructs a Gather layer. For embedding-style nodes whose name
// maps to a known weight parameter, weights are embedded in the layer. All
// other Gather nodes operate as general ONNX Gather (axis-0 indexing).
func BuildGather[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	// Derive weight patterns from the node name. ONNX node names use "/"
	// separators (e.g. "/model/embed_tokens/Gather") while parameter names
	// use "." separators (e.g. "model.embed_tokens.weight"). Normalize the
	// node name so the pattern matches the parameter.
	normalized := strings.ReplaceAll(strings.TrimPrefix(name, "/"), "/", ".")
	weightPatterns := []string{
		name + ".weight",
		strings.TrimSuffix(name, "/Gather") + ".weight",
		strings.TrimSuffix(normalized, ".Gather") + ".weight",
	}
	for _, pattern := range weightPatterns {
		if param, exists := params[pattern]; exists {
			return NewWithWeights[T](engine, param.Value), nil
		}
	}

	// General-purpose Gather: no embedded weights, takes (data, indices) inputs.
	return New[T](engine), nil
}
