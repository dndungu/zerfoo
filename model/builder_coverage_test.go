package model

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zmf"
)

// ---------------------------------------------------------------------------
// helper: encode float32 values to little-endian bytes
// ---------------------------------------------------------------------------

func float32Bytes(vals ...float32) []byte {
	buf := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// ---------------------------------------------------------------------------
// Auto-input nodes: attention_mask, position_ids, past_key_values
// ---------------------------------------------------------------------------

func TestBuildFromZMF_AutoInputNodes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("TestAutoInputOp", passthroughBuilder())
	defer UnregisterLayer("TestAutoInputOp")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input_ids", Shape: []int64{1, 8}},
				{Name: "attention_mask", Shape: []int64{1, 8}},
				{Name: "position_ids", Shape: []int64{1, 8}},
				{Name: "past_key_values.0.key", Shape: []int64{1, 4, 0, 16}},
				{Name: "past_key_values.0.value", Shape: []int64{1, 4, 0, 16}},
			},
			Nodes: []*zmf.Node{
				{Name: "op", OpType: "TestAutoInputOp", Inputs: []string{"input_ids"}},
			},
			Outputs: []*zmf.ValueInfo{{Name: "op"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

func TestBuildFromZMF_AutoInputNodes_SmallDims(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("TestSmallDimOp", passthroughBuilder())
	defer UnregisterLayer("TestSmallDimOp")

	// past_key_values with < 4 dims (no numHeads/headDim extracted)
	// and with zero numHeads/headDim values that trigger fallbacks
	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input_ids", Shape: []int64{1, 8}},
				{Name: "past_key_values.0.key", Shape: []int64{1, 2}}, // < 4 dims
				{Name: "past_key_values.1.key", Shape: []int64{1, 0, 0, 0}}, // zero numHeads and headDim
			},
			Nodes: []*zmf.Node{
				{Name: "op", OpType: "TestSmallDimOp", Inputs: []string{"input_ids"}},
			},
			Outputs: []*zmf.ValueInfo{{Name: "op"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

// ---------------------------------------------------------------------------
// Perm, Epsilon, Axis promotion from proto fields
// ---------------------------------------------------------------------------

func TestBuildFromZMF_PromotePermEpsilonAxis(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	var capturedAttrs map[string]any
	RegisterLayer("TestPromoteOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		attrs map[string]any,
	) (graph.Node[float32], error) {
		capturedAttrs = attrs
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("TestPromoteOp")

	epsilon := float32(1e-6)
	axis := int64(2)

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{
					Name:    "promote_node",
					OpType:  "TestPromoteOp",
					Inputs:  []string{"input"},
					Perm:    []int64{1, 0, 2},
					Epsilon: &epsilon,
					Axis:    &axis,
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "promote_node"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}

	if capturedAttrs == nil {
		t.Fatal("expected attributes to be captured")
	}
	if _, ok := capturedAttrs["perm"]; !ok {
		t.Error("perm not promoted into attributes")
	}
	if _, ok := capturedAttrs["epsilon"]; !ok {
		t.Error("epsilon not promoted into attributes")
	}
	if _, ok := capturedAttrs["axis"]; !ok {
		t.Error("axis not promoted into attributes")
	}
}

// ---------------------------------------------------------------------------
// Global attributes via WithGlobalAttributes
// ---------------------------------------------------------------------------

func TestBuildFromZMF_GlobalAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	var capturedAttrs map[string]any
	RegisterLayer("TestGlobalAttrOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		attrs map[string]any,
	) (graph.Node[float32], error) {
		capturedAttrs = attrs
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("TestGlobalAttrOp")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{Name: "ga_node", OpType: "TestGlobalAttrOp", Inputs: []string{"input"}},
			},
			Outputs: []*zmf.ValueInfo{{Name: "ga_node"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel,
		WithGlobalAttributes(map[string]any{"rope_scaling": 1.5, "max_seq_len": 2048}))
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}

	if capturedAttrs["rope_scaling"] != 1.5 {
		t.Errorf("rope_scaling = %v, want 1.5", capturedAttrs["rope_scaling"])
	}
	if capturedAttrs["max_seq_len"] != 2048 {
		t.Errorf("max_seq_len = %v, want 2048", capturedAttrs["max_seq_len"])
	}
}

// ---------------------------------------------------------------------------
// Output aliases for non-Constant nodes
// ---------------------------------------------------------------------------

func TestBuildFromZMF_OutputAliases(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("TestAliasOp", passthroughBuilder())
	defer UnregisterLayer("TestAliasOp")

	// Node "producer" has output alias "alias_out". Downstream references "alias_out".
	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{
					Name:    "producer",
					OpType:  "TestAliasOp",
					Inputs:  []string{"input"},
					Outputs: []string{"alias_out"},
				},
				{
					Name:   "consumer",
					OpType: "TestAliasOp",
					Inputs: []string{"alias_out"},
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "consumer"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

// ---------------------------------------------------------------------------
// Reshape with resolved constant node (not parameter)
// ---------------------------------------------------------------------------

func TestBuildFromZMF_Reshape_WithConstantShapeNode(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("Reshape", passthroughBuilder())
	defer UnregisterLayer("Reshape")

	// Create a Constant node that provides the shape tensor, then a Reshape
	// that references it as 2nd input. This triggers the resolveParam path
	// (lines 293-324) where the shape comes from an instantiated constant node.
	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{6}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "shape_const",
					OpType: "Constant",
					Attributes: map[string]*zmf.Attribute{
						"value": {Value: &zmf.Attribute_Tensor{Tensor: &zmf.Tensor{
							Dtype: zmf.Tensor_FLOAT32,
							Shape: []int64{2},
							Data:  float32Bytes(2, 3),
						}}},
					},
					Outputs: []string{"shape_const_out"},
				},
				{
					Name:   "reshape_node",
					OpType: "Reshape",
					Inputs: []string{"input", "shape_const_out"},
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "reshape_node"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

// ---------------------------------------------------------------------------
// Unsqueeze with axes from input tensor (2 inputs)
// ---------------------------------------------------------------------------

func TestBuildFromZMF_Unsqueeze_AxesFromInput(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("Unsqueeze", passthroughBuilder())
	defer UnregisterLayer("Unsqueeze")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"axes_param": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{1},
					Data:  float32Bytes(0),
				},
			},
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{3}},
				{Name: "axes_param", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "unsqueeze_node",
					OpType: "Unsqueeze",
					Inputs: []string{"input", "axes_param"},
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "unsqueeze_node"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

// ---------------------------------------------------------------------------
// Unsqueeze with constant-promoted axes attribute (1 input, rebuildWithPromotedAxes)
// ---------------------------------------------------------------------------

func TestBuildFromZMF_Unsqueeze_PromotedAxes(t *testing.T) {
	tests := []struct {
		name    string
		attrKey string
		axes    []int64
	}{
		{"slash_prefix", "/Constant_output_0", []int64{0}},
		{"onnx_prefix", "onnx::Gather_919", []int64{1}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](ops)

			RegisterLayer("Unsqueeze", passthroughBuilder())
			defer UnregisterLayer("Unsqueeze")

			zmfModel := &zmf.Model{
				Graph: &zmf.Graph{
					Inputs: []*zmf.ValueInfo{
						{Name: "input", Shape: []int64{3}},
					},
					Nodes: []*zmf.Node{
						{
							Name:   "unsqueeze_node",
							OpType: "Unsqueeze",
							Inputs: []string{"input"},
							Attributes: map[string]*zmf.Attribute{
								tc.attrKey: {
									Value: &zmf.Attribute_Ints{
										Ints: &zmf.Ints{Val: tc.axes},
									},
								},
							},
						},
					},
					Outputs: []*zmf.ValueInfo{{Name: "unsqueeze_node"}},
				},
			}

			g, err := BuildFromZMF[float32](engine, ops, zmfModel)
			if err != nil {
				t.Fatalf("BuildFromZMF failed: %v", err)
			}
			if g == nil {
				t.Fatal("expected non-nil graph")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Unsqueeze with constant-promoted axes from constant node input
// ---------------------------------------------------------------------------

func TestBuildFromZMF_Unsqueeze_AxesFromConstantNode(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("Unsqueeze", passthroughBuilder())
	defer UnregisterLayer("Unsqueeze")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{3}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "axes_const",
					OpType: "Constant",
					Attributes: map[string]*zmf.Attribute{
						"value": {Value: &zmf.Attribute_Tensor{Tensor: &zmf.Tensor{
							Dtype: zmf.Tensor_FLOAT32,
							Shape: []int64{1},
							Data:  float32Bytes(0),
						}}},
					},
					Outputs: []string{"axes_const_out"},
				},
				{
					Name:   "unsqueeze_node",
					OpType: "Unsqueeze",
					Inputs: []string{"input", "axes_const_out"},
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "unsqueeze_node"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

// ---------------------------------------------------------------------------
// Parameter resolution: direct param, resolvedName, and _transposed suffix
// ---------------------------------------------------------------------------

func TestBuildFromZMF_ParamResolution(t *testing.T) {
	tests := []struct {
		name      string
		opType    string
		paramName string
		inputRef  string
	}{
		{
			name:      "direct_param_lookup",
			opType:    "TestDirectParamOp",
			paramName: "weight",
			inputRef:  "weight",
		},
		{
			name:      "transposed_suffix_fallback",
			opType:    "TestTransposedOp",
			paramName: "my_weight",
			inputRef:  "my_weight_transposed",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](ops)

			RegisterLayer(tc.opType, passthroughBuilder())
			defer UnregisterLayer(tc.opType)

			zmfModel := &zmf.Model{
				Graph: &zmf.Graph{
					Parameters: map[string]*zmf.Tensor{
						tc.paramName: {
							Dtype: zmf.Tensor_FLOAT32,
							Shape: []int64{1},
							Data:  float32Bytes(1),
						},
					},
					Inputs: []*zmf.ValueInfo{
						{Name: "input", Shape: []int64{1}},
					},
					Nodes: []*zmf.Node{
						{
							Name:   "op_node",
							OpType: tc.opType,
							Inputs: []string{"input", tc.inputRef},
						},
					},
					Outputs: []*zmf.ValueInfo{{Name: "op_node"}},
				},
			}

			g, err := BuildFromZMF[float32](engine, ops, zmfModel)
			if err != nil {
				t.Fatalf("BuildFromZMF failed: %v", err)
			}
			if g == nil {
				t.Fatal("expected non-nil graph")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Logits output resolution fallback
// ---------------------------------------------------------------------------

func TestBuildFromZMF_LogitsOutputResolution(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("MatMul", passthroughBuilder())
	defer UnregisterLayer("MatMul")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{Name: "/lm_head/MatMul", OpType: "MatMul", Inputs: []string{"input"}},
			},
			Outputs: []*zmf.ValueInfo{{Name: "logits"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

// ---------------------------------------------------------------------------
// getNodeNames (currently 0% covered)
// ---------------------------------------------------------------------------

func TestGetNodeNames_Coverage(t *testing.T) {
	val, _ := tensor.New[float32]([]int{1}, []float32{1})
	nodes := map[string]graph.Node[float32]{
		"a": &parameterNode[float32]{value: val},
		"b": &parameterNode[float32]{value: val},
	}

	names := getNodeNames(nodes)
	if len(names) != 2 {
		t.Errorf("getNodeNames returned %d names, want 2", len(names))
	}
}
