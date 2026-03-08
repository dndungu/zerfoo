package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestGraph_EngineProxy_SetGet(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	proxy := compute.NewEngineProxy[float32](engine)

	builder := NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 2})
	node := &mockFloat32Node{
		forwardFunc: func(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
			return inputs[0], nil
		},
	}
	out := builder.AddNode(node, input)
	g, err := builder.Build(out)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if got := g.EngineProxy(); got != nil {
		t.Fatal("EngineProxy should be nil before SetEngineProxy")
	}

	g.SetEngineProxy(proxy)

	if got := g.EngineProxy(); got != proxy {
		t.Fatal("EngineProxy should return the proxy that was set")
	}
}

func TestGraph_EngineProxy_ForwardMatchesNonProxy(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Build graph without proxy.
	addNode := &addFloat32Node{engine: engine}
	builder1 := NewBuilder[float32](engine)
	in1 := builder1.Input([]int{1, 3})
	in2 := builder1.Input([]int{1, 3})
	out1 := builder1.AddNode(addNode, in1, in2)
	g1, err := builder1.Build(out1)
	if err != nil {
		t.Fatalf("Build without proxy: %v", err)
	}

	// Build graph with proxy.
	proxy := compute.NewEngineProxy[float32](engine)
	addNodeProxy := &addFloat32Node{engine: proxy}
	builder2 := NewBuilder[float32](proxy)
	in3 := builder2.Input([]int{1, 3})
	in4 := builder2.Input([]int{1, 3})
	out2 := builder2.AddNode(addNodeProxy, in3, in4)
	g2, err := builder2.Build(out2)
	if err != nil {
		t.Fatalf("Build with proxy: %v", err)
	}
	g2.SetEngineProxy(proxy)

	a, err := tensor.New[float32]([]int{1, 3}, []float32{1, 2, 3})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	b, err := tensor.New[float32]([]int{1, 3}, []float32{4, 5, 6})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ctx := context.Background()
	result1, err := g1.Forward(ctx, a, b)
	if err != nil {
		t.Fatalf("Forward without proxy: %v", err)
	}

	result2, err := g2.Forward(ctx, a, b)
	if err != nil {
		t.Fatalf("Forward with proxy: %v", err)
	}

	if result1.Size() != result2.Size() {
		t.Fatalf("size mismatch: %d vs %d", result1.Size(), result2.Size())
	}

	for i := 0; i < result1.Size(); i++ {
		v1, _ := result1.At(i)
		v2, _ := result2.At(i)
		if v1 != v2 {
			t.Fatalf("value mismatch at index %d: %v vs %v", i, v1, v2)
		}
	}
}

// mockFloat32Node is a simple pass-through node for float32.
type mockFloat32Node struct {
	forwardFunc func(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)
}

func (m *mockFloat32Node) OpType() string                              { return "MockFloat32" }
func (m *mockFloat32Node) Attributes() map[string]interface{}          { return nil }
func (m *mockFloat32Node) OutputShape() []int                          { return nil }
func (m *mockFloat32Node) Parameters() []*Parameter[float32]           { return nil }

func (m *mockFloat32Node) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if m.forwardFunc != nil {
		return m.forwardFunc(ctx, inputs...)
	}
	return inputs[0], nil
}

func (m *mockFloat32Node) Backward(_ context.Context, _ types.BackwardMode, grad *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return []*tensor.TensorNumeric[float32]{grad}, nil
}

// addFloat32Node computes element-wise addition using the engine.
type addFloat32Node struct {
	engine compute.Engine[float32]
}

func (n *addFloat32Node) OpType() string                              { return "Add" }
func (n *addFloat32Node) Attributes() map[string]interface{}          { return nil }
func (n *addFloat32Node) OutputShape() []int                          { return nil }
func (n *addFloat32Node) Parameters() []*Parameter[float32]           { return nil }

func (n *addFloat32Node) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return n.engine.Add(ctx, inputs[0], inputs[1])
}

func (n *addFloat32Node) Backward(_ context.Context, _ types.BackwardMode, grad *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return []*tensor.TensorNumeric[float32]{grad, grad}, nil
}
