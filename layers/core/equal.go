package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Equal represents an element-wise equality comparison. Output is 1 for true, 0 for false.
type Equal[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
}

func (e *Equal[T]) OpType() string                  { return "Equal" }
func (e *Equal[T]) Attributes() map[string]any       { return nil }
func (e *Equal[T]) OutputShape() []int               { return nil }
func (e *Equal[T]) Parameters() []*graph.Parameter[T] { return nil }

func (e *Equal[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Equal requires 2 inputs, got %d", len(inputs))
	}
	a, b := inputs[0].Data(), inputs[1].Data()
	if len(a) != len(b) {
		return nil, fmt.Errorf("Equal: input sizes differ (%d vs %d)", len(a), len(b))
	}
	out := make([]T, len(a))
	one := e.ops.One()
	for i := range a {
		if a[i] == b[i] {
			out[i] = one
		}
	}
	return tensor.New(inputs[0].Shape(), out)
}

func (e *Equal[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Equal backward not implemented")
}

// BuildEqual constructs an Equal node from attributes.
func BuildEqual[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Equal[T]{engine: engine, ops: ops}, nil
}

var _ graph.Node[float32] = (*Equal[float32])(nil)
