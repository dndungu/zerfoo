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

// Or represents an element-wise logical OR. Output is 1 when either input is
// nonzero, 0 otherwise.
type Or[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
}

func (o *Or[T]) OpType() string                  { return "Or" }
func (o *Or[T]) Attributes() map[string]any       { return nil }
func (o *Or[T]) OutputShape() []int               { return nil }
func (o *Or[T]) Parameters() []*graph.Parameter[T] { return nil }

func (o *Or[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Or requires 2 inputs, got %d", len(inputs))
	}
	one := o.ops.One()
	var zero T

	aShape, bShape := inputs[0].Shape(), inputs[1].Shape()
	a, b := inputs[0].Data(), inputs[1].Data()

	outShape, err := broadcastShapeChecked(aShape, bShape)
	if err != nil {
		return nil, fmt.Errorf("Or: %w", err)
	}
	outSize := 1
	for _, d := range outShape {
		outSize *= d
	}

	aStrides := broadcastStrides(aShape, outShape)
	bStrides := broadcastStrides(bShape, outShape)

	out := make([]T, outSize)
	for i := range out {
		ai := broadcastIndex(i, outShape, aStrides)
		bi := broadcastIndex(i, outShape, bStrides)
		if a[ai] != zero || b[bi] != zero {
			out[i] = one
		}
	}
	return tensor.New(outShape, out)
}

func (o *Or[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Or backward not implemented")
}

// BuildOr constructs an Or node from attributes.
func BuildOr[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Or[T]{engine: engine, ops: ops}, nil
}

var _ graph.Node[float32] = (*Or[float32])(nil)
