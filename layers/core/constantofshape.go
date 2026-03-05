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

// ConstantOfShape creates a tensor of a given shape filled with a constant value.
type ConstantOfShape[T tensor.Numeric] struct {
	engine compute.Engine[T]
	value  T
}

func (c *ConstantOfShape[T]) OpType() string { return "ConstantOfShape" }
func (c *ConstantOfShape[T]) Attributes() map[string]any {
	return map[string]any{"value": c.value}
}
func (c *ConstantOfShape[T]) OutputShape() []int               { return nil }
func (c *ConstantOfShape[T]) Parameters() []*graph.Parameter[T] { return nil }

func (c *ConstantOfShape[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("ConstantOfShape requires 1 input (shape), got %d", len(inputs))
	}

	shapeData := inputs[0].Data()
	shape := make([]int, len(shapeData))
	size := 1
	for i, v := range shapeData {
		shape[i] = int(v)
		size *= shape[i]
	}

	out := make([]T, size)
	for i := range out {
		out[i] = c.value
	}

	return tensor.New(shape, out)
}

func (c *ConstantOfShape[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("ConstantOfShape backward not implemented")
}

// BuildConstantOfShape constructs a ConstantOfShape node from attributes.
func BuildConstantOfShape[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], attrs map[string]any,
) (graph.Node[T], error) {
	var value T
	if v, ok := attrs["value"]; ok {
		switch val := v.(type) {
		case float64:
			value = ops.FromFloat64(val)
		case float32:
			value = ops.FromFloat32(val)
		case int64:
			value = ops.FromFloat64(float64(val))
		}
	}
	return &ConstantOfShape[T]{engine: engine, value: value}, nil
}

var _ graph.Node[float32] = (*ConstantOfShape[float32])(nil)
