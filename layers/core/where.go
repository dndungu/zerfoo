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

// Where selects elements from two tensors based on a condition tensor.
// Output[i] = x[i] if condition[i] != 0, else y[i].
type Where[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (w *Where[T]) OpType() string                  { return "Where" }
func (w *Where[T]) Attributes() map[string]any       { return nil }
func (w *Where[T]) OutputShape() []int               { return nil }
func (w *Where[T]) Parameters() []*graph.Parameter[T] { return nil }

func (w *Where[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 3 {
		return nil, fmt.Errorf("Where requires 3 inputs (condition, x, y), got %d", len(inputs))
	}
	cond, x, y := inputs[0].Data(), inputs[1].Data(), inputs[2].Data()
	n := len(cond)
	if len(x) != n || len(y) != n {
		return nil, fmt.Errorf("Where: input sizes differ (cond=%d, x=%d, y=%d)", n, len(x), len(y))
	}
	out := make([]T, n)
	for i := range cond {
		if cond[i] != 0 {
			out[i] = x[i]
		} else {
			out[i] = y[i]
		}
	}
	return tensor.New(inputs[1].Shape(), out)
}

func (w *Where[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Where backward not implemented")
}

// BuildWhere constructs a Where node from attributes.
func BuildWhere[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Where[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*Where[float32])(nil)
