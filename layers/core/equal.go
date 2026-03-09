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

func (e *Equal[T]) OpType() string                   { return "Equal" }
func (e *Equal[T]) Attributes() map[string]any       { return nil }
func (e *Equal[T]) OutputShape() []int               { return nil }
func (e *Equal[T]) Parameters() []*graph.Parameter[T] { return nil }

func (e *Equal[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return binaryCompare("Equal", inputs, e.ops.One(), func(a, b float64) bool { return a == b })
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

// Greater represents an element-wise greater-than comparison. Output is 1 for true, 0 for false.
type Greater[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
}

func (g *Greater[T]) OpType() string                   { return "Greater" }
func (g *Greater[T]) Attributes() map[string]any       { return nil }
func (g *Greater[T]) OutputShape() []int               { return nil }
func (g *Greater[T]) Parameters() []*graph.Parameter[T] { return nil }

func (g *Greater[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return binaryCompare("Greater", inputs, g.ops.One(), func(a, b float64) bool { return a > b })
}

func (g *Greater[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Greater backward not implemented")
}

// BuildGreater constructs a Greater node from attributes.
func BuildGreater[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Greater[T]{engine: engine, ops: ops}, nil
}

// LessOrEqual represents an element-wise less-than-or-equal comparison.
// Output is 1 for true, 0 for false.
type LessOrEqual[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
}

func (l *LessOrEqual[T]) OpType() string                   { return "LessOrEqual" }
func (l *LessOrEqual[T]) Attributes() map[string]any       { return nil }
func (l *LessOrEqual[T]) OutputShape() []int               { return nil }
func (l *LessOrEqual[T]) Parameters() []*graph.Parameter[T] { return nil }

func (l *LessOrEqual[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return binaryCompare("LessOrEqual", inputs, l.ops.One(), func(a, b float64) bool { return a <= b })
}

func (l *LessOrEqual[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("LessOrEqual backward not implemented")
}

// BuildLessOrEqual constructs a LessOrEqual node from attributes.
func BuildLessOrEqual[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &LessOrEqual[T]{engine: engine, ops: ops}, nil
}

var (
	_ graph.Node[float32] = (*Equal[float32])(nil)
	_ graph.Node[float32] = (*Greater[float32])(nil)
	_ graph.Node[float32] = (*LessOrEqual[float32])(nil)
)
