package activations

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Tanh implements the hyperbolic tangent activation function using engine.Tanh
// directly, which is visible to the tracing compiler (unlike the UnaryOp-based
// BaseActivation path).
type Tanh[T tensor.Numeric] struct {
	graph.NoParameters[T]
	engine    compute.Engine[T]
	ops       numeric.Arithmetic[T]
	lastInput *tensor.TensorNumeric[T]
}

// NewTanh creates a new Tanh activation function.
func NewTanh[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *Tanh[T] {
	return &Tanh[T]{engine: engine, ops: ops}
}

// OpType returns "Tanh".
func (t *Tanh[T]) OpType() string { return "Tanh" }

// Attributes returns an empty attribute map.
func (t *Tanh[T]) Attributes() map[string]interface{} { return nil }

// OutputShape returns the output shape of the activation.
func (t *Tanh[T]) OutputShape() []int {
	if t.lastInput != nil {
		return t.lastInput.Shape()
	}
	return nil
}

// Forward applies tanh using engine.Tanh (traced as "Tanh", not "UnaryOp").
func (t *Tanh[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Tanh: expected 1 input, got %d", len(inputs))
	}
	t.lastInput = inputs[0]
	return t.engine.Tanh(ctx, t.lastInput)
}

// Backward computes the gradient of tanh: dy/dx = 1 - tanh(x)^2.
func (t *Tanh[T]) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// tanh(x)
	tanhX, err := t.engine.Tanh(ctx, t.lastInput)
	if err != nil {
		return nil, err
	}
	// tanh(x)^2
	tanhSq, err := t.engine.Mul(ctx, tanhX, tanhX)
	if err != nil {
		return nil, err
	}
	// 1 - tanh(x)^2
	negTanhSq, err := t.engine.MulScalar(ctx, tanhSq, t.ops.FromFloat64(-1))
	if err != nil {
		return nil, err
	}
	derivative, err := t.engine.AddScalar(ctx, negTanhSq, t.ops.One())
	if err != nil {
		return nil, err
	}
	// inputGrad = derivative * outputGradient
	inputGrad, err := t.engine.Mul(ctx, outputGradient, derivative)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}

// BuildTanh constructs a Tanh activation node from attributes.
func BuildTanh[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewTanh(engine, ops), nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Tanh[float32])(nil)
