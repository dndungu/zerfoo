// Package transpose provides the Transpose layer for the Zerfoo ML framework.
package transpose

import (
	"context"
	"unsafe"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Transpose represents a transpose operation.
type Transpose[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	perm        []int
	outputShape []int

	// Cache for constant inputs: when the same input data pointer is seen
	// on consecutive calls, return the cached result instead of re-transposing.
	cachedResult *tensor.TensorNumeric[T]
	cachedInPtr  uintptr
}

// OpType returns the operation type.
func (t *Transpose[T]) OpType() string {
	return "Transpose"
}

// Attributes returns the attributes.
func (t *Transpose[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"perm": t.perm,
	}
}

// New creates a new Transpose layer.
func New[T tensor.Numeric](engine compute.Engine[T], axes []int) *Transpose[T] {
	return &Transpose[T]{
		engine: engine,
		perm:   axes,
	}
}

// OutputShape returns the output shape of the Transpose layer.
func (t *Transpose[T]) OutputShape() []int {
	return t.outputShape
}

// Parameters returns no trainable parameters for the Transpose layer.
func (t *Transpose[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the transpose operation.
func (t *Transpose[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()

	// If perm is nil, use the ONNX default: reverse all axes.
	perm := t.perm
	if perm == nil {
		perm = make([]int, len(shape))
		for i := range perm {
			perm[i] = len(shape) - 1 - i
		}
		t.perm = perm
	}

	outputShape := make([]int, len(shape))
	for i, axis := range perm {
		outputShape[i] = shape[axis]
	}

	t.outputShape = outputShape

	// Cache hit: if input data pointer matches the previous call, return cached result.
	data := input.Data()
	inPtr := uintptr(unsafe.Pointer(&data[0]))
	if t.cachedResult != nil && t.cachedInPtr == inPtr {
		return t.cachedResult, nil
	}

	// Transpose the input tensor.
	transposed, err := t.engine.Transpose(ctx, input, perm)
	if err != nil {
		return nil, err
	}

	t.cachedResult = transposed
	t.cachedInPtr = inPtr
	return transposed, nil
}

// Backward computes the gradients for the Transpose layer.
func (t *Transpose[T]) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The gradient w.r.t. the input is the gradient transposed by the inverse permutation.
	inv := make([]int, len(t.perm))
	for i, p := range t.perm {
		inv[p] = i
	}

	gradInput, err := t.engine.Transpose(ctx, outputGradient, inv)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{gradInput}, nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Transpose[float32])(nil)
