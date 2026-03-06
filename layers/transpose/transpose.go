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

	// Cache for constant inputs: return cached result when either the
	// same data pointer or same tensor object is seen. Data pointer
	// handles the common case (Go allocator reuse). Tensor pointer
	// handles Q4-backed tensors whose Data() allocates a new slice.
	cachedResult *tensor.TensorNumeric[T]
	cachedInPtr  uintptr
	cachedInput  *tensor.TensorNumeric[T]
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

	// Cache hit: check both data pointer (for dense tensors where Go may
	// reuse memory) and tensor identity (for Q4-backed tensors where Data()
	// allocates a new slice on each call).
	if t.cachedResult != nil {
		if t.cachedInput == input {
			return t.cachedResult, nil
		}
		data := input.Data()
		inPtr := uintptr(unsafe.Pointer(&data[0]))
		if t.cachedInPtr == inPtr {
			return t.cachedResult, nil
		}
	}

	// Transpose the input tensor.
	transposed, err := t.engine.Transpose(ctx, input, perm)
	if err != nil {
		return nil, err
	}

	t.cachedResult = transposed
	t.cachedInput = input
	data := input.Data()
	t.cachedInPtr = uintptr(unsafe.Pointer(&data[0]))
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
