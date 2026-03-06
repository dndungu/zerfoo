// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// MatMul is a layer that performs matrix multiplication of two tensors.
type MatMul[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int

	// cachedBTranspose caches the transposed B operand when B requires
	// transposition (constant weight case). Set on first Forward call
	// and reused on subsequent calls to avoid transposing every time.
	cachedBTranspose *tensor.TensorNumeric[T]
	cachedBPtr       uintptr                  // data pointer of the B tensor that was transposed
	cachedB          *tensor.TensorNumeric[T] // the B tensor that was transposed
}

// NewMatMul creates a new MatMul layer.
func NewMatMul[T tensor.Numeric](engine compute.Engine[T]) *MatMul[T] {
	return &MatMul[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the MatMul layer.
func (m *MatMul[T]) OutputShape() []int {
	return m.outputShape
}

// Parameters returns no trainable parameters for the MatMul layer.
func (m *MatMul[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the matrix multiplication.
func (m *MatMul[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MatMul layer requires exactly 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]

	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return nil, fmt.Errorf("MatMul requires at least 2D tensors, got %dD and %dD", len(aShape), len(bShape))
	}

	// For a @ b, a's last dim must match b's second-to-last dim.
	if aShape[len(aShape)-1] != bShape[len(bShape)-2] {
		// Check if this is a case where b needs to be transposed (2D only).
		if len(bShape) == 2 && aShape[len(aShape)-1] == bShape[1] {
			bTransposed, err := m.getCachedTranspose(ctx, b)
			if err != nil {
				return nil, fmt.Errorf("failed to transpose second operand: %w", err)
			}

			result, err := m.engine.MatMul(ctx, a, bTransposed)
			if err != nil {
				return nil, err
			}

			m.outputShape = result.Shape()
			return result, nil
		}

		return nil, fmt.Errorf("incompatible dimensions for matrix multiplication: %v x %v", aShape, bShape)
	}

	result, err := m.engine.MatMul(ctx, a, b)
	if err != nil {
		return nil, err
	}

	m.outputShape = result.Shape()

	return result, nil
}

// getCachedTranspose returns the transposed B matrix, caching it for reuse
// when the same B tensor (identified by data pointer) is passed on subsequent calls.
// This avoids re-transposing constant weight matrices on every forward pass.
func (m *MatMul[T]) getCachedTranspose(ctx context.Context, b *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.cachedBTranspose != nil {
		if m.cachedB == b {
			return m.cachedBTranspose, nil
		}
		bPtr := uintptr(unsafe.Pointer(&b.Data()[0]))
		if m.cachedBPtr == bPtr {
			return m.cachedBTranspose, nil
		}
	}
	transposed, err := m.engine.Transpose(ctx, b, []int{1, 0})
	if err != nil {
		return nil, err
	}
	m.cachedBTranspose = transposed
	m.cachedB = b
	m.cachedBPtr = uintptr(unsafe.Pointer(&b.Data()[0]))
	return transposed, nil
}

// Backward computes the gradients for the MatMul layer.
func (m *MatMul[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		panic("MatMul layer requires exactly 2 inputs")
	}

	a := inputs[0]
	b := inputs[1]

	// Gradient w.r.t. a: outputGradient @ b^T
	gradA, err := m.engine.MatMul(ctx, outputGradient, b)
	if err != nil {
		return nil, err
	}

	// Gradient w.r.t. b: a^T @ outputGradient
	gradB, err := m.engine.MatMul(ctx, a, outputGradient)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{gradA, gradB}, nil
}

// OpType returns the operation type of the MatMul layer.
func (m *MatMul[T]) OpType() string {
	return "MatMul"
}

// Attributes returns nil for the MatMul layer.
func (m *MatMul[T]) Attributes() map[string]interface{} {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*MatMul[float32])(nil)
