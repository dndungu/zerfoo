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

// ScatterND scatters updates into a copy of the data tensor at indices.
type ScatterND[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (s *ScatterND[T]) OpType() string                  { return "ScatterND" }
func (s *ScatterND[T]) Attributes() map[string]any       { return nil }
func (s *ScatterND[T]) OutputShape() []int               { return nil }
func (s *ScatterND[T]) Parameters() []*graph.Parameter[T] { return nil }

func (s *ScatterND[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 3 {
		return nil, fmt.Errorf("ScatterND requires 3 inputs (data, indices, updates), got %d", len(inputs))
	}

	data := inputs[0].Data()
	indices := inputs[1].Data()
	updates := inputs[2].Data()
	dataShape := inputs[0].Shape()
	indicesShape := inputs[1].Shape()

	// Copy data to output.
	out := make([]T, len(data))
	copy(out, data)

	// Compute strides for the data tensor.
	strides := make([]int, len(dataShape))
	strides[len(strides)-1] = 1
	for i := len(strides) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * dataShape[i+1]
	}

	// The last dimension of indices is the index depth.
	indexDepth := indicesShape[len(indicesShape)-1]

	// Number of scatter operations.
	numScatters := len(indices) / indexDepth

	// Elements per scatter update.
	elemPerUpdate := len(updates) / numScatters

	for i := range numScatters {
		// Compute flat offset from multi-dimensional index.
		offset := 0
		for d := range indexDepth {
			idx := int(indices[i*indexDepth+d])
			offset += idx * strides[d]
		}

		// Copy update elements.
		for j := range elemPerUpdate {
			if offset+j < len(out) {
				out[offset+j] = updates[i*elemPerUpdate+j]
			}
		}
	}

	return tensor.New(dataShape, out)
}

func (s *ScatterND[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("ScatterND backward not implemented")
}

// BuildScatterND constructs a ScatterND node from attributes.
func BuildScatterND[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &ScatterND[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*ScatterND[float32])(nil)
