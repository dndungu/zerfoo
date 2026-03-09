package core

import (
	"fmt"

	"github.com/zerfoo/zerfoo/tensor"
)

// binaryCompare performs an element-wise comparison of two tensors with full
// N-D NumPy-style broadcasting. The predicate receives (a[i], b[i]) as float64
// and returns true when the output element should be set to one.
func binaryCompare[T tensor.Numeric](
	opName string,
	inputs []*tensor.TensorNumeric[T],
	one T,
	pred func(float64, float64) bool,
) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("%s requires 2 inputs, got %d", opName, len(inputs))
	}

	aShape, bShape := inputs[0].Shape(), inputs[1].Shape()
	a, b := inputs[0].Data(), inputs[1].Data()

	outShape, err := broadcastShapeChecked(aShape, bShape)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", opName, err)
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
		if pred(float64(a[ai]), float64(b[bi])) {
			out[i] = one
		}
	}
	return tensor.New(outShape, out)
}
