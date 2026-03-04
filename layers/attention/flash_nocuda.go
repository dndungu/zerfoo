//go:build !(cuda && cutlass)

package attention

import "github.com/zerfoo/zerfoo/tensor"

// tryFlashForward is the fallback when CUTLASS flash attention is not
// available. It always returns (nil, nil) to signal that the caller should
// use the naive attention path.
func tryFlashForward[T tensor.Numeric](
	_, _, _ *tensor.TensorNumeric[T],
	_ int,
	_ bool,
) (*tensor.TensorNumeric[T], error) {
	return nil, nil
}
