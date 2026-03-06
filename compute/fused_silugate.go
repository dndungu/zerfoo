package compute

import (
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/tensor"
)

// FusedSiLUGate computes silu(gate) * up in a single element-wise pass.
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
// gate and up must have the same shape.
// This avoids materializing separate sigmoid, mul, and mul intermediate tensors.
func FusedSiLUGate(gate, up *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	gShape := gate.Shape()
	uShape := up.Shape()
	if len(gShape) != len(uShape) {
		return nil, fmt.Errorf("FusedSiLUGate: shape rank mismatch: gate %v vs up %v", gShape, uShape)
	}
	for i := range gShape {
		if gShape[i] != uShape[i] {
			return nil, fmt.Errorf("FusedSiLUGate: shape mismatch at dim %d: gate %v vs up %v", i, gShape, uShape)
		}
	}

	gData := gate.Data()
	uData := up.Data()
	outData := make([]float32, len(gData))

	for i, g := range gData {
		// silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
		sig := float32(1.0 / (1.0 + math.Exp(-float64(g))))
		outData[i] = g * sig * uData[i]
	}

	return tensor.New(gShape, outData)
}
