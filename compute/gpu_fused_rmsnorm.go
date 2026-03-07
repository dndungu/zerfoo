//go:build cuda

package compute

import (
	"github.com/zerfoo/zerfoo/tensor"
)

// FusedRMSNormGPU implements the FusedRMSNormer interface for GPUEngine.
// Uses the fused GPU kernel when input is GPU-resident, falls back to CPU otherwise.
func (e *GPUEngine[T]) FusedRMSNormGPU(input, weight *tensor.TensorNumeric[float32], epsilon float32) (*tensor.TensorNumeric[float32], error) {
	// Only use GPU path when input is GPU-resident.
	if _, ok := input.GetStorage().(*tensor.GPUStorage[float32]); !ok {
		out, _, err := FusedRMSNorm(input, weight, epsilon)
		return out, err
	}

	e.setDevice()

	shape := input.Shape()
	D := shape[len(shape)-1]
	total := input.Size()
	rows := total / D

	// We need float32-specific device pointers. Cast through any.
	f32Engine, ok := any(e).(*GPUEngine[float32])
	if !ok {
		out, _, err := FusedRMSNorm(input, weight, epsilon)
		return out, err
	}

	devIn, cleanupIn, err := getDevicePtr(f32Engine, input)
	if err != nil {
		out, _, ferr := FusedRMSNorm(input, weight, epsilon)
		return out, ferr
	}
	defer cleanupIn()

	devWeight, cleanupWeight, err := getDevicePtr(f32Engine, weight)
	if err != nil {
		out, _, ferr := FusedRMSNorm(input, weight, epsilon)
		return out, ferr
	}
	defer cleanupWeight()

	outByteSize := total * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, outByteSize)
	if err != nil {
		out, _, ferr := FusedRMSNorm(input, weight, epsilon)
		return out, ferr
	}

	if err := e.kernels.RMSNorm(devIn, devWeight, devOut, epsilon, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outByteSize)
		return nil, err
	}

	return makeGPUResult[float32](f32Engine, shape, devOut, total)
}
