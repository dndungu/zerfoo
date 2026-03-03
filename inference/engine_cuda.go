//go:build cuda

package inference

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
)

// createEngine returns a compute engine for the given device string.
// With CUDA support, "cuda" and "cuda:N" create a GPUEngine on the specified
// device; "cpu" creates a CPUEngine.
func createEngine(device string) (compute.Engine[float32], error) {
	devType, deviceID, err := parseDevice(device)
	if err != nil {
		return nil, err
	}
	if devType == "cpu" {
		return compute.NewCPUEngine[float32](numeric.Float32Ops{}), nil
	}
	return compute.NewGPUEngine[float32](numeric.Float32Ops{}, deviceID)
}
