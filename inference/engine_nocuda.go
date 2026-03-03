//go:build !cuda

package inference

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
)

// createEngine returns a compute engine for the given device string.
// Without CUDA support, only "cpu" is valid.
func createEngine(device string) (compute.Engine[float32], error) {
	devType, _, err := parseDevice(device)
	if err != nil {
		return nil, err
	}
	if devType == "cuda" {
		return nil, fmt.Errorf("CUDA device requested but binary built without cuda build tag")
	}
	return compute.NewCPUEngine[float32](numeric.Float32Ops{}), nil
}
