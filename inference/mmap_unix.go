//go:build unix

package inference

import (
	"io"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
)

// loadZMFWithMmap loads a ZMF model using mmap and returns the model graph
// plus an io.Closer that releases the mapping. The closer must remain open
// for the lifetime of the Model.
func loadZMFWithMmap(
	eng compute.Engine[float32],
	zmfPath string,
	buildOpts []model.BuildOption,
) (*model.Model[float32], io.Closer, error) {
	zmfModel, r, err := model.LoadZMFMmap(zmfPath)
	if err != nil {
		return nil, nil, err
	}

	mdl, err := model.BuildModelFromProto(eng, numeric.Float32Ops{}, zmfModel, buildOpts...)
	if err != nil {
		_ = r.Close()
		return nil, nil, err
	}

	return mdl, r, nil
}
