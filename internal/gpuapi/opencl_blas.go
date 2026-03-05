//go:build opencl

package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/clblast"
)

// OpenCLBlas implements the BLAS interface using CLBlast.
type OpenCLBlas struct {
	handle *clblast.Handle
}

// NewOpenCLBlas creates a new CLBlast BLAS adapter.
// queue and context are the OpenCL command queue and context pointers.
func NewOpenCLBlas(queue, context unsafe.Pointer) *OpenCLBlas {
	return &OpenCLBlas{
		handle: clblast.NewHandle(queue, context),
	}
}

func (b *OpenCLBlas) Sgemm(m, n, k int, alpha float32, a, bPtr unsafe.Pointer, beta float32, c unsafe.Pointer) error {
	return b.handle.Sgemm(m, n, k, alpha, a, bPtr, beta, c)
}

func (b *OpenCLBlas) SetStream(s Stream) error {
	return b.handle.SetStream(s.Ptr())
}

func (b *OpenCLBlas) Destroy() error {
	return b.handle.Destroy()
}

// Compile-time interface assertion.
var _ BLAS = (*OpenCLBlas)(nil)
