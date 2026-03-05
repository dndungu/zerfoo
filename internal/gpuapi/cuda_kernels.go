//go:build cuda

package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
)

// CUDAKernels implements the KernelRunner interface using custom CUDA kernels.
type CUDAKernels struct{}

// NewCUDAKernels returns a new CUDA kernel runner adapter.
func NewCUDAKernels() *CUDAKernels {
	return &CUDAKernels{}
}

func streamPtr(s Stream) unsafe.Pointer {
	if s == nil {
		return nil
	}
	return s.Ptr()
}

func (k *CUDAKernels) Add(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Add(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) Sub(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Sub(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) Mul(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Mul(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) Div(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Div(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) Pow(base, exp, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Pow(base, exp, c, n, streamPtr(s))
}

func (k *CUDAKernels) Exp(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Exp(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) Log(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Log(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) Sqrt(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Sqrt(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) Rsqrt(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Rsqrt(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) Tanh(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Tanh(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) TanhPrime(a, upstream, c unsafe.Pointer, n int, s Stream) error {
	return kernels.TanhPrime(a, upstream, c, n, streamPtr(s))
}

func (k *CUDAKernels) AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.AddScalar(a, scalar, c, n, streamPtr(s))
}

func (k *CUDAKernels) MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.MulScalar(a, scalar, c, n, streamPtr(s))
}

func (k *CUDAKernels) DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.DivScalar(a, scalar, c, n, streamPtr(s))
}

func (k *CUDAKernels) Fill(data unsafe.Pointer, value float32, n int, s Stream) error {
	return kernels.Fill(data, value, n, streamPtr(s))
}

func (k *CUDAKernels) SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, s Stream) error {
	return kernels.SumAxis(input, output, outer, inner, axisSize, streamPtr(s))
}

func (k *CUDAKernels) Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, s Stream) error {
	return kernels.Softmax(input, output, outer, inner, axisSize, streamPtr(s))
}

// Compile-time interface assertion.
var _ KernelRunner = (*CUDAKernels)(nil)
