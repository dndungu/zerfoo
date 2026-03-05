//go:build rocm

package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/hip/kernels"
)

// ROCmKernels implements the KernelRunner interface using custom HIP kernels.
type ROCmKernels struct{}

// NewROCmKernels returns a new ROCm kernel runner adapter.
func NewROCmKernels() *ROCmKernels {
	return &ROCmKernels{}
}

func rocmStreamPtr(s Stream) unsafe.Pointer {
	if s == nil {
		return nil
	}
	return s.Ptr()
}

func (k *ROCmKernels) Add(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Add(a, b, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Sub(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Sub(a, b, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Mul(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Mul(a, b, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Div(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Div(a, b, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Pow(base, exp, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Pow(base, exp, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Exp(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Exp(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Log(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Log(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Sqrt(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Sqrt(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Rsqrt(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Rsqrt(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Tanh(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Tanh(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) TanhPrime(a, upstream, c unsafe.Pointer, n int, s Stream) error {
	return kernels.TanhPrime(a, upstream, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.AddScalar(a, scalar, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.MulScalar(a, scalar, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.DivScalar(a, scalar, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Fill(data unsafe.Pointer, value float32, n int, s Stream) error {
	return kernels.Fill(data, value, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, s Stream) error {
	return kernels.SumAxis(input, output, outer, inner, axisSize, rocmStreamPtr(s))
}

func (k *ROCmKernels) Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, s Stream) error {
	return kernels.Softmax(input, output, outer, inner, axisSize, rocmStreamPtr(s))
}

// Compile-time interface assertion.
var _ KernelRunner = (*ROCmKernels)(nil)
