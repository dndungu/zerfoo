//go:build opencl

package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/opencl/kernels"
)

// OpenCLKernels implements the KernelRunner interface using OpenCL kernels.
// Kernels are compiled from .cl source at initialization time.
type OpenCLKernels struct {
	prog *kernels.Program
}

// NewOpenCLKernels compiles the elementwise kernels and returns a runner.
// ctx, dev, and queue are the OpenCL context, device, and command queue pointers.
func NewOpenCLKernels(ctx, dev, queue unsafe.Pointer) (*OpenCLKernels, error) {
	prog, err := kernels.Compile(ctx, dev, queue)
	if err != nil {
		return nil, err
	}
	return &OpenCLKernels{prog: prog}, nil
}

// Destroy releases the compiled OpenCL program.
func (k *OpenCLKernels) Destroy() {
	if k.prog != nil {
		k.prog.Destroy()
	}
}

func (k *OpenCLKernels) Add(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Add(a, b, c, n)
}

func (k *OpenCLKernels) Sub(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Sub(a, b, c, n)
}

func (k *OpenCLKernels) Mul(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Mul(a, b, c, n)
}

func (k *OpenCLKernels) Div(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Div(a, b, c, n)
}

func (k *OpenCLKernels) Pow(base, exp, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Pow(base, exp, c, n)
}

func (k *OpenCLKernels) Exp(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Exp(a, c, n)
}

func (k *OpenCLKernels) Log(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Log(a, c, n)
}

func (k *OpenCLKernels) Sqrt(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Sqrt(a, c, n)
}

func (k *OpenCLKernels) Rsqrt(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Rsqrt(a, c, n)
}

func (k *OpenCLKernels) Tanh(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Tanh(a, c, n)
}

func (k *OpenCLKernels) TanhPrime(a, upstream, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.TanhPrime(a, upstream, c, n)
}

func (k *OpenCLKernels) AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.AddScalar(a, scalar, c, n)
}

func (k *OpenCLKernels) MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.MulScalar(a, scalar, c, n)
}

func (k *OpenCLKernels) DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.DivScalar(a, scalar, c, n)
}

func (k *OpenCLKernels) Fill(data unsafe.Pointer, value float32, n int, _ Stream) error {
	return k.prog.Fill(data, value, n)
}

func (k *OpenCLKernels) SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, _ Stream) error {
	return k.prog.SumAxis(input, output, outer, inner, axisSize)
}

func (k *OpenCLKernels) Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, _ Stream) error {
	return k.prog.Softmax(input, output, outer, inner, axisSize)
}

// Compile-time interface assertion.
var _ KernelRunner = (*OpenCLKernels)(nil)
