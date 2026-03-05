package gpuapi

import "unsafe"

// KernelRunner abstracts GPU compute kernels for elementwise, scalar,
// reduction, and utility operations. Each vendor provides an implementation
// using its own kernel compilation toolchain (CUDA .cu, HIP .hip, OpenCL .cl).
type KernelRunner interface {
	// Binary elementwise operations: c[i] = op(a[i], b[i])
	Add(a, b, c unsafe.Pointer, n int, stream Stream) error
	Sub(a, b, c unsafe.Pointer, n int, stream Stream) error
	Mul(a, b, c unsafe.Pointer, n int, stream Stream) error
	Div(a, b, c unsafe.Pointer, n int, stream Stream) error
	Pow(base, exp, c unsafe.Pointer, n int, stream Stream) error

	// Unary elementwise operations: c[i] = op(a[i])
	Exp(a, c unsafe.Pointer, n int, stream Stream) error
	Log(a, c unsafe.Pointer, n int, stream Stream) error
	Sqrt(a, c unsafe.Pointer, n int, stream Stream) error
	Rsqrt(a, c unsafe.Pointer, n int, stream Stream) error
	Tanh(a, c unsafe.Pointer, n int, stream Stream) error

	// TanhPrime: c[i] = (1 - tanh(a[i])^2) * upstream[i]
	TanhPrime(a, upstream, c unsafe.Pointer, n int, stream Stream) error

	// Scalar operations: c[i] = op(a[i], scalar)
	AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error
	MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error
	DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error

	// Fill sets all n elements to value.
	Fill(data unsafe.Pointer, value float32, n int, stream Stream) error

	// SumAxis reduces along one axis: output[outer][inner] = sum(input[outer][k][inner], k=0..axisSize-1).
	SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, stream Stream) error

	// Softmax computes softmax along one axis.
	Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, stream Stream) error

	// GemmQ4F32 performs Q4_0 dequant-GEMM: C = dequant(A_q4) * B.
	// A_q4 is packed Q4_0 blocks, B is [K,N] float32, C is [M,N] float32.
	GemmQ4F32(aQ4, b, c unsafe.Pointer, m, k, n int, stream Stream) error
}
