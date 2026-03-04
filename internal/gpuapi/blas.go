package gpuapi

import "unsafe"

// BLAS abstracts GPU-accelerated Basic Linear Algebra Subprograms.
// Each vendor (cuBLAS, rocBLAS, CLBlast) provides an implementation.
type BLAS interface {
	// Sgemm performs single-precision general matrix multiplication:
	//   C = alpha * A * B + beta * C
	// where A is m x k, B is k x n, and C is m x n.
	// All matrices are in row-major order. The implementation handles
	// the row-major to column-major conversion internally.
	Sgemm(m, n, k int, alpha float32,
		a unsafe.Pointer, lda int,
		b unsafe.Pointer, ldb int,
		beta float32,
		c unsafe.Pointer, ldc int,
	) error

	// SetStream associates the BLAS handle with an asynchronous stream.
	SetStream(stream Stream) error

	// Destroy releases the BLAS handle resources.
	Destroy() error
}
