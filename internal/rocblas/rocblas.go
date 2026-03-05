//go:build rocm

package rocblas

/*
#cgo LDFLAGS: -lrocblas
#include <rocblas/rocblas.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Handle wraps a rocBLAS handle.
type Handle struct {
	h C.rocblas_handle
}

// CreateHandle creates a new rocBLAS context handle.
func CreateHandle() (*Handle, error) {
	var h C.rocblas_handle

	status := C.rocblas_create_handle(&h)
	if status != C.rocblas_status_success {
		return nil, fmt.Errorf("rocblas_create_handle failed with status %d", int(status))
	}

	return &Handle{h: h}, nil
}

// Destroy releases the rocBLAS handle resources.
func (h *Handle) Destroy() error {
	status := C.rocblas_destroy_handle(h.h)
	if status != C.rocblas_status_success {
		return fmt.Errorf("rocblas_destroy_handle failed with status %d", int(status))
	}

	return nil
}

// SetStream associates a HIP stream with this rocBLAS handle.
// All subsequent rocBLAS operations will execute on the given stream.
// Pass nil to use the default stream.
func (h *Handle) SetStream(streamPtr unsafe.Pointer) error {
	status := C.rocblas_set_stream(h.h, C.hipStream_t(streamPtr))
	if status != C.rocblas_status_success {
		return fmt.Errorf("rocblas_set_stream failed with status %d", int(status))
	}

	return nil
}

// Sgemm performs single-precision general matrix multiplication.
//
// This function handles the row-major to column-major conversion internally.
// rocBLAS uses column-major order, but Go uses row-major. The trick:
//
//	For row-major C = A * B (m x n = m x k * k x n):
//	Call rocblas_sgemm with B as first arg and A as second, swapping m/n,
//	because in column-major: B^T * A^T = (A * B)^T, and since rocBLAS reads
//	row-major data as the transpose of what it expects, this yields the
//	correct row-major result in C.
//
// Parameters (in row-major terms):
//
//	m     - rows of A and C
//	n     - columns of B and C
//	k     - columns of A / rows of B
//	alpha - scalar multiplier for A*B
//	a     - device pointer to A (m x k, row-major)
//	b     - device pointer to B (k x n, row-major)
//	beta  - scalar multiplier for C
//	c     - device pointer to C (m x n, row-major), output
func Sgemm(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	// Row-major to column-major conversion (same strategy as cuBLAS):
	// rocblas_sgemm(handle, transB, transA, n, m, k, alpha, B, n, A, k, beta, C, n)
	status := C.rocblas_sgemm(
		h.h,
		C.rocblas_operation_none, // transB = no-transpose
		C.rocblas_operation_none, // transA = no-transpose
		C.rocblas_int(n),         // rows of op(B) and C (col-major)
		C.rocblas_int(m),         // cols of op(A) and C (col-major)
		C.rocblas_int(k),         // inner dimension
		(*C.float)(unsafe.Pointer(&cAlpha)),
		(*C.float)(b),    // B comes first
		C.rocblas_int(n), // leading dimension of B
		(*C.float)(a),    // A comes second
		C.rocblas_int(k), // leading dimension of A
		(*C.float)(unsafe.Pointer(&cBeta)),
		(*C.float)(c),
		C.rocblas_int(n), // leading dimension of C
	)

	if status != C.rocblas_status_success {
		return fmt.Errorf("rocblas_sgemm failed with status %d", int(status))
	}

	return nil
}
