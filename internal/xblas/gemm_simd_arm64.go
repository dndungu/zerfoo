//go:build arm64

package xblas

import "unsafe"

// vdotf32 computes the dot product of two float32 vectors using NEON.
// n must be > 0. a and b must point to at least n float32 values.
// Implemented in gemm_simd_arm64.s.
//
//go:noescape
func vdotf32(a, b unsafe.Pointer, n int) float32

// sgemmAccRowNeon computes c[j] += aVal * b[j] for j = 0..n-1 using NEON.
// Implemented in gemm_simd_arm64.s.
//
//go:noescape
func sgemmAccRowNeon(c, b unsafe.Pointer, aVal float32, n int)

// sgemmAccRow is the platform-agnostic name used by GemmQ4F32Fused.
func sgemmAccRow(c, b unsafe.Pointer, aVal float32, n int) {
	sgemmAccRowNeon(c, b, aVal, n)
}

const tileK = 256

// SgemmSimd computes C = A*B using NEON-accelerated operations.
// A is m×k, B is k×n, C is m×n. All row-major.
func SgemmSimd(m, n, k int, a, b, c []float32) {
	if m == 0 || n == 0 || k == 0 {
		return
	}

	// Tile along K to keep B panel in L2 cache.
	for p0 := 0; p0 < k; p0 += tileK {
		p1 := min(p0+tileK, k)

		for i := range m {
			cRow := unsafe.Pointer(&c[i*n])
			for p := p0; p < p1; p++ {
				if aVal := a[i*k+p]; aVal != 0 {
					sgemmAccRowNeon(cRow, unsafe.Pointer(&b[p*n]), aVal, n)
				}
			}
		}
	}
}
