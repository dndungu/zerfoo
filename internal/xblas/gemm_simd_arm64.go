//go:build arm64

package xblas

import (
	"runtime"
	"sync"
	"unsafe"
)

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

// gemvParallelThreshold is the minimum N*K for M=1 GEMV parallelization.
// Below this threshold, goroutine overhead outweighs the benefit.
const gemvParallelThreshold = 256 * 256

// SgemmSimd computes C = A*B using NEON-accelerated operations.
// A is m×k, B is k×n, C is m×n. All row-major.
// For M=1 GEMV with large N*K, parallelizes across output columns.
func SgemmSimd(m, n, k int, a, b, c []float32) {
	if m == 0 || n == 0 || k == 0 {
		return
	}

	// M=1 GEMV: parallelize across output columns (N dimension).
	// Each goroutine computes a contiguous chunk of C independently.
	if m == 1 && n*k >= gemvParallelThreshold {
		nCores := runtime.NumCPU()
		if nCores > n/16 {
			nCores = n / 16 // need at least 16 cols per core
		}
		if nCores > 1 {
			sgemmGemvParallel(n, k, a, b, c, nCores)
			return
		}
	}

	// General path: tile along K.
	sgemmTiled(m, n, k, a, b, c)
}

// sgemmTiled is the single-threaded SGEMM with K-tiling.
func sgemmTiled(m, n, k int, a, b, c []float32) {
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

// sgemmGemvParallel splits M=1 GEMV across nCores goroutines along N.
func sgemmGemvParallel(n, k int, a, b, c []float32, nCores int) {
	chunkSize := (n + nCores - 1) / nCores
	var wg sync.WaitGroup
	for t := range nCores {
		nStart := t * chunkSize
		nEnd := min(nStart+chunkSize, n)
		if nStart >= n {
			break
		}
		wg.Add(1)
		go func(nStart, nEnd int) {
			defer wg.Done()
			chunk := nEnd - nStart
			for p0 := 0; p0 < k; p0 += tileK {
				p1 := min(p0+tileK, k)
				cPtr := unsafe.Pointer(&c[nStart])
				for p := p0; p < p1; p++ {
					if aVal := a[p]; aVal != 0 {
						sgemmAccRowNeon(cPtr, unsafe.Pointer(&b[p*n+nStart]), aVal, chunk)
					}
				}
			}
		}(nStart, nEnd)
	}
	wg.Wait()
}
