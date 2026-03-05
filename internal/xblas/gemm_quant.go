package xblas

import "github.com/zerfoo/zerfoo/tensor"

// GemmQ4F32 computes C = A * B where A is Q4_0 quantized and B, C are float32.
// A has logical shape (m, k), B has shape (k, n), C has shape (m, n).
// Dequantizes A once upfront, then uses the float32 data for the GEMM.
func GemmQ4F32(m, n, k int, a *tensor.Q4Storage, b, c []float32) {
	// Single dequantize of entire A matrix.
	af32 := make([]float32, a.Len())
	a.Dequantize(af32)

	for i := range m {
		row := af32[i*k : i*k+k]
		for j := range n {
			var sum float32
			for p := range k {
				sum += row[p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// GemmQ8F32 computes C = A * B where A is Q8_0 quantized and B, C are float32.
func GemmQ8F32(m, n, k int, a *tensor.Q8Storage, b, c []float32) {
	af32 := make([]float32, a.Len())
	a.Dequantize(af32)

	for i := range m {
		row := af32[i*k : i*k+k]
		for j := range n {
			var sum float32
			for p := range k {
				sum += row[p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}
