package xblas

import "github.com/zerfoo/zerfoo/tensor"

// GemmQ4F32 computes C = A * B where A is Q4_0 quantized and B, C are float32.
// A has logical shape (m, k), B has shape (k, n), C has shape (m, n).
// Dequantizes A once upfront, then uses SIMD-accelerated SGEMM.
func GemmQ4F32(m, n, k int, a *tensor.Q4Storage, b, c []float32) {
	af32 := make([]float32, a.Len())
	a.Dequantize(af32)
	SgemmSimd(m, n, k, af32, b, c)
}

// GemmQ8F32 computes C = A * B where A is Q8_0 quantized and B, C are float32.
// Dequantizes A once upfront, then uses SIMD-accelerated SGEMM.
func GemmQ8F32(m, n, k int, a *tensor.Q8Storage, b, c []float32) {
	af32 := make([]float32, a.Len())
	a.Dequantize(af32)
	SgemmSimd(m, n, k, af32, b, c)
}
