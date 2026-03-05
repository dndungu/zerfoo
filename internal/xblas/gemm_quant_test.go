package xblas

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

type quantGemmTestCase struct {
	name    string
	m, n, k int
	maxErr  float32
}

func makeTestInputs(m, k, n int) ([]float32, []float32) {
	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	b := make([]float32, k*n)
	for i := range b {
		b[i] = float32(i%5-2) * 0.1
	}
	return aF32, b
}

func assertClose(t *testing.T, got, want []float32, maxErr float32) {
	t.Helper()
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		if diff > maxErr {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

var gemmSizes = []quantGemmTestCase{
	{"1x1x32", 1, 1, 32, 0},
	{"2x2x32", 2, 2, 32, 0},
	{"4x4x64", 4, 4, 64, 0},
	{"8x8x128", 8, 8, 128, 0},
}

func TestGemmQ4F32_Correctness(t *testing.T) {
	for _, tt := range gemmSizes {
		tt.maxErr = 0.15
		t.Run(tt.name, func(t *testing.T) {
			aF32, b := makeTestInputs(tt.m, tt.k, tt.n)
			aQ4 := tensor.QuantizeQ4(aF32)

			got := make([]float32, tt.m*tt.n)
			GemmQ4F32(tt.m, tt.n, tt.k, aQ4, b, got)

			// Reference: dequantize then float32 GEMM.
			af32Full := make([]float32, tt.m*tt.k)
			aQ4.Dequantize(af32Full)
			want := make([]float32, tt.m*tt.n)
			GemmF32(tt.m, tt.n, tt.k, af32Full, b, want)

			assertClose(t, got, want, tt.maxErr)
		})
	}
}

func TestGemmQ8F32_Correctness(t *testing.T) {
	for _, tt := range gemmSizes {
		tt.maxErr = 0.02
		t.Run(tt.name, func(t *testing.T) {
			aF32, b := makeTestInputs(tt.m, tt.k, tt.n)
			aQ8 := tensor.QuantizeQ8(aF32)

			got := make([]float32, tt.m*tt.n)
			GemmQ8F32(tt.m, tt.n, tt.k, aQ8, b, got)

			af32Full := make([]float32, tt.m*tt.k)
			aQ8.Dequantize(af32Full)
			want := make([]float32, tt.m*tt.n)
			GemmF32(tt.m, tt.n, tt.k, af32Full, b, want)

			assertClose(t, got, want, tt.maxErr)
		})
	}
}

func BenchmarkGemmQ4F32(b *testing.B) {
	for _, size := range []int{512, 1024} {
		b.Run(benchLabel(size), func(b *testing.B) {
			aF32, bf32 := makeTestInputs(size, size, size)
			aQ4 := tensor.QuantizeQ4(aF32)
			c := make([]float32, size*size)

			b.ResetTimer()
			for range b.N {
				GemmQ4F32(size, size, size, aQ4, bf32, c)
			}
		})
	}
}

func BenchmarkGemmQ8F32(b *testing.B) {
	for _, size := range []int{512, 1024} {
		b.Run(benchLabel(size), func(b *testing.B) {
			aF32, bf32 := makeTestInputs(size, size, size)
			aQ8 := tensor.QuantizeQ8(aF32)
			c := make([]float32, size*size)

			b.ResetTimer()
			for range b.N {
				GemmQ8F32(size, size, size, aQ8, bf32, c)
			}
		})
	}
}

func benchLabel(n int) string {
	switch {
	case n >= 1024:
		return "1k"
	default:
		return "512"
	}
}
