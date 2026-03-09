package xblas

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/tensor"
)

// TestQ4DotRow_MatchesDequantize compares the fast q4DotRow path
// (reads float16 scale from raw struct bytes) against the Dequantize path
// (uses Float16.ToFloat32()). These paths must produce identical results
// for model output correctness.
func TestQ4DotRow_MatchesDequantize(t *testing.T) {
	tests := []struct {
		name string
		k    int
	}{
		{"k=32", 32},
		{"k=64", 64},
		{"k=256", 256},
		{"k=1024", 1024},
		{"k=2304", 2304}, // Gemma 3 2B head dim * num_kv_heads
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create realistic weight-like data.
			weights := make([]float32, tt.k)
			for i := range weights {
				weights[i] = float32(i%13-6) * 0.01
			}

			q4 := tensor.QuantizeQ4(weights)

			// Create activation vector.
			x := make([]float32, tt.k)
			for i := range x {
				x[i] = float32(i%7-3) * 0.1
			}

			blocksPerRow := tt.k / 32

			// Path 1: q4DotRow (production path via BlockPtr).
			got := q4DotRow(unsafe.Pointer(q4.BlockPtr(0)), &x[0], blocksPerRow)

			// Path 2: Dequantize + manual dot (reference path).
			deq := make([]float32, tt.k)
			q4.Dequantize(deq)
			var want float32
			for i := range tt.k {
				want += deq[i] * x[i]
			}

			diff := float32(math.Abs(float64(got - want)))
			relErr := diff / (float32(math.Abs(float64(want))) + 1e-10)
			if relErr > 1e-4 {
				t.Errorf("q4DotRow=%f, dequant+dot=%f, diff=%f, relErr=%f", got, want, diff, relErr)
			}
		})
	}
}

// TestQ4DotRowScalar_MatchesDequantize is the same test but forces the scalar path.
func TestQ4DotRowScalar_MatchesDequantize(t *testing.T) {
	k := 256
	weights := make([]float32, k)
	for i := range weights {
		weights[i] = float32(i%13-6) * 0.01
	}

	q4 := tensor.QuantizeQ4(weights)

	x := make([]float32, k)
	for i := range x {
		x[i] = float32(i%7-3) * 0.1
	}

	blocksPerRow := k / 32

	got := q4DotRowScalar(unsafe.Pointer(q4.BlockPtr(0)), &x[0], blocksPerRow)

	deq := make([]float32, k)
	q4.Dequantize(deq)
	var want float32
	for i := range k {
		want += deq[i] * x[i]
	}

	diff := float32(math.Abs(float64(got - want)))
	relErr := diff / (float32(math.Abs(float64(want))) + 1e-10)
	if relErr > 1e-4 {
		t.Errorf("q4DotRowScalar=%f, dequant+dot=%f, diff=%f, relErr=%f", got, want, diff, relErr)
	}
}

// TestGemmF32Q4NT_MatchesDequantize compares the full GemmF32Q4NT
// against dequantize+transpose+regular SGEMM.
func TestGemmF32Q4NT_MatchesDequantize(t *testing.T) {
	m, n, k := 1, 64, 256 // typical decode: 1 token, 64 output, 256 inner dim

	// Create weight matrix [N, K] and quantize.
	bF32 := make([]float32, n*k)
	for i := range bF32 {
		bF32[i] = float32(i%17-8) * 0.005
	}
	bQ4 := tensor.QuantizeQ4(bF32)

	// Create activation matrix [M, K].
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(i%11-5) * 0.1
	}

	// Path 1: GemmF32Q4NT (production path).
	cFast := make([]float32, m*n)
	GemmF32Q4NT(m, n, k, a, bQ4, cFast)

	// Path 2: Dequantize B, transpose, regular SGEMM.
	bDeq := make([]float32, n*k)
	bQ4.Dequantize(bDeq)
	bT := make([]float32, k*n)
	for r := range n {
		for c := range k {
			bT[c*n+r] = bDeq[r*k+c]
		}
	}
	cRef := make([]float32, m*n)
	SgemmSimd(m, n, k, a, bT, cRef)

	// Compare.
	var maxDiff float32
	var maxRelErr float32
	for j := range m * n {
		diff := float32(math.Abs(float64(cFast[j] - cRef[j])))
		if diff > maxDiff {
			maxDiff = diff
		}
		denom := float32(math.Abs(float64(cRef[j]))) + 1e-10
		if re := diff / denom; re > maxRelErr {
			maxRelErr = re
		}
	}

	t.Logf("GemmF32Q4NT vs dequant+SGEMM: maxDiff=%f, maxRelErr=%f", maxDiff, maxRelErr)
	if maxRelErr > 1e-3 {
		// Print first few mismatches.
		for j := range min(10, m*n) {
			t.Logf("  c[%d]: fast=%f ref=%f", j, cFast[j], cRef[j])
		}
		t.Errorf("too much divergence: maxRelErr=%f", maxRelErr)
	}
}
