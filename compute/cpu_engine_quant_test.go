package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestCPUEngine_MatMul_QuantizedStorage(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	m, k, n := 2, 32, 3

	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	bData := make([]float32, k*n)
	for i := range bData {
		bData[i] = float32(i%5-2) * 0.1
	}
	b, err := tensor.New[float32]([]int{k, n}, bData)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		stor   tensor.Storage[float32]
		maxErr float32
	}{
		{"Q4_0", tensor.QuantizeQ4(aF32), 0.15},
		{"Q8_0", tensor.QuantizeQ8(aF32), 0.02},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := tensor.NewWithStorage([]int{m, k}, tt.stor)
			if err != nil {
				t.Fatal(err)
			}

			result, err := engine.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("MatMul failed: %v", err)
			}

			if result.Shape()[0] != m || result.Shape()[1] != n {
				t.Errorf("shape = %v, want [%d %d]", result.Shape(), m, n)
			}

			// Reference: dequantize then float32 GEMM.
			refA, _ := tensor.New[float32]([]int{m, k}, tt.stor.Slice())
			refResult, err := engine.MatMul(ctx, refA, b)
			if err != nil {
				t.Fatal(err)
			}

			got := result.Data()
			want := refResult.Data()
			for i := range got {
				diff := float32(math.Abs(float64(got[i] - want[i])))
				if diff > tt.maxErr {
					t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
				}
			}
		})
	}
}

func TestCPUEngine_MatMul_Q4Storage_Batched(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	batch, m, k, n := 2, 1, 32, 4

	aF32 := make([]float32, batch*m*k)
	for i := range aF32 {
		aF32[i] = float32(i%9-4) * 0.05
	}
	q4 := tensor.QuantizeQ4(aF32)
	a, err := tensor.NewWithStorage([]int{batch, m, k}, q4)
	if err != nil {
		t.Fatal(err)
	}

	bData := make([]float32, k*n)
	for i := range bData {
		bData[i] = float32(i%5-2) * 0.1
	}
	b, err := tensor.New[float32]([]int{k, n}, bData)
	if err != nil {
		t.Fatal(err)
	}

	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("Batched MatMul with Q4 failed: %v", err)
	}

	shape := result.Shape()
	if shape[0] != batch || shape[1] != m || shape[2] != n {
		t.Errorf("output shape = %v, want [%d %d %d]", shape, batch, m, n)
	}
}
