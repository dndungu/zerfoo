//go:build cuda

package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestGemmQ4F32_Correctness(t *testing.T) {
	// Small matrix: M=2, K=32, N=4.
	// K=32 so each row of A is exactly 1 Q4 block.
	M, K, N := 2, 32, 4

	// Create float32 source data for A.
	aF32 := make([]float32, M*K)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}

	// Quantize A to Q4.
	aQ4 := tensor.QuantizeQ4(aF32)
	aBytes := aQ4.RawBytes()

	// Create B matrix.
	bF32 := make([]float32, K*N)
	for i := range bF32 {
		bF32[i] = float32(i%5-2) * 0.1
	}

	// Compute reference: dequant(A) * B on CPU.
	aDequant := aQ4.Slice()
	ref := make([]float32, M*N)
	for i := range M {
		for j := range N {
			var sum float32
			for k := range K {
				sum += aDequant[i*K+k] * bF32[k*N+j]
			}
			ref[i*N+j] = sum
		}
	}

	// Allocate device memory.
	devA, err := cuda.Malloc(len(aBytes))
	if err != nil {
		t.Fatalf("cuda.Malloc A: %v", err)
	}
	defer cuda.Free(devA)

	devB, err := cuda.Malloc(K * N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc B: %v", err)
	}
	defer cuda.Free(devB)

	devC, err := cuda.Malloc(M * N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc C: %v", err)
	}
	defer cuda.Free(devC)

	// Copy H2D.
	if err := cuda.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), K*N*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	// Run kernel.
	if err := GemmQ4F32(devA, devB, devC, M, K, N, nil); err != nil {
		t.Fatalf("GemmQ4F32: %v", err)
	}

	// Sync.
	if err := cuda.DeviceSynchronize(); err != nil {
		t.Fatalf("DeviceSynchronize: %v", err)
	}

	// Copy D2H.
	got := make([]float32, M*N)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devC, M*N*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy C: %v", err)
	}

	// Compare with Q4 tolerance.
	const tol = 0.15
	for i := range got {
		if diff := math.Abs(float64(got[i] - ref[i])); diff > tol {
			t.Errorf("C[%d] = %f, want %f (diff %f > tol %f)", i, got[i], ref[i], diff, tol)
		}
	}
}

func TestGemmQ4F32_LargerMatrix(t *testing.T) {
	M, K, N := 64, 128, 64

	aF32 := make([]float32, M*K)
	for i := range aF32 {
		aF32[i] = float32(i%11-5) * 0.05
	}
	aQ4 := tensor.QuantizeQ4(aF32)
	aBytes := aQ4.RawBytes()
	aDequant := aQ4.Slice()

	bF32 := make([]float32, K*N)
	for i := range bF32 {
		bF32[i] = float32(i%9-4) * 0.05
	}

	// Reference.
	ref := make([]float32, M*N)
	for i := range M {
		for j := range N {
			var sum float32
			for k := range K {
				sum += aDequant[i*K+k] * bF32[k*N+j]
			}
			ref[i*N+j] = sum
		}
	}

	devA, err := cuda.Malloc(len(aBytes))
	if err != nil {
		t.Fatalf("cuda.Malloc A: %v", err)
	}
	defer cuda.Free(devA)

	devB, err := cuda.Malloc(K * N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc B: %v", err)
	}
	defer cuda.Free(devB)

	devC, err := cuda.Malloc(M * N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc C: %v", err)
	}
	defer cuda.Free(devC)

	if err := cuda.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), K*N*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	if err := GemmQ4F32(devA, devB, devC, M, K, N, nil); err != nil {
		t.Fatalf("GemmQ4F32: %v", err)
	}
	if err := cuda.DeviceSynchronize(); err != nil {
		t.Fatalf("DeviceSynchronize: %v", err)
	}

	got := make([]float32, M*N)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devC, M*N*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy C: %v", err)
	}

	const tol = 0.2
	maxDiff := 0.0
	for i := range got {
		diff := math.Abs(float64(got[i] - ref[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tol {
			t.Errorf("C[%d] = %f, want %f (diff %f)", i, got[i], ref[i], diff)
			if t.Failed() {
				break // Don't flood output.
			}
		}
	}
	t.Logf("max diff: %f", maxDiff)
}

func BenchmarkGemmQ4F32_1024(b *testing.B) {
	M, K, N := 1024, 1024, 1024

	aF32 := make([]float32, M*K)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.01
	}
	aQ4 := tensor.QuantizeQ4(aF32)
	aBytes := aQ4.RawBytes()

	bF32 := make([]float32, K*N)
	for i := range bF32 {
		bF32[i] = float32(i%5-2) * 0.01
	}

	devA, _ := cuda.Malloc(len(aBytes))
	defer cuda.Free(devA)
	devB, _ := cuda.Malloc(K * N * 4)
	defer cuda.Free(devB)
	devC, _ := cuda.Malloc(M * N * 4)
	defer cuda.Free(devC)

	_ = cuda.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), K*N*4, cuda.MemcpyHostToDevice)

	b.ResetTimer()
	for b.Loop() {
		_ = GemmQ4F32(devA, devB, devC, M, K, N, nil)
	}
	_ = cuda.DeviceSynchronize()

	elapsed := b.Elapsed()
	// Q4 GEMM effective FLOPS: 2*M*K*N per iteration.
	flops := 2.0 * float64(M) * float64(K) * float64(N) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}
