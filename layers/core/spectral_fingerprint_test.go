package core_test

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
	"github.com/zerfoo/zerfoo/types"
)

func approxEqual(a, b, eps float32) bool {
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return false
	}

	return float32(math.Abs(float64(a-b))) <= eps
}

func TestSpectralFingerprint_ForwardSimple(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	window := 8
	topK := 3
	layer, err := core.NewSpectralFingerprint[float32](engine, ops, window, topK)
	testutils.AssertNoError(t, err, "expected no error creating SpectralFingerprint")

	// Batch=2
	// Row 0: pure cosine at k=1 => |X[1]| = W/2
	// Row 1: constant signal => only DC, bins 1..topK should be ~0
	batch := 2

	inData := make([]float32, batch*window)
	for n := range window {
		inData[n] = float32(math.Cos(2 * math.Pi * float64(n) / float64(window))) // row 0
	}

	for n := range window {
		inData[window+n] = 1.0 // row 1 constant
	}

	inTensor, err := tensor.New([]int{batch, window}, inData)
	testutils.AssertNoError(t, err, "create input tensor")

	out, err := layer.Forward(context.Background(), inTensor)
	testutils.AssertNoError(t, err, "forward should succeed")

	shape := out.Shape()
	testutils.AssertTrue(t, len(shape) == 2 && shape[0] == batch && shape[1] == topK, "unexpected output shape")

	outD := out.Data()
	expectedMag := float32(window) / 2.0 // |X[1]|
	// Row 0 expectations: bin1 ~= W/2, others near 0
	eps := float32(1e-4)
	testutils.AssertTrue(t, approxEqual(outD[0*topK+0], expectedMag, eps), "bin 1 magnitude mismatch")
	testutils.AssertTrue(t, approxEqual(outD[0*topK+1], 0, 1e-3), "bin 2 should be ~0")
	testutils.AssertTrue(t, approxEqual(outD[0*topK+2], 0, 1e-3), "bin 3 should be ~0")
	// Row 1 expectations: all ~0
	testutils.AssertTrue(t, approxEqual(outD[1*topK+0], 0, 1e-3), "row1 bin1 should be ~0")
	testutils.AssertTrue(t, approxEqual(outD[1*topK+1], 0, 1e-3), "row1 bin2 should be ~0")
	testutils.AssertTrue(t, approxEqual(outD[1*topK+2], 0, 1e-3), "row1 bin3 should be ~0")
}

func TestSpectralFingerprint_Backward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	layer, err := core.NewSpectralFingerprint[float32](engine, ops, 4, 2)
	testutils.AssertNoError(t, err, "create layer")

	in, _ := tensor.New([]int{1, 4}, []float32{1, 2, 3, 4})
	out, _ := layer.Forward(context.Background(), in)
	grad, _ := tensor.New(out.Shape(), make([]float32, out.Size()))

	inGrads, err := layer.Backward(context.Background(), types.FullBackprop, grad, in)
	testutils.AssertNoError(t, err, "backward should not error")
	testutils.AssertTrue(t, len(inGrads) == 1, "expected 1 gradient tensor for 1 input")
	// Gradient should be all zeros since spectral fingerprint is non-differentiable
	for _, v := range inGrads[0].Data() {
		testutils.AssertTrue(t, v == 0, "expected zero gradient for non-differentiable layer")
	}
}

func TestSpectralFingerprint_BuilderRegistry(t *testing.T) {
	// Ensure registration
	registry.RegisterAll()

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	builder, err := model.GetLayerBuilder[float32]("SpectralFingerprint")
	testutils.AssertNoError(t, err, "get builder")

	attrs := map[string]interface{}{"window": 8, "top_k": 2}
	n, err := builder(engine, ops, "sf", nil, attrs)
	testutils.AssertNoError(t, err, "builder should construct node")
	testutils.AssertTrue(t, n.OpType() == "SpectralFingerprint", "op type mismatch")
}

// TestSpectralFingerprint_Parity verifies engine.MatMul-based Forward produces
// outputs within 1e-5 of hand-computed DFT magnitudes.
func TestSpectralFingerprint_Parity(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := compute.NewCPUEngine[float32](ops)
	const eps float32 = 1e-5

	tests := []struct {
		name   string
		window int
		topK   int
		input  []float32 // flat [batch, window]
		batch  int
	}{
		{
			name:   "pure_cos_k1",
			window: 8,
			topK:   3,
			batch:  1,
			input: func() []float32 {
				d := make([]float32, 8)
				for n := range 8 {
					d[n] = float32(math.Cos(2 * math.Pi * float64(n) / 8))
				}
				return d
			}(),
		},
		{
			name:   "constant_signal",
			window: 8,
			topK:   3,
			batch:  1,
			input:  []float32{1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name:   "batch2_mixed",
			window: 4,
			topK:   2,
			batch:  2,
			input:  []float32{1, 0, -1, 0, 0, 0, 0, 0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			layer, err := core.NewSpectralFingerprint[float32](eng, ops, tc.window, tc.topK)
			if err != nil {
				t.Fatalf("create layer: %v", err)
			}

			in, err := tensor.New([]int{tc.batch, tc.window}, tc.input)
			if err != nil {
				t.Fatalf("create input: %v", err)
			}

			out, err := layer.Forward(context.Background(), in)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}

			outData := out.Data()

			// Compute reference DFT magnitudes.
			idx := 0
			for b := range tc.batch {
				for k := 1; k <= tc.topK; k++ {
					var ref float32
					if k < tc.window {
						var re, im float64
						for n := range tc.window {
							angle := -2 * math.Pi * float64(k) * float64(n) / float64(tc.window)
							x := float64(tc.input[b*tc.window+n])
							re += x * math.Cos(angle)
							im += x * math.Sin(angle)
						}
						ref = float32(math.Sqrt(re*re + im*im))
					}
					got := outData[idx]
					if !approxEqual(got, ref, eps) {
						t.Errorf("batch %d bin %d: got %v want %v", b, k, got, ref)
					}
					idx++
				}
			}
		})
	}
}

// Statically assert that the type implements the graph.Node interface.
