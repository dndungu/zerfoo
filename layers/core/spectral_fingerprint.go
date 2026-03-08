// Package core provides core neural network layer implementations.
package core

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// SpectralFingerprint computes FFT/DFT magnitude features over a fixed window.
//
// Input shape:  [batch, window]
// Output shape: [batch, topK] where bins 1..topK (non-DC) magnitudes are returned
// (if topK >= window, bins beyond window-1 are returned as zeros).
//
// The layer is stateless and intended primarily for feature engineering. We do
// not propagate gradients through this transformation (Backward returns nil),
// treating it as a non-differentiable pre-processing step.
//
// For generalization to higher ranks or different axes, extend this layer as
// needed. This initial implementation focuses on the common case of a 2D input
// with a single spectral axis equal to `window`.
//
// Note: Computation is performed in the numeric type T using provided ops.
// Cos/Sin are computed in float64 then converted into T using ops.FromFloat64.
// Magnitude is computed as sqrt(re^2 + im^2) in T using ops.
//
// OpType: "SpectralFingerprint"
// Attributes: {"window": int, "top_k": int}
// Parameters: none
//
// Example
//   in:  [N, W]
//   out: [N, K] with out[:, k-1] = |DFT(series)[k]| for k in 1..K
//
// DFT definition used:
//   X[k] = sum_{n=0}^{W-1} x[n] * exp(-j*2*pi*k*n/W)
//   |X[k]| = sqrt(Re^2 + Im^2)
//
// where we explicitly compute cos/sin components.

// SpectralFingerprint implements a layer that extracts the top-K frequency components
// from the input using FFT, creating a fixed-size spectral signature for variable-length inputs.
type SpectralFingerprint[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	window      int
	topK        int
	outputShape []int

	// Precomputed Fourier basis matrices of shape [window, topK].
	cosBasis *tensor.TensorNumeric[T]
	sinBasis *tensor.TensorNumeric[T]
}

// NewSpectralFingerprint creates a new SpectralFingerprint layer.
func NewSpectralFingerprint[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], window, topK int) (*SpectralFingerprint[T], error) {
	if window <= 1 {
		return nil, fmt.Errorf("window must be > 1, got %d", window)
	}

	if topK <= 0 {
		return nil, fmt.Errorf("topK must be > 0, got %d", topK)
	}

	// Precompute Fourier basis matrices [window, topK].
	// Column k-1 corresponds to DFT bin k (1-indexed).
	cosData := make([]T, window*topK)
	sinData := make([]T, window*topK)
	for k := 1; k <= topK; k++ {
		col := k - 1
		if k >= window {
			// Bins beyond window-1 are zero (already zero-initialized).
			continue
		}
		for n := range window {
			angle := -2 * math.Pi * float64(k) * float64(n) / float64(window)
			cosData[n*topK+col] = ops.FromFloat64(math.Cos(angle))
			sinData[n*topK+col] = ops.FromFloat64(math.Sin(angle))
		}
	}

	cosBasis, err := tensor.New([]int{window, topK}, cosData)
	if err != nil {
		return nil, fmt.Errorf("failed to create cos basis: %w", err)
	}
	sinBasis, err := tensor.New([]int{window, topK}, sinData)
	if err != nil {
		return nil, fmt.Errorf("failed to create sin basis: %w", err)
	}

	return &SpectralFingerprint[T]{
		engine:   engine,
		ops:      ops,
		window:   window,
		topK:     topK,
		cosBasis: cosBasis,
		sinBasis: sinBasis,
	}, nil
}

// OutputShape returns the last computed output shape.
func (s *SpectralFingerprint[T]) OutputShape() []int { return s.outputShape }

// Parameters returns no trainable parameters for SpectralFingerprint.
func (s *SpectralFingerprint[T]) Parameters() []*graph.Parameter[T] { return nil }

// OpType returns the operation type.
func (s *SpectralFingerprint[T]) OpType() string { return "SpectralFingerprint" }

// Attributes returns the attributes of the layer.
func (s *SpectralFingerprint[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"window": s.window,
		"top_k":  s.topK,
	}
}

// Forward computes spectral magnitudes for bins 1..topK for each row in the batch.
// Input must be [batch, window]. If input window dimension is larger than the configured
// window, only the last `window` elements are used. If smaller, an error is returned.
func (s *SpectralFingerprint[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SpectralFingerprint expects exactly 1 input, got %d", len(inputs))
	}

	in := inputs[0]

	shape := in.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("input must be 2D [batch, window], got shape %v", shape)
	}

	batch := shape[0]
	w := shape[1]
	if w < s.window {
		return nil, fmt.Errorf("input window (%d) smaller than configured window (%d)", w, s.window)
	}

	// If input is wider than configured window, slice to the last s.window columns.
	x := in
	if w > s.window {
		start := w - s.window
		inData := in.Data()
		sliceData := make([]T, batch*s.window)
		for b := range batch {
			copy(sliceData[b*s.window:], inData[b*w+start:b*w+start+s.window])
		}
		var err error
		x, err = tensor.New([]int{batch, s.window}, sliceData)
		if err != nil {
			return nil, fmt.Errorf("failed to slice input window: %w", err)
		}
	}

	// re = x @ cosBasis  [batch, topK]
	re, err := s.engine.MatMul(ctx, x, s.cosBasis)
	if err != nil {
		return nil, fmt.Errorf("failed to compute real DFT: %w", err)
	}

	// im = x @ sinBasis  [batch, topK]
	im, err := s.engine.MatMul(ctx, x, s.sinBasis)
	if err != nil {
		return nil, fmt.Errorf("failed to compute imaginary DFT: %w", err)
	}

	// mag = sqrt(re^2 + im^2)
	re2, err := s.engine.Mul(ctx, re, re)
	if err != nil {
		return nil, fmt.Errorf("failed to compute re^2: %w", err)
	}
	im2, err := s.engine.Mul(ctx, im, im)
	if err != nil {
		return nil, fmt.Errorf("failed to compute im^2: %w", err)
	}
	sum, err := s.engine.Add(ctx, re2, im2)
	if err != nil {
		return nil, fmt.Errorf("failed to compute re^2 + im^2: %w", err)
	}
	out, err := s.engine.Sqrt(ctx, sum)
	if err != nil {
		return nil, fmt.Errorf("failed to compute sqrt: %w", err)
	}

	s.outputShape = out.Shape()

	return out, nil
}

// Backward returns no gradients (treated as non-differentiable feature transform).
func (s *SpectralFingerprint[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) == 0 {
		return []*tensor.TensorNumeric[T]{}, nil
	}
	shape := inputs[0].Shape()
	size := 1
	for _, d := range shape {
		size *= d
	}
	zeroGrad, err := tensor.New[T](shape, make([]T, size))
	if err != nil {
		return nil, fmt.Errorf("spectral fingerprint backward: %w", err)
	}
	return []*tensor.TensorNumeric[T]{zeroGrad}, nil
}

// Statically assert that the type implements the graph.Node interface.
