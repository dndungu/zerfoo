package embeddings

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// RotaryPositionalEmbedding applies Rotary Positional Embedding to a tensor.
type RotaryPositionalEmbedding[T tensor.Numeric] struct {
	engine    compute.Engine[T]
	headDim   int
	cosAngles *tensor.TensorNumeric[T]
	sinAngles *tensor.TensorNumeric[T]
	// Cached input for backward pass
	inputShape  []int
	xRot0Slice  *tensor.TensorNumeric[T]
	xRot1Slice  *tensor.TensorNumeric[T]
	outputShape []int
	// attnScaleFactor is the YaRN attention scaling factor (1.0 when no YaRN).
	attnScaleFactor float64
}

// RotaryPositionalEmbeddingOptions holds configuration options for RotaryPositionalEmbedding layers.
type RotaryPositionalEmbeddingOptions struct {
	Base       float64 // Base for the inverse frequency calculation (theta parameter)
	YaRN       bool    // Whether to apply YaRN scaling
	YaRNFactor float64 // YaRN scaling factor (e.g. 4.0 for 4x context extension)
	YaRNOrigML int     // Original max sequence length before scaling
}

// RotaryPositionalEmbeddingOption is a functional option for configuring RotaryPositionalEmbedding layers.
type RotaryPositionalEmbeddingOption func(*RotaryPositionalEmbeddingOptions)

// WithRotaryBase sets the base (theta) parameter for the inverse frequency calculation.
func WithRotaryBase(base float64) RotaryPositionalEmbeddingOption {
	return func(opts *RotaryPositionalEmbeddingOptions) {
		opts.Base = base
	}
}

// WithYaRNScaling enables YaRN (Yet another RoPE extensioN) scaling.
// factor is the context extension factor (e.g. 4.0 for 4x).
// origMaxLen is the original maximum sequence length before scaling.
func WithYaRNScaling(factor float64, origMaxLen int) RotaryPositionalEmbeddingOption {
	return func(opts *RotaryPositionalEmbeddingOptions) {
		opts.YaRN = true
		opts.YaRNFactor = factor
		opts.YaRNOrigML = origMaxLen
	}
}

// NewRotaryPositionalEmbedding creates a new RotaryPositionalEmbedding layer.
// headDim: The dimension of the head. Must be even.
// seqLen: The maximum sequence length this embedding will be applied to.
// engine: The compute engine to use for tensor operations.
func NewRotaryPositionalEmbedding[T tensor.Numeric](
	ctx context.Context,
	engine compute.Engine[T],
	headDim int,
	seqLen int,
	options ...RotaryPositionalEmbeddingOption,
) (*RotaryPositionalEmbedding[T], error) {
	if headDim%2 != 0 {
		return nil, fmt.Errorf("head dimension (%d) must be even for RoPE", headDim)
	}

	// Apply functional options
	opts := &RotaryPositionalEmbeddingOptions{
		Base: 10000.0, // Default base value (theta)
	}
	for _, option := range options {
		option(opts)
	}

	// Create position indices: [0, 1, ..., seq_len-1]
	positions := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		positions[i] = i
	}

	// Create inverse frequencies: 1 / (base^(2i/head_dim))
	ops := engine.Ops()
	halfDim := headDim / 2
	invFreqs64 := make([]float64, halfDim)
	for i := 0; i < halfDim; i++ {
		invFreqs64[i] = 1.0 / math.Pow(opts.Base, float64(2*i)/float64(headDim))
	}

	// Apply YaRN scaling to inverse frequencies if enabled.
	attnScaleFactor := 1.0
	if opts.YaRN {
		attnScaleFactor = math.Sqrt(1 + math.Log(opts.YaRNFactor)/math.Log(float64(opts.YaRNOrigML)))
		origML := float64(opts.YaRNOrigML)
		for i := 0; i < halfDim; i++ {
			wavelength := 2 * math.Pi / invFreqs64[i]
			if wavelength > opts.YaRNFactor*origML {
				// Low frequency: scale by 1/factor
				invFreqs64[i] /= opts.YaRNFactor
			} else if wavelength >= origML {
				// Intermediate frequency: linearly interpolate
				ratio := (wavelength - origML) / (opts.YaRNFactor*origML - origML)
				invFreqs64[i] = invFreqs64[i] * (1 - ratio) + invFreqs64[i]/opts.YaRNFactor*ratio
			}
			// High frequency (wavelength < origMaxLen): keep unchanged
		}
	}

	// Precompute cos and sin of angles using float64 and convert to T
	size := seqLen * halfDim
	cosData := make([]T, size)
	sinData := make([]T, size)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < halfDim; j++ {
			angle := float64(positions[i]) * invFreqs64[j]
			idx := i*halfDim + j
			cosData[idx] = ops.FromFloat64(math.Cos(angle))
			sinData[idx] = ops.FromFloat64(math.Sin(angle))
		}
	}

	cosAngles, err := tensor.New[T]([]int{seqLen, halfDim}, cosData)
	if err != nil {
		return nil, err
	}

	sinAngles, err := tensor.New[T]([]int{seqLen, halfDim}, sinData)
	if err != nil {
		return nil, err
	}

	return &RotaryPositionalEmbedding[T]{
		engine:          engine,
		headDim:         headDim,
		cosAngles:       cosAngles,
		sinAngles:       sinAngles,
		attnScaleFactor: attnScaleFactor,
	}, nil
}

// OutputShape returns the output shape of the RoPE layer.
func (rpe *RotaryPositionalEmbedding[T]) OutputShape() []int {
	return rpe.outputShape
}

// Parameters returns no trainable parameters for RoPE.
func (rpe *RotaryPositionalEmbedding[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward applies Rotary Positional Embedding to the input tensor.
func (rpe *RotaryPositionalEmbedding[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("RotaryPositionalEmbedding expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	rpe.inputShape = input.Shape()

	rpe.outputShape = input.Shape()
	if len(rpe.inputShape) < 2 {
		return nil, fmt.Errorf("input tensor must have at least 2 dimensions, got %d", len(rpe.inputShape))
	}

	seqLen := rpe.inputShape[1]

	// Slice cos and sin angles to match the input sequence length
	cosAngles, err := rpe.cosAngles.Slice([2]int{0, seqLen}, [2]int{0, rpe.headDim / 2})
	if err != nil {
		return nil, err
	}

	sinAngles, err := rpe.sinAngles.Slice([2]int{0, seqLen}, [2]int{0, rpe.headDim / 2})
	if err != nil {
		return nil, err
	}

	// Split input into two halves: x_rot0, x_rot1
	rpe.xRot0Slice, err = input.Slice([2]int{0, rpe.inputShape[0]}, [2]int{0, seqLen}, [2]int{0, rpe.headDim / 2})
	if err != nil {
		return nil, err
	}

	rpe.xRot1Slice, err = input.Slice([2]int{0, rpe.inputShape[0]}, [2]int{0, seqLen}, [2]int{rpe.headDim / 2, rpe.headDim})
	if err != nil {
		return nil, err
	}

	// Apply rotation:
	// x_rot0 * cos(angles) - x_rot1 * sin(angles)
	// x_rot1 * cos(angles) + x_rot0 * sin(angles)

	// Term 1: x_rot0 * cos(angles)
	term1, err := rpe.engine.Mul(ctx, rpe.xRot0Slice, cosAngles) // Broadcasting cosAngles
	if err != nil {
		return nil, err
	}

	// Term 2: x_rot1 * sin(angles)
	term2, err := rpe.engine.Mul(ctx, rpe.xRot1Slice, sinAngles) // Broadcasting sinAngles
	if err != nil {
		return nil, err
	}

	// rotated_x0 = term1 - term2
	rotatedX0, err := rpe.engine.Sub(ctx, term1, term2)
	if err != nil {
		return nil, err
	}

	// rotated_x1 = x_rot1 * cos(angles) + x_rot0 * sin(angles)
	mul1, err := rpe.engine.Mul(ctx, rpe.xRot1Slice, cosAngles)
	if err != nil {
		return nil, err
	}

	mul2, err := rpe.engine.Mul(ctx, rpe.xRot0Slice, sinAngles)
	if err != nil {
		return nil, err
	}

	rotatedX1, err := rpe.engine.Add(ctx, mul1, mul2)
	if err != nil {
		return nil, err
	}

	// Concatenate rotated halves
	output, err := rpe.engine.Concat(ctx, []*tensor.TensorNumeric[T]{rotatedX0, rotatedX1}, 2)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for RoPE.
// Shapes are derived from dOut so that a single RoPE instance can be shared
// across Q and K paths whose batch dimensions differ.
func (rpe *RotaryPositionalEmbedding[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	dShape := dOut.Shape()
	batchDim := dShape[0]
	seqLen := dShape[1]

	// Slice cos and sin angles to match the input sequence length
	cosAngles, err := rpe.cosAngles.Slice([2]int{0, seqLen}, [2]int{0, rpe.headDim / 2})
	if err != nil {
		return nil, err
	}

	sinAngles, err := rpe.sinAngles.Slice([2]int{0, seqLen}, [2]int{0, rpe.headDim / 2})
	if err != nil {
		return nil, err
	}

	// Split dOut into d_rotated_x0, d_rotated_x1
	dRotatedX0, err := dOut.Slice([2]int{0, batchDim}, [2]int{0, seqLen}, [2]int{0, rpe.headDim / 2})
	if err != nil {
		return nil, err
	}

	dRotatedX1, err := dOut.Slice([2]int{0, batchDim}, [2]int{0, seqLen}, [2]int{rpe.headDim / 2, rpe.headDim})
	if err != nil {
		return nil, err
	}

	// Gradients for x_rot0 and x_rot1
	// dL/dx_rot0 = d_rotated_x0 * cos(angles) + d_rotated_x1 * sin(angles)
	mul1, err := rpe.engine.Mul(ctx, dRotatedX0, cosAngles)
	if err != nil {
		return nil, err
	}

	mul2, err := rpe.engine.Mul(ctx, dRotatedX1, sinAngles)
	if err != nil {
		return nil, err
	}

	dLdxRot0, err := rpe.engine.Add(ctx, mul1, mul2)
	if err != nil {
		return nil, err
	}

	// dL/dx_rot1 = d_rotated_x1 * cos(angles) - d_rotated_x0 * sin(angles)
	mul3, err := rpe.engine.Mul(ctx, dRotatedX1, cosAngles)
	if err != nil {
		return nil, err
	}

	mul4, err := rpe.engine.Mul(ctx, dRotatedX0, sinAngles)
	if err != nil {
		return nil, err
	}

	dLdxRot1, err := rpe.engine.Sub(ctx, mul3, mul4)
	if err != nil {
		return nil, err
	}

	// Concatenate gradients for x_rot0 and x_rot1
	dInput, err := rpe.engine.Concat(ctx, []*tensor.TensorNumeric[T]{dLdxRot0, dLdxRot1}, 2)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// OpType returns the operation type of the RotaryPositionalEmbedding layer.
func (rpe *RotaryPositionalEmbedding[T]) OpType() string {
	return "RotaryPositionalEmbedding"
}

// Attributes returns the attributes of the RotaryPositionalEmbedding layer.
func (rpe *RotaryPositionalEmbedding[T]) Attributes() map[string]interface{} {
	return nil
}

// AttentionScaleFactor returns the YaRN attention scaling factor.
// Returns 1.0 when YaRN is not enabled.
func (rpe *RotaryPositionalEmbedding[T]) AttentionScaleFactor() float64 {
	if rpe.attnScaleFactor == 0 {
		return 1.0
	}
	return rpe.attnScaleFactor
}

// Scale scales the positional embeddings by a given factor.
func (rpe *RotaryPositionalEmbedding[T]) Scale(ctx context.Context, factor float64) error {
	ops := rpe.engine.Ops()
	scaledCos, err := rpe.engine.MulScalar(ctx, rpe.cosAngles, ops.FromFloat64(factor), nil)
	if err != nil {
		return err
	}
	rpe.cosAngles = scaledCos

	scaledSin, err := rpe.engine.MulScalar(ctx, rpe.sinAngles, ops.FromFloat64(factor), nil)
	if err != nil {
		return err
	}
	rpe.sinAngles = scaledSin

	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*RotaryPositionalEmbedding[float32])(nil)
