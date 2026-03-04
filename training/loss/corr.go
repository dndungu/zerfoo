package loss

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

// CorrLoss computes -PearsonCorrelation(predictions, targets) as a differentiable
// scalar loss. Minimizing this loss maximizes the Pearson correlation between
// predictions and targets. Since Numerai targets are rank-normalized, Pearson
// closely approximates Spearman rank correlation.
//
// Forward: loss = -sum(p_c * t_c) / (sqrt(sum(p_c^2) * sum(t_c^2)) + eps)
//
//	where p_c = p - mean(p), t_c = t - mean(t)
//
// Backward: grad_i = -(1/N) * (t_c_i / denom - corr * p_c_i / sum_pp) * dOut
type CorrLoss[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	predictions *tensor.TensorNumeric[T]
	targets     *tensor.TensorNumeric[T]
}

// NewCorrLoss creates a new correlation loss function.
func NewCorrLoss[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *CorrLoss[T] {
	return &CorrLoss[T]{engine: engine, ops: ops}
}

// Forward computes -PearsonCorrelation(predictions, targets).
func (c *CorrLoss[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("CorrLoss expects 2 inputs, got %d", len(inputs))
	}
	predictions := inputs[0]
	targets := inputs[1]
	c.predictions = predictions
	c.targets = targets

	pData := predictions.Data()
	tData := targets.Data()
	n := float64(len(pData))

	// Compute means
	var sumP, sumT float64
	for i := range pData {
		sumP += float64(pData[i])
		sumT += float64(tData[i])
	}
	meanP := sumP / n
	meanT := sumT / n

	// Compute centered dot products
	var sumPT, sumPP, sumTT float64
	for i := range pData {
		pc := float64(pData[i]) - meanP
		tc := float64(tData[i]) - meanT
		sumPT += pc * tc
		sumPP += pc * pc
		sumTT += tc * tc
	}

	eps := 1e-8
	denom := math.Sqrt(sumPP*sumTT) + eps
	corr := sumPT / denom
	loss := T(-corr)

	result, err := tensor.New[T]([]int{1}, []T{loss})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Backward computes the gradient of -PearsonCorrelation with respect to predictions.
// Returns [dPredictions, dTargets(zeros)].
func (c *CorrLoss[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	preds := c.predictions
	targs := c.targets
	if len(inputs) > 0 {
		preds = inputs[0]
		if len(inputs) > 1 {
			targs = inputs[1]
		}
	}
	if preds == nil || targs == nil {
		return nil, graph.ErrInvalidInputCount
	}

	pData := preds.Data()
	tData := targs.Data()
	scale := float64(dOut.Data()[0])

	// Compute means
	n := float64(len(pData))
	var sumP, sumT float64
	for i := range pData {
		sumP += float64(pData[i])
		sumT += float64(tData[i])
	}
	meanP := sumP / n
	meanT := sumT / n

	// Compute centered values and sums
	var sumPT, sumPP, sumTT float64
	pc := make([]float64, len(pData))
	tc := make([]float64, len(pData))
	for i := range pData {
		pc[i] = float64(pData[i]) - meanP
		tc[i] = float64(tData[i]) - meanT
		sumPT += pc[i] * tc[i]
		sumPP += pc[i] * pc[i]
		sumTT += tc[i] * tc[i]
	}

	eps := 1e-8
	denom := math.Sqrt(sumPP*sumTT) + eps
	corr := sumPT / denom

	// d(-corr)/d(p_i) = -(t_c_i / denom - corr * p_c_i / sumPP) * dOut
	gradData := make([]T, len(pData))
	sumPPEps := sumPP + eps
	for i := range pData {
		g := -(tc[i]/denom - corr*pc[i]/sumPPEps) * scale
		gradData[i] = T(g)
	}

	gradPred, err := tensor.New[T](preds.Shape(), gradData)
	if err != nil {
		return nil, err
	}

	zeroGrad, err := tensor.New[T](targs.Shape(), make([]T, len(tData)))
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{gradPred, zeroGrad}, nil
}

// OutputShape returns [1] (scalar loss).
func (c *CorrLoss[T]) OutputShape() []int {
	return []int{1}
}

// OpType returns "CorrLoss".
func (c *CorrLoss[T]) OpType() string {
	return "CorrLoss"
}

// Attributes returns nil (no configurable attributes).
func (c *CorrLoss[T]) Attributes() map[string]any {
	return nil
}

// Parameters returns nil (no trainable parameters).
func (c *CorrLoss[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Statically assert that CorrLoss implements graph.Node.
var _ graph.Node[float32] = (*CorrLoss[float32])(nil)
