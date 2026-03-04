//go:build rocm

package miopen

/*
#cgo LDFLAGS: -lMIOpen
#include <miopen/miopen.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// statusError converts a MIOpen status to a Go error, or nil on success.
func statusError(status C.miopenStatus_t, context string) error {
	if status == C.miopenStatusSuccess {
		return nil
	}
	return fmt.Errorf("%s: MIOpen status %d", context, int(status))
}

// --- Data types and enums ---

// DataType mirrors miopenDataType_t.
type DataType int

const (
	Float32 DataType = C.miopenFloat
	Float16 DataType = C.miopenHalf
)

// TensorLayout selects NCHW or NHWC.
type TensorLayout int

const (
	NCHW TensorLayout = 0
	NHWC TensorLayout = 1
)

// ActivationMode mirrors miopenActivationMode_t.
type ActivationMode int

const (
	ActivationReLU    ActivationMode = C.miopenActivationRELU
	ActivationSigmoid ActivationMode = C.miopenActivationLOGISTIC
	ActivationTanh    ActivationMode = C.miopenActivationTANH
	ActivationELU     ActivationMode = C.miopenActivationELU
)

// PoolingMode mirrors miopenPoolingMode_t.
type PoolingMode int

const (
	PoolingMax                     PoolingMode = C.miopenPoolingMax
	PoolingAverageCountIncludePad  PoolingMode = C.miopenPoolingAverageInclusive
	PoolingAverageCountExcludePad  PoolingMode = C.miopenPoolingAverage
)

// BatchNormMode mirrors miopenBatchNormMode_t.
type BatchNormMode int

const (
	BatchNormPerActivation BatchNormMode = C.miopenBNPerActivation
	BatchNormSpatial       BatchNormMode = C.miopenBNSpatial
)

// ConvMode mirrors miopenConvolutionMode_t.
type ConvMode int

const (
	ConvolutionMode ConvMode = C.miopenConvolution
	TransposeMode   ConvMode = C.miopenTranspose
)

// SoftmaxAlgorithm selects the softmax computation variant.
type SoftmaxAlgorithm int

const (
	SoftmaxAccurate SoftmaxAlgorithm = C.MIOPEN_SOFTMAX_ACCURATE
	SoftmaxLog      SoftmaxAlgorithm = C.MIOPEN_SOFTMAX_LOG
	SoftmaxFast     SoftmaxAlgorithm = C.MIOPEN_SOFTMAX_FAST
)

// SoftmaxMode selects the dimension for softmax.
type SoftmaxMode int

const (
	SoftmaxModeChannel  SoftmaxMode = C.MIOPEN_SOFTMAX_MODE_CHANNEL
	SoftmaxModeInstance SoftmaxMode = C.MIOPEN_SOFTMAX_MODE_INSTANCE
)

// --- Handle ---

// Handle wraps a miopenHandle_t.
type Handle struct {
	h C.miopenHandle_t
}

// CreateHandle creates a new MIOpen handle.
func CreateHandle() (*Handle, error) {
	var h C.miopenHandle_t
	if err := statusError(C.miopenCreate(&h), "miopenCreate"); err != nil {
		return nil, err
	}
	return &Handle{h: h}, nil
}

// SetStream associates a HIP stream with this MIOpen handle.
func (h *Handle) SetStream(streamPtr unsafe.Pointer) error {
	return statusError(C.miopenSetStream(h.h, C.hipStream_t(streamPtr)), "miopenSetStream")
}

// Destroy releases the MIOpen handle.
func (h *Handle) Destroy() error {
	return statusError(C.miopenDestroy(h.h), "miopenDestroy")
}

// --- Tensor Descriptor ---

// TensorDescriptor wraps miopenTensorDescriptor_t.
type TensorDescriptor struct {
	d C.miopenTensorDescriptor_t
}

// CreateTensorDescriptor creates a new tensor descriptor.
func CreateTensorDescriptor() (*TensorDescriptor, error) {
	var d C.miopenTensorDescriptor_t
	if err := statusError(C.miopenCreateTensorDescriptor(&d), "miopenCreateTensorDescriptor"); err != nil {
		return nil, err
	}
	return &TensorDescriptor{d: d}, nil
}

// Set4d configures a 4D NCHW tensor descriptor.
func (t *TensorDescriptor) Set4d(dt DataType, n, c, h, w int) error {
	return statusError(
		C.miopenSet4dTensorDescriptor(t.d, C.miopenDataType_t(dt), C.int(n), C.int(c), C.int(h), C.int(w)),
		"miopenSet4dTensorDescriptor",
	)
}

// Destroy releases the tensor descriptor.
func (t *TensorDescriptor) Destroy() error {
	return statusError(C.miopenDestroyTensorDescriptor(t.d), "miopenDestroyTensorDescriptor")
}

// --- Convolution Descriptor ---

// ConvolutionDescriptor wraps miopenConvolutionDescriptor_t.
type ConvolutionDescriptor struct {
	d C.miopenConvolutionDescriptor_t
}

// CreateConvolutionDescriptor creates a new convolution descriptor.
func CreateConvolutionDescriptor() (*ConvolutionDescriptor, error) {
	var d C.miopenConvolutionDescriptor_t
	if err := statusError(C.miopenCreateConvolutionDescriptor(&d), "miopenCreateConvolutionDescriptor"); err != nil {
		return nil, err
	}
	return &ConvolutionDescriptor{d: d}, nil
}

// Set2d configures a 2D convolution descriptor.
func (c *ConvolutionDescriptor) Set2d(padH, padW, strH, strW, dilH, dilW int, mode ConvMode) error {
	return statusError(
		C.miopenInitConvolutionDescriptor(c.d, C.miopenConvolutionMode_t(mode),
			C.int(padH), C.int(padW), C.int(strH), C.int(strW), C.int(dilH), C.int(dilW)),
		"miopenInitConvolutionDescriptor",
	)
}

// SetGroupCount sets the group count for grouped convolutions.
func (c *ConvolutionDescriptor) SetGroupCount(groups int) error {
	return statusError(
		C.miopenSetConvolutionGroupCount(c.d, C.int(groups)),
		"miopenSetConvolutionGroupCount",
	)
}

// Destroy releases the convolution descriptor.
func (c *ConvolutionDescriptor) Destroy() error {
	return statusError(C.miopenDestroyConvolutionDescriptor(c.d), "miopenDestroyConvolutionDescriptor")
}

// --- Activation Descriptor ---

// ActivationDescriptor wraps miopenActivationDescriptor_t.
type ActivationDescriptor struct {
	d C.miopenActivationDescriptor_t
}

// CreateActivationDescriptor creates a new activation descriptor.
func CreateActivationDescriptor() (*ActivationDescriptor, error) {
	var d C.miopenActivationDescriptor_t
	if err := statusError(C.miopenCreateActivationDescriptor(&d), "miopenCreateActivationDescriptor"); err != nil {
		return nil, err
	}
	return &ActivationDescriptor{d: d}, nil
}

// Set configures the activation descriptor.
func (a *ActivationDescriptor) Set(mode ActivationMode, alpha, beta, gamma float64) error {
	return statusError(
		C.miopenSetActivationDescriptor(a.d, C.miopenActivationMode_t(mode),
			C.double(alpha), C.double(beta), C.double(gamma)),
		"miopenSetActivationDescriptor",
	)
}

// Destroy releases the activation descriptor.
func (a *ActivationDescriptor) Destroy() error {
	return statusError(C.miopenDestroyActivationDescriptor(a.d), "miopenDestroyActivationDescriptor")
}

// --- Pooling Descriptor ---

// PoolingDescriptor wraps miopenPoolingDescriptor_t.
type PoolingDescriptor struct {
	d C.miopenPoolingDescriptor_t
}

// CreatePoolingDescriptor creates a new pooling descriptor.
func CreatePoolingDescriptor() (*PoolingDescriptor, error) {
	var d C.miopenPoolingDescriptor_t
	if err := statusError(C.miopenCreatePoolingDescriptor(&d), "miopenCreatePoolingDescriptor"); err != nil {
		return nil, err
	}
	return &PoolingDescriptor{d: d}, nil
}

// Set2d configures a 2D pooling descriptor.
func (p *PoolingDescriptor) Set2d(mode PoolingMode, windowH, windowW, padH, padW, strideH, strideW int) error {
	return statusError(
		C.miopenSet2dPoolingDescriptor(p.d, C.miopenPoolingMode_t(mode),
			C.int(windowH), C.int(windowW), C.int(padH), C.int(padW), C.int(strideH), C.int(strideW)),
		"miopenSet2dPoolingDescriptor",
	)
}

// Destroy releases the pooling descriptor.
func (p *PoolingDescriptor) Destroy() error {
	return statusError(C.miopenDestroyPoolingDescriptor(p.d), "miopenDestroyPoolingDescriptor")
}

// --- Forward Operations ---

// ConvolutionForwardGetWorkspaceSize returns the workspace size in bytes for
// the forward convolution algorithm search.
func (h *Handle) ConvolutionForwardGetWorkspaceSize(
	xDesc *TensorDescriptor,
	wDesc *TensorDescriptor,
	convDesc *ConvolutionDescriptor,
	yDesc *TensorDescriptor,
) (int, error) {
	var size C.size_t
	err := statusError(
		C.miopenConvolutionForwardGetWorkSpaceSize(h.h, wDesc.d, xDesc.d, convDesc.d, yDesc.d, &size),
		"miopenConvolutionForwardGetWorkSpaceSize",
	)
	return int(size), err
}

// FindConvolutionForwardAlgorithm finds the best convolution algorithm.
// Returns the algorithm enum value suitable for ConvolutionForward.
func (h *Handle) FindConvolutionForwardAlgorithm(
	xDesc *TensorDescriptor, x unsafe.Pointer,
	wDesc *TensorDescriptor, w unsafe.Pointer,
	convDesc *ConvolutionDescriptor,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	workspace unsafe.Pointer, wsSize int,
) (C.miopenConvFwdAlgorithm_t, error) {
	var result C.miopenConvAlgoPerf_t
	var returnedCount C.int
	err := statusError(
		C.miopenFindConvolutionForwardAlgorithm(h.h, xDesc.d, x, wDesc.d, w,
			convDesc.d, yDesc.d, y, 1, &returnedCount, &result,
			workspace, C.size_t(wsSize), false),
		"miopenFindConvolutionForwardAlgorithm",
	)
	if err != nil {
		return 0, err
	}
	return result.fwd_algo, nil
}

// ConvolutionForward performs the forward convolution.
func (h *Handle) ConvolutionForward(
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	wDesc *TensorDescriptor, w unsafe.Pointer,
	convDesc *ConvolutionDescriptor,
	algo C.miopenConvFwdAlgorithm_t,
	workspace unsafe.Pointer, wsSize int,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)
	return statusError(
		C.miopenConvolutionForward(h.h,
			unsafe.Pointer(&cAlpha),
			xDesc.d, x, wDesc.d, w, convDesc.d,
			algo,
			unsafe.Pointer(&cBeta),
			yDesc.d, y,
			workspace, C.size_t(wsSize)),
		"miopenConvolutionForward",
	)
}

// BatchNormalizationForwardInference performs batch normalization in inference mode.
func (h *Handle) BatchNormalizationForwardInference(
	mode BatchNormMode,
	alpha, beta float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	bnScaleBiasMeanVarDesc *TensorDescriptor,
	scale, bias, mean, variance unsafe.Pointer,
	epsilon float64,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)
	return statusError(
		C.miopenBatchNormalizationForwardInference(h.h,
			C.miopenBatchNormMode_t(mode),
			unsafe.Pointer(&cAlpha), unsafe.Pointer(&cBeta),
			xDesc.d, x, yDesc.d, y,
			bnScaleBiasMeanVarDesc.d,
			scale, bias, mean, variance,
			C.double(epsilon)),
		"miopenBatchNormalizationForwardInference",
	)
}

// ActivationForward applies an activation function.
func (h *Handle) ActivationForward(
	actDesc *ActivationDescriptor,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)
	return statusError(
		C.miopenActivationForward(h.h, actDesc.d,
			unsafe.Pointer(&cAlpha), xDesc.d, x,
			unsafe.Pointer(&cBeta), yDesc.d, y),
		"miopenActivationForward",
	)
}

// PoolingForward performs 2D pooling.
func (h *Handle) PoolingForward(
	poolDesc *PoolingDescriptor,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	workspaceIndex bool,
	workspace unsafe.Pointer, wsSize int,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)
	return statusError(
		C.miopenPoolingForward(h.h, poolDesc.d,
			unsafe.Pointer(&cAlpha), xDesc.d, x,
			unsafe.Pointer(&cBeta), yDesc.d, y,
			C.bool(workspaceIndex), workspace, C.size_t(wsSize)),
		"miopenPoolingForward",
	)
}

// SoftmaxForward computes softmax.
func (h *Handle) SoftmaxForward(
	algo SoftmaxAlgorithm, mode SoftmaxMode,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)
	return statusError(
		C.miopenSoftmaxForward_V2(h.h,
			unsafe.Pointer(&cAlpha), xDesc.d, x,
			unsafe.Pointer(&cBeta), yDesc.d, y,
			C.miopenSoftmaxAlgorithm_t(algo), C.miopenSoftmaxMode_t(mode)),
		"miopenSoftmaxForward_V2",
	)
}

// OpTensor adds tensors: y = alpha * b + beta * y.
// MIOpen uses miopenOpTensor with miopenTensorOpAdd for bias addition.
func (h *Handle) OpTensorAdd(
	alpha float32,
	bDesc *TensorDescriptor, b unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)
	cZero := C.float(0)
	// miopenOpTensor: C = op(alpha1 * A, alpha2 * B) + beta * C
	// For add bias: y = alpha * b + beta * y, set A=b, B=b (ignored for add), C=y
	var opDesc C.miopenTensorOp_t = C.miopenTensorOpAdd
	return statusError(
		C.miopenOpTensor(h.h, opDesc,
			unsafe.Pointer(&cAlpha), bDesc.d, b,
			unsafe.Pointer(&cZero), bDesc.d, b,
			unsafe.Pointer(&cBeta), yDesc.d, y),
		"miopenOpTensor(add)",
	)
}

// GetPoolingForwardOutputDim returns the output dimensions for a pooling operation.
func (h *Handle) GetPoolingForwardOutputDim(
	poolDesc *PoolingDescriptor,
	xDesc *TensorDescriptor,
) (n, c, outH, outW int, err error) {
	var cn, cc, ch, cw C.int
	err = statusError(
		C.miopenGetPoolingForwardOutputDim(poolDesc.d, xDesc.d, &cn, &cc, &ch, &cw),
		"miopenGetPoolingForwardOutputDim",
	)
	return int(cn), int(cc), int(ch), int(cw), err
}

// PoolingGetWorkSpaceSize returns the workspace size for pooling.
func (h *Handle) PoolingGetWorkSpaceSize(yDesc *TensorDescriptor) (int, error) {
	var size C.size_t
	err := statusError(
		C.miopenPoolingGetWorkSpaceSize(yDesc.d, &size),
		"miopenPoolingGetWorkSpaceSize",
	)
	return int(size), err
}
