//go:build cuda

// Package cudnn provides CGo bindings for the NVIDIA cuDNN library. It
// exposes handle lifecycle, tensor/filter/convolution/activation/pooling
// descriptors, and forward operation bindings needed for GPU-accelerated
// neural network primitives.
package cudnn

/*
#cgo LDFLAGS: -lcudnn
#include <cudnn.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// statusError converts a cudnnStatus_t to a Go error. Returns nil on success.
func statusError(status C.cudnnStatus_t, context string) error {
	if status == C.CUDNN_STATUS_SUCCESS {
		return nil
	}
	return fmt.Errorf("%s: %s", context, C.GoString(C.cudnnGetErrorString(status)))
}

// --- Data types ---

// DataType maps to cudnnDataType_t.
type DataType int

const (
	Float32 DataType = C.CUDNN_DATA_FLOAT
	Float64 DataType = C.CUDNN_DATA_DOUBLE
	Float16 DataType = C.CUDNN_DATA_HALF
	Int32   DataType = C.CUDNN_DATA_INT32
	Int8    DataType = C.CUDNN_DATA_INT8
)

// TensorFormat maps to cudnnTensorFormat_t.
type TensorFormat int

const (
	NCHW TensorFormat = C.CUDNN_TENSOR_NCHW
	NHWC TensorFormat = C.CUDNN_TENSOR_NHWC
)

// ActivationMode maps to cudnnActivationMode_t.
type ActivationMode int

const (
	ActivationSigmoid    ActivationMode = C.CUDNN_ACTIVATION_SIGMOID
	ActivationReLU       ActivationMode = C.CUDNN_ACTIVATION_RELU
	ActivationTanh       ActivationMode = C.CUDNN_ACTIVATION_TANH
	ActivationClippedReLU ActivationMode = C.CUDNN_ACTIVATION_CLIPPED_RELU
	ActivationELU        ActivationMode = C.CUDNN_ACTIVATION_ELU
)

// PoolingMode maps to cudnnPoolingMode_t.
type PoolingMode int

const (
	PoolingMax                    PoolingMode = C.CUDNN_POOLING_MAX
	PoolingAverageCountIncludePad PoolingMode = C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
	PoolingAverageCountExcludePad PoolingMode = C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
)

// NanPropagation maps to cudnnNanPropagation_t.
type NanPropagation int

const (
	NotPropagateNan NanPropagation = C.CUDNN_NOT_PROPAGATE_NAN
	PropagateNan    NanPropagation = C.CUDNN_PROPAGATE_NAN
)

// ConvolutionMode maps to cudnnConvolutionMode_t.
type ConvolutionMode int

const (
	Convolution      ConvolutionMode = C.CUDNN_CONVOLUTION
	CrossCorrelation ConvolutionMode = C.CUDNN_CROSS_CORRELATION
)

// BatchNormMode maps to cudnnBatchNormMode_t.
type BatchNormMode int

const (
	BatchNormPerActivation BatchNormMode = C.CUDNN_BATCHNORM_PER_ACTIVATION
	BatchNormSpatial       BatchNormMode = C.CUDNN_BATCHNORM_SPATIAL
)

// SoftmaxAlgorithm maps to cudnnSoftmaxAlgorithm_t.
type SoftmaxAlgorithm int

const (
	SoftmaxFast     SoftmaxAlgorithm = C.CUDNN_SOFTMAX_FAST
	SoftmaxAccurate SoftmaxAlgorithm = C.CUDNN_SOFTMAX_ACCURATE
	SoftmaxLog      SoftmaxAlgorithm = C.CUDNN_SOFTMAX_LOG
)

// SoftmaxMode maps to cudnnSoftmaxMode_t.
type SoftmaxMode int

const (
	SoftmaxModeInstance SoftmaxMode = C.CUDNN_SOFTMAX_MODE_INSTANCE
	SoftmaxModeChannel  SoftmaxMode = C.CUDNN_SOFTMAX_MODE_CHANNEL
)

// ConvFwdAlgo maps to cudnnConvolutionFwdAlgo_t.
type ConvFwdAlgo int

const (
	ConvFwdAlgoImplicitGemm        ConvFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
	ConvFwdAlgoImplicitPrecompGemm ConvFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	ConvFwdAlgoGemm                ConvFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM
	ConvFwdAlgoFFT                 ConvFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_FFT
	ConvFwdAlgoWinograd            ConvFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
)

// --- Handle ---

// Handle wraps a cudnnHandle_t. Create one per GPUEngine.
type Handle struct {
	h C.cudnnHandle_t
}

// CreateHandle creates a new cuDNN handle. The caller must call Destroy when
// done. The handle is implicitly associated with the current CUDA device.
func CreateHandle() (*Handle, error) {
	var h C.cudnnHandle_t
	if err := statusError(C.cudnnCreate(&h), "cudnnCreate"); err != nil {
		return nil, err
	}
	return &Handle{h: h}, nil
}

// SetStream associates the handle with a CUDA stream. The stream parameter
// is a cudaStream_t passed as unsafe.Pointer.
func (h *Handle) SetStream(stream unsafe.Pointer) error {
	return statusError(
		C.cudnnSetStream(h.h, C.cudaStream_t(stream)),
		"cudnnSetStream",
	)
}

// Destroy releases the cuDNN handle resources.
func (h *Handle) Destroy() error {
	return statusError(C.cudnnDestroy(h.h), "cudnnDestroy")
}

// --- TensorDescriptor ---

// TensorDescriptor wraps a cudnnTensorDescriptor_t describing tensor layout
// and data type.
type TensorDescriptor struct {
	d C.cudnnTensorDescriptor_t
}

// CreateTensorDescriptor allocates a new tensor descriptor.
func CreateTensorDescriptor() (*TensorDescriptor, error) {
	var d C.cudnnTensorDescriptor_t
	if err := statusError(C.cudnnCreateTensorDescriptor(&d), "cudnnCreateTensorDescriptor"); err != nil {
		return nil, err
	}
	return &TensorDescriptor{d: d}, nil
}

// Set4d sets the tensor descriptor to a 4D layout (N, C, H, W).
func (t *TensorDescriptor) Set4d(format TensorFormat, dtype DataType, n, c, h, w int) error {
	return statusError(
		C.cudnnSetTensor4dDescriptor(
			t.d,
			C.cudnnTensorFormat_t(format),
			C.cudnnDataType_t(dtype),
			C.int(n), C.int(c), C.int(h), C.int(w),
		),
		"cudnnSetTensor4dDescriptor",
	)
}

// SetNd sets the tensor descriptor to an N-dimensional layout with explicit
// dimensions and strides.
func (t *TensorDescriptor) SetNd(dtype DataType, dims, strides []int) error {
	if len(dims) != len(strides) {
		return fmt.Errorf("cudnnSetTensorNdDescriptor: dims length %d != strides length %d", len(dims), len(strides))
	}
	n := len(dims)
	cDims := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cDims[i] = C.int(dims[i])
		cStrides[i] = C.int(strides[i])
	}
	return statusError(
		C.cudnnSetTensorNdDescriptor(
			t.d,
			C.cudnnDataType_t(dtype),
			C.int(n),
			&cDims[0],
			&cStrides[0],
		),
		"cudnnSetTensorNdDescriptor",
	)
}

// Destroy releases the tensor descriptor.
func (t *TensorDescriptor) Destroy() error {
	return statusError(C.cudnnDestroyTensorDescriptor(t.d), "cudnnDestroyTensorDescriptor")
}

// --- FilterDescriptor ---

// FilterDescriptor wraps a cudnnFilterDescriptor_t describing convolution
// filter (weight) layout.
type FilterDescriptor struct {
	d C.cudnnFilterDescriptor_t
}

// CreateFilterDescriptor allocates a new filter descriptor.
func CreateFilterDescriptor() (*FilterDescriptor, error) {
	var d C.cudnnFilterDescriptor_t
	if err := statusError(C.cudnnCreateFilterDescriptor(&d), "cudnnCreateFilterDescriptor"); err != nil {
		return nil, err
	}
	return &FilterDescriptor{d: d}, nil
}

// Set4d sets the filter descriptor to a 4D layout (K, C, H, W) where K is
// the number of output channels.
func (f *FilterDescriptor) Set4d(dtype DataType, format TensorFormat, k, c, h, w int) error {
	return statusError(
		C.cudnnSetFilter4dDescriptor(
			f.d,
			C.cudnnDataType_t(dtype),
			C.cudnnTensorFormat_t(format),
			C.int(k), C.int(c), C.int(h), C.int(w),
		),
		"cudnnSetFilter4dDescriptor",
	)
}

// Destroy releases the filter descriptor.
func (f *FilterDescriptor) Destroy() error {
	return statusError(C.cudnnDestroyFilterDescriptor(f.d), "cudnnDestroyFilterDescriptor")
}

// --- ConvolutionDescriptor ---

// ConvolutionDescriptor wraps a cudnnConvolutionDescriptor_t describing
// convolution parameters (padding, stride, dilation).
type ConvolutionDescriptor struct {
	d C.cudnnConvolutionDescriptor_t
}

// CreateConvolutionDescriptor allocates a new convolution descriptor.
func CreateConvolutionDescriptor() (*ConvolutionDescriptor, error) {
	var d C.cudnnConvolutionDescriptor_t
	if err := statusError(C.cudnnCreateConvolutionDescriptor(&d), "cudnnCreateConvolutionDescriptor"); err != nil {
		return nil, err
	}
	return &ConvolutionDescriptor{d: d}, nil
}

// Set2d configures the convolution descriptor for 2D convolution.
func (c *ConvolutionDescriptor) Set2d(padH, padW, strideH, strideW, dilationH, dilationW int, mode ConvolutionMode, dtype DataType) error {
	return statusError(
		C.cudnnSetConvolution2dDescriptor(
			c.d,
			C.int(padH), C.int(padW),
			C.int(strideH), C.int(strideW),
			C.int(dilationH), C.int(dilationW),
			C.cudnnConvolutionMode_t(mode),
			C.cudnnDataType_t(dtype),
		),
		"cudnnSetConvolution2dDescriptor",
	)
}

// SetGroupCount sets the number of groups for grouped convolution.
func (c *ConvolutionDescriptor) SetGroupCount(groups int) error {
	return statusError(
		C.cudnnSetConvolutionGroupCount(c.d, C.int(groups)),
		"cudnnSetConvolutionGroupCount",
	)
}

// Destroy releases the convolution descriptor.
func (c *ConvolutionDescriptor) Destroy() error {
	return statusError(C.cudnnDestroyConvolutionDescriptor(c.d), "cudnnDestroyConvolutionDescriptor")
}

// --- ActivationDescriptor ---

// ActivationDescriptor wraps a cudnnActivationDescriptor_t describing an
// activation function.
type ActivationDescriptor struct {
	d C.cudnnActivationDescriptor_t
}

// CreateActivationDescriptor allocates a new activation descriptor.
func CreateActivationDescriptor() (*ActivationDescriptor, error) {
	var d C.cudnnActivationDescriptor_t
	if err := statusError(C.cudnnCreateActivationDescriptor(&d), "cudnnCreateActivationDescriptor"); err != nil {
		return nil, err
	}
	return &ActivationDescriptor{d: d}, nil
}

// Set configures the activation descriptor with the given mode, NaN
// propagation, and ceiling for clipped ReLU.
func (a *ActivationDescriptor) Set(mode ActivationMode, nanProp NanPropagation, coef float64) error {
	return statusError(
		C.cudnnSetActivationDescriptor(
			a.d,
			C.cudnnActivationMode_t(mode),
			C.cudnnNanPropagation_t(nanProp),
			C.double(coef),
		),
		"cudnnSetActivationDescriptor",
	)
}

// Destroy releases the activation descriptor.
func (a *ActivationDescriptor) Destroy() error {
	return statusError(C.cudnnDestroyActivationDescriptor(a.d), "cudnnDestroyActivationDescriptor")
}

// --- PoolingDescriptor ---

// PoolingDescriptor wraps a cudnnPoolingDescriptor_t describing a pooling
// operation.
type PoolingDescriptor struct {
	d C.cudnnPoolingDescriptor_t
}

// CreatePoolingDescriptor allocates a new pooling descriptor.
func CreatePoolingDescriptor() (*PoolingDescriptor, error) {
	var d C.cudnnPoolingDescriptor_t
	if err := statusError(C.cudnnCreatePoolingDescriptor(&d), "cudnnCreatePoolingDescriptor"); err != nil {
		return nil, err
	}
	return &PoolingDescriptor{d: d}, nil
}

// Set2d configures the pooling descriptor for 2D pooling.
func (p *PoolingDescriptor) Set2d(mode PoolingMode, nanProp NanPropagation, windowH, windowW, padH, padW, strideH, strideW int) error {
	return statusError(
		C.cudnnSetPooling2dDescriptor(
			p.d,
			C.cudnnPoolingMode_t(mode),
			C.cudnnNanPropagation_t(nanProp),
			C.int(windowH), C.int(windowW),
			C.int(padH), C.int(padW),
			C.int(strideH), C.int(strideW),
		),
		"cudnnSetPooling2dDescriptor",
	)
}

// Destroy releases the pooling descriptor.
func (p *PoolingDescriptor) Destroy() error {
	return statusError(C.cudnnDestroyPoolingDescriptor(p.d), "cudnnDestroyPoolingDescriptor")
}

// --- Forward Operations ---

// ConvolutionForward performs a forward convolution.
// alpha and beta are scaling factors (typically 1.0 and 0.0).
// workspace is a device pointer to pre-allocated workspace memory.
func (h *Handle) ConvolutionForward(
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	wDesc *FilterDescriptor, w unsafe.Pointer,
	convDesc *ConvolutionDescriptor,
	algo ConvFwdAlgo,
	workspace unsafe.Pointer, workspaceSize int,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	a := C.float(alpha)
	b := C.float(beta)
	return statusError(
		C.cudnnConvolutionForward(
			h.h,
			unsafe.Pointer(&a),
			xDesc.d, x,
			wDesc.d, w,
			convDesc.d,
			C.cudnnConvolutionFwdAlgo_t(algo),
			workspace, C.size_t(workspaceSize),
			unsafe.Pointer(&b),
			yDesc.d, y,
		),
		"cudnnConvolutionForward",
	)
}

// GetConvolutionForwardWorkspaceSize returns the workspace size in bytes
// required for the given convolution algorithm.
func (h *Handle) GetConvolutionForwardWorkspaceSize(
	xDesc *TensorDescriptor,
	wDesc *FilterDescriptor,
	convDesc *ConvolutionDescriptor,
	yDesc *TensorDescriptor,
	algo ConvFwdAlgo,
) (int, error) {
	var size C.size_t
	if err := statusError(
		C.cudnnGetConvolutionForwardWorkspaceSize(
			h.h,
			xDesc.d,
			wDesc.d,
			convDesc.d,
			yDesc.d,
			C.cudnnConvolutionFwdAlgo_t(algo),
			&size,
		),
		"cudnnGetConvolutionForwardWorkspaceSize",
	); err != nil {
		return 0, err
	}
	return int(size), nil
}

// BatchNormalizationForwardInference performs batch normalization in inference
// mode (using pre-computed running mean and variance).
func (h *Handle) BatchNormalizationForwardInference(
	mode BatchNormMode,
	alpha, beta float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	bnScaleBiasMeanVarDesc *TensorDescriptor,
	bnScale, bnBias unsafe.Pointer,
	estimatedMean, estimatedVariance unsafe.Pointer,
	epsilon float64,
) error {
	a := C.float(alpha)
	b := C.float(beta)
	return statusError(
		C.cudnnBatchNormalizationForwardInference(
			h.h,
			C.cudnnBatchNormMode_t(mode),
			unsafe.Pointer(&a),
			unsafe.Pointer(&b),
			xDesc.d, x,
			yDesc.d, y,
			bnScaleBiasMeanVarDesc.d,
			bnScale, bnBias,
			estimatedMean, estimatedVariance,
			C.double(epsilon),
		),
		"cudnnBatchNormalizationForwardInference",
	)
}

// ActivationForward applies an activation function element-wise.
func (h *Handle) ActivationForward(
	actDesc *ActivationDescriptor,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	a := C.float(alpha)
	b := C.float(beta)
	return statusError(
		C.cudnnActivationForward(
			h.h,
			actDesc.d,
			unsafe.Pointer(&a),
			xDesc.d, x,
			unsafe.Pointer(&b),
			yDesc.d, y,
		),
		"cudnnActivationForward",
	)
}

// PoolingForward performs a forward pooling operation.
func (h *Handle) PoolingForward(
	poolDesc *PoolingDescriptor,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	a := C.float(alpha)
	b := C.float(beta)
	return statusError(
		C.cudnnPoolingForward(
			h.h,
			poolDesc.d,
			unsafe.Pointer(&a),
			xDesc.d, x,
			unsafe.Pointer(&b),
			yDesc.d, y,
		),
		"cudnnPoolingForward",
	)
}

// AddTensor adds a bias tensor to the output: y = alpha*b + beta*y.
// This is commonly used to add bias after convolution.
func (h *Handle) AddTensor(
	alpha float32,
	bDesc *TensorDescriptor, b unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	a := C.float(alpha)
	bt := C.float(beta)
	return statusError(
		C.cudnnAddTensor(
			h.h,
			unsafe.Pointer(&a),
			bDesc.d, b,
			unsafe.Pointer(&bt),
			yDesc.d, y,
		),
		"cudnnAddTensor",
	)
}

// SoftmaxForward computes softmax over the channel dimension.
func (h *Handle) SoftmaxForward(
	algo SoftmaxAlgorithm,
	mode SoftmaxMode,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	a := C.float(alpha)
	b := C.float(beta)
	return statusError(
		C.cudnnSoftmaxForward(
			h.h,
			C.cudnnSoftmaxAlgorithm_t(algo),
			C.cudnnSoftmaxMode_t(mode),
			unsafe.Pointer(&a),
			xDesc.d, x,
			unsafe.Pointer(&b),
			yDesc.d, y,
		),
		"cudnnSoftmaxForward",
	)
}
