//go:build cuda

package compute

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cudnn"
	"github.com/zerfoo/zerfoo/tensor"
)

// Conv2dForward performs 2D convolution using cuDNN.
// x must be [N, C_in, H, W], w must be [C_out, C_in/groups, kH, kW].
// bias is optional (nil to skip). pads is [top, left, bottom, right].
// Returns error if padding is asymmetric (cuDNN requires symmetric padding).
func (e *GPUEngine[T]) Conv2dForward(
	_ context.Context,
	x, w *tensor.TensorNumeric[T],
	bias *tensor.TensorNumeric[T],
	strides [2]int,
	pads [4]int,
	dilations [2]int,
	groups int,
) (*tensor.TensorNumeric[T], error) {
	// Only float32 has a cuDNN path.
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("Conv2dForward: only float32 supported, got %T", zero)
	}

	e.setDevice()

	xShape := x.Shape()
	wShape := w.Shape()
	if len(xShape) != 4 {
		return nil, fmt.Errorf("Conv2dForward: x must be 4D [N,C,H,W], got %v", xShape)
	}
	if len(wShape) != 4 {
		return nil, fmt.Errorf("Conv2dForward: w must be 4D [C_out,C_in/g,kH,kW], got %v", wShape)
	}

	// cuDNN requires symmetric padding.
	padH, padW := pads[0], pads[1]
	if pads[0] != pads[2] || pads[1] != pads[3] {
		return nil, fmt.Errorf("Conv2dForward: cuDNN requires symmetric padding, got [%d,%d,%d,%d]", pads[0], pads[1], pads[2], pads[3])
	}

	n, cIn, inH, inW := xShape[0], xShape[1], xShape[2], xShape[3]
	cOut, cInG, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]
	sH, sW := strides[0], strides[1]
	dH, dW := dilations[0], dilations[1]

	// Compute output dimensions.
	outH := (inH+2*padH-dH*(kH-1)-1)/sH + 1
	outW := (inW+2*padW-dW*(kW-1)-1)/sW + 1

	_ = cIn  // used implicitly via cInG * groups
	_ = cInG // validated by cuDNN

	// --- Descriptors ---

	xDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: create xDesc: %w", err)
	}
	defer xDesc.Destroy()
	if err := xDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, cIn, inH, inW); err != nil {
		return nil, fmt.Errorf("Conv2dForward: set xDesc: %w", err)
	}

	wDesc, err := cudnn.CreateFilterDescriptor()
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: create wDesc: %w", err)
	}
	defer wDesc.Destroy()
	if err := wDesc.Set4d(cudnn.Float32, cudnn.NCHW, cOut, cInG, kH, kW); err != nil {
		return nil, fmt.Errorf("Conv2dForward: set wDesc: %w", err)
	}

	convDesc, err := cudnn.CreateConvolutionDescriptor()
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: create convDesc: %w", err)
	}
	defer convDesc.Destroy()
	if err := convDesc.Set2d(padH, padW, sH, sW, dH, dW, cudnn.CrossCorrelation, cudnn.Float32); err != nil {
		return nil, fmt.Errorf("Conv2dForward: set convDesc: %w", err)
	}
	if groups > 1 {
		if err := convDesc.SetGroupCount(groups); err != nil {
			return nil, fmt.Errorf("Conv2dForward: set group count: %w", err)
		}
	}

	yDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: create yDesc: %w", err)
	}
	defer yDesc.Destroy()
	if err := yDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, cOut, outH, outW); err != nil {
		return nil, fmt.Errorf("Conv2dForward: set yDesc: %w", err)
	}

	// --- Algorithm and workspace ---

	algo := cudnn.ConvFwdAlgoImplicitGemm

	wsSize, err := e.cudnnHandle.GetConvolutionForwardWorkspaceSize(xDesc, wDesc, convDesc, yDesc, algo)
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: workspace size query: %w", err)
	}

	var wsPtr unsafe.Pointer
	if wsSize > 0 {
		wsPtr, err = e.pool.Alloc(e.deviceID, wsSize)
		if err != nil {
			return nil, fmt.Errorf("Conv2dForward: workspace alloc: %w", err)
		}
		defer e.pool.Free(e.deviceID, wsPtr, wsSize)
	}

	// --- Device pointers ---

	devX, cleanupX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: getDevicePtr(x): %w", err)
	}
	defer cleanupX()

	devW, cleanupW, err := getDevicePtr(e, w)
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: getDevicePtr(w): %w", err)
	}
	defer cleanupW()

	outElems := n * cOut * outH * outW
	devY, err := e.pool.Alloc(e.deviceID, outElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: output alloc: %w", err)
	}

	// --- cuDNN forward ---

	if err := e.cudnnHandle.ConvolutionForward(
		1.0, // alpha
		xDesc, devX,
		wDesc, devW,
		convDesc,
		algo,
		wsPtr, wsSize,
		0.0, // beta
		yDesc, devY,
	); err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		return nil, fmt.Errorf("Conv2dForward: cudnnConvolutionForward: %w", err)
	}

	// --- Add bias ---

	if bias != nil {
		bShape := bias.Shape()
		if len(bShape) != 1 || bShape[0] != cOut {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: bias must be [%d], got %v", cOut, bShape)
		}

		bDesc, err := cudnn.CreateTensorDescriptor()
		if err != nil {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: create bDesc: %w", err)
		}
		defer bDesc.Destroy()
		if err := bDesc.Set4d(cudnn.NCHW, cudnn.Float32, 1, cOut, 1, 1); err != nil {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: set bDesc: %w", err)
		}

		devB, cleanupB, err := getDevicePtr(e, bias)
		if err != nil {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: getDevicePtr(bias): %w", err)
		}
		defer cleanupB()

		// cudnnAddTensor adds bias to the output: y = alpha*b + beta*y
		if err := e.cudnnHandle.AddTensor(1.0, bDesc, devB, 1.0, yDesc, devY); err != nil {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: cudnnAddTensor (bias): %w", err)
		}
	}

	// --- Synchronize and wrap ---

	if e.stream != nil {
		if err := e.stream.Synchronize(); err != nil {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: stream sync: %w", err)
		}
	}

	return makeGPUResult[T](e, []int{n, cOut, outH, outW}, devY, outElems)
}

// BatchNormForwardInference performs batch normalization in inference mode
// using pre-computed running mean and variance via cuDNN.
// x must be [N, C, H, W]. scale, bias, mean, variance must each be [C].
func (e *GPUEngine[T]) BatchNormForwardInference(
	_ context.Context,
	x, scale, bias, mean, variance *tensor.TensorNumeric[T],
	epsilon float64,
) (*tensor.TensorNumeric[T], error) {
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("BatchNormForwardInference: only float32 supported")
	}
	e.setDevice()

	xShape := x.Shape()
	if len(xShape) != 4 {
		return nil, fmt.Errorf("BatchNormForwardInference: x must be 4D, got %v", xShape)
	}
	n, c, h, w := xShape[0], xShape[1], xShape[2], xShape[3]

	xDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: create xDesc: %w", err)
	}
	defer xDesc.Destroy()
	if err := xDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, c, h, w); err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: set xDesc: %w", err)
	}

	yDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: create yDesc: %w", err)
	}
	defer yDesc.Destroy()
	if err := yDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, c, h, w); err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: set yDesc: %w", err)
	}

	bnDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: create bnDesc: %w", err)
	}
	defer bnDesc.Destroy()
	if err := bnDesc.Set4d(cudnn.NCHW, cudnn.Float32, 1, c, 1, 1); err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: set bnDesc: %w", err)
	}

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devScale, cleanScale, err := getDevicePtr(e, scale)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(scale): %w", err)
	}
	defer cleanScale()

	devBias, cleanBias, err := getDevicePtr(e, bias)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(bias): %w", err)
	}
	defer cleanBias()

	devMean, cleanMean, err := getDevicePtr(e, mean)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(mean): %w", err)
	}
	defer cleanMean()

	devVar, cleanVar, err := getDevicePtr(e, variance)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(var): %w", err)
	}
	defer cleanVar()

	outElems := n * c * h * w
	devY, err := e.pool.Alloc(e.deviceID, outElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: output alloc: %w", err)
	}

	if err := e.cudnnHandle.BatchNormalizationForwardInference(
		cudnn.BatchNormSpatial,
		1.0, 0.0,
		xDesc, devX,
		yDesc, devY,
		bnDesc,
		devScale, devBias,
		devMean, devVar,
		epsilon,
	); err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		return nil, fmt.Errorf("BatchNormForwardInference: %w", err)
	}

	if e.stream != nil {
		if err := e.stream.Synchronize(); err != nil {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("BatchNormForwardInference: stream sync: %w", err)
		}
	}

	return makeGPUResult[T](e, xShape, devY, outElems)
}

// CudnnActivationForward applies an activation function via cuDNN.
// mode selects the activation: cudnn.ActivationReLU, ActivationSigmoid, ActivationTanh.
// The input tensor shape is preserved in the output.
func (e *GPUEngine[T]) CudnnActivationForward(
	_ context.Context,
	x *tensor.TensorNumeric[T],
	mode cudnn.ActivationMode,
) (*tensor.TensorNumeric[T], error) {
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("CudnnActivationForward: only float32 supported")
	}
	e.setDevice()

	shape := x.Shape()
	numElems := 1
	for _, d := range shape {
		numElems *= d
	}

	// Pack shape into 4D for cuDNN (N=1, C=numElems, H=1, W=1 for 1D/2D/3D).
	n, c, h, w := 1, 1, 1, 1
	switch len(shape) {
	case 4:
		n, c, h, w = shape[0], shape[1], shape[2], shape[3]
	case 3:
		n, c, h = shape[0], shape[1], shape[2]
	case 2:
		n, c = shape[0], shape[1]
	case 1:
		c = shape[0]
	default:
		// Flatten to 1D.
		c = numElems
	}

	xDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: create xDesc: %w", err)
	}
	defer xDesc.Destroy()
	if err := xDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, c, h, w); err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: set xDesc: %w", err)
	}

	yDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: create yDesc: %w", err)
	}
	defer yDesc.Destroy()
	if err := yDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, c, h, w); err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: set yDesc: %w", err)
	}

	actDesc, err := cudnn.CreateActivationDescriptor()
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: create actDesc: %w", err)
	}
	defer actDesc.Destroy()
	if err := actDesc.Set(mode, cudnn.NotPropagateNan, 0.0); err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: set actDesc: %w", err)
	}

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devY, err := e.pool.Alloc(e.deviceID, numElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: output alloc: %w", err)
	}

	if err := e.cudnnHandle.ActivationForward(actDesc, 1.0, xDesc, devX, 0.0, yDesc, devY); err != nil {
		e.pool.Free(e.deviceID, devY, numElems*f32Size)
		return nil, fmt.Errorf("CudnnActivationForward: %w", err)
	}

	if e.stream != nil {
		if err := e.stream.Synchronize(); err != nil {
			e.pool.Free(e.deviceID, devY, numElems*f32Size)
			return nil, fmt.Errorf("CudnnActivationForward: stream sync: %w", err)
		}
	}

	return makeGPUResult[T](e, shape, devY, numElems)
}

// CudnnPoolingForward performs 2D pooling via cuDNN.
// x must be [N, C, H, W]. Returns [N, C, outH, outW].
func (e *GPUEngine[T]) CudnnPoolingForward(
	_ context.Context,
	x *tensor.TensorNumeric[T],
	mode cudnn.PoolingMode,
	windowH, windowW, padH, padW, strideH, strideW int,
) (*tensor.TensorNumeric[T], error) {
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("CudnnPoolingForward: only float32 supported")
	}
	e.setDevice()

	xShape := x.Shape()
	if len(xShape) != 4 {
		return nil, fmt.Errorf("CudnnPoolingForward: x must be 4D, got %v", xShape)
	}
	n, c, inH, inW := xShape[0], xShape[1], xShape[2], xShape[3]
	outH := (inH+2*padH-windowH)/strideH + 1
	outW := (inW+2*padW-windowW)/strideW + 1

	xDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: create xDesc: %w", err)
	}
	defer xDesc.Destroy()
	if err := xDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, c, inH, inW); err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: set xDesc: %w", err)
	}

	yDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: create yDesc: %w", err)
	}
	defer yDesc.Destroy()
	if err := yDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, c, outH, outW); err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: set yDesc: %w", err)
	}

	poolDesc, err := cudnn.CreatePoolingDescriptor()
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: create poolDesc: %w", err)
	}
	defer poolDesc.Destroy()
	if err := poolDesc.Set2d(mode, cudnn.NotPropagateNan, windowH, windowW, padH, padW, strideH, strideW); err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: set poolDesc: %w", err)
	}

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	outElems := n * c * outH * outW
	devY, err := e.pool.Alloc(e.deviceID, outElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: output alloc: %w", err)
	}

	if err := e.cudnnHandle.PoolingForward(poolDesc, 1.0, xDesc, devX, 0.0, yDesc, devY); err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		return nil, fmt.Errorf("CudnnPoolingForward: %w", err)
	}

	if e.stream != nil {
		if err := e.stream.Synchronize(); err != nil {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("CudnnPoolingForward: stream sync: %w", err)
		}
	}

	return makeGPUResult[T](e, []int{n, c, outH, outW}, devY, outElems)
}

// CudnnSoftmaxForward computes softmax via cuDNN over the channel dimension.
// x must be [N, C, H, W] (or reshaped to fit).
func (e *GPUEngine[T]) CudnnSoftmaxForward(
	_ context.Context,
	x *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("CudnnSoftmaxForward: only float32 supported")
	}
	e.setDevice()

	shape := x.Shape()
	numElems := 1
	for _, d := range shape {
		numElems *= d
	}

	// cuDNN softmax operates over the C dimension in NCHW.
	// For a typical [batch, seq, vocab] tensor, map to [batch*seq, vocab, 1, 1].
	n, c, h, w := 1, 1, 1, 1
	switch len(shape) {
	case 4:
		n, c, h, w = shape[0], shape[1], shape[2], shape[3]
	case 3:
		n = shape[0] * shape[1]
		c = shape[2]
	case 2:
		n, c = shape[0], shape[1]
	case 1:
		c = shape[0]
	}

	xDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("CudnnSoftmaxForward: create xDesc: %w", err)
	}
	defer xDesc.Destroy()
	if err := xDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, c, h, w); err != nil {
		return nil, fmt.Errorf("CudnnSoftmaxForward: set xDesc: %w", err)
	}

	yDesc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, fmt.Errorf("CudnnSoftmaxForward: create yDesc: %w", err)
	}
	defer yDesc.Destroy()
	if err := yDesc.Set4d(cudnn.NCHW, cudnn.Float32, n, c, h, w); err != nil {
		return nil, fmt.Errorf("CudnnSoftmaxForward: set yDesc: %w", err)
	}

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("CudnnSoftmaxForward: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devY, err := e.pool.Alloc(e.deviceID, numElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("CudnnSoftmaxForward: output alloc: %w", err)
	}

	if err := e.cudnnHandle.SoftmaxForward(
		cudnn.SoftmaxAccurate, cudnn.SoftmaxModeChannel,
		1.0, xDesc, devX, 0.0, yDesc, devY,
	); err != nil {
		e.pool.Free(e.deviceID, devY, numElems*f32Size)
		return nil, fmt.Errorf("CudnnSoftmaxForward: %w", err)
	}

	if e.stream != nil {
		if err := e.stream.Synchronize(); err != nil {
			e.pool.Free(e.deviceID, devY, numElems*f32Size)
			return nil, fmt.Errorf("CudnnSoftmaxForward: stream sync: %w", err)
		}
	}

	return makeGPUResult[T](e, shape, devY, numElems)
}
