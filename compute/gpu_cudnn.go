//go:build cuda

package compute

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/tensor"
)

// Conv2dForward performs 2D convolution using the GPU DNN backend.
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
	// Only float32 has a DNN path.
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

	// DNN requires symmetric padding.
	padH, padW := pads[0], pads[1]
	if pads[0] != pads[2] || pads[1] != pads[3] {
		return nil, fmt.Errorf("Conv2dForward: DNN requires symmetric padding, got [%d,%d,%d,%d]", pads[0], pads[1], pads[2], pads[3])
	}

	n, cIn, inH, inW := xShape[0], xShape[1], xShape[2], xShape[3]
	cOut, _, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]
	sH, sW := strides[0], strides[1]
	dH, dW := dilations[0], dilations[1]

	// Compute output dimensions.
	outH := (inH+2*padH-dH*(kH-1)-1)/sH + 1
	outW := (inW+2*padW-dW*(kW-1)-1)/sW + 1

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

	// --- Bias pointer ---
	var devB unsafe.Pointer
	var cleanupB func()
	if bias != nil {
		bShape := bias.Shape()
		if len(bShape) != 1 || bShape[0] != cOut {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: bias must be [%d], got %v", cOut, bShape)
		}
		devB, cleanupB, err = getDevicePtr(e, bias)
		if err != nil {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: getDevicePtr(bias): %w", err)
		}
		defer cleanupB()
	}

	// --- DNN forward ---

	if err := e.dnn.ConvForward(
		devX, [4]int{n, cIn, inH, inW},
		devW, [4]int{wShape[0], wShape[1], wShape[2], wShape[3]},
		devB,
		devY, [4]int{n, cOut, outH, outW},
		[2]int{padH, padW}, strides, dilations,
		groups,
		e.stream,
	); err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		return nil, fmt.Errorf("Conv2dForward: %w", err)
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
// using pre-computed running mean and variance via the GPU DNN backend.
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

	if err := e.dnn.BatchNormForwardInference(
		devX, [4]int{n, c, h, w},
		devScale, devBias, devMean, devVar,
		c,
		epsilon,
		devY,
		e.stream,
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

// CudnnActivationForward applies an activation function via the GPU DNN backend.
// mode selects the activation: ActivationReLU, ActivationSigmoid, ActivationTanh.
// The input tensor shape is preserved in the output.
func (e *GPUEngine[T]) CudnnActivationForward(
	_ context.Context,
	x *tensor.TensorNumeric[T],
	mode gpuapi.ActivationMode,
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

	// Pack shape into 4D for DNN (N=1, C=numElems, H=1, W=1 for 1D/2D/3D).
	n4, c4, h4, w4 := 1, 1, 1, 1
	switch len(shape) {
	case 4:
		n4, c4, h4, w4 = shape[0], shape[1], shape[2], shape[3]
	case 3:
		n4, c4, h4 = shape[0], shape[1], shape[2]
	case 2:
		n4, c4 = shape[0], shape[1]
	case 1:
		c4 = shape[0]
	default:
		// Flatten to 1D.
		c4 = numElems
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

	if err := e.dnn.ActivationForward(mode, devX, [4]int{n4, c4, h4, w4}, devY, e.stream); err != nil {
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

// CudnnPoolingForward performs 2D pooling via the GPU DNN backend.
// x must be [N, C, H, W]. Returns [N, C, outH, outW].
func (e *GPUEngine[T]) CudnnPoolingForward(
	_ context.Context,
	x *tensor.TensorNumeric[T],
	mode gpuapi.PoolingMode,
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

	if err := e.dnn.PoolingForward(
		mode,
		devX, [4]int{n, c, inH, inW},
		devY, [4]int{n, c, outH, outW},
		windowH, windowW, padH, padW, strideH, strideW,
		e.stream,
	); err != nil {
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

// CudnnSoftmaxForward computes softmax via the GPU DNN backend over the channel dimension.
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

	// DNN softmax operates over the C dimension in NCHW.
	n4, c4, h4, w4 := 1, 1, 1, 1
	switch len(shape) {
	case 4:
		n4, c4, h4, w4 = shape[0], shape[1], shape[2], shape[3]
	case 3:
		n4 = shape[0] * shape[1]
		c4 = shape[2]
	case 2:
		n4, c4 = shape[0], shape[1]
	case 1:
		c4 = shape[0]
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

	if err := e.dnn.SoftmaxForward(devX, [4]int{n4, c4, h4, w4}, devY, e.stream); err != nil {
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
