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
