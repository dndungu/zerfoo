//go:build cuda

package compute

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/tensor"
)

const f32Size = int(unsafe.Sizeof(float32(0)))

// getDevicePtr returns a GPU device pointer for the tensor's data.
// If the tensor has GPUStorage, returns Ptr() directly (zero-copy).
// If the tensor has CPUStorage, allocates device memory from the pool,
// copies H2D, and returns a cleanup function that returns the buffer to the pool.
func getDevicePtr[T tensor.Numeric](e *GPUEngine[T], t *tensor.TensorNumeric[T]) (unsafe.Pointer, func(), error) {
	if gs, ok := t.GetStorage().(*tensor.GPUStorage[T]); ok {
		return gs.Ptr(), func() {}, nil
	}

	// CPUStorage path: allocate from pool, copy H2D.
	data := t.Data()
	n := len(data)
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	byteSize := n * elemSize

	devPtr, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return nil, nil, err
	}

	if err := e.runtime.Memcpy(devPtr, unsafe.Pointer(&data[0]), byteSize, gpuapi.MemcpyHostToDevice); err != nil {
		e.pool.Free(e.deviceID, devPtr, byteSize)

		return nil, nil, err
	}

	cleanup := func() {
		e.pool.Free(e.deviceID, devPtr, byteSize)
	}

	return devPtr, cleanup, nil
}

// makeGPUResult creates a tensor with GPUStorage wrapping the given device pointer.
// The device pointer is NOT freed when the storage is freed; the caller retains
// ownership through the pool.
func makeGPUResult[T tensor.Numeric](e *GPUEngine[T], shape []int, devPtr unsafe.Pointer, numElems int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	gs, err := tensor.NewGPUStorageFromPtr[T](devPtr, numElems, e.deviceID)
	if err != nil {
		return nil, err
	}

	if len(dst) > 0 && dst[0] != nil {
		dst[0].SetStorage(gs)
		dst[0].SetShape(shape)

		return dst[0], nil
	}

	t, err := tensor.NewWithStorage[T](shape, gs)
	if err != nil {
		return nil, err
	}

	return t, nil
}

// gpuBroadcastOp runs a broadcast binary kernel on two float32 tensors.
// Supports 2D broadcasting: row broadcast ([1,D] op [M,D]), column broadcast
// ([M,1] op [M,D]), and same-shape. Falls back to CPU for unsupported patterns.
func gpuBroadcastOp[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a, b *tensor.TensorNumeric[T],
	kernelFn func(devA, devB, devC unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream gpuapi.Stream) error,
	cpuFallback func(context.Context, *tensor.TensorNumeric[T], *tensor.TensorNumeric[T], ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error),
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	// Flatten to 2D for broadcast analysis.
	// For N-D tensors, treat as [product(all-but-last), last].
	aM, aD := flattenTo2D(aShape)
	bM, bD := flattenTo2D(bShape)

	// Determine output shape and broadcast strides.
	var M, D, saRow, saCol, sbRow, sbCol int

	switch {
	case aM == bM && aD == bD:
		// Same shape.
		M, D = aM, aD
		saRow, saCol = aD, 1
		sbRow, sbCol = bD, 1
	case bM == 1 && aD == bD:
		// b is row-broadcast: [1,D] op [M,D].
		M, D = aM, aD
		saRow, saCol = aD, 1
		sbRow, sbCol = 0, 1
	case aM == 1 && aD == bD:
		// a is row-broadcast: [1,D] op [M,D].
		M, D = bM, bD
		saRow, saCol = 0, 1
		sbRow, sbCol = bD, 1
	case aM == bM && bD == 1:
		// b is column-broadcast: [M,1] op [M,D].
		M, D = aM, aD
		saRow, saCol = aD, 1
		sbRow, sbCol = 1, 0
	case aM == bM && aD == 1:
		// a is column-broadcast: [M,1] op [M,D].
		M, D = bM, bD
		saRow, saCol = 1, 0
		sbRow, sbCol = bD, 1
	default:
		// Unsupported broadcast pattern.
		return cpuFallback(ctx, a, b, dst...)
	}

	// Compute proper N-D broadcast output shape (NumPy rules).
	outShape := broadcastShape(aShape, bShape)

	e.setDevice()

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return nil, err
	}
	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return nil, err
	}
	defer cleanupB()

	outElems := M * D
	byteSize := outElems * f32Size
	devC, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return nil, err
	}

	if err := kernelFn(devA, devB, devC, saRow, saCol, sbRow, sbCol, M, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, byteSize)
		return nil, err
	}

	return makeGPUResult[T](e, outShape, devC, outElems, dst...)
}

// flattenTo2D flattens an N-D shape to [M, D] where M = product of all dims except last, D = last dim.
func flattenTo2D(shape []int) (int, int) {
	if len(shape) == 0 {
		return 1, 1
	}
	D := shape[len(shape)-1]
	M := 1
	for i := 0; i < len(shape)-1; i++ {
		M *= shape[i]
	}
	return M, D
}

// gpuBinaryOp runs a binary kernel on two equal-length float32 tensors.
// Uses the device-resident pipeline: inputs via getDevicePtr, output as GPUStorage.
func gpuBinaryOp[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a, b *tensor.TensorNumeric[T],
	kernelFn func(devA, devB, devC unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("GPU kernel: unsupported type %T", zero)
	}

	n := a.GetStorage().Len()
	if b.GetStorage().Len() != n {
		return nil, fmt.Errorf("GPU binary op: length mismatch %d vs %d", n, b.GetStorage().Len())
	}

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return nil, err
	}

	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return nil, err
	}

	defer cleanupB()

	byteSize := n * f32Size

	devC, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return nil, err
	}

	if err := kernelFn(devA, devB, devC, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, byteSize)

		return nil, err
	}

	return makeGPUResult[T](e, a.Shape(), devC, n, dst...)
}

// gpuUnaryOp runs a unary kernel on a float32 tensor.
func gpuUnaryOp[T tensor.Numeric](
	e *GPUEngine[T],
	a *tensor.TensorNumeric[T],
	kernelFn func(devA, devC unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("GPU kernel: unsupported type %T", zero)
	}

	n := a.GetStorage().Len()

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return nil, err
	}

	defer cleanupA()

	byteSize := n * f32Size

	devC, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return nil, err
	}

	if err := kernelFn(devA, devC, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, byteSize)

		return nil, err
	}

	return makeGPUResult[T](e, a.Shape(), devC, n, dst...)
}

// gpuScalarOp runs a scalar kernel on a float32 tensor.
func gpuScalarOp[T tensor.Numeric](
	e *GPUEngine[T],
	a *tensor.TensorNumeric[T],
	scalar float32,
	kernelFn func(devA unsafe.Pointer, scalar float32, devC unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("GPU kernel: unsupported type %T", zero)
	}

	n := a.GetStorage().Len()

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return nil, err
	}

	defer cleanupA()

	byteSize := n * f32Size

	devC, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return nil, err
	}

	if err := kernelFn(devA, scalar, devC, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, byteSize)

		return nil, err
	}

	return makeGPUResult[T](e, a.Shape(), devC, n, dst...)
}

// isFloat32 checks if the generic type T is float32.
func isFloat32[T tensor.Numeric]() bool {
	var zero T
	_, ok := any(zero).(float32)

	return ok
}

// toFloat32 converts a T value to float32 via any.
func toFloat32[T tensor.Numeric](v T) float32 {
	return any(v).(float32)
}

// --- GPU-accelerated method overrides ---

func (e *GPUEngine[T]) gpuAdd(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Add(ctx, a, b, dst...)
	}

	if sameShape(a, b) {
		e.setDevice()
		return gpuBinaryOp(e, ctx, a, b, e.kernels.Add, dst...)
	}

	return gpuBroadcastOp(e, ctx, a, b, e.kernels.AddBroadcast, e.cpu.Add, dst...)
}

func (e *GPUEngine[T]) gpuSub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Sub(ctx, a, b, dst...)
	}

	if sameShape(a, b) {
		e.setDevice()
		return gpuBinaryOp(e, ctx, a, b, e.kernels.Sub, dst...)
	}

	return gpuBroadcastOp(e, ctx, a, b, e.kernels.SubBroadcast, e.cpu.Sub, dst...)
}

func (e *GPUEngine[T]) gpuMul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Mul(ctx, a, b, dst...)
	}

	if sameShape(a, b) {
		e.setDevice()
		return gpuBinaryOp(e, ctx, a, b, e.kernels.Mul, dst...)
	}

	return gpuBroadcastOp(e, ctx, a, b, e.kernels.MulBroadcast, e.cpu.Mul, dst...)
}

func (e *GPUEngine[T]) gpuDiv(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Div(ctx, a, b, dst...)
	}

	if sameShape(a, b) {
		e.setDevice()
		return gpuBinaryOp(e, ctx, a, b, e.kernels.Div, dst...)
	}

	return gpuBroadcastOp(e, ctx, a, b, e.kernels.DivBroadcast, e.cpu.Div, dst...)
}

func (e *GPUEngine[T]) gpuPow(ctx context.Context, base, exponent *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || !sameShape(base, exponent) {
		return e.cpu.Pow(ctx, base, exponent, dst...)
	}

	e.setDevice()

	return gpuBinaryOp(e, ctx, base, exponent, e.kernels.Pow, dst...)
}

func (e *GPUEngine[T]) gpuExp(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Exp(ctx, a, dst...)
	}

	e.setDevice()

	return gpuUnaryOp(e, a, e.kernels.Exp, dst...)
}

func (e *GPUEngine[T]) gpuLog(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Log(ctx, a, dst...)
	}

	e.setDevice()

	return gpuUnaryOp(e, a, e.kernels.Log, dst...)
}

func (e *GPUEngine[T]) gpuSqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Sqrt(ctx, a, dst...)
	}

	e.setDevice()

	return gpuUnaryOp(e, a, e.kernels.Sqrt, dst...)
}

func (e *GPUEngine[T]) gpuRsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Rsqrt(ctx, a, dst...)
	}

	e.setDevice()

	return gpuUnaryOp(e, a, e.kernels.Rsqrt, dst...)
}

func (e *GPUEngine[T]) gpuTanh(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Tanh(ctx, a, dst...)
	}

	e.setDevice()

	return gpuUnaryOp(e, a, e.kernels.Tanh, dst...)
}

func (e *GPUEngine[T]) gpuTanhPrime(ctx context.Context, a, upstream *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || !sameShape(a, upstream) {
		return e.cpu.TanhPrime(ctx, a, upstream, dst...)
	}

	e.setDevice()

	return gpuBinaryOp(e, ctx, a, upstream, e.kernels.TanhPrime, dst...)
}

func (e *GPUEngine[T]) gpuAddScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.AddScalar(ctx, a, scalar, dst...)
	}

	e.setDevice()

	return gpuScalarOp(e, a, toFloat32(scalar), e.kernels.AddScalar, dst...)
}

func (e *GPUEngine[T]) gpuMulScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.MulScalar(ctx, a, scalar, dst...)
	}

	e.setDevice()

	return gpuScalarOp(e, a, toFloat32(scalar), e.kernels.MulScalar, dst...)
}

func (e *GPUEngine[T]) gpuDivScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.DivScalar(ctx, a, scalar, dst...)
	}

	e.setDevice()

	return gpuScalarOp(e, a, toFloat32(scalar), e.kernels.DivScalar, dst...)
}

func (e *GPUEngine[T]) gpuFill(ctx context.Context, t *tensor.TensorNumeric[T], value T) error {
	if !isFloat32[T]() {
		return e.cpu.Fill(ctx, t, value)
	}

	e.setDevice()

	n := t.GetStorage().Len()
	byteSize := n * f32Size

	devPtr, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return e.cpu.Fill(ctx, t, value)
	}

	if err := e.kernels.Fill(devPtr, toFloat32(value), n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devPtr, byteSize)

		return err
	}

	gs, err := tensor.NewGPUStorageFromPtr[T](devPtr, n, e.deviceID)
	if err != nil {
		e.pool.Free(e.deviceID, devPtr, byteSize)

		return err
	}

	t.SetStorage(gs)

	return nil
}

func (e *GPUEngine[T]) gpuSum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Sum(ctx, a, axis, keepDims, dst...)
	}

	e.setDevice()

	if a == nil {
		return nil, fmt.Errorf("Sum: input tensor must not be nil")
	}

	if axis < 0 {
		return e.cpu.Sum(ctx, a, axis, keepDims, dst...)
	}

	shape := a.Shape()
	rank := len(shape)

	if axis >= rank {
		return nil, fmt.Errorf("Sum: axis %d out of bounds for %d dimensions", axis, rank)
	}

	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}

	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}

	axisSize := shape[axis]
	numStripes := outer * inner

	var newShape []int
	if keepDims {
		newShape = make([]int, rank)
		for i, d := range shape {
			if i == axis {
				newShape[i] = 1
			} else {
				newShape[i] = d
			}
		}
	} else {
		for i, d := range shape {
			if i != axis {
				newShape = append(newShape, d)
			}
		}
		if len(newShape) == 0 {
			newShape = []int{1}
		}
	}

	devIn, cleanupIn, err := getDevicePtr(e, a)
	if err != nil {
		e.oomFallbackCount.Add(1)
		e.logger.Warn("Sum: GPU alloc failed, falling back to CPU", "error", err.Error())

		return e.cpu.Sum(ctx, a, axis, keepDims, dst...)
	}

	defer cleanupIn()

	outByteSize := numStripes * f32Size

	devOut, err := e.pool.Alloc(e.deviceID, outByteSize)
	if err != nil {
		e.oomFallbackCount.Add(1)
		e.logger.Warn("Sum: GPU output alloc failed, falling back to CPU", "error", err.Error())

		return e.cpu.Sum(ctx, a, axis, keepDims, dst...)
	}

	if err := e.kernels.SumAxis(devIn, devOut, outer, inner, axisSize, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outByteSize)

		return nil, err
	}

	return makeGPUResult[T](e, newShape, devOut, numStripes, dst...)
}

func (e *GPUEngine[T]) gpuReduceSum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSum(ctx, a, axis, keepDims, dst...)
}

func (e *GPUEngine[T]) gpuReduceMean(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.ReduceMean(ctx, a, axis, keepDims, dst...)
	}

	e.setDevice()

	if axis < 0 {
		return e.cpu.ReduceMean(ctx, a, axis, keepDims, dst...)
	}

	shape := a.Shape()
	rank := len(shape)

	if axis >= rank {
		return nil, fmt.Errorf("ReduceMean: axis %d out of bounds for %d dimensions", axis, rank)
	}

	sumResult, err := e.gpuSum(ctx, a, axis, keepDims)
	if err != nil {
		return nil, err
	}

	divisor := any(float32(shape[axis])).(T)

	return e.gpuDivScalar(ctx, sumResult, divisor, dst...)
}

func (e *GPUEngine[T]) gpuSoftmax(ctx context.Context, a *tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Softmax(ctx, a, axis, dst...)
	}

	e.setDevice()

	if a == nil {
		return nil, fmt.Errorf("Softmax: input tensor must not be nil")
	}

	shape := a.Shape()
	rank := len(shape)

	if rank == 0 {
		return e.cpu.Softmax(ctx, a, axis, dst...)
	}

	if axis < 0 {
		axis = rank + axis
	}

	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("Softmax: axis %d out of bounds for %d dimensions", axis, rank)
	}

	n := a.GetStorage().Len()

	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}

	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}

	axisSize := shape[axis]

	devIn, cleanupIn, err := getDevicePtr(e, a)
	if err != nil {
		e.oomFallbackCount.Add(1)
		e.logger.Warn("Softmax: GPU alloc failed, falling back to CPU", "error", err.Error())

		return e.cpu.Softmax(ctx, a, axis, dst...)
	}

	defer cleanupIn()

	byteSize := n * f32Size

	devOut, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		e.oomFallbackCount.Add(1)
		e.logger.Warn("Softmax: GPU output alloc failed, falling back to CPU", "error", err.Error())

		return e.cpu.Softmax(ctx, a, axis, dst...)
	}

	if err := e.kernels.Softmax(devIn, devOut, outer, inner, axisSize, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, byteSize)

		return nil, err
	}

	return makeGPUResult[T](e, shape, devOut, n, dst...)
}

// sameShape checks if two tensors have the same shape (for non-broadcasting GPU path).
func sameShape[T tensor.Numeric](a, b *tensor.TensorNumeric[T]) bool {
	as := a.Shape()
	bs := b.Shape()

	if len(as) != len(bs) {
		return false
	}

	for i := range as {
		if as[i] != bs[i] {
			return false
		}
	}

	return true
}
