//go:build cuda

package codegen

/*
#cgo LDFLAGS: -ldl -lcuda -lcudart
#include <dlfcn.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef int (*launch_fn)(float* workspace, float** frozen, int pos, int total_size);

static int call_launch(void* fn, float* workspace, float** frozen, int pos, int total_size) {
    return ((launch_fn)fn)(workspace, frozen, pos, total_size);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// MegakernelRunner manages a compiled megakernel .so and its GPU resources.
type MegakernelRunner struct {
	soHandle    unsafe.Pointer // dlopen handle
	launchFn    unsafe.Pointer // dlsym'd launch_megakernel
	workspace   unsafe.Pointer // GPU workspace buffer
	frozenPtrs  unsafe.Pointer // GPU array of float* pointers to frozen data
	frozenBufs  []unsafe.Pointer
	layout      WorkspaceLayout
	outputShape []int
}

// LoadMegakernel opens a compiled megakernel .so and resolves the launch symbol.
func LoadMegakernel(soPath string) (*MegakernelRunner, error) {
	cPath := C.CString(soPath)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.dlopen(cPath, C.RTLD_NOW)
	if handle == nil {
		errMsg := C.GoString(C.dlerror())
		return nil, fmt.Errorf("load megakernel: %s", errMsg)
	}

	cSym := C.CString("launch_megakernel")
	defer C.free(unsafe.Pointer(cSym))

	fn := C.dlsym(handle, cSym)
	if fn == nil {
		errMsg := C.GoString(C.dlerror())
		C.dlclose(handle)
		return nil, fmt.Errorf("resolve launch_megakernel: %s", errMsg)
	}

	return &MegakernelRunner{soHandle: handle, launchFn: fn}, nil
}

// PrepareWorkspace allocates GPU memory for the workspace and frozen slots.
func (r *MegakernelRunner) PrepareWorkspace(cfg MegakernelConfig, frozenData [][]float32) error {
	r.layout = ComputeWorkspaceLayout(cfg)

	// Allocate workspace buffer.
	wsBytes := r.layout.TotalSize * 4
	if wsBytes > 0 {
		var devPtr unsafe.Pointer
		if ret := C.cudaMalloc(&devPtr, C.size_t(wsBytes)); ret != 0 {
			return fmt.Errorf("alloc workspace (%d bytes): cuda error %d", wsBytes, ret)
		}
		r.workspace = devPtr
	}

	nFrozen := len(cfg.FrozenSlots)
	if nFrozen == 0 {
		return nil
	}
	if len(frozenData) != nFrozen {
		return fmt.Errorf("frozenData length %d != FrozenSlots length %d", len(frozenData), nFrozen)
	}

	// Allocate and upload each frozen slot to GPU.
	r.frozenBufs = make([]unsafe.Pointer, nFrozen)
	hostPtrs := make([]uintptr, nFrozen)
	for i, data := range frozenData {
		if len(data) == 0 {
			continue
		}
		var devPtr unsafe.Pointer
		nbytes := len(data) * 4
		if ret := C.cudaMalloc(&devPtr, C.size_t(nbytes)); ret != 0 {
			return fmt.Errorf("alloc frozen slot %d: cuda error %d", i, ret)
		}
		if ret := C.cudaMemcpy(devPtr, unsafe.Pointer(&data[0]), C.size_t(nbytes), C.cudaMemcpyHostToDevice); ret != 0 {
			return fmt.Errorf("upload frozen slot %d: cuda error %d", i, ret)
		}
		r.frozenBufs[i] = devPtr
		hostPtrs[i] = uintptr(devPtr)
	}

	// Upload the pointer array to GPU (8 bytes per pointer on 64-bit).
	ptrArrayBytes := nFrozen * 8
	var devPtrArray unsafe.Pointer
	if ret := C.cudaMalloc(&devPtrArray, C.size_t(ptrArrayBytes)); ret != 0 {
		return fmt.Errorf("alloc frozen ptr array: cuda error %d", ret)
	}
	if ret := C.cudaMemcpy(devPtrArray, unsafe.Pointer(&hostPtrs[0]), C.size_t(ptrArrayBytes), C.cudaMemcpyHostToDevice); ret != 0 {
		return fmt.Errorf("upload frozen ptr array: cuda error %d", ret)
	}
	r.frozenPtrs = devPtrArray

	// Store output shape for callers.
	if cfg.OutputSlot < len(cfg.SlotShapes) {
		r.outputShape = cfg.SlotShapes[cfg.OutputSlot]
	}

	return nil
}

// OutputShape returns the shape of the megakernel output slot.
func (r *MegakernelRunner) OutputShape() []int {
	return r.outputShape
}

// Launch runs the megakernel with input data and returns the output.
func (r *MegakernelRunner) Launch(inputData []float32, pos int) ([]float32, error) {
	// Copy input to workspace at InputOffset.
	if len(inputData) > 0 {
		dstPtr := unsafe.Add(r.workspace, r.layout.InputOffset*4)
		if ret := C.cudaMemcpy(dstPtr, unsafe.Pointer(&inputData[0]), C.size_t(len(inputData)*4), C.cudaMemcpyHostToDevice); ret != 0 {
			return nil, fmt.Errorf("upload input: cuda error %d", ret)
		}
	}

	// Launch the megakernel.
	ret := C.call_launch(r.launchFn,
		(*C.float)(r.workspace),
		(**C.float)(r.frozenPtrs),
		C.int(pos),
		C.int(r.layout.TotalSize),
	)
	if ret != 0 {
		return nil, fmt.Errorf("megakernel launch failed: cuda error %d", ret)
	}

	// Synchronize to ensure kernel completion.
	if ret := C.cudaDeviceSynchronize(); ret != 0 {
		return nil, fmt.Errorf("megakernel sync failed: cuda error %d", ret)
	}

	// Copy output from workspace at OutputOffset.
	output := make([]float32, r.layout.OutputSize)
	srcPtr := unsafe.Add(r.workspace, r.layout.OutputOffset*4)
	if ret := C.cudaMemcpy(unsafe.Pointer(&output[0]), srcPtr, C.size_t(r.layout.OutputSize*4), C.cudaMemcpyDeviceToHost); ret != 0 {
		return nil, fmt.Errorf("download output: cuda error %d", ret)
	}

	return output, nil
}

// InitWorkspaceSlot copies float32 data to a workspace slot at the given offset.
func (r *MegakernelRunner) InitWorkspaceSlot(offset int, data []float32) error {
	if len(data) == 0 || r.workspace == nil {
		return nil
	}
	dstPtr := unsafe.Add(r.workspace, offset*4)
	if ret := C.cudaMemcpy(dstPtr, unsafe.Pointer(&data[0]), C.size_t(len(data)*4), C.cudaMemcpyHostToDevice); ret != 0 {
		return fmt.Errorf("init workspace slot: cuda error %d", ret)
	}
	return nil
}

// ClearGPUError clears the sticky CUDA error state after a kernel failure.
func (r *MegakernelRunner) ClearGPUError() {
	C.cudaGetLastError()
	C.cudaDeviceSynchronize()
	C.cudaGetLastError()
}

// Close releases all GPU resources.
func (r *MegakernelRunner) Close() error {
	for _, buf := range r.frozenBufs {
		if buf != nil {
			C.cudaFree(buf)
		}
	}
	if r.frozenPtrs != nil {
		C.cudaFree(r.frozenPtrs)
	}
	if r.workspace != nil {
		C.cudaFree(r.workspace)
	}
	if r.soHandle != nil {
		C.dlclose(r.soHandle)
	}
	return nil
}
