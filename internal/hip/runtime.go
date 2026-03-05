//go:build rocm

package hip

/*
#cgo LDFLAGS: -lamdhip64
#include <hip/hip_runtime.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// MemcpyKind specifies the direction of a memory copy.
type MemcpyKind int

const (
	// MemcpyHostToDevice copies from host to device.
	MemcpyHostToDevice MemcpyKind = C.hipMemcpyHostToDevice
	// MemcpyDeviceToHost copies from device to host.
	MemcpyDeviceToHost MemcpyKind = C.hipMemcpyDeviceToHost
	// MemcpyDeviceToDevice copies from device to device.
	MemcpyDeviceToDevice MemcpyKind = C.hipMemcpyDeviceToDevice
)

// Malloc allocates size bytes on the HIP device and returns a device pointer.
func Malloc(size int) (unsafe.Pointer, error) {
	var devPtr unsafe.Pointer

	err := C.hipMalloc(&devPtr, C.size_t(size))
	if err != C.hipSuccess {
		return nil, fmt.Errorf("hipMalloc failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return devPtr, nil
}

// Free releases device memory previously allocated with Malloc.
func Free(devPtr unsafe.Pointer) error {
	err := C.hipFree(devPtr)
	if err != C.hipSuccess {
		return fmt.Errorf("hipFree failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return nil
}

// Memcpy copies count bytes between host and device memory.
func Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	err := C.hipMemcpy(dst, src, C.size_t(count), uint32(kind))
	if err != C.hipSuccess {
		return fmt.Errorf("hipMemcpy failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return nil
}

// GetDeviceCount returns the number of HIP-capable devices.
func GetDeviceCount() (int, error) {
	var count C.int

	err := C.hipGetDeviceCount(&count)
	if err != C.hipSuccess {
		return 0, fmt.Errorf("hipGetDeviceCount failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return int(count), nil
}

// SetDevice sets the current HIP device.
func SetDevice(deviceID int) error {
	err := C.hipSetDevice(C.int(deviceID))
	if err != C.hipSuccess {
		return fmt.Errorf("hipSetDevice failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return nil
}

// Stream wraps a hipStream_t for asynchronous kernel execution and memory transfers.
type Stream struct {
	s C.hipStream_t
}

// CreateStream creates a new HIP stream.
func CreateStream() (*Stream, error) {
	var s C.hipStream_t

	err := C.hipStreamCreate(&s)
	if err != C.hipSuccess {
		return nil, fmt.Errorf("hipStreamCreate failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return &Stream{s: s}, nil
}

// Synchronize blocks the calling CPU thread until all previously issued work
// on this stream has completed.
func (s *Stream) Synchronize() error {
	err := C.hipStreamSynchronize(s.s)
	if err != C.hipSuccess {
		return fmt.Errorf("hipStreamSynchronize failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return nil
}

// Destroy releases the HIP stream. The stream must not be used after Destroy.
func (s *Stream) Destroy() error {
	err := C.hipStreamDestroy(s.s)
	if err != C.hipSuccess {
		return fmt.Errorf("hipStreamDestroy failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return nil
}

// Ptr returns the underlying hipStream_t as an unsafe.Pointer.
// This is used to pass the stream to kernel launchers via CGO.
func (s *Stream) Ptr() unsafe.Pointer {
	return unsafe.Pointer(s.s)
}

// MemcpyPeer copies count bytes between devices using peer-to-peer transfer.
// This enables direct GPU-to-GPU copy without staging through host memory.
func MemcpyPeer(dst unsafe.Pointer, dstDevice int, src unsafe.Pointer, srcDevice int, count int) error {
	err := C.hipMemcpyPeer(dst, C.int(dstDevice), src, C.int(srcDevice), C.size_t(count))
	if err != C.hipSuccess {
		return fmt.Errorf("hipMemcpyPeer failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return nil
}

// MemcpyAsync copies count bytes asynchronously on the given stream.
func MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, stream *Stream) error {
	var hs C.hipStream_t
	if stream != nil {
		hs = stream.s
	}

	err := C.hipMemcpyAsync(dst, src, C.size_t(count), uint32(kind), hs)
	if err != C.hipSuccess {
		return fmt.Errorf("hipMemcpyAsync failed: %s", C.GoString(C.hipGetErrorString(err)))
	}

	return nil
}
