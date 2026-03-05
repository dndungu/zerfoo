//go:build opencl

package opencl

/*
#cgo LDFLAGS: -lOpenCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdlib.h>
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
	MemcpyHostToDevice MemcpyKind = iota
	// MemcpyDeviceToHost copies from device to host.
	MemcpyDeviceToHost
	// MemcpyDeviceToDevice copies from device to device.
	MemcpyDeviceToDevice
)

// Context holds an OpenCL context, device, and default command queue.
type Context struct {
	platform C.cl_platform_id
	device   C.cl_device_id
	ctx      C.cl_context
	queue    C.cl_command_queue
	deviceID int
}

// Stream wraps an OpenCL command queue.
type Stream struct {
	queue C.cl_command_queue
}

// NewContext creates an OpenCL context on the specified device.
// If deviceID is -1, the first available GPU device is used.
func NewContext(deviceID int) (*Context, error) {
	var numPlatforms C.cl_uint
	if err := C.clGetPlatformIDs(0, nil, &numPlatforms); err != C.CL_SUCCESS {
		return nil, fmt.Errorf("clGetPlatformIDs: error %d", err)
	}
	if numPlatforms == 0 {
		return nil, fmt.Errorf("no OpenCL platforms found")
	}

	platforms := make([]C.cl_platform_id, numPlatforms)
	if err := C.clGetPlatformIDs(numPlatforms, &platforms[0], nil); err != C.CL_SUCCESS {
		return nil, fmt.Errorf("clGetPlatformIDs: error %d", err)
	}

	// Find GPU devices across all platforms.
	var allDevices []C.cl_device_id
	var devicePlatforms []C.cl_platform_id
	for _, p := range platforms {
		var numDevices C.cl_uint
		ret := C.clGetDeviceIDs(p, C.CL_DEVICE_TYPE_GPU, 0, nil, &numDevices)
		if ret != C.CL_SUCCESS || numDevices == 0 {
			continue
		}
		devs := make([]C.cl_device_id, numDevices)
		if err := C.clGetDeviceIDs(p, C.CL_DEVICE_TYPE_GPU, numDevices, &devs[0], nil); err != C.CL_SUCCESS {
			continue
		}
		for _, d := range devs {
			allDevices = append(allDevices, d)
			devicePlatforms = append(devicePlatforms, p)
		}
	}

	if len(allDevices) == 0 {
		return nil, fmt.Errorf("no OpenCL GPU devices found")
	}

	dev := deviceID
	if dev < 0 {
		dev = 0
	}
	if dev >= len(allDevices) {
		return nil, fmt.Errorf("OpenCL device %d not found (have %d)", dev, len(allDevices))
	}

	selectedDevice := allDevices[dev]
	selectedPlatform := devicePlatforms[dev]

	var errCode C.cl_int
	ctx := C.clCreateContext(nil, 1, &selectedDevice, nil, nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, fmt.Errorf("clCreateContext: error %d", errCode)
	}

	queue := C.clCreateCommandQueue(ctx, selectedDevice, 0, &errCode)
	if errCode != C.CL_SUCCESS {
		C.clReleaseContext(ctx)
		return nil, fmt.Errorf("clCreateCommandQueue: error %d", errCode)
	}

	return &Context{
		platform: selectedPlatform,
		device:   selectedDevice,
		ctx:      ctx,
		queue:    queue,
		deviceID: dev,
	}, nil
}

// Destroy releases the OpenCL context and default command queue.
func (c *Context) Destroy() error {
	if c.queue != nil {
		C.clReleaseCommandQueue(c.queue)
	}
	if c.ctx != nil {
		C.clReleaseContext(c.ctx)
	}
	return nil
}

// GetDeviceCount returns the total number of OpenCL GPU devices.
func GetDeviceCount() (int, error) {
	var numPlatforms C.cl_uint
	if err := C.clGetPlatformIDs(0, nil, &numPlatforms); err != C.CL_SUCCESS {
		return 0, fmt.Errorf("clGetPlatformIDs: error %d", err)
	}

	total := 0
	platforms := make([]C.cl_platform_id, numPlatforms)
	if err := C.clGetPlatformIDs(numPlatforms, &platforms[0], nil); err != C.CL_SUCCESS {
		return 0, fmt.Errorf("clGetPlatformIDs: error %d", err)
	}

	for _, p := range platforms {
		var numDevices C.cl_uint
		ret := C.clGetDeviceIDs(p, C.CL_DEVICE_TYPE_GPU, 0, nil, &numDevices)
		if ret == C.CL_SUCCESS {
			total += int(numDevices)
		}
	}
	return total, nil
}

// Malloc allocates a device buffer of the given size.
// Returns the cl_mem handle cast to unsafe.Pointer.
func (c *Context) Malloc(size int) (unsafe.Pointer, error) {
	var errCode C.cl_int
	mem := C.clCreateBuffer(c.ctx, C.CL_MEM_READ_WRITE, C.size_t(size), nil, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, fmt.Errorf("clCreateBuffer(%d bytes): error %d", size, errCode)
	}
	return unsafe.Pointer(mem), nil
}

// Free releases a device buffer.
func (c *Context) Free(ptr unsafe.Pointer) error {
	if ptr == nil {
		return nil
	}
	mem := C.cl_mem(ptr)
	if err := C.clReleaseMemObject(mem); err != C.CL_SUCCESS {
		return fmt.Errorf("clReleaseMemObject: error %d", err)
	}
	return nil
}

// Memcpy copies data between host and device.
func (c *Context) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	switch kind {
	case MemcpyHostToDevice:
		// src is host pointer, dst is cl_mem
		mem := C.cl_mem(dst)
		if err := C.clEnqueueWriteBuffer(c.queue, mem, C.CL_TRUE, 0, C.size_t(count), src, 0, nil, nil); err != C.CL_SUCCESS {
			return fmt.Errorf("clEnqueueWriteBuffer: error %d", err)
		}
	case MemcpyDeviceToHost:
		// src is cl_mem, dst is host pointer
		mem := C.cl_mem(src)
		if err := C.clEnqueueReadBuffer(c.queue, mem, C.CL_TRUE, 0, C.size_t(count), dst, 0, nil, nil); err != C.CL_SUCCESS {
			return fmt.Errorf("clEnqueueReadBuffer: error %d", err)
		}
	case MemcpyDeviceToDevice:
		// Both are cl_mem
		srcMem := C.cl_mem(src)
		dstMem := C.cl_mem(dst)
		if err := C.clEnqueueCopyBuffer(c.queue, srcMem, dstMem, 0, 0, C.size_t(count), 0, nil, nil); err != C.CL_SUCCESS {
			return fmt.Errorf("clEnqueueCopyBuffer: error %d", err)
		}
		// Wait for copy to complete.
		C.clFinish(c.queue)
	default:
		return fmt.Errorf("unsupported MemcpyKind: %d", kind)
	}
	return nil
}

// CreateStream creates a new command queue (stream equivalent).
func (c *Context) CreateStream() (*Stream, error) {
	var errCode C.cl_int
	queue := C.clCreateCommandQueue(c.ctx, c.device, 0, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, fmt.Errorf("clCreateCommandQueue: error %d", errCode)
	}
	return &Stream{queue: queue}, nil
}

// Synchronize waits for all commands in the command queue to complete.
func (s *Stream) Synchronize() error {
	if err := C.clFinish(s.queue); err != C.CL_SUCCESS {
		return fmt.Errorf("clFinish: error %d", err)
	}
	return nil
}

// Destroy releases the command queue.
func (s *Stream) Destroy() error {
	if s.queue != nil {
		C.clReleaseCommandQueue(s.queue)
		s.queue = nil
	}
	return nil
}

// Ptr returns the underlying command queue handle as unsafe.Pointer.
func (s *Stream) Ptr() unsafe.Pointer {
	return unsafe.Pointer(s.queue)
}

// CLContext returns the underlying cl_context for kernel compilation.
func (c *Context) CLContext() unsafe.Pointer {
	return unsafe.Pointer(c.ctx)
}

// CLQueue returns the default command queue.
func (c *Context) CLQueue() unsafe.Pointer {
	return unsafe.Pointer(c.queue)
}

// CLDevice returns the underlying cl_device_id.
func (c *Context) CLDevice() unsafe.Pointer {
	return unsafe.Pointer(c.device)
}

// SetDevice is a no-op for OpenCL (device is selected at context creation).
// It stores the device ID for compatibility with the GRAL interface.
func (c *Context) SetDevice(deviceID int) error {
	c.deviceID = deviceID
	return nil
}

// DeviceID returns the device ordinal.
func (c *Context) DeviceID() int {
	return c.deviceID
}
