//go:build opencl

package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/internal/opencl"
)

// OpenCLRuntime implements the Runtime interface using OpenCL.
type OpenCLRuntime struct {
	ctx *opencl.Context
}

// NewOpenCLRuntime returns a new OpenCL runtime adapter.
func NewOpenCLRuntime() *OpenCLRuntime {
	return &OpenCLRuntime{}
}

func (r *OpenCLRuntime) DeviceType() device.Type { return device.OpenCL }

func (r *OpenCLRuntime) SetDevice(deviceID int) error {
	if r.ctx != nil {
		_ = r.ctx.Destroy()
	}
	ctx, err := opencl.NewContext(deviceID)
	if err != nil {
		return err
	}
	r.ctx = ctx
	return nil
}

func (r *OpenCLRuntime) GetDeviceCount() (int, error) {
	return opencl.GetDeviceCount()
}

func (r *OpenCLRuntime) Malloc(byteSize int) (unsafe.Pointer, error) {
	return r.ctx.Malloc(byteSize)
}

func (r *OpenCLRuntime) Free(ptr unsafe.Pointer) error {
	return r.ctx.Free(ptr)
}

func (r *OpenCLRuntime) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	return r.ctx.Memcpy(dst, src, count, openclMemcpyKind(kind))
}

func (r *OpenCLRuntime) MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, _ Stream) error {
	// OpenCL async operations use the context's default queue.
	// For simplicity, fall back to synchronous copy.
	return r.Memcpy(dst, src, count, kind)
}

func (r *OpenCLRuntime) MemcpyPeer(dst unsafe.Pointer, _ int, src unsafe.Pointer, _ int, count int) error {
	// OpenCL does not support peer-to-peer copies directly.
	// Fall back to D2H + H2D via a host buffer.
	buf := make([]byte, count)
	hostPtr := unsafe.Pointer(unsafe.SliceData(buf))
	if err := r.ctx.Memcpy(hostPtr, src, count, opencl.MemcpyDeviceToHost); err != nil {
		return err
	}
	return r.ctx.Memcpy(dst, hostPtr, count, opencl.MemcpyHostToDevice)
}

func (r *OpenCLRuntime) CreateStream() (Stream, error) {
	s, err := r.ctx.CreateStream()
	if err != nil {
		return nil, err
	}
	return &openclStreamWrapper{stream: s}, nil
}

// openclStreamWrapper adapts opencl.Stream to the GRAL Stream interface.
type openclStreamWrapper struct {
	stream *opencl.Stream
}

func (w *openclStreamWrapper) Synchronize() error { return w.stream.Synchronize() }
func (w *openclStreamWrapper) Destroy() error      { return w.stream.Destroy() }
func (w *openclStreamWrapper) Ptr() unsafe.Pointer  { return w.stream.Ptr() }

// openclMemcpyKind converts a GRAL MemcpyKind to an OpenCL MemcpyKind.
func openclMemcpyKind(kind MemcpyKind) opencl.MemcpyKind {
	switch kind {
	case MemcpyHostToDevice:
		return opencl.MemcpyHostToDevice
	case MemcpyDeviceToHost:
		return opencl.MemcpyDeviceToHost
	case MemcpyDeviceToDevice:
		return opencl.MemcpyDeviceToDevice
	default:
		return opencl.MemcpyHostToDevice
	}
}

// Compile-time interface assertion.
var _ Runtime = (*OpenCLRuntime)(nil)
