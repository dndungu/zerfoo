package cuda

import (
	"fmt"
	"sync"
)

// CUDALib holds dlopen handles and resolved function pointers for
// CUDA runtime functions. All function pointers are resolved at Open()
// time via dlsym. The actual calls go through platform-specific ccall
// implementations that do NOT use CGo (zero runtime.cgocall overhead).
type CUDALib struct {
	handle uintptr // dlopen handle for libcudart

	// CUDA runtime function pointers
	cudaMalloc             uintptr
	cudaFree               uintptr
	cudaMemcpy             uintptr
	cudaMemcpyAsync        uintptr
	cudaMallocManaged      uintptr
	cudaStreamCreate       uintptr
	cudaStreamSynchronize  uintptr
	cudaStreamDestroy      uintptr
	cudaGetDeviceCount     uintptr
	cudaSetDevice          uintptr
	cudaGetErrorString     uintptr
	cudaGetDeviceProperties uintptr
	cudaMemcpyPeer         uintptr
}

var (
	globalLib  *CUDALib
	globalOnce sync.Once
	errGlobal  error
)

// cudartPaths lists the shared library names to try, in order.
// On Linux, libcudart.so is the standard name. The versioned
// name (libcudart.so.12) is tried first for specificity.
var cudartPaths = []string{
	"libcudart.so.12",
	"libcudart.so",
}

// Open loads libcudart via dlopen and resolves all CUDA runtime
// function pointers via dlsym. Returns an error if CUDA is not
// available (library not found or symbols missing).
func Open() (*CUDALib, error) {
	lib := &CUDALib{}

	// Try each library path until one succeeds.
	var lastErr string
	for _, path := range cudartPaths {
		h := dlopenImpl(path, rtldLazy|rtldGlobal)
		if h != 0 {
			lib.handle = h
			break
		}
		lastErr = dlerrorImpl()
	}
	if lib.handle == 0 {
		return nil, fmt.Errorf("cuda: dlopen libcudart failed: %s", lastErr)
	}

	// Resolve all required function pointers.
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"cudaMalloc", &lib.cudaMalloc},
		{"cudaFree", &lib.cudaFree},
		{"cudaMemcpy", &lib.cudaMemcpy},
		{"cudaMemcpyAsync", &lib.cudaMemcpyAsync},
		{"cudaMallocManaged", &lib.cudaMallocManaged},
		{"cudaStreamCreate", &lib.cudaStreamCreate},
		{"cudaStreamSynchronize", &lib.cudaStreamSynchronize},
		{"cudaStreamDestroy", &lib.cudaStreamDestroy},
		{"cudaGetDeviceCount", &lib.cudaGetDeviceCount},
		{"cudaSetDevice", &lib.cudaSetDevice},
		{"cudaGetErrorString", &lib.cudaGetErrorString},
		{"cudaGetDeviceProperties", &lib.cudaGetDeviceProperties},
		{"cudaMemcpyPeer", &lib.cudaMemcpyPeer},
	}
	for _, s := range syms {
		addr := dlsymImpl(lib.handle, s.name)
		if addr == 0 {
			_ = lib.Close()
			return nil, fmt.Errorf("cuda: dlsym %s failed: %s", s.name, dlerrorImpl())
		}
		*s.ptr = addr
	}

	return lib, nil
}

// Close releases the dlopen handle.
func (lib *CUDALib) Close() error {
	if lib.handle != 0 {
		dlcloseImpl(lib.handle)
		lib.handle = 0
	}
	return nil
}

// Available returns true if CUDA runtime is loadable on this machine.
// The result is cached after the first call.
func Available() bool {
	globalOnce.Do(func() {
		globalLib, errGlobal = Open()
	})
	return errGlobal == nil
}

// Lib returns the global CUDALib instance, or nil if CUDA is not available.
func Lib() *CUDALib {
	if !Available() {
		return nil
	}
	return globalLib
}

// dlopen flags
const (
	rtldLazy   = 0x1
	rtldGlobal = 0x100
)
