//go:build !arm64

package xblas

// q4DotBlock on non-arm64 platforms uses the scalar fallback.
func q4DotBlock(packed *byte, scale float32, x *float32, n int) float32 {
	return q4DotBlockScalar(packed, scale, x, n)
}
