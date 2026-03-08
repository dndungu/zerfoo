//go:build !arm64

package xblas

import (
	"math"
	"unsafe"
)

func RMSNormF32(out, x, weight *float32, D int, eps float32) float32 {
	xSlice := unsafe.Slice(x, D)
	wSlice := unsafe.Slice(weight, D)
	oSlice := unsafe.Slice(out, D)
	var sumSq float32
	for i := range D {
		sumSq += xSlice[i] * xSlice[i]
	}
	scale := float32(1.0 / math.Sqrt(float64(sumSq/float32(D)+eps)))
	for i := range D {
		oSlice[i] = xSlice[i] * scale * wSlice[i]
	}
	return scale
}
