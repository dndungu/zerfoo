package xblas

import "unsafe"

// q4DotBlock computes the dot product of one Q4 block (32 quantized values)
// with 32 float32 activation values. The Q4 block consists of 16 packed bytes
// (two 4-bit values per byte), and a float32 scale factor.
//
// This fuses dequantization and dot product: instead of dequantizing into a
// [32]float32 buffer then computing the dot, nibbles are extracted and
// multiplied directly.
//
// On ARM64, q4DotBlockSIMD provides a NEON-accelerated version.
// This file provides the portable scalar fallback.
func q4DotBlock(packed *byte, scale float32, x *float32, n int) float32 {
	return q4DotBlockScalar(packed, scale, x, n)
}

// q4DotBlockScalar is the pure-Go scalar implementation.
func q4DotBlockScalar(packed *byte, scale float32, x *float32, n int) float32 {
	if n > 32 {
		n = 32
	}
	data := unsafe.Slice(packed, 16)
	xSlice := unsafe.Slice(x, n)

	var sum float32
	for p := range 16 {
		if p*2 >= n {
			break
		}
		byteVal := data[p]
		lo := float32(int(byteVal&0x0F) - 8)
		hi := float32(int(byteVal>>4) - 8)
		sum += lo * xSlice[p*2]
		if p*2+1 < n {
			sum += hi * xSlice[p*2+1]
		}
	}
	return sum * scale
}
