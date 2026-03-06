//go:build arm64

package xblas

// q4DotBlockSIMD computes the dot product of one Q4 block (32 quantized values)
// with 32 float32 activation values using NEON. The Q4 block consists of 16
// packed bytes (two 4-bit values per byte) and a float32 scale factor.
//
// This fuses dequantization and dot product in NEON registers: nibbles are
// extracted, converted to float32, and immediately multiplied with activations
// via FMLA — no intermediate buffer is written to memory.
//
// Implemented in q4dot_arm64.s.
//
//go:noescape
func q4DotBlockSIMD(packed *byte, scale float32, x *float32) float32

// q4DotBlock dispatches to the NEON implementation on arm64.
func q4DotBlock(packed *byte, scale float32, x *float32, n int) float32 {
	if n >= 32 {
		return q4DotBlockSIMD(packed, scale, x)
	}
	return q4DotBlockScalar(packed, scale, x, n)
}
