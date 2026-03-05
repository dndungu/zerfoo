package xblas

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/tensor"
)

// GemmQ4F32 computes C = A * B where A is Q4_0 quantized and B, C are float32.
// A has logical shape (m, k), B has shape (k, n), C has shape (m, n).
// Uses the fused dequant+multiply path that avoids heap-allocating the full
// dequantized A matrix. Falls back to dequant+SgemmSimd for non-32-aligned K.
func GemmQ4F32(m, n, k int, a *tensor.Q4Storage, b, c []float32) {
	GemmQ4F32Fused(m, n, k, a, b, c)
}

// GemmQ8F32 computes C = A * B where A is Q8_0 quantized and B, C are float32.
// Dequantizes A once upfront, then uses SIMD-accelerated SGEMM.
func GemmQ8F32(m, n, k int, a *tensor.Q8Storage, b, c []float32) {
	af32 := make([]float32, a.Len())
	a.Dequantize(af32)
	SgemmSimd(m, n, k, af32, b, c)
}

// GemmQ4F32Fused computes C = dequant(A) * B with fused dequant+multiply.
// Instead of allocating a full M*K dequantized buffer, it dequantizes one
// Q4 block (32 values = 128 bytes) at a time into a stack buffer, then
// multiplies using SIMD-accelerated sgemmAccRow. This eliminates the O(M*K)
// heap allocation of GemmQ4F32, making it faster for decode (M=1) where
// the dequant allocation dominates, and equally fast for larger M.
// K must be a multiple of 32; falls back to GemmQ4F32 otherwise.
func GemmQ4F32Fused(m, n, k int, a *tensor.Q4Storage, b, c []float32) {
	if k%32 != 0 {
		GemmQ4F32(m, n, k, a, b, c)
		return
	}

	blocksPerRow := k / 32

	// Zero output.
	for i := range c {
		c[i] = 0
	}

	// Stack buffer for one dequantized Q4 block (32 float32 = 128 bytes).
	var buf [32]float32

	for i := range m {
		cRow := c[i*n : (i+1)*n]
		for bi := range blocksPerRow {
			blkIdx := i*blocksPerRow + bi
			scale := a.BlockScaleF32(blkIdx)
			if scale == 0 {
				continue
			}

			// Dequantize one block into stack buffer.
			dequantQ4Block(a.BlockData(blkIdx), scale, &buf)

			// Accumulate: c[j] += buf[p] * b[(kBase+p)*n + j] for p=0..31
			kBase := bi * 32
			for p := range 32 {
				if aVal := buf[p]; aVal != 0 {
					sgemmAccRow(unsafe.Pointer(&cRow[0]), unsafe.Pointer(&b[(kBase+p)*n]), aVal, n)
				}
			}
		}
	}
}

// dequantQ4Block unpacks 16 packed bytes into 32 float32 values.
func dequantQ4Block(data *byte, scale float32, buf *[32]float32) {
	packed := unsafe.Slice(data, 16)
	for p := range 16 {
		byteVal := packed[p]
		buf[p*2] = float32(int(byteVal&0x0F)-8) * scale
		buf[p*2+1] = float32(int(byteVal>>4)-8) * scale
	}
}
