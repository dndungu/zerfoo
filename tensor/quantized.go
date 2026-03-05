package tensor

import (
	"math"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/device"
)

const q4BlockSize = 32

// q4Block represents a single Q4_0 quantization block.
// Format: 2 bytes float16 scale + 16 bytes packed 4-bit data = 18 bytes per 32 values.
type q4Block struct {
	scale float16.Float16
	data  [16]byte // 32 x 4-bit values packed into 16 bytes
}

// Q4Storage holds Q4_0 quantized tensor data on CPU.
type Q4Storage struct {
	blocks []q4Block
	len    int // number of logical float32 elements (before padding)
}

// QuantizeQ4 quantizes a float32 slice into Q4_0 format.
// The input is padded to a multiple of 32 if necessary.
func QuantizeQ4(src []float32) *Q4Storage {
	n := len(src)
	nBlocks := (n + q4BlockSize - 1) / q4BlockSize
	blocks := make([]q4Block, nBlocks)

	for bi := range nBlocks {
		offset := bi * q4BlockSize

		// Find absmax for this block.
		var absMax float32
		for j := range q4BlockSize {
			idx := offset + j
			var v float32
			if idx < n {
				v = src[idx]
			}
			if av := float32(math.Abs(float64(v))); av > absMax {
				absMax = av
			}
		}

		// Compute scale: maps [-absMax, absMax] to [-8, 7] (signed 4-bit range).
		var scale float32
		if absMax > 0 {
			scale = absMax / 7.0
		}
		blocks[bi].scale = float16.FromFloat32(scale)

		// Quantize values to 4-bit signed integers and pack.
		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		for j := 0; j < q4BlockSize; j += 2 {
			var v0, v1 float32
			if offset+j < n {
				v0 = src[offset+j]
			}
			if offset+j+1 < n {
				v1 = src[offset+j+1]
			}

			q0 := clampInt(int(math.Round(float64(v0*invScale))), -8, 7)
			q1 := clampInt(int(math.Round(float64(v1*invScale))), -8, 7)

			// Pack two 4-bit signed values into one byte.
			// Low nibble = q0+8, high nibble = q1+8 (unsigned offset).
			blocks[bi].data[j/2] = byte(q0+8) | (byte(q1+8) << 4)
		}
	}

	return &Q4Storage{blocks: blocks, len: n}
}

// Dequantize unpacks Q4_0 blocks into dst. len(dst) must be >= q.Len().
func (q *Q4Storage) Dequantize(dst []float32) {
	for bi, blk := range q.blocks {
		scale := blk.scale.ToFloat32()
		offset := bi * q4BlockSize
		for j := 0; j < q4BlockSize; j += 2 {
			packed := blk.data[j/2]
			q0 := int(packed&0x0F) - 8
			q1 := int(packed>>4) - 8

			if idx := offset + j; idx < q.len {
				dst[idx] = float32(q0) * scale
			}
			if idx := offset + j + 1; idx < q.len {
				dst[idx] = float32(q1) * scale
			}
		}
	}
}

// Len returns the number of logical float32 elements.
func (q *Q4Storage) Len() int { return q.len }

// NumBlocks returns the number of Q4_0 blocks.
func (q *Q4Storage) NumBlocks() int { return len(q.blocks) }

// ByteSize returns the raw byte size of the quantized data.
// Each block is 18 bytes (2 byte scale + 16 bytes packed data).
func (q *Q4Storage) ByteSize() int { return len(q.blocks) * 18 }

// Slice returns a dequantized float32 copy of the data.
func (q *Q4Storage) Slice() []float32 {
	dst := make([]float32, q.len)
	q.Dequantize(dst)
	return dst
}

// Set is not supported on quantized storage (weights are immutable).
func (q *Q4Storage) Set(_ []float32) { panic("Q4Storage is immutable") }

// DeviceType returns device.CPU.
func (q *Q4Storage) DeviceType() device.Type { return device.CPU }

// Ensure Q4Storage implements Storage[float32].
var _ Storage[float32] = (*Q4Storage)(nil)

// ---------------------------------------------------------------------------
// Q8_0 format: 32 values per block.
// Each block = 4 bytes float32 scale + 32 bytes int8 data = 36 bytes per 32 values.
// ---------------------------------------------------------------------------

const q8BlockSize = 32

type q8Block struct {
	scale float32
	data  [32]int8
}

// Q8Storage holds Q8_0 quantized tensor data on CPU.
type Q8Storage struct {
	blocks []q8Block
	len    int
}

// QuantizeQ8 quantizes a float32 slice into Q8_0 format.
func QuantizeQ8(src []float32) *Q8Storage {
	n := len(src)
	nBlocks := (n + q8BlockSize - 1) / q8BlockSize
	blocks := make([]q8Block, nBlocks)

	for bi := range nBlocks {
		offset := bi * q8BlockSize

		var absMax float32
		for j := range q8BlockSize {
			idx := offset + j
			var v float32
			if idx < n {
				v = src[idx]
			}
			if av := float32(math.Abs(float64(v))); av > absMax {
				absMax = av
			}
		}

		var scale float32
		if absMax > 0 {
			scale = absMax / 127.0
		}
		blocks[bi].scale = scale

		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		for j := range q8BlockSize {
			var v float32
			if offset+j < n {
				v = src[offset+j]
			}
			blocks[bi].data[j] = int8(clampInt(int(math.Round(float64(v*invScale))), -128, 127))
		}
	}

	return &Q8Storage{blocks: blocks, len: n}
}

// Dequantize unpacks Q8_0 blocks into dst.
func (q *Q8Storage) Dequantize(dst []float32) {
	for bi, blk := range q.blocks {
		offset := bi * q8BlockSize
		for j := range q8BlockSize {
			idx := offset + j
			if idx >= q.len {
				break
			}
			dst[idx] = float32(blk.data[j]) * blk.scale
		}
	}
}

// Len returns the number of logical float32 elements.
func (q *Q8Storage) Len() int { return q.len }

// NumBlocks returns the number of Q8_0 blocks.
func (q *Q8Storage) NumBlocks() int { return len(q.blocks) }

// ByteSize returns the raw byte size of the quantized data.
func (q *Q8Storage) ByteSize() int { return len(q.blocks) * 36 }

// Slice returns a dequantized float32 copy of the data.
func (q *Q8Storage) Slice() []float32 {
	dst := make([]float32, q.len)
	q.Dequantize(dst)
	return dst
}

// Set is not supported on quantized storage (weights are immutable).
func (q *Q8Storage) Set(_ []float32) { panic("Q8Storage is immutable") }

// DeviceType returns device.CPU.
func (q *Q8Storage) DeviceType() device.Type { return device.CPU }

// Ensure Q8Storage implements Storage[float32].
var _ Storage[float32] = (*Q8Storage)(nil)

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
