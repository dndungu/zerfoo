package tensor

import (
	"encoding/binary"
	"fmt"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/device"
)

// Q4_K format: super-blocks of 256 values.
// Each super-block is 144 bytes:
//   - 2 bytes: fp16 d (super-block scale)
//   - 2 bytes: fp16 dmin (super-block min)
//   - 12 bytes: packed 6-bit scales and mins for 8 sub-blocks
//   - 128 bytes: 256 x 4-bit quantized values packed
//
// Reference: llama.cpp ggml-quants.c dequantize_row_q4_K
const (
	q4KSuperBlockSize  = 256
	q4KBlockBytes      = 144
	q4KNumSubBlocks    = 8
	q4KSubBlockSize    = 32
)

// DequantizeQ4K dequantizes one Q4_K super-block (144 bytes) into 256 float32 values.
func DequantizeQ4K(raw []byte, dst []float32) {
	d := float16.FromBits(binary.LittleEndian.Uint16(raw[0:2])).ToFloat32()
	dmin := float16.FromBits(binary.LittleEndian.Uint16(raw[2:4])).ToFloat32()

	// Decode 6-bit scales and mins for each sub-block.
	// Bytes 4-11: low 4 bits of scale and min for each sub-block.
	// Bytes 12-15: high 2 bits packed.
	var scales [q4KNumSubBlocks]uint8
	var mins [q4KNumSubBlocks]uint8

	for i := range q4KNumSubBlocks {
		scales[i] = raw[4+i] & 0xF
		mins[i] = raw[4+i] >> 4
	}
	// Unpack high 2 bits.
	scales[0] |= (raw[12] & 3) << 4
	scales[1] |= ((raw[12] >> 2) & 3) << 4
	scales[2] |= ((raw[12] >> 4) & 3) << 4
	scales[3] |= ((raw[12] >> 6) & 3) << 4
	scales[4] |= (raw[13] & 3) << 4
	scales[5] |= ((raw[13] >> 2) & 3) << 4
	scales[6] |= ((raw[13] >> 4) & 3) << 4
	scales[7] |= ((raw[13] >> 6) & 3) << 4

	mins[0] |= (raw[14] & 3) << 4
	mins[1] |= ((raw[14] >> 2) & 3) << 4
	mins[2] |= ((raw[14] >> 4) & 3) << 4
	mins[3] |= ((raw[14] >> 6) & 3) << 4
	mins[4] |= (raw[15] & 3) << 4
	mins[5] |= ((raw[15] >> 2) & 3) << 4
	mins[6] |= ((raw[15] >> 4) & 3) << 4
	mins[7] |= ((raw[15] >> 6) & 3) << 4

	// Dequantize 256 values.
	qdata := raw[16:] // 128 bytes of packed 4-bit values
	for sb := range q4KNumSubBlocks {
		sc := d * float32(scales[sb])
		mn := dmin * float32(mins[sb])
		off := sb * q4KSubBlockSize

		for j := 0; j < q4KSubBlockSize; j += 2 {
			byteIdx := (off + j) / 2
			packed := qdata[byteIdx]
			q0 := packed & 0xF
			q1 := packed >> 4
			dst[off+j] = sc*float32(q0) - mn
			dst[off+j+1] = sc*float32(q1) - mn
		}
	}
}

// Q4KStorage holds Q4_K quantized tensor data on CPU.
type Q4KStorage struct {
	raw []byte // raw super-block data
	len int    // number of logical float32 elements
}

// NewQ4KStorageFromRaw creates Q4KStorage from raw super-block data.
func NewQ4KStorageFromRaw(raw []byte, numElements int) (*Q4KStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + q4KSuperBlockSize - 1) / q4KSuperBlockSize
	need := nBlocks * q4KBlockBytes
	if len(raw) < need {
		return nil, fmt.Errorf("Q4_K raw data too short: need %d bytes for %d blocks, got %d", need, nBlocks, len(raw))
	}
	data := make([]byte, need)
	copy(data, raw[:need])
	return &Q4KStorage{raw: data, len: numElements}, nil
}

// Dequantize unpacks all Q4_K super-blocks into dst.
func (q *Q4KStorage) Dequantize(dst []float32) {
	nBlocks := (q.len + q4KSuperBlockSize - 1) / q4KSuperBlockSize
	for bi := range nBlocks {
		blockRaw := q.raw[bi*q4KBlockBytes : (bi+1)*q4KBlockBytes]
		off := bi * q4KSuperBlockSize
		remaining := q.len - off
		if remaining >= q4KSuperBlockSize {
			DequantizeQ4K(blockRaw, dst[off:off+q4KSuperBlockSize])
		} else {
			var tmp [q4KSuperBlockSize]float32
			DequantizeQ4K(blockRaw, tmp[:])
			copy(dst[off:], tmp[:remaining])
		}
	}
}

func (q *Q4KStorage) Len() int                    { return q.len }
func (q *Q4KStorage) Slice() []float32             { dst := make([]float32, q.len); q.Dequantize(dst); return dst }
func (q *Q4KStorage) Set(_ []float32)              { panic("Q4KStorage is immutable") }
func (q *Q4KStorage) DeviceType() device.Type      { return device.CPU }

var _ Storage[float32] = (*Q4KStorage)(nil)

// Q6_K format: super-blocks of 256 values.
// Each super-block is 210 bytes:
//   - 128 bytes: ql (low 4 bits of each 6-bit value)
//   - 64 bytes: qh (high 2 bits of each 6-bit value)
//   - 16 bytes: int8 scales for 16 sub-blocks of 16 values
//   - 2 bytes: fp16 d (super-block scale)
//
// Reference: llama.cpp ggml-quants.c dequantize_row_q6_K
const (
	q6KSuperBlockSize = 256
	q6KBlockBytes     = 210
)

// DequantizeQ6K dequantizes one Q6_K super-block (210 bytes) into 256 float32 values.
func DequantizeQ6K(raw []byte, dst []float32) {
	ql := raw[0:128]   // low 4 bits
	qh := raw[128:192] // high 2 bits
	sc := raw[192:208] // int8 scales for 16 sub-blocks
	d := float16.FromBits(binary.LittleEndian.Uint16(raw[208:210])).ToFloat32()

	for i := range 256 {
		// Extract 6-bit quantized value.
		qlByte := ql[i/2]
		var q4 uint8
		if i%2 == 0 {
			q4 = qlByte & 0xF
		} else {
			q4 = qlByte >> 4
		}

		// High 2 bits: qh stores 4 x 2-bit values per byte.
		qhByte := qh[i/4]
		shift := uint(i%4) * 2
		q2 := (qhByte >> shift) & 3

		q6 := int8(q4 | (q2 << 4)) // 6-bit unsigned: 0..63
		q6 -= 32                     // center to signed: -32..31

		subBlock := i / 16
		scale := int8(sc[subBlock])
		dst[i] = d * float32(scale) * float32(q6)
	}
}

// Q6KStorage holds Q6_K quantized tensor data on CPU.
type Q6KStorage struct {
	raw []byte
	len int
}

// NewQ6KStorageFromRaw creates Q6KStorage from raw super-block data.
func NewQ6KStorageFromRaw(raw []byte, numElements int) (*Q6KStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + q6KSuperBlockSize - 1) / q6KSuperBlockSize
	need := nBlocks * q6KBlockBytes
	if len(raw) < need {
		return nil, fmt.Errorf("Q6_K raw data too short: need %d bytes for %d blocks, got %d", need, nBlocks, len(raw))
	}
	data := make([]byte, need)
	copy(data, raw[:need])
	return &Q6KStorage{raw: data, len: numElements}, nil
}

func (q *Q6KStorage) Dequantize(dst []float32) {
	nBlocks := (q.len + q6KSuperBlockSize - 1) / q6KSuperBlockSize
	for bi := range nBlocks {
		blockRaw := q.raw[bi*q6KBlockBytes : (bi+1)*q6KBlockBytes]
		off := bi * q6KSuperBlockSize
		remaining := q.len - off
		if remaining >= q6KSuperBlockSize {
			DequantizeQ6K(blockRaw, dst[off:off+q6KSuperBlockSize])
		} else {
			var tmp [q6KSuperBlockSize]float32
			DequantizeQ6K(blockRaw, tmp[:])
			copy(dst[off:], tmp[:remaining])
		}
	}
}

func (q *Q6KStorage) Len() int               { return q.len }
func (q *Q6KStorage) Slice() []float32        { dst := make([]float32, q.len); q.Dequantize(dst); return dst }
func (q *Q6KStorage) Set(_ []float32)         { panic("Q6KStorage is immutable") }
func (q *Q6KStorage) DeviceType() device.Type { return device.CPU }

var _ Storage[float32] = (*Q6KStorage)(nil)

// Q5_K format: super-blocks of 256 values.
// Each super-block is 176 bytes:
//   - 2 bytes: fp16 d (super-block scale)
//   - 2 bytes: fp16 dmin (super-block min)
//   - 12 bytes: packed 6-bit scales and mins for 8 sub-blocks (same as Q4_K)
//   - 128 bytes: ql (low 4 bits of each 5-bit value)
//   - 32 bytes: qh (high 1 bit of each value, packed)
//
// Reference: llama.cpp ggml-quants.c dequantize_row_q5_K
const (
	q5KSuperBlockSize = 256
	q5KBlockBytes     = 176
)

// DequantizeQ5K dequantizes one Q5_K super-block (176 bytes) into 256 float32 values.
func DequantizeQ5K(raw []byte, dst []float32) {
	d := float16.FromBits(binary.LittleEndian.Uint16(raw[0:2])).ToFloat32()
	dmin := float16.FromBits(binary.LittleEndian.Uint16(raw[2:4])).ToFloat32()

	// Decode 6-bit scales and mins (same layout as Q4_K).
	var scales [q4KNumSubBlocks]uint8
	var mins [q4KNumSubBlocks]uint8
	for i := range q4KNumSubBlocks {
		scales[i] = raw[4+i] & 0xF
		mins[i] = raw[4+i] >> 4
	}
	scales[0] |= (raw[12] & 3) << 4
	scales[1] |= ((raw[12] >> 2) & 3) << 4
	scales[2] |= ((raw[12] >> 4) & 3) << 4
	scales[3] |= ((raw[12] >> 6) & 3) << 4
	scales[4] |= (raw[13] & 3) << 4
	scales[5] |= ((raw[13] >> 2) & 3) << 4
	scales[6] |= ((raw[13] >> 4) & 3) << 4
	scales[7] |= ((raw[13] >> 6) & 3) << 4
	mins[0] |= (raw[14] & 3) << 4
	mins[1] |= ((raw[14] >> 2) & 3) << 4
	mins[2] |= ((raw[14] >> 4) & 3) << 4
	mins[3] |= ((raw[14] >> 6) & 3) << 4
	mins[4] |= (raw[15] & 3) << 4
	mins[5] |= ((raw[15] >> 2) & 3) << 4
	mins[6] |= ((raw[15] >> 4) & 3) << 4
	mins[7] |= ((raw[15] >> 6) & 3) << 4

	ql := raw[16:144]   // 128 bytes: low 4 bits
	qh := raw[144:176]  // 32 bytes: high 1 bit (256 bits)

	for sb := range q4KNumSubBlocks {
		sc := d * float32(scales[sb])
		mn := dmin * float32(mins[sb])
		off := sb * q4KSubBlockSize

		for j := 0; j < q4KSubBlockSize; j += 2 {
			byteIdx := (off + j) / 2
			packed := ql[byteIdx]
			q0 := uint8(packed & 0xF)
			q1 := uint8(packed >> 4)

			// Add high bit from qh.
			bitIdx0 := off + j
			bitIdx1 := off + j + 1
			if qh[bitIdx0/8]&(1<<(uint(bitIdx0)%8)) != 0 {
				q0 |= 16
			}
			if qh[bitIdx1/8]&(1<<(uint(bitIdx1)%8)) != 0 {
				q1 |= 16
			}

			dst[off+j] = sc*float32(q0) - mn
			dst[off+j+1] = sc*float32(q1) - mn
		}
	}
}

// Q5KStorage holds Q5_K quantized tensor data on CPU.
type Q5KStorage struct {
	raw []byte
	len int
}

// NewQ5KStorageFromRaw creates Q5KStorage from raw super-block data.
func NewQ5KStorageFromRaw(raw []byte, numElements int) (*Q5KStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + q5KSuperBlockSize - 1) / q5KSuperBlockSize
	need := nBlocks * q5KBlockBytes
	if len(raw) < need {
		return nil, fmt.Errorf("Q5_K raw data too short: need %d bytes for %d blocks, got %d", need, nBlocks, len(raw))
	}
	data := make([]byte, need)
	copy(data, raw[:need])
	return &Q5KStorage{raw: data, len: numElements}, nil
}

func (q *Q5KStorage) Dequantize(dst []float32) {
	nBlocks := (q.len + q5KSuperBlockSize - 1) / q5KSuperBlockSize
	for bi := range nBlocks {
		blockRaw := q.raw[bi*q5KBlockBytes : (bi+1)*q5KBlockBytes]
		off := bi * q5KSuperBlockSize
		remaining := q.len - off
		if remaining >= q5KSuperBlockSize {
			DequantizeQ5K(blockRaw, dst[off:off+q5KSuperBlockSize])
		} else {
			var tmp [q5KSuperBlockSize]float32
			DequantizeQ5K(blockRaw, tmp[:])
			copy(dst[off:], tmp[:remaining])
		}
	}
}

func (q *Q5KStorage) Len() int               { return q.len }
func (q *Q5KStorage) Slice() []float32        { dst := make([]float32, q.len); q.Dequantize(dst); return dst }
func (q *Q5KStorage) Set(_ []float32)         { panic("Q5KStorage is immutable") }
func (q *Q5KStorage) DeviceType() device.Type { return device.CPU }

var _ Storage[float32] = (*Q5KStorage)(nil)
