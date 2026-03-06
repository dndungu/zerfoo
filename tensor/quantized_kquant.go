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
