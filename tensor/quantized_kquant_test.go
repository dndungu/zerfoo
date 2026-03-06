package tensor

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/float16"
)

// buildQ4KBlock constructs a raw Q4_K super-block for testing.
// It quantizes 256 float32 values and returns 144 bytes.
func buildQ4KBlock(values []float32) []byte {
	if len(values) != 256 {
		panic("buildQ4KBlock requires exactly 256 values")
	}

	// Find overall scale and min.
	// Q4_K uses 8 sub-blocks of 32 values each.
	// Each sub-block has a 6-bit scale and 6-bit min.
	const numSubBlocks = 8
	const subBlockSize = 32

	subScales := make([]float32, numSubBlocks)
	subMins := make([]float32, numSubBlocks)

	for sb := range numSubBlocks {
		off := sb * subBlockSize
		minVal := values[off]
		maxVal := values[off]
		for j := 1; j < subBlockSize; j++ {
			v := values[off+j]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		if minVal > 0 {
			minVal = 0
		}
		subScales[sb] = (maxVal - minVal) / 15.0
		subMins[sb] = -minVal
	}

	// Find the super-block scale and min.
	var maxScale, maxMin float32
	for sb := range numSubBlocks {
		if subScales[sb] > maxScale {
			maxScale = subScales[sb]
		}
		if subMins[sb] > maxMin {
			maxMin = subMins[sb]
		}
	}

	d := maxScale / 63.0
	dmin := maxMin / 63.0

	// Quantize sub-block scales and mins to 6 bits.
	scalesQ := make([]uint8, numSubBlocks)
	minsQ := make([]uint8, numSubBlocks)
	for sb := range numSubBlocks {
		if d > 0 {
			scalesQ[sb] = uint8(math.Round(float64(subScales[sb] / d)))
			if scalesQ[sb] > 63 {
				scalesQ[sb] = 63
			}
		}
		if dmin > 0 {
			minsQ[sb] = uint8(math.Round(float64(subMins[sb] / dmin)))
			if minsQ[sb] > 63 {
				minsQ[sb] = 63
			}
		}
	}

	raw := make([]byte, 144)

	// Bytes 0-1: fp16 d
	binary.LittleEndian.PutUint16(raw[0:2], float16.FromFloat32(d).Bits())
	// Bytes 2-3: fp16 dmin
	binary.LittleEndian.PutUint16(raw[2:4], float16.FromFloat32(dmin).Bits())

	// Bytes 4-15: packed scales and mins (12 bytes).
	// Layout from ggml: for sub-blocks 0-3, low 4 bits of scale and min in bytes 4-7.
	// For sub-blocks 4-7, low 4 bits in bytes 8-11.
	// High 2 bits of scales in bytes 12-13 (but only 12 bytes total, so packed differently).
	//
	// Exact layout from ggml-quants.c:
	// scales[i] for i<4: low 6 bits stored as: raw[4+i] = (scalesQ[i] & 0xF) | ((minsQ[i] & 0xF) << 4)
	// scales[i] for i>=4: raw[4+i] = (scalesQ[i] & 0xF) | ((minsQ[i] & 0xF) << 4)
	// High 2 bits: raw[4+8] to raw[4+11] store the high bits.
	// raw[12] = (scalesQ[0]>>4)&3 | ((scalesQ[1]>>4)&3)<<2 | ((scalesQ[2]>>4)&3)<<4 | ((scalesQ[3]>>4)&3)<<6
	// raw[13] = (scalesQ[4]>>4)&3 | ((scalesQ[5]>>4)&3)<<2 | ((scalesQ[6]>>4)&3)<<4 | ((scalesQ[7]>>4)&3)<<6
	// raw[14] = (minsQ[0]>>4)&3 | ((minsQ[1]>>4)&3)<<2 | ((minsQ[2]>>4)&3)<<4 | ((minsQ[3]>>4)&3)<<6
	// raw[15] = (minsQ[4]>>4)&3 | ((minsQ[5]>>4)&3)<<2 | ((minsQ[6]>>4)&3)<<4 | ((minsQ[7]>>4)&3)<<6
	for i := range 8 {
		raw[4+i] = (scalesQ[i] & 0xF) | ((minsQ[i] & 0xF) << 4)
	}
	raw[12] = (scalesQ[0]>>4)&3 | ((scalesQ[1]>>4)&3)<<2 | ((scalesQ[2]>>4)&3)<<4 | ((scalesQ[3]>>4)&3)<<6
	raw[13] = (scalesQ[4]>>4)&3 | ((scalesQ[5]>>4)&3)<<2 | ((scalesQ[6]>>4)&3)<<4 | ((scalesQ[7]>>4)&3)<<6
	raw[14] = (minsQ[0]>>4)&3 | ((minsQ[1]>>4)&3)<<2 | ((minsQ[2]>>4)&3)<<4 | ((minsQ[3]>>4)&3)<<6
	raw[15] = (minsQ[4]>>4)&3 | ((minsQ[5]>>4)&3)<<2 | ((minsQ[6]>>4)&3)<<4 | ((minsQ[7]>>4)&3)<<6

	// Bytes 16-143: 256 packed 4-bit quantized values (128 bytes).
	// Quantize each value using its sub-block's reconstructed scale and min.
	for sb := range numSubBlocks {
		off := sb * subBlockSize
		sc := d * float32(scalesQ[sb])
		mn := dmin * float32(minsQ[sb])
		var invScale float32
		if sc > 0 {
			invScale = 1.0 / sc
		}
		for j := 0; j < subBlockSize; j += 2 {
			v0 := values[off+j]
			v1 := values[off+j+1]
			q0 := clampInt(int(math.Round(float64((v0+mn)*invScale))), 0, 15)
			q1 := clampInt(int(math.Round(float64((v1+mn)*invScale))), 0, 15)
			byteIdx := (off + j) / 2
			raw[16+byteIdx] = byte(q0) | (byte(q1) << 4)
		}
	}

	return raw
}

func TestDequantizeQ4K_RoundTrip(t *testing.T) {
	// Create 256 test values with a known pattern.
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(math.Sin(float64(i)*0.1)) * 2.0
	}

	raw := buildQ4KBlock(values)
	dst := make([]float32, 256)
	DequantizeQ4K(raw, dst)

	// Check values are close (Q4_K has limited precision).
	maxErr := float32(0.0)
	for i := range values {
		err := float32(math.Abs(float64(dst[i] - values[i])))
		if err > maxErr {
			maxErr = err
		}
	}
	// Q4_K with 4-bit per value should be within ~0.5 of the original
	// for values in [-2, 2] range.
	if maxErr > 0.6 {
		t.Errorf("max dequantization error %f exceeds 0.6", maxErr)
	}
}

func TestDequantizeQ4K_Zeros(t *testing.T) {
	values := make([]float32, 256)
	raw := buildQ4KBlock(values)
	dst := make([]float32, 256)
	DequantizeQ4K(raw, dst)

	for i, v := range dst {
		if v != 0 {
			t.Errorf("dst[%d] = %f, want 0", i, v)
			break
		}
	}
}

func TestNewQ4KStorageFromRaw(t *testing.T) {
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(i) * 0.01
	}

	raw := buildQ4KBlock(values)
	storage, err := NewQ4KStorageFromRaw(raw, 256)
	if err != nil {
		t.Fatalf("NewQ4KStorageFromRaw: %v", err)
	}

	if storage.Len() != 256 {
		t.Errorf("Len() = %d, want 256", storage.Len())
	}

	slice := storage.Slice()
	if len(slice) != 256 {
		t.Fatalf("Slice() len = %d, want 256", len(slice))
	}

	// Verify dequantized values are reasonable.
	maxErr := float32(0.0)
	for i := range values {
		err := float32(math.Abs(float64(slice[i] - values[i])))
		if err > maxErr {
			maxErr = err
		}
	}
	if maxErr > 0.3 {
		t.Errorf("max error %f exceeds 0.3 for linear ramp", maxErr)
	}
}

func TestNewQ4KStorageFromRaw_InvalidSize(t *testing.T) {
	_, err := NewQ4KStorageFromRaw(make([]byte, 10), 256)
	if err == nil {
		t.Fatal("expected error for short raw data")
	}
}

// buildQ6KBlock constructs a raw Q6_K super-block for testing.
func buildQ6KBlock(values []float32) []byte {
	if len(values) != 256 {
		panic("buildQ6KBlock requires exactly 256 values")
	}

	raw := make([]byte, 210)

	// Find overall absmax to set d.
	var absMax float32
	for _, v := range values {
		if av := float32(math.Abs(float64(v))); av > absMax {
			absMax = av
		}
	}

	// 16 sub-blocks of 16 values, each with an int8 scale.
	d := absMax / (31.0 * 127.0)
	if d == 0 {
		return raw
	}

	subScales := make([]int8, 16)
	for sb := range 16 {
		off := sb * 16
		var subMax float32
		for j := range 16 {
			if av := float32(math.Abs(float64(values[off+j]))); av > subMax {
				subMax = av
			}
		}
		sc := subMax / (d * 31.0)
		if sc > 127 {
			sc = 127
		}
		subScales[sb] = int8(math.Round(float64(sc)))
	}

	// Quantize each value to 6-bit signed: -32..31
	for i := range 256 {
		sb := i / 16
		sc := d * float32(subScales[sb])
		var q int
		if sc > 0 {
			q = clampInt(int(math.Round(float64(values[i]/sc)))+32, 0, 63)
		} else {
			q = 32
		}
		// Store low 4 bits in ql.
		q4 := byte(q & 0xF)
		if i%2 == 0 {
			raw[i/2] = (raw[i/2] & 0xF0) | q4
		} else {
			raw[i/2] = (raw[i/2] & 0x0F) | (q4 << 4)
		}
		// Store high 2 bits in qh.
		q2 := byte((q >> 4) & 3)
		shift := uint(i%4) * 2
		raw[128+i/4] |= q2 << shift
	}

	// Write int8 scales.
	for i, sc := range subScales {
		raw[192+i] = byte(sc)
	}

	// Write fp16 d.
	binary.LittleEndian.PutUint16(raw[208:210], float16.FromFloat32(d).Bits())

	return raw
}

func TestDequantizeQ6K_RoundTrip(t *testing.T) {
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(math.Sin(float64(i)*0.1)) * 2.0
	}

	raw := buildQ6KBlock(values)
	dst := make([]float32, 256)
	DequantizeQ6K(raw, dst)

	maxErr := float32(0.0)
	for i := range values {
		err := float32(math.Abs(float64(dst[i] - values[i])))
		if err > maxErr {
			maxErr = err
		}
	}
	// Q6_K has higher precision than Q4_K.
	if maxErr > 0.3 {
		t.Errorf("max dequantization error %f exceeds 0.3", maxErr)
	}
}

func TestDequantizeQ6K_Zeros(t *testing.T) {
	values := make([]float32, 256)
	raw := buildQ6KBlock(values)
	dst := make([]float32, 256)
	DequantizeQ6K(raw, dst)

	for i, v := range dst {
		if v != 0 {
			t.Errorf("dst[%d] = %f, want 0", i, v)
			break
		}
	}
}

func TestNewQ6KStorageFromRaw(t *testing.T) {
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(i) * 0.01
	}

	raw := buildQ6KBlock(values)
	storage, err := NewQ6KStorageFromRaw(raw, 256)
	if err != nil {
		t.Fatalf("NewQ6KStorageFromRaw: %v", err)
	}

	if storage.Len() != 256 {
		t.Errorf("Len() = %d, want 256", storage.Len())
	}

	slice := storage.Slice()
	if len(slice) != 256 {
		t.Fatalf("Slice() len = %d, want 256", len(slice))
	}
}

// buildQ5KBlock constructs a raw Q5_K super-block for testing.
func buildQ5KBlock(values []float32) []byte {
	if len(values) != 256 {
		panic("buildQ5KBlock requires exactly 256 values")
	}

	const numSubBlocks = 8
	const subBlockSize = 32

	subScales := make([]float32, numSubBlocks)
	subMins := make([]float32, numSubBlocks)

	for sb := range numSubBlocks {
		off := sb * subBlockSize
		minVal := values[off]
		maxVal := values[off]
		for j := 1; j < subBlockSize; j++ {
			v := values[off+j]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		if minVal > 0 {
			minVal = 0
		}
		subScales[sb] = (maxVal - minVal) / 31.0 // 5-bit: 0..31
		subMins[sb] = -minVal
	}

	var maxScale, maxMin float32
	for sb := range numSubBlocks {
		if subScales[sb] > maxScale {
			maxScale = subScales[sb]
		}
		if subMins[sb] > maxMin {
			maxMin = subMins[sb]
		}
	}

	d := maxScale / 63.0
	dmin := maxMin / 63.0

	scalesQ := make([]uint8, numSubBlocks)
	minsQ := make([]uint8, numSubBlocks)
	for sb := range numSubBlocks {
		if d > 0 {
			scalesQ[sb] = uint8(min(63, int(math.Round(float64(subScales[sb]/d)))))
		}
		if dmin > 0 {
			minsQ[sb] = uint8(min(63, int(math.Round(float64(subMins[sb]/dmin)))))
		}
	}

	raw := make([]byte, 176)

	binary.LittleEndian.PutUint16(raw[0:2], float16.FromFloat32(d).Bits())
	binary.LittleEndian.PutUint16(raw[2:4], float16.FromFloat32(dmin).Bits())

	for i := range 8 {
		raw[4+i] = (scalesQ[i] & 0xF) | ((minsQ[i] & 0xF) << 4)
	}
	raw[12] = (scalesQ[0]>>4)&3 | ((scalesQ[1]>>4)&3)<<2 | ((scalesQ[2]>>4)&3)<<4 | ((scalesQ[3]>>4)&3)<<6
	raw[13] = (scalesQ[4]>>4)&3 | ((scalesQ[5]>>4)&3)<<2 | ((scalesQ[6]>>4)&3)<<4 | ((scalesQ[7]>>4)&3)<<6
	raw[14] = (minsQ[0]>>4)&3 | ((minsQ[1]>>4)&3)<<2 | ((minsQ[2]>>4)&3)<<4 | ((minsQ[3]>>4)&3)<<6
	raw[15] = (minsQ[4]>>4)&3 | ((minsQ[5]>>4)&3)<<2 | ((minsQ[6]>>4)&3)<<4 | ((minsQ[7]>>4)&3)<<6

	// Quantize to 5-bit.
	for sb := range numSubBlocks {
		off := sb * subBlockSize
		sc := d * float32(scalesQ[sb])
		mn := dmin * float32(minsQ[sb])
		var invScale float32
		if sc > 0 {
			invScale = 1.0 / sc
		}
		for j := 0; j < subBlockSize; j += 2 {
			v0 := values[off+j]
			v1 := values[off+j+1]
			q0 := clampInt(int(math.Round(float64((v0+mn)*invScale))), 0, 31)
			q1 := clampInt(int(math.Round(float64((v1+mn)*invScale))), 0, 31)

			// Low 4 bits go to ql.
			byteIdx := (off + j) / 2
			raw[16+byteIdx] = byte(q0&0xF) | (byte(q1&0xF) << 4)

			// High 1 bit goes to qh.
			bitIdx0 := off + j
			bitIdx1 := off + j + 1
			if q0&16 != 0 {
				raw[144+bitIdx0/8] |= 1 << (uint(bitIdx0) % 8)
			}
			if q1&16 != 0 {
				raw[144+bitIdx1/8] |= 1 << (uint(bitIdx1) % 8)
			}
		}
	}

	return raw
}

func TestDequantizeQ5K_RoundTrip(t *testing.T) {
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(math.Sin(float64(i)*0.1)) * 2.0
	}

	raw := buildQ5KBlock(values)
	dst := make([]float32, 256)
	DequantizeQ5K(raw, dst)

	maxErr := float32(0.0)
	for i := range values {
		err := float32(math.Abs(float64(dst[i] - values[i])))
		if err > maxErr {
			maxErr = err
		}
	}
	// Q5_K should be between Q4_K and Q6_K in precision.
	if maxErr > 0.5 {
		t.Errorf("max dequantization error %f exceeds 0.5", maxErr)
	}
}

func TestNewQ5KStorageFromRaw(t *testing.T) {
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(i) * 0.01
	}

	raw := buildQ5KBlock(values)
	storage, err := NewQ5KStorageFromRaw(raw, 256)
	if err != nil {
		t.Fatalf("NewQ5KStorageFromRaw: %v", err)
	}

	if storage.Len() != 256 {
		t.Errorf("Len() = %d, want 256", storage.Len())
	}

	slice := storage.Slice()
	if len(slice) != 256 {
		t.Fatalf("Slice() len = %d, want 256", len(slice))
	}
}
