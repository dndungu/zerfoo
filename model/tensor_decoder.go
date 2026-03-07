// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zmf"
)

// decodeBFloat16Bits reads a 2-byte little-endian BFloat16 bit pattern from p.
func decodeBFloat16Bits(p []byte) float16.BFloat16 {
	return float16.BFloat16FromBits(binary.LittleEndian.Uint16(p))
}

// DecodeTensor converts a ZMF Tensor protobuf message into a Zerfoo Tensor.
func DecodeTensor[T tensor.Numeric](tensorProto *zmf.Tensor) (*tensor.TensorNumeric[T], error) {
	shape := tensor.ConvertInt64ToInt(tensorProto.Shape)
	size := tensor.Product(shape)

	var zero T

	// Guard against empty data when shape implies non-zero elements.
	if size > 0 && len(tensorProto.Data) == 0 {
		return nil, fmt.Errorf("tensor has shape %v (size %d) but no data; external data may not have been loaded",
			tensorProto.Shape, size)
	}

	switch tensorProto.Dtype {
	case zmf.Tensor_FLOAT32:
		// Decode raw bytes into []float32
		if len(tensorProto.Data)%4 != 0 {
			return nil, fmt.Errorf("invalid float32 data length: must be a multiple of 4, got %d", len(tensorProto.Data))
		}

		f32 := make([]float32, size)
		for i := 0; i < size; i++ {
			bits := binary.LittleEndian.Uint32(tensorProto.Data[i*4 : i*4+4])
			f32[i] = math.Float32frombits(bits)
		}

		switch any(zero).(type) {
		case float32:
			data := any(f32).([]T)
			return tensor.New[T](shape, data)
		case float16.Float16:
			f16 := make([]float16.Float16, size)
			for i, v := range f32 {
				f16[i] = float16.FromFloat32(v)
			}
			data := any(f16).([]T)
			return tensor.New[T](shape, data)
		case float16.BFloat16:
			bf := make([]float16.BFloat16, size)
			for i, v := range f32 {
				bf[i] = float16.BFloat16FromFloat32(v)
			}
			data := any(bf).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for FLOAT32 source", zero)
		}

	case zmf.Tensor_FLOAT16:
		if len(tensorProto.Data)%2 != 0 {
			return nil, fmt.Errorf("invalid float16 data length: must be a multiple of 2, got %d", len(tensorProto.Data))
		}

		f16 := make([]float16.Float16, size)
		for i := 0; i < size; i++ {
			bits := binary.LittleEndian.Uint16(tensorProto.Data[i*2 : i*2+2])
			f16[i] = float16.FromBits(bits)
		}

		switch any(zero).(type) {
		case float16.Float16:
			data := any(f16).([]T)
			return tensor.New[T](shape, data)
		case float32:
			f32 := make([]float32, size)
			for i, v := range f16 {
				f32[i] = v.ToFloat32()
			}
			data := any(f32).([]T)
			return tensor.New[T](shape, data)
		case float16.BFloat16:
			bf := make([]float16.BFloat16, size)
			for i, v := range f16 {
				bf[i] = float16.BFloat16FromFloat32(v.ToFloat32())
			}
			data := any(bf).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for FLOAT16 source", zero)
		}

	case zmf.Tensor_INT8:
		if size != len(tensorProto.Data) {
			return nil, fmt.Errorf("invalid int8 data length: expected %d, got %d", size, len(tensorProto.Data))
		}

		switch any(zero).(type) {
		case int8:
			vals := make([]int8, size)
			for i := 0; i < size; i++ {
				vals[i] = int8(tensorProto.Data[i])
			}
			data := any(vals).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for INT8 source", zero)
		}

	case zmf.Tensor_BFLOAT16:
		if len(tensorProto.Data)%2 != 0 {
			return nil, fmt.Errorf("invalid bfloat16 data length: must be a multiple of 2, got %d", len(tensorProto.Data))
		}

		switch any(zero).(type) {
		case float16.BFloat16:
			vals := make([]float16.BFloat16, size)
			for i := 0; i < size; i++ {
				vals[i] = decodeBFloat16Bits(tensorProto.Data[i*2 : i*2+2])
			}
			data := any(vals).([]T)
			return tensor.New[T](shape, data)
		case float32:
			f32 := make([]float32, size)
			for i := 0; i < size; i++ {
				f32[i] = decodeBFloat16Bits(tensorProto.Data[i*2 : i*2+2]).ToFloat32()
			}
			data := any(f32).([]T)
			return tensor.New[T](shape, data)
		case float16.Float16:
			f16 := make([]float16.Float16, size)
			for i := 0; i < size; i++ {
				f16[i] = float16.FromFloat32(decodeBFloat16Bits(tensorProto.Data[i*2 : i*2+2]).ToFloat32())
			}
			data := any(f16).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for BFLOAT16 source", zero)
		}

	case zmf.Tensor_INT32: //nolint:dupl // INT32 and INT64 cases are structurally similar but differ in byte width and target type.
		if len(tensorProto.Data)%4 != 0 {
			return nil, fmt.Errorf("invalid int32 data length: must be a multiple of 4, got %d", len(tensorProto.Data))
		}

		switch any(zero).(type) {
		case int32:
			vals := make([]int32, size)
			for i := 0; i < size; i++ {
				vals[i] = int32(binary.LittleEndian.Uint32(tensorProto.Data[i*4 : i*4+4]))
			}
			data := any(vals).([]T)
			return tensor.New[T](shape, data)
		case float32:
			f32 := make([]float32, size)
			for i := 0; i < size; i++ {
				f32[i] = float32(int32(binary.LittleEndian.Uint32(tensorProto.Data[i*4 : i*4+4])))
			}
			data := any(f32).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for INT32 source", zero)
		}

	case zmf.Tensor_INT64: //nolint:dupl // INT32 and INT64 cases are structurally similar but differ in byte width and target type.
		if len(tensorProto.Data)%8 != 0 {
			return nil, fmt.Errorf("invalid int64 data length: must be a multiple of 8, got %d", len(tensorProto.Data))
		}

		switch any(zero).(type) {
		case int64:
			vals := make([]int64, size)
			for i := 0; i < size; i++ {
				vals[i] = int64(binary.LittleEndian.Uint64(tensorProto.Data[i*8 : i*8+8]))
			}
			data := any(vals).([]T)
			return tensor.New[T](shape, data)
		case float32:
			f32 := make([]float32, size)
			for i := 0; i < size; i++ {
				f32[i] = float32(int64(binary.LittleEndian.Uint64(tensorProto.Data[i*8 : i*8+8])))
			}
			data := any(f32).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for INT64 source", zero)
		}

	case zmf.Tensor_FLOAT64:
		if len(tensorProto.Data)%8 != 0 {
			return nil, fmt.Errorf("invalid float64 data length: must be a multiple of 8, got %d", len(tensorProto.Data))
		}

		switch any(zero).(type) {
		case float64:
			vals := make([]float64, size)
			for i := 0; i < size; i++ {
				vals[i] = math.Float64frombits(binary.LittleEndian.Uint64(tensorProto.Data[i*8 : i*8+8]))
			}
			data := any(vals).([]T)
			return tensor.New[T](shape, data)
		case float32:
			f32 := make([]float32, size)
			for i := 0; i < size; i++ {
				f32[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(tensorProto.Data[i*8 : i*8+8])))
			}
			data := any(f32).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for FLOAT64 source", zero)
		}

	case zmf.Tensor_UINT8:
		if size != len(tensorProto.Data) {
			return nil, fmt.Errorf("invalid uint8 data length: expected %d, got %d", size, len(tensorProto.Data))
		}

		switch any(zero).(type) {
		case uint8:
			vals := make([]uint8, size)
			copy(vals, tensorProto.Data)
			data := any(vals).([]T)
			return tensor.New[T](shape, data)
		default:
			return nil, fmt.Errorf("unsupported destination type %T for UINT8 source", zero)
		}

	case zmf.Tensor_Q4_0:
		// When T is float32, keep weights in Q4 storage so the compute engine
		// can dispatch to the fused Q4×F32 GEMM kernel instead of dequantizing
		// up front and running full-precision GEMM.
		switch any(zero).(type) {
		case float32:
			q4, err := tensor.NewQ4StorageFromRaw(tensorProto.Data, size)
			if err != nil {
				return nil, fmt.Errorf("Q4_0 decode: %w", err)
			}
			t, err := tensor.NewWithStorage[float32](shape, q4)
			if err != nil {
				return nil, err
			}
			return any(t).(*tensor.TensorNumeric[T]), nil
		default:
			f32, err := decodeQ4Blocks(tensorProto.Data, size)
			if err != nil {
				return nil, err
			}
			return castFloat32ToT[T](shape, f32, zero, "Q4_0")
		}

	case zmf.Tensor_Q8_0:
		f32, err := decodeQ8Blocks(tensorProto.Data, size)
		if err != nil {
			return nil, err
		}
		return castFloat32ToT[T](shape, f32, zero, "Q8_0")

	default:
		return nil, fmt.Errorf("unsupported tensor dtype: %s", tensorProto.Dtype)
	}
}

// castFloat32ToT converts dequantized float32 values to the target tensor type T.
func castFloat32ToT[T tensor.Numeric](shape []int, f32 []float32, zero T, srcName string) (*tensor.TensorNumeric[T], error) {
	switch any(zero).(type) {
	case float32:
		data := any(f32).([]T)
		return tensor.New[T](shape, data)
	case float16.Float16:
		f16 := make([]float16.Float16, len(f32))
		for i, v := range f32 {
			f16[i] = float16.FromFloat32(v)
		}
		data := any(f16).([]T)
		return tensor.New[T](shape, data)
	case float16.BFloat16:
		bf := make([]float16.BFloat16, len(f32))
		for i, v := range f32 {
			bf[i] = float16.BFloat16FromFloat32(v)
		}
		data := any(bf).([]T)
		return tensor.New[T](shape, data)
	default:
		return nil, fmt.Errorf("unsupported destination type %T for %s source", zero, srcName)
	}
}

// decodeQ4Blocks decodes Q4_0 quantized blocks into float32 values.
// Q4_0 block layout: 2 bytes float16 scale + 16 bytes packed 4-bit data = 18 bytes per 32 values.
func decodeQ4Blocks(data []byte, size int) ([]float32, error) {
	const blockBytes = 18
	const blockSize = 32

	nBlocks := (size + blockSize - 1) / blockSize
	if len(data) < nBlocks*blockBytes {
		return nil, fmt.Errorf("Q4_0 data too short: need %d bytes for %d blocks, got %d", nBlocks*blockBytes, nBlocks, len(data))
	}

	f32 := make([]float32, size)
	for bi := range nBlocks {
		off := bi * blockBytes
		scale := float16.FromBits(binary.LittleEndian.Uint16(data[off : off+2])).ToFloat32()
		for j := 0; j < blockSize; j += 2 {
			packed := data[off+2+j/2]
			q0 := int(packed&0x0F) - 8
			q1 := int(packed>>4) - 8

			if idx := bi*blockSize + j; idx < size {
				f32[idx] = float32(q0) * scale
			}
			if idx := bi*blockSize + j + 1; idx < size {
				f32[idx] = float32(q1) * scale
			}
		}
	}
	return f32, nil
}

// decodeQ8Blocks decodes Q8_0 quantized blocks into float32 values.
// Q8_0 block layout: 4 bytes float32 scale + 32 bytes int8 data = 36 bytes per 32 values.
func decodeQ8Blocks(data []byte, size int) ([]float32, error) {
	const blockBytes = 36
	const blockSize = 32

	nBlocks := (size + blockSize - 1) / blockSize
	if len(data) < nBlocks*blockBytes {
		return nil, fmt.Errorf("Q8_0 data too short: need %d bytes for %d blocks, got %d", nBlocks*blockBytes, nBlocks, len(data))
	}

	f32 := make([]float32, size)
	for bi := range nBlocks {
		off := bi * blockBytes
		scale := math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4]))
		for j := range blockSize {
			idx := bi*blockSize + j
			if idx >= size {
				break
			}
			f32[idx] = float32(int8(data[off+4+j])) * scale
		}
	}
	return f32, nil
}
