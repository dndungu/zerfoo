package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/tensor"
)

// LoadTensors reads tensor data from a parsed GGUF file and returns them as
// float32 tensors keyed by name. Quantized tensors (Q4_0, Q8_0) are stored
// using their native quantized storage types for memory efficiency.
func LoadTensors(f *File, r io.ReadSeeker) (map[string]*tensor.TensorNumeric[float32], error) {
	result := make(map[string]*tensor.TensorNumeric[float32], len(f.Tensors))

	for i := range f.Tensors {
		ti := &f.Tensors[i]

		// Compute number of elements.
		numElements := 1
		for _, d := range ti.Dimensions {
			numElements *= int(d)
		}

		// Convert dimensions to int for tensor API.
		shape := make([]int, len(ti.Dimensions))
		for j, d := range ti.Dimensions {
			shape[j] = int(d)
		}

		// Compute byte size of this tensor's data.
		dataSize, err := tensorByteSize(ti.Type, numElements)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}

		// Seek to tensor data.
		offset := f.DataOffset + int64(ti.Offset)
		if _, err := r.Seek(offset, io.SeekStart); err != nil {
			return nil, fmt.Errorf("tensor %q: seek to offset %d: %w", ti.Name, offset, err)
		}

		// Read raw data.
		raw := make([]byte, dataSize)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, fmt.Errorf("tensor %q: read %d bytes: %w", ti.Name, dataSize, err)
		}

		// Decode based on type.
		t, err := decodeTensor(ti.Type, shape, numElements, raw)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}
		result[ti.Name] = t
	}

	return result, nil
}

// tensorByteSize returns the number of bytes needed for a tensor of the given type and element count.
func tensorByteSize(typ GGMLType, numElements int) (int, error) {
	switch typ {
	case GGMLTypeF32:
		return numElements * 4, nil
	case GGMLTypeF16:
		return numElements * 2, nil
	case GGMLTypeQ4_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 18, nil // 2 bytes scale + 16 bytes data
	case GGMLTypeQ8_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 34, nil // 2 bytes fp16 scale + 32 bytes int8 (GGUF format)
	case GGMLTypeQ4_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 144, nil // 4 bytes header + 12 bytes scales + 128 bytes data
	default:
		return 0, fmt.Errorf("unsupported GGML type %d", typ)
	}
}

// decodeTensor converts raw bytes into a tensor based on GGML type.
func decodeTensor(typ GGMLType, shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	switch typ {
	case GGMLTypeF32:
		return decodeF32Tensor(shape, numElements, raw)
	case GGMLTypeF16:
		return decodeF16Tensor(shape, numElements, raw)
	case GGMLTypeQ4_0:
		return decodeQ4Tensor(shape, numElements, raw)
	case GGMLTypeQ8_0:
		return decodeQ8Tensor(shape, numElements, raw)
	case GGMLTypeQ4_K:
		return decodeQ4KTensor(shape, numElements, raw)
	default:
		return nil, fmt.Errorf("unsupported GGML type %d", typ)
	}
}

func decodeF32Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	data := make([]float32, numElements)
	for i := range numElements {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
	}
	return tensor.New[float32](shape, data)
}

func decodeF16Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	data := make([]float32, numElements)
	for i := range numElements {
		bits := binary.LittleEndian.Uint16(raw[i*2 : i*2+2])
		data[i] = float16.FromBits(bits).ToFloat32()
	}
	return tensor.New[float32](shape, data)
}

func decodeQ4Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	q4, err := tensor.NewQ4StorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q4_0 decode: %w", err)
	}
	return tensor.NewWithStorage[float32](shape, q4)
}

func decodeQ4KTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	q4k, err := tensor.NewQ4KStorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q4_K decode: %w", err)
	}
	return tensor.NewWithStorage[float32](shape, q4k)
}

func decodeQ8Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	// GGUF Q8_0 format: 34 bytes per block (2-byte fp16 scale + 32 int8 quants).
	// Zerfoo Q8Storage uses float32 scales. Convert.
	const ggufQ8BlockBytes = 34
	nBlocks := (numElements + 31) / 32

	scales := make([]float32, nBlocks)
	quants := make([]int8, nBlocks*32)

	for bi := range nBlocks {
		off := bi * ggufQ8BlockBytes
		f16Bits := binary.LittleEndian.Uint16(raw[off : off+2])
		scales[bi] = float16.FromBits(f16Bits).ToFloat32()
		for j := range 32 {
			quants[bi*32+j] = int8(raw[off+2+j])
		}
	}

	q8, err := tensor.NewQ8StorageFromBlocks(scales, quants, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q8_0 decode: %w", err)
	}
	return tensor.NewWithStorage[float32](shape, q8)
}
