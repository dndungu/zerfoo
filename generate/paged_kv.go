package generate

import (
	"fmt"

	"github.com/zerfoo/zerfoo/tensor"
)

// PagedKVCache stores key-value tensors for autoregressive generation using
// block-level allocation from a BlockPool. Instead of pre-allocating the full
// maxSeqLen per sequence, blocks of blockSize tokens are allocated on demand,
// reducing memory waste for concurrent sequences of varying length.
//
// Each sequence gets its own PagedKVCache. The cache supports batch=1 only;
// multi-sequence serving allocates one PagedKVCache per sequence.
type PagedKVCache[T tensor.Numeric] struct {
	pool      *BlockPool[T]
	numLayers int
	blockSize int
	headDim   int

	// blockTable holds the allocated blocks in order.
	// Blocks are shared across layers: block.K and block.V are laid out as
	// [numLayers * blockSize * headDim], so layer L at position P within
	// a block is at offset L*blockSize*headDim + P*headDim.
	blockTable []*Block[T]

	// layerCursors tracks the number of tokens appended per layer.
	layerCursors []int
}

// NewPagedKVCache creates a paged KV cache backed by the given block pool.
func NewPagedKVCache[T tensor.Numeric](pool *BlockPool[T], numLayers int) *PagedKVCache[T] {
	return &PagedKVCache[T]{
		pool:         pool,
		numLayers:    numLayers,
		blockSize:    pool.blockSize,
		headDim:      pool.headDim,
		layerCursors: make([]int, numLayers),
	}
}

// SeqLen returns the number of token positions stored in the cache,
// based on layer 0's cursor. Returns 0 if the cache is empty.
func (c *PagedKVCache[T]) SeqLen() int {
	if c.numLayers == 0 {
		return 0
	}
	return c.layerCursors[0]
}

// Append writes new key and value data for the given layer. The tensors must
// have shape [1, seqLen, headDim] (batch=1). Data is written into the current
// block; a new block is allocated from the pool when the current one fills up.
func (c *PagedKVCache[T]) Append(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	if layer < 0 || layer >= c.numLayers {
		return fmt.Errorf("layer %d out of range [0, %d)", layer, c.numLayers)
	}

	shape := newK.Shape()
	if len(shape) != 3 {
		return fmt.Errorf("expected 3D tensor [batch, seq, dim], got %dD", len(shape))
	}
	batch, seqLen, dim := shape[0], shape[1], shape[2]
	if batch != 1 {
		return fmt.Errorf("paged KV cache requires batch=1, got %d", batch)
	}
	if dim != c.headDim {
		return fmt.Errorf("headDim mismatch: pool has %d, tensor has %d", c.headDim, dim)
	}

	cursor := c.layerCursors[layer]
	kData := newK.Data()
	vData := newV.Data()

	for pos := range seqLen {
		globalPos := cursor + pos
		blockIdx := globalPos / c.blockSize
		posInBlock := globalPos % c.blockSize

		// Allocate new block if needed.
		for blockIdx >= len(c.blockTable) {
			b, err := c.pool.Alloc()
			if err != nil {
				return fmt.Errorf("alloc block: %w", err)
			}
			c.blockTable = append(c.blockTable, b)
		}

		block := c.blockTable[blockIdx]

		// Write K and V at the layer's region within the block.
		// Layout: [numLayers][blockSize][headDim]
		offset := layer*c.blockSize*c.headDim + posInBlock*c.headDim
		srcOffset := pos * c.headDim
		copy(block.K[offset:offset+c.headDim], kData[srcOffset:srcOffset+c.headDim])
		copy(block.V[offset:offset+c.headDim], vData[srcOffset:srcOffset+c.headDim])

		if posInBlock+1 > block.Used {
			block.Used = posInBlock + 1
		}
	}

	c.layerCursors[layer] = cursor + seqLen
	return nil
}

// GetKV returns the cached key and value tensors for the given layer,
// gathered into contiguous [1, seqLen, headDim] tensors. Returns false if
// the layer is out of range or the cache is empty for that layer.
func (c *PagedKVCache[T]) GetKV(layer int) (*LayerKV[T], bool) {
	if layer < 0 || layer >= c.numLayers {
		return nil, false
	}
	seqLen := c.layerCursors[layer]
	if seqLen == 0 {
		return nil, false
	}

	kOut := make([]T, seqLen*c.headDim)
	vOut := make([]T, seqLen*c.headDim)

	for pos := range seqLen {
		blockIdx := pos / c.blockSize
		posInBlock := pos % c.blockSize
		block := c.blockTable[blockIdx]

		srcOffset := layer*c.blockSize*c.headDim + posInBlock*c.headDim
		dstOffset := pos * c.headDim
		copy(kOut[dstOffset:dstOffset+c.headDim], block.K[srcOffset:srcOffset+c.headDim])
		copy(vOut[dstOffset:dstOffset+c.headDim], block.V[srcOffset:srcOffset+c.headDim])
	}

	kTensor, err := tensor.New([]int{1, seqLen, c.headDim}, kOut)
	if err != nil {
		return nil, false
	}
	vTensor, err := tensor.New([]int{1, seqLen, c.headDim}, vOut)
	if err != nil {
		return nil, false
	}

	return &LayerKV[T]{Key: kTensor, Value: vTensor}, true
}

// Free returns all allocated blocks to the pool and resets the cache.
func (c *PagedKVCache[T]) Free() {
	for _, b := range c.blockTable {
		c.pool.Free(b)
	}
	c.blockTable = c.blockTable[:0]
	for i := range c.layerCursors {
		c.layerCursors[i] = 0
	}
}
