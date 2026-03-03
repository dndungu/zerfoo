package generate

import (
	"fmt"

	"github.com/zerfoo/zerfoo/tensor"
)

// LayerKV holds the cached key and value tensors for a single attention layer.
type LayerKV struct {
	Key   *tensor.TensorNumeric[float32]
	Value *tensor.TensorNumeric[float32]
}

// KVCache stores key-value tensors for all attention layers during autoregressive generation.
// Each call to Update appends new K/V along the sequence dimension (axis 1).
type KVCache struct {
	layers []LayerKV
}

// NewKVCache creates a KVCache for the specified number of layers.
func NewKVCache(numLayers int) *KVCache {
	return &KVCache{
		layers: make([]LayerKV, numLayers),
	}
}

// NumLayers returns the number of layers in the cache.
func (c *KVCache) NumLayers() int {
	return len(c.layers)
}

// Get returns the cached key-value pair for the given layer.
// Returns false if the layer has not been populated yet.
func (c *KVCache) Get(layer int) (*LayerKV, bool) {
	if layer < 0 || layer >= len(c.layers) {
		return nil, false
	}
	lkv := &c.layers[layer]
	if lkv.Key == nil {
		return nil, false
	}
	return lkv, true
}

// Update appends new key and value tensors to the cache for the given layer.
// Tensors are expected to have shape [batch, seq_len, dim].
// The new tensors are concatenated along axis 1 (sequence dimension).
func (c *KVCache) Update(layer int, newK, newV *tensor.TensorNumeric[float32]) error {
	if layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, len(c.layers))
	}
	lkv := &c.layers[layer]

	if lkv.Key == nil {
		// First update: store directly.
		lkv.Key = newK
		lkv.Value = newV
		return nil
	}

	// Concatenate along sequence dimension (axis 1).
	var err error
	lkv.Key, err = concatAxis1(lkv.Key, newK)
	if err != nil {
		return fmt.Errorf("concat key for layer %d: %w", layer, err)
	}
	lkv.Value, err = concatAxis1(lkv.Value, newV)
	if err != nil {
		return fmt.Errorf("concat value for layer %d: %w", layer, err)
	}
	return nil
}

// SeqLen returns the current cached sequence length.
// Returns 0 if the cache is empty.
func (c *KVCache) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	lkv := &c.layers[0]
	if lkv.Key == nil {
		return 0
	}
	shape := lkv.Key.Shape()
	if len(shape) < 2 {
		return 0
	}
	return shape[1] // [batch, seq_len, dim]
}

// Reset clears all cached tensors.
func (c *KVCache) Reset() {
	for i := range c.layers {
		c.layers[i].Key = nil
		c.layers[i].Value = nil
	}
}

// concatAxis1 concatenates two 3D tensors along axis 1 (sequence dimension).
// Both tensors must have matching batch and dim dimensions.
func concatAxis1(a, b *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shapeA := a.Shape()
	shapeB := b.Shape()

	if len(shapeA) != 3 || len(shapeB) != 3 {
		return nil, fmt.Errorf("expected 3D tensors, got %dD and %dD", len(shapeA), len(shapeB))
	}
	if shapeA[0] != shapeB[0] {
		return nil, fmt.Errorf("batch dimension mismatch: %d vs %d", shapeA[0], shapeB[0])
	}
	if shapeA[2] != shapeB[2] {
		return nil, fmt.Errorf("feature dimension mismatch: %d vs %d", shapeA[2], shapeB[2])
	}

	batch := shapeA[0]
	seqA := shapeA[1]
	seqB := shapeB[1]
	dim := shapeA[2]
	newSeq := seqA + seqB

	dataA := a.Data()
	dataB := b.Data()
	newData := make([]float32, batch*newSeq*dim)

	for b := range batch {
		// Copy from A: batch b, all seqA positions.
		srcOffA := b * seqA * dim
		dstOff := b * newSeq * dim
		copy(newData[dstOff:dstOff+seqA*dim], dataA[srcOffA:srcOffA+seqA*dim])

		// Copy from B: batch b, all seqB positions.
		srcOffB := b * seqB * dim
		dstOff2 := dstOff + seqA*dim
		copy(newData[dstOff2:dstOff2+seqB*dim], dataB[srcOffB:srcOffB+seqB*dim])
	}

	return tensor.New([]int{batch, newSeq, dim}, newData)
}
