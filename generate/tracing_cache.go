package generate

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// TracingCacheProvider wraps a CacheProvider and records KV cache operations
// into a Tracer for graph compilation. Each Update records KVCacheAppendK and
// KVCacheAppendV ops; each Get records KVCacheGetK and KVCacheGetV ops.
type TracingCacheProvider[T tensor.Numeric] struct {
	real   CacheProvider[T]
	tracer *compute.Tracer[T]
}

// NewTracingCacheProvider creates a TracingCacheProvider that delegates to real
// and records cache operations into tracer.
func NewTracingCacheProvider[T tensor.Numeric](real CacheProvider[T], tracer *compute.Tracer[T]) *TracingCacheProvider[T] {
	return &TracingCacheProvider[T]{real: real, tracer: tracer}
}

// Update delegates to the underlying cache and records KVCacheAppendK and
// KVCacheAppendV traced ops with the layer index.
func (tc *TracingCacheProvider[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	err := tc.real.Update(layer, newK, newV)
	if err != nil {
		return err
	}
	extra := map[string]any{"layer": layer}
	tc.tracer.Record("KVCacheAppendK", []*tensor.TensorNumeric[T]{newK}, newK, extra)
	tc.tracer.Record("KVCacheAppendV", []*tensor.TensorNumeric[T]{newV}, newV, extra)
	return nil
}

// Get delegates to the underlying cache and records KVCacheGetK and
// KVCacheGetV traced ops with the layer index and returned tensors.
func (tc *TracingCacheProvider[T]) Get(layer int) (*LayerKV[T], bool) {
	kv, ok := tc.real.Get(layer)
	if !ok {
		return nil, false
	}
	extra := map[string]any{"layer": layer}
	tc.tracer.Record("KVCacheGetK", nil, kv.Key, extra)
	tc.tracer.Record("KVCacheGetV", nil, kv.Value, extra)
	return kv, true
}

// SeqLen delegates to the underlying cache and records a KVCacheSeqLen op.
// Since SeqLen returns a scalar int (not a tensor), the tracer is marked
// opaque to signal that compilation cannot fully inline this operation.
func (tc *TracingCacheProvider[T]) SeqLen() int {
	n := tc.real.SeqLen()
	tc.tracer.MarkOpaque()
	return n
}

// Reset delegates to the underlying cache.
func (tc *TracingCacheProvider[T]) Reset() {
	tc.real.Reset()
}

// Truncate delegates to the underlying cache.
func (tc *TracingCacheProvider[T]) Truncate(newSeqLen int) {
	tc.real.Truncate(newSeqLen)
}
