package generate

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// mockCache is a minimal CacheProvider for testing TracingCacheProvider.
type mockCache[T tensor.Numeric] struct {
	layers  map[int]*LayerKV[T]
	seqLen  int
	updated []mockUpdate[T]
	resets  int
}

type mockUpdate[T tensor.Numeric] struct {
	layer int
	k, v  *tensor.TensorNumeric[T]
}

func newMockCache[T tensor.Numeric]() *mockCache[T] {
	return &mockCache[T]{layers: make(map[int]*LayerKV[T])}
}

func (m *mockCache[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	m.updated = append(m.updated, mockUpdate[T]{layer: layer, k: newK, v: newV})
	m.layers[layer] = &LayerKV[T]{Key: newK, Value: newV}
	m.seqLen++
	return nil
}

func (m *mockCache[T]) Get(layer int) (*LayerKV[T], bool) {
	kv, ok := m.layers[layer]
	return kv, ok
}

func (m *mockCache[T]) SeqLen() int {
	return m.seqLen
}

func (m *mockCache[T]) Reset() {
	m.resets++
	m.layers = make(map[int]*LayerKV[T])
	m.seqLen = 0
}

func (m *mockCache[T]) Truncate(newSeqLen int) {
	if newSeqLen < m.seqLen {
		m.seqLen = newSeqLen
	}
}

func newTensor(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, s := range shape {
		size *= s
	}
	t, err := tensor.New[float32](shape, make([]float32, size))
	if err != nil {
		panic(err)
	}
	return t
}

func TestTracingCacheProviderDelegatesUpdate(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	k := newTensor([]int{1, 1, 4})
	v := newTensor([]int{1, 1, 4})

	if err := tc.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	if len(mock.updated) != 1 {
		t.Fatalf("expected 1 update, got %d", len(mock.updated))
	}
	if mock.updated[0].layer != 0 {
		t.Errorf("layer = %d, want 0", mock.updated[0].layer)
	}
	if mock.updated[0].k != k {
		t.Error("key tensor not delegated")
	}
	if mock.updated[0].v != v {
		t.Error("value tensor not delegated")
	}
}

func TestTracingCacheProviderDelegatesGet(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	// Get on empty cache should return false.
	_, ok := tc.Get(0)
	if ok {
		t.Error("expected Get to return false on empty cache")
	}

	// Populate and retrieve.
	k := newTensor([]int{1, 1, 4})
	v := newTensor([]int{1, 1, 4})
	if err := tc.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	kv, ok := tc.Get(0)
	if !ok {
		t.Fatal("expected Get to return true after Update")
	}
	if kv.Key != k || kv.Value != v {
		t.Error("Get did not return expected tensors")
	}
}

func TestTracingCacheProviderDelegatesSeqLen(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	if tc.SeqLen() != 0 {
		t.Errorf("SeqLen = %d, want 0", tc.SeqLen())
	}

	k := newTensor([]int{1, 1, 4})
	v := newTensor([]int{1, 1, 4})
	_ = tc.Update(0, k, v)

	if tc.SeqLen() != 1 {
		t.Errorf("SeqLen = %d, want 1", tc.SeqLen())
	}
}

func TestTracingCacheProviderDelegatesReset(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	tc.Reset()
	if mock.resets != 1 {
		t.Errorf("resets = %d, want 1", mock.resets)
	}
}

func TestTracingCacheProviderDelegatesTruncate(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	k := newTensor([]int{1, 1, 4})
	v := newTensor([]int{1, 1, 4})
	_ = tc.Update(0, k, v)
	_ = tc.Update(0, k, v)

	tc.Truncate(1)
	if mock.seqLen != 1 {
		t.Errorf("seqLen after Truncate = %d, want 1", mock.seqLen)
	}
}

func TestTracingCacheProviderRecordsUpdateOps(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	k := newTensor([]int{1, 1, 4})
	v := newTensor([]int{1, 1, 4})
	_ = tc.Update(2, k, v)

	ops := tracer.TracedOps()
	if len(ops) != 2 {
		t.Fatalf("expected 2 traced ops, got %d", len(ops))
	}

	tests := []struct {
		idx    int
		opName string
	}{
		{0, "KVCacheAppendK"},
		{1, "KVCacheAppendV"},
	}
	for _, tt := range tests {
		op := ops[tt.idx]
		if op.OpName != tt.opName {
			t.Errorf("ops[%d].OpName = %q, want %q", tt.idx, op.OpName, tt.opName)
		}
		layer, ok := op.ExtraArgs["layer"]
		if !ok {
			t.Errorf("ops[%d] missing layer ExtraArg", tt.idx)
		} else if layer != 2 {
			t.Errorf("ops[%d] layer = %v, want 2", tt.idx, layer)
		}
	}
}

func TestTracingCacheProviderRecordsGetOps(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	k := newTensor([]int{1, 1, 4})
	v := newTensor([]int{1, 1, 4})
	_ = tc.Update(3, k, v)

	// Clear the Update ops from the trace by noting their count.
	updateOps := len(tracer.TracedOps())

	_, _ = tc.Get(3)

	ops := tracer.TracedOps()
	getOps := ops[updateOps:]
	if len(getOps) != 2 {
		t.Fatalf("expected 2 Get traced ops, got %d", len(getOps))
	}

	tests := []struct {
		idx    int
		opName string
	}{
		{0, "KVCacheGetK"},
		{1, "KVCacheGetV"},
	}
	for _, tt := range tests {
		op := getOps[tt.idx]
		if op.OpName != tt.opName {
			t.Errorf("getOps[%d].OpName = %q, want %q", tt.idx, op.OpName, tt.opName)
		}
		layer, ok := op.ExtraArgs["layer"]
		if !ok {
			t.Errorf("getOps[%d] missing layer ExtraArg", tt.idx)
		} else if layer != 3 {
			t.Errorf("getOps[%d] layer = %v, want 3", tt.idx, layer)
		}
	}
}

func TestTracingCacheProviderSeqLenMarksOpaque(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	if tracer.HasOpaqueOps() {
		t.Error("tracer should not have opaque ops before SeqLen")
	}

	_ = tc.SeqLen()

	if !tracer.HasOpaqueOps() {
		t.Error("tracer should have opaque ops after SeqLen")
	}
}

func TestTracingCacheProviderGetMissNoOps(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	_, ok := tc.Get(0)
	if ok {
		t.Error("expected Get to return false on empty cache")
	}

	ops := tracer.TracedOps()
	if len(ops) != 0 {
		t.Errorf("expected 0 traced ops for cache miss, got %d", len(ops))
	}
}

func TestTracingCacheProviderMultipleLayerOps(t *testing.T) {
	mock := newMockCache[float32]()
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](mock, tracer)

	for layer := range 3 {
		k := newTensor([]int{1, 1, 4})
		v := newTensor([]int{1, 1, 4})
		if err := tc.Update(layer, k, v); err != nil {
			t.Fatalf("Update layer %d: %v", layer, err)
		}
	}

	ops := tracer.TracedOps()
	// 3 layers * 2 ops (AppendK + AppendV) = 6 ops
	if len(ops) != 6 {
		t.Fatalf("expected 6 traced ops, got %d", len(ops))
	}

	for i := range 3 {
		appendK := ops[i*2]
		appendV := ops[i*2+1]
		if appendK.OpName != "KVCacheAppendK" {
			t.Errorf("ops[%d].OpName = %q, want KVCacheAppendK", i*2, appendK.OpName)
		}
		if appendV.OpName != "KVCacheAppendV" {
			t.Errorf("ops[%d].OpName = %q, want KVCacheAppendV", i*2+1, appendV.OpName)
		}
		if appendK.ExtraArgs["layer"] != i {
			t.Errorf("ops[%d] layer = %v, want %d", i*2, appendK.ExtraArgs["layer"], i)
		}
	}
}
