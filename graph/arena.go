package graph

import "github.com/zerfoo/zerfoo/tensor"

// BufferArena pre-allocates tensor buffers for use by an ExecutionPlan.
// All buffers are created once and reused across Run() calls.
type BufferArena[T tensor.Numeric] struct {
	buffers []*tensor.TensorNumeric[T]
}

// NewBufferArena pre-allocates one tensor per shape.
func NewBufferArena[T tensor.Numeric](shapes [][]int) *BufferArena[T] {
	a := &BufferArena[T]{
		buffers: make([]*tensor.TensorNumeric[T], len(shapes)),
	}
	for i, shape := range shapes {
		t, _ := tensor.New[T](shape, nil)
		a.buffers[i] = t
	}
	return a
}

// Get returns the pre-allocated buffer at index idx.
func (a *BufferArena[T]) Get(idx int) *tensor.TensorNumeric[T] {
	return a.buffers[idx]
}

// Reset zeros all buffer data for the next execution step.
func (a *BufferArena[T]) Reset() {
	for _, buf := range a.buffers {
		data := buf.Data()
		clear(data)
	}
}
