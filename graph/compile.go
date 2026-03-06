package graph

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/tensor"
)

// Instruction is a single pre-resolved operation in a compiled execution plan.
// It holds a direct kernel function pointer and pre-computed buffer indices,
// eliminating interface dispatch, shape validation, and map lookups at runtime.
type Instruction[T tensor.Numeric] struct {
	Kernel    func(ctx context.Context, inputs []*tensor.TensorNumeric[T], output *tensor.TensorNumeric[T]) error
	InputIdx  []int // indices into the buffer arena
	OutputIdx int   // index into the buffer arena
}

// ExecutionPlan is a compiled, flat instruction sequence that replaces the
// interpreted node-by-node Forward() loop. All buffers are pre-allocated.
type ExecutionPlan[T tensor.Numeric] struct {
	instructions []Instruction[T]
	arena        *BufferArena[T]
	inputIdx     []int // which arena slots receive graph inputs
	outputIdx    int   // which arena slot holds the final output
}

// Run executes the compiled plan. It copies input data into pre-allocated
// arena buffers, executes each instruction in sequence, and returns the output.
func (p *ExecutionPlan[T]) Run(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != len(p.inputIdx) {
		return nil, fmt.Errorf("compiled plan: expected %d inputs, got %d", len(p.inputIdx), len(inputs))
	}

	p.arena.Reset()

	// Copy input data into arena slots.
	for i, idx := range p.inputIdx {
		dst := p.arena.Get(idx)
		copy(dst.Data(), inputs[i].Data())
	}

	// Execute each instruction.
	for i := range p.instructions {
		inst := &p.instructions[i]
		ins := make([]*tensor.TensorNumeric[T], len(inst.InputIdx))
		for j, idx := range inst.InputIdx {
			ins[j] = p.arena.Get(idx)
		}
		out := p.arena.Get(inst.OutputIdx)
		if err := inst.Kernel(ctx, ins, out); err != nil {
			return nil, fmt.Errorf("instruction %d: %w", i, err)
		}
	}

	return p.arena.Get(p.outputIdx), nil
}
