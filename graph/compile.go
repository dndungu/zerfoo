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
	OpName    string // for error reporting
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
			return nil, fmt.Errorf("instruction %d (%s): %w", i, inst.OpName, err)
		}
	}

	return p.arena.Get(p.outputIdx), nil
}

// Compile pre-compiles the graph into a flat ExecutionPlan. It runs one
// Forward() pass to determine tensor shapes, then assigns buffer indices
// and creates instruction kernels for each node.
func (g *Graph[T]) Compile(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*ExecutionPlan[T], error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if len(inputs) != len(g.inputs) {
		return nil, fmt.Errorf("compile: expected %d inputs, got %d", len(g.inputs), len(inputs))
	}

	// Step 1: Run one Forward() to populate memo with shapes.
	g.memo = make(map[Node[T]]*tensor.TensorNumeric[T])
	for i, n := range g.inputs {
		g.memo[n] = inputs[i]
	}
	for _, n := range g.nodes {
		if _, ok := n.(*inputNode[T]); ok {
			continue
		}
		nodeInputs := make([]*tensor.TensorNumeric[T], len(g.dependencies[n]))
		for i, dep := range g.dependencies[n] {
			nodeInputs[i] = g.memo[dep]
		}
		output, err := n.Forward(ctx, nodeInputs...)
		if err != nil {
			return nil, fmt.Errorf("compile forward: node %s: %w", n.OpType(), err)
		}
		g.memo[n] = output
	}

	// Step 2: Assign buffer index to each node in topological order.
	nodeIdx := make(map[Node[T]]int, len(g.nodes))
	for i, n := range g.nodes {
		nodeIdx[n] = i
	}

	// Step 3: Collect shapes and build the arena.
	shapes := make([][]int, len(g.nodes))
	for i, n := range g.nodes {
		if t := g.memo[n]; t != nil {
			shapes[i] = t.Shape()
		} else {
			shapes[i] = []int{1} // fallback for nodes without output
		}
	}
	arena := NewBufferArena[T](shapes)

	// Step 4: Populate slots. Input slots keep pre-allocated buffers (filled
	// each Run). Parameter/constant slots are frozen with their weight tensors.
	inputSlots := make([]int, len(g.inputs))
	for i, n := range g.inputs {
		inputSlots[i] = nodeIdx[n]
	}
	for _, n := range g.nodes {
		if isConstantNode[T](n) {
			idx := nodeIdx[n]
			arena.Set(idx, g.memo[n], true) // frozen: model weights
		}
	}

	// Step 5: Create instructions for each compute node.
	var instructions []Instruction[T]
	for _, n := range g.nodes {
		if _, ok := n.(*inputNode[T]); ok {
			continue
		}
		if isConstantNode[T](n) {
			continue
		}

		outIdx := nodeIdx[n]
		depIndices := make([]int, len(g.dependencies[n]))
		for i, dep := range g.dependencies[n] {
			depIndices[i] = nodeIdx[dep]
		}

		// Create a kernel that calls the node's Forward() and copies the
		// result into the pre-allocated output buffer.
		kernel := func(ctx context.Context, ins []*tensor.TensorNumeric[T], out *tensor.TensorNumeric[T]) error {
			result, err := n.Forward(ctx, ins...)
			if err != nil {
				return err
			}
			copy(out.Data(), result.Data())
			return nil
		}

		instructions = append(instructions, Instruction[T]{
			Kernel:    kernel,
			InputIdx:  depIndices,
			OutputIdx: outIdx,
			OpName:    n.OpType(),
		})
	}

	return &ExecutionPlan[T]{
		instructions: instructions,
		arena:        arena,
		inputIdx:     inputSlots,
		outputIdx:    nodeIdx[g.output],
	}, nil
}
