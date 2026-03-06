package graph

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/tensor"
)

// Instruction is a single pre-resolved operation in a compiled execution plan.
// It holds a direct function that calls node.Forward() with pre-computed
// buffer indices, eliminating dependency map lookups and memo operations.
type Instruction[T tensor.Numeric] struct {
	Forward   func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
	InputIdx  []int  // indices into the slot array
	OutputIdx int    // index into the slot array
	OpName    string // for error reporting
}

// ExecutionPlan is a compiled, flat instruction sequence that replaces the
// interpreted node-by-node Forward() loop. Node outputs are stored in an
// indexed slot array instead of a map, eliminating map lookups.
type ExecutionPlan[T tensor.Numeric] struct {
	instructions []Instruction[T]
	slots        []*tensor.TensorNumeric[T] // indexed output storage
	inputIdx     []int                      // which slots receive graph inputs
	outputIdx    int                        // which slot holds the final output
	frozenIdx    []int                      // slots holding frozen data (params)
}

// Run executes the compiled plan. It sets input tensors into the slot array,
// executes each instruction in sequence, and returns the output.
func (p *ExecutionPlan[T]) Run(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != len(p.inputIdx) {
		return nil, fmt.Errorf("compiled plan: expected %d inputs, got %d", len(p.inputIdx), len(inputs))
	}

	// Set input tensors directly into slots (no copy needed).
	for i, idx := range p.inputIdx {
		p.slots[idx] = inputs[i]
	}

	// Execute each instruction: gather inputs by index, call Forward, store result.
	for i := range p.instructions {
		inst := &p.instructions[i]
		ins := make([]*tensor.TensorNumeric[T], len(inst.InputIdx))
		for j, idx := range inst.InputIdx {
			ins[j] = p.slots[idx]
		}
		result, err := inst.Forward(ctx, ins)
		if err != nil {
			return nil, fmt.Errorf("instruction %d (%s): %w", i, inst.OpName, err)
		}
		p.slots[inst.OutputIdx] = result
	}

	return p.slots[p.outputIdx], nil
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

	// Step 1: Get tensor shapes. Use existing memo from the last Forward()
	// if available (avoids re-running Forward which would corrupt model state
	// like attention KV caches). Otherwise, run one Forward() to populate memo.
	if len(g.memo) == 0 {
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
	}

	// Step 2: Assign slot index to each node in topological order.
	nodeIdx := make(map[Node[T]]int, len(g.nodes))
	for i, n := range g.nodes {
		nodeIdx[n] = i
	}

	// Step 3: Create slot array and populate frozen slots (params/constants).
	slots := make([]*tensor.TensorNumeric[T], len(g.nodes))
	var frozenIdx []int
	inputSlots := make([]int, len(g.inputs))
	for i, n := range g.inputs {
		inputSlots[i] = nodeIdx[n]
	}
	for _, n := range g.nodes {
		if isConstantNode[T](n) {
			idx := nodeIdx[n]
			slots[idx] = g.memo[n] // frozen: model weights
			frozenIdx = append(frozenIdx, idx)
		}
	}

	// Step 4: Create instructions for each compute node.
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

		fwd := func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return n.Forward(ctx, inputs...)
		}
		instructions = append(instructions, Instruction[T]{
			Forward:   fwd,
			InputIdx:  depIndices,
			OutputIdx: outIdx,
			OpName:    n.OpType(),
		})
	}

	return &ExecutionPlan[T]{
		instructions: instructions,
		slots:        slots,
		inputIdx:     inputSlots,
		outputIdx:    nodeIdx[g.output],
		frozenIdx:    frozenIdx,
	}, nil
}
