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
	slotShapes   [][]int                    // shapes from warmup pass
	inputIdx     []int                      // which slots receive graph inputs
	outputIdx    int                        // which slot holds the final output
	frozenIdx    []int                      // slots holding frozen data (params)
}

// InstructionMeta is the exported metadata for a single compiled instruction.
// It contains everything needed by a code generator without exposing the
// Forward() closure.
type InstructionMeta struct {
	OpName    string // operation type (e.g. "Add", "MatMulNBits", "RMSNorm")
	InputIdx  []int  // slot indices for inputs
	OutputIdx int    // slot index for the output
}

// FrozenSlot describes a slot that holds frozen (constant) data such as
// model weights. The Data field holds the tensor from the warmup pass.
type FrozenSlot[T tensor.Numeric] struct {
	SlotIdx int
	Data    *tensor.TensorNumeric[T]
}

// Instructions returns exported metadata for each compute instruction in
// the plan. The order matches the execution order.
func (p *ExecutionPlan[T]) Instructions() []InstructionMeta {
	metas := make([]InstructionMeta, len(p.instructions))
	for i, inst := range p.instructions {
		idx := make([]int, len(inst.InputIdx))
		copy(idx, inst.InputIdx)
		metas[i] = InstructionMeta{
			OpName:    inst.OpName,
			InputIdx:  idx,
			OutputIdx: inst.OutputIdx,
		}
	}
	return metas
}

// SlotShapes returns the shape of each slot as determined during compilation.
// Nil entries indicate slots that were not populated during the warmup pass.
func (p *ExecutionPlan[T]) SlotShapes() [][]int {
	out := make([][]int, len(p.slotShapes))
	for i, s := range p.slotShapes {
		if s != nil {
			cp := make([]int, len(s))
			copy(cp, s)
			out[i] = cp
		}
	}
	return out
}

// FrozenSlots returns the frozen (constant/parameter) slots and their data.
func (p *ExecutionPlan[T]) FrozenSlots() []FrozenSlot[T] {
	frozen := make([]FrozenSlot[T], len(p.frozenIdx))
	for i, idx := range p.frozenIdx {
		frozen[i] = FrozenSlot[T]{
			SlotIdx: idx,
			Data:    p.slots[idx],
		}
	}
	return frozen
}

// InputSlots returns the slot indices that receive graph inputs.
func (p *ExecutionPlan[T]) InputSlots() []int {
	idx := make([]int, len(p.inputIdx))
	copy(idx, p.inputIdx)
	return idx
}

// OutputSlot returns the slot index that holds the final output.
func (p *ExecutionPlan[T]) OutputSlot() int {
	return p.outputIdx
}

// Run executes the compiled plan. It sets input tensors into the slot array,
// executes each instruction in sequence, and returns the output.
func (p *ExecutionPlan[T]) Run(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != len(p.inputIdx) {
		return nil, fmt.Errorf("compiled plan: expected %d inputs, got %d", len(p.inputIdx), len(inputs))
	}

	// Use local slot copy so concurrent Run() calls are safe.
	slots := make([]*tensor.TensorNumeric[T], len(p.slots))
	copy(slots, p.slots) // copies frozen slot pointers (params)

	for i, idx := range p.inputIdx {
		slots[idx] = inputs[i]
	}

	// Execute each instruction: gather inputs by index, call Forward, store result.
	for i := range p.instructions {
		inst := &p.instructions[i]
		ins := make([]*tensor.TensorNumeric[T], len(inst.InputIdx))
		for j, idx := range inst.InputIdx {
			ins[j] = slots[idx]
		}
		result, err := inst.Forward(ctx, ins)
		if err != nil {
			return nil, fmt.Errorf("instruction %d (%s): %w", i, inst.OpName, err)
		}
		slots[inst.OutputIdx] = result
	}

	return slots[p.outputIdx], nil
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

	// Step 5: Record slot shapes from warmup memo.
	slotShapes := make([][]int, len(g.nodes))
	for n, t := range g.memo {
		if idx, ok := nodeIdx[n]; ok && t != nil {
			slotShapes[idx] = t.Shape()
		}
	}

	return &ExecutionPlan[T]{
		instructions: instructions,
		slots:        slots,
		slotShapes:   slotShapes,
		inputIdx:     inputSlots,
		outputIdx:    nodeIdx[g.output],
		frozenIdx:    frozenIdx,
	}, nil
}
