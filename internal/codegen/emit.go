package codegen

import (
	"fmt"
	"slices"
	"strings"

	"github.com/zerfoo/zerfoo/graph"
)

// FrozenSlotMeta describes a frozen (constant/weight) slot for the emitter.
type FrozenSlotMeta struct {
	SlotIdx int
}

// MegakernelConfig holds all information needed to emit a megakernel .cu file.
type MegakernelConfig struct {
	Instructions []graph.InstructionMeta
	SlotShapes   [][]int
	FrozenSlots  []FrozenSlotMeta
	InputSlots   []int
	OutputSlot   int
}

// slotSize returns the total number of elements for a slot shape.
func slotSize(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// EmitMegakernel generates a complete CUDA .cu source string from the
// compiled instruction tape. Returns an error if any op is unsupported.
func EmitMegakernel(cfg MegakernelConfig) (string, error) {
	var b strings.Builder

	// Header
	b.WriteString("#include \"megakernel_ops.cu\"\n\n")

	// Build set of used slots.
	usedSlots := make(map[int]bool)
	for _, inst := range cfg.Instructions {
		for _, idx := range inst.InputIdx {
			usedSlots[idx] = true
		}
		usedSlots[inst.OutputIdx] = true
	}
	for _, idx := range cfg.InputSlots {
		usedSlots[idx] = true
	}
	usedSlots[cfg.OutputSlot] = true

	// Frozen slot set for parameter declarations.
	frozenSet := make(map[int]bool)
	for _, f := range cfg.FrozenSlots {
		frozenSet[f.SlotIdx] = true
	}

	// Kernel signature: input pointer, output pointer, frozen pointers, pos.
	b.WriteString("__global__ void megakernel(\n")
	b.WriteString("    const float* __restrict__ input,\n")
	b.WriteString("    float* __restrict__ output,\n")
	for _, f := range cfg.FrozenSlots {
		fmt.Fprintf(&b, "    const float* __restrict__ frozen_%d,\n", f.SlotIdx)
	}
	b.WriteString("    int pos,\n")
	b.WriteString("    int num_elements\n")
	b.WriteString(") {\n")

	// Thread index.
	b.WriteString("  int tid = blockIdx.x * blockDim.x + threadIdx.x;\n")
	b.WriteString("  if (tid >= num_elements) return;\n\n")

	// Declare slot registers for non-frozen, non-input slots.
	for idx := range usedSlots {
		if frozenSet[idx] {
			continue
		}
		size := 0
		if idx < len(cfg.SlotShapes) && cfg.SlotShapes[idx] != nil {
			size = slotSize(cfg.SlotShapes[idx])
		}
		if size == 0 {
			size = 1
		}
		// Input slots load from the input pointer.
		if slices.Contains(cfg.InputSlots, idx) {
			fmt.Fprintf(&b, "  // slot_%d: input (size=%d)\n", idx, size)
			fmt.Fprintf(&b, "  float slot_%d_val = input[tid];\n", idx)
		} else {
			fmt.Fprintf(&b, "  float slot_%d_val = 0.0f; // intermediate (size=%d)\n", idx, size)
		}
	}
	b.WriteString("\n")

	// Emit instructions.
	for i, inst := range cfg.Instructions {
		// Build slot info for inputs.
		inputs := make([]SlotInfo, len(inst.InputIdx))
		for j, idx := range inst.InputIdx {
			if idx < len(cfg.SlotShapes) && cfg.SlotShapes[idx] != nil {
				inputs[j] = SlotInfo{Shape: cfg.SlotShapes[idx]}
			}
		}

		code, err := Emit(inst, inputs)
		if err != nil {
			return "", fmt.Errorf("instruction %d (%s): %w", i, inst.OpName, err)
		}
		fmt.Fprintf(&b, "  // [%d] %s\n", i, inst.OpName)
		b.WriteString(code)
		b.WriteString("\n")
	}

	// Write output.
	b.WriteString("\n")
	fmt.Fprintf(&b, "  output[tid] = slot_%d[tid];\n", cfg.OutputSlot)
	b.WriteString("}\n")

	return b.String(), nil
}
