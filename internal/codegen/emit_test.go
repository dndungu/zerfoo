package codegen

import (
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
)

func TestEmitMegakernel(t *testing.T) {
	// Simple graph: slot0 (input) + slot1 (frozen) -> slot2 (output)
	instructions := []graph.InstructionMeta{
		{OpName: "Add", InputIdx: []int{0, 1}, OutputIdx: 2},
	}
	slotShapes := [][]int{
		{1, 4}, // input
		{1, 4}, // frozen weight
		{1, 4}, // output
	}
	frozen := []FrozenSlotMeta{
		{SlotIdx: 1},
	}
	cfg := MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		FrozenSlots:  frozen,
		InputSlots:   []int{0},
		OutputSlot:   2,
	}

	code, err := EmitMegakernel(cfg)
	if err != nil {
		t.Fatalf("EmitMegakernel: %v", err)
	}

	// Verify key parts of the generated code.
	checks := []string{
		"__global__",
		"megakernel",
		"slot_0",
		"slot_1",
		"slot_2",
		"+",
	}
	for _, want := range checks {
		if !strings.Contains(code, want) {
			t.Errorf("generated code missing %q", want)
		}
	}
}

func TestEmitMegakernelUnsupportedOp(t *testing.T) {
	instructions := []graph.InstructionMeta{
		{OpName: "FancyNewOp", InputIdx: []int{0}, OutputIdx: 1},
	}
	slotShapes := [][]int{{1, 4}, {1, 4}}
	cfg := MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		InputSlots:   []int{0},
		OutputSlot:   1,
	}

	_, err := EmitMegakernel(cfg)
	if err == nil {
		t.Fatal("expected error for unsupported op")
	}
	if !strings.Contains(err.Error(), "unsupported") {
		t.Errorf("error should mention 'unsupported': %v", err)
	}
}

func TestEmitMegakernelMultiOp(t *testing.T) {
	// input -> Mul(input, frozen) -> Exp -> output
	instructions := []graph.InstructionMeta{
		{OpName: "Mul", InputIdx: []int{0, 1}, OutputIdx: 2},
		{OpName: "Exp", InputIdx: []int{2}, OutputIdx: 3},
	}
	slotShapes := [][]int{
		{1, 2048}, // input
		{1, 2048}, // frozen weight
		{1, 2048}, // intermediate
		{1, 2048}, // output
	}
	frozen := []FrozenSlotMeta{{SlotIdx: 1}}
	cfg := MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		FrozenSlots:  frozen,
		InputSlots:   []int{0},
		OutputSlot:   3,
	}

	code, err := EmitMegakernel(cfg)
	if err != nil {
		t.Fatalf("EmitMegakernel: %v", err)
	}

	// Both ops should appear in order.
	mulIdx := strings.Index(code, "slot_2[tid] = slot_0[tid] * slot_1[tid]")
	expIdx := strings.Index(code, "expf(slot_2[tid])")
	if mulIdx < 0 {
		t.Error("missing Mul instruction")
	}
	if expIdx < 0 {
		t.Error("missing Exp instruction")
	}
	if mulIdx >= 0 && expIdx >= 0 && mulIdx >= expIdx {
		t.Error("Mul should appear before Exp")
	}
}
