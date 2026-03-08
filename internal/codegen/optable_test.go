package codegen

import (
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
)

func TestEmitterRegistered(t *testing.T) {
	// All Gemma 3 ops should have emitters.
	type opSpec struct {
		name      string
		numInputs int
	}
	gemmaOps := []opSpec{
		{"Add", 2}, {"Sub", 2}, {"Mul", 2}, {"Div", 2}, {"Pow", 2},
		{"Exp", 1}, {"Log", 1}, {"Sqrt", 1}, {"Rsqrt", 1}, {"Tanh", 1},
		{"Neg", 1}, {"Abs", 1}, {"Silu", 1},
		{"AddScalar", 1}, {"MulScalar", 1}, {"SubScalar", 1}, {"DivScalar", 1}, {"PowScalar", 1},
		{"RMSNorm", 2}, {"Softmax", 1},
		{"ReduceSum", 1}, {"ReduceMean", 1},
		{"Slice", 1}, {"Repeat", 1},
		{"MatMul", 2}, {"MatMulNBits", 2},
		{"Gather", 2}, {"Concat", 1}, {"Reshape", 1}, {"Transpose", 1},
	}
	for _, op := range gemmaOps {
		inputIdx := make([]int, op.numInputs)
		for i := range inputIdx {
			inputIdx[i] = i
		}
		meta := graph.InstructionMeta{OpName: op.name, InputIdx: inputIdx, OutputIdx: 10}
		slots := make([]SlotInfo, op.numInputs)
		for i := range slots {
			slots[i] = SlotInfo{Shape: []int{1, 2048}}
		}
		_, err := Emit(meta, slots)
		if err != nil {
			t.Errorf("op %q: %v", op.name, err)
		}
	}
}

func TestEmitterUnsupportedOp(t *testing.T) {
	meta := graph.InstructionMeta{OpName: "UnknownFancyOp"}
	info := SlotInfo{Shape: []int{1, 4}}
	_, err := Emit(meta, []SlotInfo{info})
	if err == nil {
		t.Error("expected error for unsupported op, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported") {
		t.Errorf("error should contain 'unsupported': %v", err)
	}
}

func TestEmitterOutputFormat(t *testing.T) {
	tests := []struct {
		op      string
		inputs  int
		wantSub string // substring that should appear in emitted code
	}{
		{"Add", 2, "slot_"},
		{"Exp", 1, "expf"},
		{"MulScalar", 1, "slot_"},
		{"RMSNorm", 2, "dev_rmsnorm"},
		{"Softmax", 1, "dev_softmax"},
		{"ReduceSum", 1, "dev_reduce_sum"},
		{"ReduceMean", 1, "dev_reduce_mean"},
		{"Slice", 1, "dev_slice"},
		{"Repeat", 1, "dev_repeat"},
		{"MatMul", 2, "dev_gemv"},
		{"Gather", 2, "dev_gather"},
	}
	for _, tc := range tests {
		meta := graph.InstructionMeta{
			OpName:    tc.op,
			InputIdx:  make([]int, tc.inputs),
			OutputIdx: 10,
		}
		slots := make([]SlotInfo, tc.inputs)
		for i := range slots {
			slots[i] = SlotInfo{Shape: []int{1, 2048}}
		}
		code, err := Emit(meta, slots)
		if err != nil {
			t.Errorf("op %q: %v", tc.op, err)
			continue
		}
		if !strings.Contains(code, tc.wantSub) {
			t.Errorf("op %q: output %q missing %q", tc.op, code, tc.wantSub)
		}
	}
}
