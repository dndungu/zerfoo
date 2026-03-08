package compute

import (
	"reflect"
	"sort"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

func newTestTensor(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float32, size)
	t, err := tensor.New[float32](shape, data)
	if err != nil {
		panic(err)
	}
	return t
}

func TestTracerTensorIdentity(t *testing.T) {
	tr := NewTracer[float32](nil)
	a := newTestTensor([]int{2, 3})

	slot1 := tr.slotFor(a)
	slot2 := tr.slotFor(a)

	if slot1 != slot2 {
		t.Errorf("same tensor got different slots: %d vs %d", slot1, slot2)
	}
}

func TestTracerDifferentTensors(t *testing.T) {
	tr := NewTracer[float32](nil)
	a := newTestTensor([]int{2, 3})
	b := newTestTensor([]int{2, 3})

	slotA := tr.slotFor(a)
	slotB := tr.slotFor(b)

	if slotA == slotB {
		t.Errorf("different tensors got same slot: %d", slotA)
	}
}

func TestTracerFrozenRegistration(t *testing.T) {
	w1 := newTestTensor([]int{4, 4})
	w2 := newTestTensor([]int{8, 4})

	tr := NewTracer[float32]([]*tensor.TensorNumeric[float32]{w1, w2})

	frozen := tr.FrozenSlots()
	sort.Ints(frozen)

	if len(frozen) != 2 {
		t.Fatalf("expected 2 frozen slots, got %d", len(frozen))
	}
	if frozen[0] != 0 || frozen[1] != 1 {
		t.Errorf("expected frozen slots [0, 1], got %v", frozen)
	}

	// Frozen tensors should reuse their pre-registered slots.
	slot := tr.slotFor(w1)
	if slot != 0 {
		t.Errorf("frozen tensor w1 got slot %d, want 0", slot)
	}
}

func TestTracerFrozenNilSkipped(t *testing.T) {
	w1 := newTestTensor([]int{2, 2})
	tr := NewTracer[float32]([]*tensor.TensorNumeric[float32]{nil, w1})

	frozen := tr.FrozenSlots()
	if len(frozen) != 1 {
		t.Fatalf("expected 1 frozen slot, got %d", len(frozen))
	}
}

func TestTracerExtraArgs(t *testing.T) {
	tests := []struct {
		name   string
		opName string
		extra  map[string]any
	}{
		{
			name:   "Softmax",
			opName: "Softmax",
			extra:  map[string]any{"axis": 1},
		},
		{
			name:   "Transpose",
			opName: "Transpose",
			extra:  map[string]any{"axes": []int{0, 2, 1, 3}},
		},
		{
			name:   "Reshape",
			opName: "Reshape",
			extra:  map[string]any{"shape": []int{1, 8, 1, 256}},
		},
		{
			name:   "MulScalar",
			opName: "MulScalar",
			extra:  map[string]any{"scalar": 0.044715},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := NewTracer[float32](nil)
			a := newTestTensor([]int{2, 3})
			out := newTestTensor([]int{2, 3})

			tr.Record(tt.opName, []*tensor.TensorNumeric[float32]{a}, out, tt.extra)

			ops := tr.TracedOps()
			if len(ops) != 1 {
				t.Fatalf("expected 1 op, got %d", len(ops))
			}
			if ops[0].OpName != tt.opName {
				t.Errorf("op name = %q, want %q", ops[0].OpName, tt.opName)
			}
			for k, v := range tt.extra {
				got, ok := ops[0].ExtraArgs[k]
				if !ok {
					t.Errorf("missing ExtraArgs key %q", k)
					continue
				}
				if !reflect.DeepEqual(got, v) {
					t.Errorf("ExtraArgs[%q] = %v, want %v", k, got, v)
				}
			}
		})
	}
}

func TestTracerOpsOrder(t *testing.T) {
	tr := NewTracer[float32](nil)
	a := newTestTensor([]int{2, 3})
	b := newTestTensor([]int{2, 3})
	c := newTestTensor([]int{2, 3})

	tr.Record("Add", []*tensor.TensorNumeric[float32]{a, b}, c, nil)
	tr.Record("Mul", []*tensor.TensorNumeric[float32]{c, a}, b, nil)
	tr.Record("Softmax", []*tensor.TensorNumeric[float32]{b}, a, map[string]any{"axis": 1})

	ops := tr.TracedOps()
	if len(ops) != 3 {
		t.Fatalf("expected 3 ops, got %d", len(ops))
	}

	expected := []string{"Add", "Mul", "Softmax"}
	for i, op := range ops {
		if op.OpName != expected[i] {
			t.Errorf("ops[%d].OpName = %q, want %q", i, op.OpName, expected[i])
		}
	}
}

func TestTracerSlotShapes(t *testing.T) {
	tr := NewTracer[float32](nil)
	a := newTestTensor([]int{2, 3})
	b := newTestTensor([]int{3, 4})
	out := newTestTensor([]int{2, 4})

	tr.Record("MatMul", []*tensor.TensorNumeric[float32]{a, b}, out, nil)

	shapes := tr.SlotShapes()
	op := tr.TracedOps()[0]

	aShape := shapes[op.InputIDs[0]]
	if !reflect.DeepEqual(aShape, []int{2, 3}) {
		t.Errorf("input a shape = %v, want [2 3]", aShape)
	}
	bShape := shapes[op.InputIDs[1]]
	if !reflect.DeepEqual(bShape, []int{3, 4}) {
		t.Errorf("input b shape = %v, want [3 4]", bShape)
	}
	outShape := shapes[op.OutputID]
	if !reflect.DeepEqual(outShape, []int{2, 4}) {
		t.Errorf("output shape = %v, want [2 4]", outShape)
	}
}

func TestTracerFrozenSlots(t *testing.T) {
	w := newTestTensor([]int{4, 4})
	tr := NewTracer[float32]([]*tensor.TensorNumeric[float32]{w})

	frozen := tr.FrozenSlots()
	if len(frozen) != 1 {
		t.Fatalf("expected 1 frozen slot, got %d", len(frozen))
	}
	if frozen[0] != 0 {
		t.Errorf("frozen slot = %d, want 0", frozen[0])
	}
}

func TestTracerHasOpaqueOps(t *testing.T) {
	tr := NewTracer[float32](nil)

	if tr.HasOpaqueOps() {
		t.Error("new tracer should not have opaque ops")
	}

	tr.MarkOpaque()

	if !tr.HasOpaqueOps() {
		t.Error("tracer should have opaque ops after MarkOpaque")
	}
}

func TestTracerRecordGather(t *testing.T) {
	// Pre-register params as frozen weight.
	params := newTestTensor([]int{4, 8})
	tr := NewTracer[float32]([]*tensor.TensorNumeric[float32]{params})

	indices, _ := tensor.New[int]([]int{3}, []int{0, 1, 2})
	output := newTestTensor([]int{3, 8})

	tr.RecordGather(params, indices, output)

	ops := tr.TracedOps()
	if len(ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(ops))
	}
	op := ops[0]
	if op.OpName != "Gather" {
		t.Errorf("op name = %q, want Gather", op.OpName)
	}
	if len(op.InputIDs) != 2 {
		t.Fatalf("expected 2 input IDs, got %d", len(op.InputIDs))
	}
	// params should be frozen slot 0
	if op.InputIDs[0] != 0 {
		t.Errorf("params slot = %d, want 0 (frozen)", op.InputIDs[0])
	}
	// indices should get a new slot
	if op.InputIDs[1] == op.InputIDs[0] {
		t.Error("indices slot should differ from params slot")
	}
	// output should get its own slot
	if op.OutputID == op.InputIDs[0] || op.OutputID == op.InputIDs[1] {
		t.Error("output slot should differ from input slots")
	}
}

func TestTracerSlotForIntTensorIdentity(t *testing.T) {
	tr := NewTracer[float32](nil)
	indices, _ := tensor.New[int]([]int{3}, []int{0, 1, 2})

	slot1 := tr.slotForIntTensor(indices)
	slot2 := tr.slotForIntTensor(indices)

	if slot1 != slot2 {
		t.Errorf("same int tensor got different slots: %d vs %d", slot1, slot2)
	}
}

func TestTracerSlotForIntTensorShape(t *testing.T) {
	tr := NewTracer[float32](nil)
	indices, _ := tensor.New[int]([]int{2, 3}, []int{0, 1, 2, 3, 4, 5})

	slot := tr.slotForIntTensor(indices)
	shapes := tr.SlotShapes()

	if !reflect.DeepEqual(shapes[slot], []int{2, 3}) {
		t.Errorf("int tensor shape = %v, want [2 3]", shapes[slot])
	}
}

func TestTracerRecordMultiOutput(t *testing.T) {
	tests := []struct {
		name       string
		numOutputs int
	}{
		{"two outputs", 2},
		{"three outputs", 3},
		{"four outputs", 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := NewTracer[float32](nil)
			input := newTestTensor([]int{tt.numOutputs * 2, 3})
			outputs := make([]*tensor.TensorNumeric[float32], tt.numOutputs)
			for i := range outputs {
				outputs[i] = newTestTensor([]int{2, 3})
			}

			tr.RecordMultiOutput("Split", []*tensor.TensorNumeric[float32]{input}, outputs, map[string]any{
				"numSplits": tt.numOutputs,
				"axis":      0,
			})

			ops := tr.TracedOps()
			if len(ops) != 1 {
				t.Fatalf("expected 1 op, got %d", len(ops))
			}
			op := ops[0]
			if op.OpName != "Split" {
				t.Errorf("op name = %q, want Split", op.OpName)
			}
			if len(op.OutputIDs) != tt.numOutputs {
				t.Fatalf("expected %d OutputIDs, got %d", tt.numOutputs, len(op.OutputIDs))
			}
			// All output slots should be unique.
			seen := make(map[int]bool)
			for i, id := range op.OutputIDs {
				if seen[id] {
					t.Errorf("duplicate OutputID %d at index %d", id, i)
				}
				seen[id] = true
			}
			// Output slots should differ from input slot.
			for _, id := range op.OutputIDs {
				if id == op.InputIDs[0] {
					t.Errorf("output slot %d should differ from input slot %d", id, op.InputIDs[0])
				}
			}
			// OutputID should be zero (not used for multi-output).
			if op.OutputID != 0 {
				t.Errorf("OutputID = %d, want 0 for multi-output op", op.OutputID)
			}
		})
	}
}

func TestTracerRecordMultiOutputShapes(t *testing.T) {
	tr := NewTracer[float32](nil)
	input := newTestTensor([]int{6, 4})
	out1 := newTestTensor([]int{2, 4})
	out2 := newTestTensor([]int{2, 4})
	out3 := newTestTensor([]int{2, 4})

	tr.RecordMultiOutput("Split", []*tensor.TensorNumeric[float32]{input},
		[]*tensor.TensorNumeric[float32]{out1, out2, out3}, nil)

	shapes := tr.SlotShapes()
	op := tr.TracedOps()[0]
	for i, id := range op.OutputIDs {
		shape := shapes[id]
		if !reflect.DeepEqual(shape, []int{2, 4}) {
			t.Errorf("output %d shape = %v, want [2 4]", i, shape)
		}
	}
}

func TestTracerRecordMultipleInputs(t *testing.T) {
	tr := NewTracer[float32](nil)
	a := newTestTensor([]int{2, 3})
	b := newTestTensor([]int{2, 3})
	out := newTestTensor([]int{2, 3})

	tr.Record("Add", []*tensor.TensorNumeric[float32]{a, b}, out, nil)

	ops := tr.TracedOps()
	if len(ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(ops))
	}
	if len(ops[0].InputIDs) != 2 {
		t.Errorf("expected 2 input IDs, got %d", len(ops[0].InputIDs))
	}
	if ops[0].InputIDs[0] == ops[0].InputIDs[1] {
		t.Error("different input tensors should have different slot IDs")
	}
}
