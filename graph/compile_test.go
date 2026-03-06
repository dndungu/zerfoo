package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

func TestExecutionPlanRun(t *testing.T) {
	// Build a simple 2-instruction plan: input -> double -> add10
	// Buffer 0: input, Buffer 1: doubled, Buffer 2: result
	inputShape := []int{1, 4}
	arena := NewBufferArena[float32]([][]int{inputShape, inputShape, inputShape})

	double := func(_ context.Context, inputs []*tensor.TensorNumeric[float32], output *tensor.TensorNumeric[float32]) error {
		in := inputs[0].Data()
		out := output.Data()
		for i := range in {
			out[i] = in[i] * 2
		}
		return nil
	}
	add10 := func(_ context.Context, inputs []*tensor.TensorNumeric[float32], output *tensor.TensorNumeric[float32]) error {
		in := inputs[0].Data()
		out := output.Data()
		for i := range in {
			out[i] = in[i] + 10
		}
		return nil
	}

	plan := &ExecutionPlan[float32]{
		instructions: []Instruction[float32]{
			{Kernel: double, InputIdx: []int{0}, OutputIdx: 1},
			{Kernel: add10, InputIdx: []int{1}, OutputIdx: 2},
		},
		arena:     arena,
		inputIdx:  []int{0},
		outputIdx: 2,
	}

	input, err := tensor.New[float32](inputShape, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}

	result, err := plan.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := []float32{12, 14, 16, 18} // (x*2)+10
	got := result.Data()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestExecutionPlanRunDiamond(t *testing.T) {
	// Diamond: input -> branch1 (*2), input -> branch2 (*3), merge (branch1 + branch2)
	shape := []int{1, 3}
	arena := NewBufferArena[float32]([][]int{shape, shape, shape, shape})

	mul2 := func(_ context.Context, inputs []*tensor.TensorNumeric[float32], output *tensor.TensorNumeric[float32]) error {
		for i, v := range inputs[0].Data() {
			output.Data()[i] = v * 2
		}
		return nil
	}
	mul3 := func(_ context.Context, inputs []*tensor.TensorNumeric[float32], output *tensor.TensorNumeric[float32]) error {
		for i, v := range inputs[0].Data() {
			output.Data()[i] = v * 3
		}
		return nil
	}
	add := func(_ context.Context, inputs []*tensor.TensorNumeric[float32], output *tensor.TensorNumeric[float32]) error {
		a := inputs[0].Data()
		b := inputs[1].Data()
		out := output.Data()
		for i := range a {
			out[i] = a[i] + b[i]
		}
		return nil
	}

	plan := &ExecutionPlan[float32]{
		instructions: []Instruction[float32]{
			{Kernel: mul2, InputIdx: []int{0}, OutputIdx: 1},
			{Kernel: mul3, InputIdx: []int{0}, OutputIdx: 2},
			{Kernel: add, InputIdx: []int{1, 2}, OutputIdx: 3},
		},
		arena:     arena,
		inputIdx:  []int{0},
		outputIdx: 3,
	}

	input, _ := tensor.New[float32](shape, []float32{1, 2, 3})
	result, err := plan.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := []float32{5, 10, 15} // x*2 + x*3 = x*5
	got := result.Data()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestExecutionPlanRunReuse(t *testing.T) {
	// Verify multiple runs produce correct results (arena.Reset works)
	shape := []int{1, 2}
	arena := NewBufferArena[float32]([][]int{shape, shape})

	double := func(_ context.Context, inputs []*tensor.TensorNumeric[float32], output *tensor.TensorNumeric[float32]) error {
		for i, v := range inputs[0].Data() {
			output.Data()[i] = v * 2
		}
		return nil
	}

	plan := &ExecutionPlan[float32]{
		instructions: []Instruction[float32]{
			{Kernel: double, InputIdx: []int{0}, OutputIdx: 1},
		},
		arena:     arena,
		inputIdx:  []int{0},
		outputIdx: 1,
	}

	ctx := context.Background()
	tests := []struct {
		input []float32
		want  []float32
	}{
		{[]float32{1, 2}, []float32{2, 4}},
		{[]float32{5, 10}, []float32{10, 20}},
		{[]float32{-1, 0}, []float32{-2, 0}},
	}
	for _, tt := range tests {
		in, _ := tensor.New[float32](shape, tt.input)
		result, err := plan.Run(ctx, in)
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		got := result.Data()
		for i := range tt.want {
			if got[i] != tt.want[i] {
				t.Errorf("input %v: result[%d] = %v, want %v", tt.input, i, got[i], tt.want[i])
			}
		}
	}
}
