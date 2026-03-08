package compute

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// mockTracer records trace events for testing.
type mockTracer[T tensor.Numeric] struct {
	events      []traceEvent[T]
	multiEvents []multiOutputEvent[T]
}

type traceEvent[T tensor.Numeric] struct {
	opName string
	inputs []*tensor.TensorNumeric[T]
	output *tensor.TensorNumeric[T]
	extra  map[string]any
}

type multiOutputEvent[T tensor.Numeric] struct {
	opName  string
	inputs  []*tensor.TensorNumeric[T]
	outputs []*tensor.TensorNumeric[T]
	extra   map[string]any
}

func (m *mockTracer[T]) Record(opName string, inputs []*tensor.TensorNumeric[T], output *tensor.TensorNumeric[T], extra map[string]any) {
	m.events = append(m.events, traceEvent[T]{opName: opName, inputs: inputs, output: output, extra: extra})
}

func (m *mockTracer[T]) RecordMultiOutput(opName string, inputs []*tensor.TensorNumeric[T], outputs []*tensor.TensorNumeric[T], extra map[string]any) {
	m.multiEvents = append(m.multiEvents, multiOutputEvent[T]{opName: opName, inputs: inputs, outputs: outputs, extra: extra})
}

func (m *mockTracer[T]) RecordGather(params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T]) {
	m.events = append(m.events, traceEvent[T]{opName: "Gather", inputs: []*tensor.TensorNumeric[T]{params}, output: output, extra: nil})
}

func newTestProxy() (*EngineProxy[int], *mockTracer[int]) {
	cpu := NewCPUEngine[int](numeric.IntOps{})
	proxy := NewEngineProxy[int](cpu)
	tracer := &mockTracer[int]{}
	proxy.StartTracing(tracer)
	return proxy, tracer
}

func TestEngineProxy_TracedBinaryOps(t *testing.T) {
	tests := []struct {
		name   string
		opName string
		call   func(ctx context.Context, p *EngineProxy[int], a, b *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error)
	}{
		{"Add", "Add", func(ctx context.Context, p *EngineProxy[int], a, b *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Add(ctx, a, b)
		}},
		{"Sub", "Sub", func(ctx context.Context, p *EngineProxy[int], a, b *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Sub(ctx, a, b)
		}},
		{"Mul", "Mul", func(ctx context.Context, p *EngineProxy[int], a, b *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Mul(ctx, a, b)
		}},
		{"Div", "Div", func(ctx context.Context, p *EngineProxy[int], a, b *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Div(ctx, a, b)
		}},
		{"Pow", "Pow", func(ctx context.Context, p *EngineProxy[int], a, b *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Pow(ctx, a, b)
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			proxy, tracer := newTestProxy()
			a, _ := tensor.New[int]([]int{2}, []int{4, 6})
			b, _ := tensor.New[int]([]int{2}, []int{2, 3})
			ctx := context.Background()

			_, err := tc.call(ctx, proxy, a, b)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(tracer.events) != 1 {
				t.Fatalf("expected 1 trace event, got %d", len(tracer.events))
			}
			if tracer.events[0].opName != tc.opName {
				t.Errorf("expected opName %q, got %q", tc.opName, tracer.events[0].opName)
			}
			if len(tracer.events[0].inputs) != 2 {
				t.Errorf("expected 2 inputs, got %d", len(tracer.events[0].inputs))
			}
			if tracer.events[0].output == nil {
				t.Error("expected non-nil output")
			}
		})
	}
}

func TestEngineProxy_TracedUnaryOps(t *testing.T) {
	tests := []struct {
		name   string
		opName string
		call   func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error)
	}{
		{"Exp", "Exp", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Exp(ctx, a)
		}},
		{"Log", "Log", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Log(ctx, a)
		}},
		{"Tanh", "Tanh", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Tanh(ctx, a)
		}},
		{"Sqrt", "Sqrt", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Sqrt(ctx, a)
		}},
		{"Rsqrt", "Rsqrt", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Rsqrt(ctx, a)
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			proxy, tracer := newTestProxy()
			a, _ := tensor.New[int]([]int{2}, []int{4, 9})
			ctx := context.Background()

			_, err := tc.call(ctx, proxy, a)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(tracer.events) != 1 {
				t.Fatalf("expected 1 trace event, got %d", len(tracer.events))
			}
			if tracer.events[0].opName != tc.opName {
				t.Errorf("expected opName %q, got %q", tc.opName, tracer.events[0].opName)
			}
			if len(tracer.events[0].inputs) != 1 {
				t.Errorf("expected 1 input, got %d", len(tracer.events[0].inputs))
			}
		})
	}
}

func TestEngineProxy_TracedScalarOps(t *testing.T) {
	tests := []struct {
		name   string
		opName string
		call   func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error)
	}{
		{"MulScalar", "MulScalar", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.MulScalar(ctx, a, 2)
		}},
		{"AddScalar", "AddScalar", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.AddScalar(ctx, a, 10)
		}},
		{"DivScalar", "DivScalar", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.DivScalar(ctx, a, 2)
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			proxy, tracer := newTestProxy()
			a, _ := tensor.New[int]([]int{2}, []int{4, 6})
			ctx := context.Background()

			_, err := tc.call(ctx, proxy, a)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(tracer.events) != 1 {
				t.Fatalf("expected 1 trace event, got %d", len(tracer.events))
			}
			if tracer.events[0].opName != tc.opName {
				t.Errorf("expected opName %q, got %q", tc.opName, tracer.events[0].opName)
			}
		})
	}
}

func TestEngineProxy_TracedReductionOps(t *testing.T) {
	tests := []struct {
		name   string
		opName string
		call   func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error)
	}{
		{"Softmax", "Softmax", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Softmax(ctx, a, 0)
		}},
		{"ReduceSum", "ReduceSum", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.ReduceSum(ctx, a, 0, false)
		}},
		{"ReduceMean", "ReduceMean", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.ReduceMean(ctx, a, 0, false)
		}},
		{"Sum", "Sum", func(ctx context.Context, p *EngineProxy[int], a *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return p.Sum(ctx, a, 0, false)
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			proxy, tracer := newTestProxy()
			a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
			ctx := context.Background()

			_, err := tc.call(ctx, proxy, a)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(tracer.events) != 1 {
				t.Fatalf("expected 1 trace event, got %d", len(tracer.events))
			}
			if tracer.events[0].opName != tc.opName {
				t.Errorf("expected opName %q, got %q", tc.opName, tracer.events[0].opName)
			}
		})
	}
}

func TestEngineProxy_TracedShapeOps(t *testing.T) {
	ctx := context.Background()

	t.Run("Reshape", func(t *testing.T) {
		proxy, tracer := newTestProxy()
		a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
		_, err := proxy.Reshape(ctx, a, []int{4})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tracer.events) != 1 || tracer.events[0].opName != "Reshape" {
			t.Errorf("expected Reshape trace, got %v", tracer.events)
		}
	})

	t.Run("Transpose", func(t *testing.T) {
		proxy, tracer := newTestProxy()
		a, _ := tensor.New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
		_, err := proxy.Transpose(ctx, a, []int{1, 0})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tracer.events) != 1 || tracer.events[0].opName != "Transpose" {
			t.Errorf("expected Transpose trace, got %v", tracer.events)
		}
	})

	t.Run("Concat", func(t *testing.T) {
		proxy, tracer := newTestProxy()
		a, _ := tensor.New[int]([]int{2}, []int{1, 2})
		b, _ := tensor.New[int]([]int{2}, []int{3, 4})
		_, err := proxy.Concat(ctx, []*tensor.TensorNumeric[int]{a, b}, 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tracer.events) != 1 {
			t.Fatalf("expected 1 Concat trace event, got %d", len(tracer.events))
		}
		ev := tracer.events[0]
		if ev.opName != "Concat" {
			t.Errorf("expected opName Concat, got %q", ev.opName)
		}
		if len(ev.inputs) != 2 {
			t.Errorf("expected 2 inputs for Concat, got %d", len(ev.inputs))
		}
		if ev.inputs[0] != a || ev.inputs[1] != b {
			t.Error("Concat inputs should match provided tensors")
		}
		if ev.output == nil {
			t.Error("expected non-nil output")
		}
	})

	t.Run("Split", func(t *testing.T) {
		proxy, tracer := newTestProxy()
		a, _ := tensor.New[int]([]int{4}, []int{1, 2, 3, 4})
		results, err := proxy.Split(ctx, a, 2, 0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Split should produce a single multi-output event, not per-output events.
		if len(tracer.events) != 0 {
			t.Errorf("expected 0 single-output events for Split, got %d", len(tracer.events))
		}
		if len(tracer.multiEvents) != 1 {
			t.Fatalf("expected 1 multi-output event, got %d", len(tracer.multiEvents))
		}
		ev := tracer.multiEvents[0]
		if ev.opName != "Split" {
			t.Errorf("expected opName Split, got %q", ev.opName)
		}
		if len(ev.inputs) != 1 {
			t.Errorf("expected 1 input, got %d", len(ev.inputs))
		}
		if len(ev.outputs) != 2 {
			t.Errorf("expected 2 outputs, got %d", len(ev.outputs))
		}
		if ev.outputs[0] != results[0] || ev.outputs[1] != results[1] {
			t.Error("output tensors should match Split results")
		}
		if ev.extra["numSplits"] != 2 {
			t.Errorf("expected numSplits=2, got %v", ev.extra["numSplits"])
		}
		if ev.extra["axis"] != 0 {
			t.Errorf("expected axis=0, got %v", ev.extra["axis"])
		}
	})

	t.Run("Repeat", func(t *testing.T) {
		proxy, tracer := newTestProxy()
		a, _ := tensor.New[int]([]int{2}, []int{1, 2})
		_, err := proxy.Repeat(ctx, a, 0, 3)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tracer.events) != 1 || tracer.events[0].opName != "Repeat" {
			t.Errorf("expected Repeat trace, got %v", tracer.events)
		}
	})
}

func TestEngineProxy_MatMul(t *testing.T) {
	proxy, tracer := newTestProxy()
	a, _ := tensor.New[int]([]int{2, 3}, []int{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[int]([]int{3, 2}, []int{7, 8, 9, 10, 11, 12})
	ctx := context.Background()

	result, err := proxy.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify result matches direct CPUEngine call.
	cpu := NewCPUEngine[int](numeric.IntOps{})
	expected, _ := cpu.MatMul(ctx, a, b)
	if !reflect.DeepEqual(result.Data(), expected.Data()) {
		t.Errorf("MatMul result mismatch: got %v, expected %v", result.Data(), expected.Data())
	}

	if len(tracer.events) != 1 {
		t.Fatalf("expected 1 trace event, got %d", len(tracer.events))
	}
	if tracer.events[0].opName != "MatMul" {
		t.Errorf("expected opName MatMul, got %q", tracer.events[0].opName)
	}
	if len(tracer.events[0].inputs) != 2 {
		t.Errorf("expected 2 inputs, got %d", len(tracer.events[0].inputs))
	}
}

func TestEngineProxy_Gather_Traced(t *testing.T) {
	proxy, tracer := newTestProxy()
	ctx := context.Background()

	// params: 3x2 embedding table
	params, _ := tensor.New[int]([]int{3, 2}, []int{10, 20, 30, 40, 50, 60})
	// indices: [0, 2]
	indices, _ := tensor.New[int]([]int{2}, []int{0, 2})
	// output: 2x2
	output, _ := tensor.New[int]([]int{2, 2}, []int{0, 0, 0, 0})

	err := proxy.Gather(ctx, params, indices, output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(tracer.events) != 1 {
		t.Fatalf("expected 1 trace event, got %d", len(tracer.events))
	}
	ev := tracer.events[0]
	if ev.opName != "Gather" {
		t.Errorf("expected opName Gather, got %q", ev.opName)
	}
	if ev.output == nil {
		t.Error("expected non-nil output")
	}
	// Verify result matches direct CPU engine call.
	cpu := NewCPUEngine[int](numeric.IntOps{})
	expected, _ := tensor.New[int]([]int{2, 2}, []int{0, 0, 0, 0})
	_ = cpu.Gather(ctx, params, indices, expected)
	if !reflect.DeepEqual(output.Data(), expected.Data()) {
		t.Errorf("Gather result mismatch: got %v, expected %v", output.Data(), expected.Data())
	}
}

func TestEngineProxy_Gather_NotTracedWhenOff(t *testing.T) {
	cpu := NewCPUEngine[int](numeric.IntOps{})
	proxy := NewEngineProxy[int](cpu)
	ctx := context.Background()

	params, _ := tensor.New[int]([]int{3, 2}, []int{10, 20, 30, 40, 50, 60})
	indices, _ := tensor.New[int]([]int{2}, []int{0, 2})
	output, _ := tensor.New[int]([]int{2, 2}, []int{0, 0, 0, 0})

	err := proxy.Gather(ctx, params, indices, output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// No panic with nil tracer.
}

func TestEngineProxy_NonTracedMethods(t *testing.T) {
	proxy, tracer := newTestProxy()
	ctx := context.Background()

	t.Run("Ops", func(t *testing.T) {
		ops := proxy.Ops()
		if ops == nil {
			t.Error("expected non-nil Ops")
		}
		if len(tracer.events) != 0 {
			t.Errorf("Ops should not be traced, got %d events", len(tracer.events))
		}
	})

	t.Run("UnaryOp", func(t *testing.T) {
		tracer.events = nil
		a, _ := tensor.New[int]([]int{2}, []int{1, 2})
		_, err := proxy.UnaryOp(ctx, a, func(v int) int { return v * 2 })
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tracer.events) != 0 {
			t.Errorf("UnaryOp should not be traced, got %d events", len(tracer.events))
		}
	})

	t.Run("Zero", func(t *testing.T) {
		tracer.events = nil
		a, _ := tensor.New[int]([]int{2}, []int{1, 2})
		err := proxy.Zero(ctx, a)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tracer.events) != 0 {
			t.Errorf("Zero should not be traced, got %d events", len(tracer.events))
		}
	})

	t.Run("Fill", func(t *testing.T) {
		tracer.events = nil
		a, _ := tensor.New[int]([]int{2}, []int{0, 0})
		err := proxy.Fill(ctx, a, 42)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tracer.events) != 0 {
			t.Errorf("Fill should not be traced, got %d events", len(tracer.events))
		}
	})

	t.Run("Copy", func(t *testing.T) {
		tracer.events = nil
		src, _ := tensor.New[int]([]int{2}, []int{1, 2})
		dst, _ := tensor.New[int]([]int{2}, []int{0, 0})
		err := proxy.Copy(ctx, dst, src)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tracer.events) != 0 {
			t.Errorf("Copy should not be traced, got %d events", len(tracer.events))
		}
	})

	t.Run("TanhPrime", func(t *testing.T) {
		tracer.events = nil
		a, _ := tensor.New[int]([]int{2}, []int{1, 2})
		up, _ := tensor.New[int]([]int{2}, []int{1, 1})
		_, err := proxy.TanhPrime(ctx, a, up)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tracer.events) != 0 {
			t.Errorf("TanhPrime should not be traced, got %d events", len(tracer.events))
		}
	})
}

func TestEngineProxy_NoTracerProducesNoTrace(t *testing.T) {
	cpu := NewCPUEngine[int](numeric.IntOps{})
	proxy := NewEngineProxy[int](cpu)
	// No tracer set -- tracing is off.
	ctx := context.Background()

	a, _ := tensor.New[int]([]int{2}, []int{1, 2})
	b, _ := tensor.New[int]([]int{2}, []int{3, 4})

	_, err := proxy.Add(ctx, a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = proxy.MatMul(ctx,
		func() *tensor.TensorNumeric[int] { t2, _ := tensor.New[int]([]int{1, 2}, []int{1, 2}); return t2 }(),
		func() *tensor.TensorNumeric[int] { t2, _ := tensor.New[int]([]int{2, 1}, []int{3, 4}); return t2 }(),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// No panic, no trace recorded -- just verifying no nil pointer dereference.
}

func TestEngineProxy_StopTracing(t *testing.T) {
	proxy, tracer := newTestProxy()
	ctx := context.Background()

	a, _ := tensor.New[int]([]int{2}, []int{1, 2})
	b, _ := tensor.New[int]([]int{2}, []int{3, 4})

	_, _ = proxy.Add(ctx, a, b)
	if len(tracer.events) != 1 {
		t.Fatalf("expected 1 event before stop, got %d", len(tracer.events))
	}

	proxy.StopTracing()
	_, _ = proxy.Sub(ctx, a, b)
	if len(tracer.events) != 1 {
		t.Errorf("expected no new events after StopTracing, got %d", len(tracer.events))
	}
}

func TestEngineProxy_SplitWithRealTracer(t *testing.T) {
	cpu := NewCPUEngine[int](numeric.IntOps{})
	proxy := NewEngineProxy[int](cpu)
	tr := NewTracer[int](nil)
	proxy.StartTracing(tr)
	ctx := context.Background()

	tests := []struct {
		name      string
		inputSize int
		numSplits int
	}{
		{"split into 2", 4, 2},
		{"split into 3", 6, 3},
		{"split into 4", 8, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			localTr := NewTracer[int](nil)
			localProxy := NewEngineProxy[int](cpu)
			localProxy.StartTracing(localTr)

			data := make([]int, tt.inputSize)
			for i := range data {
				data[i] = i + 1
			}
			a, _ := tensor.New[int]([]int{tt.inputSize}, data)
			_, err := localProxy.Split(ctx, a, tt.numSplits, 0)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			ops := localTr.TracedOps()
			if len(ops) != 1 {
				t.Fatalf("expected 1 traced op, got %d", len(ops))
			}
			op := ops[0]
			if op.OpName != "Split" {
				t.Errorf("op name = %q, want Split", op.OpName)
			}
			if len(op.OutputIDs) != tt.numSplits {
				t.Fatalf("expected %d OutputIDs, got %d", tt.numSplits, len(op.OutputIDs))
			}
			// All output slots must be unique and different from input.
			seen := map[int]bool{op.InputIDs[0]: true}
			for _, id := range op.OutputIDs {
				if seen[id] {
					t.Errorf("duplicate or input-matching slot %d", id)
				}
				seen[id] = true
			}
		})
	}
}

func TestEngineProxy_ConcatWithRealTracer(t *testing.T) {
	cpu := NewCPUEngine[int](numeric.IntOps{})
	proxy := NewEngineProxy[int](cpu)
	tr := NewTracer[int](nil)
	proxy.StartTracing(tr)
	ctx := context.Background()

	a, _ := tensor.New[int]([]int{2}, []int{1, 2})
	b, _ := tensor.New[int]([]int{3}, []int{3, 4, 5})
	c, _ := tensor.New[int]([]int{2}, []int{6, 7})

	result, err := proxy.Concat(ctx, []*tensor.TensorNumeric[int]{a, b, c}, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ops := tr.TracedOps()
	if len(ops) != 1 {
		t.Fatalf("expected 1 traced op, got %d", len(ops))
	}
	op := ops[0]
	if op.OpName != "Concat" {
		t.Errorf("op name = %q, want Concat", op.OpName)
	}
	if len(op.InputIDs) != 3 {
		t.Fatalf("expected 3 input IDs, got %d", len(op.InputIDs))
	}
	// All input slots must be unique.
	seen := make(map[int]bool)
	for _, id := range op.InputIDs {
		if seen[id] {
			t.Errorf("duplicate input slot %d", id)
		}
		seen[id] = true
	}
	// Output slot must differ from all inputs.
	if seen[op.OutputID] {
		t.Errorf("output slot %d should differ from input slots", op.OutputID)
	}
	// Verify result data.
	expected := []int{1, 2, 3, 4, 5, 6, 7}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("Concat result = %v, want %v", result.Data(), expected)
	}
}

func TestEngineProxy_SplitNotTracedWhenOff(t *testing.T) {
	cpu := NewCPUEngine[int](numeric.IntOps{})
	proxy := NewEngineProxy[int](cpu)
	ctx := context.Background()

	a, _ := tensor.New[int]([]int{4}, []int{1, 2, 3, 4})
	_, err := proxy.Split(ctx, a, 2, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// No panic with nil tracer.
}

func TestEngineProxy_ResultsMatchCPUEngine(t *testing.T) {
	cpu := NewCPUEngine[int](numeric.IntOps{})
	proxy := NewEngineProxy[int](cpu)
	tracer := &mockTracer[int]{}
	proxy.StartTracing(tracer)
	ctx := context.Background()

	t.Run("Add", func(t *testing.T) {
		a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
		b, _ := tensor.New[int]([]int{2, 2}, []int{5, 6, 7, 8})
		proxyResult, _ := proxy.Add(ctx, a, b)
		cpuResult, _ := cpu.Add(ctx, a, b)
		if !reflect.DeepEqual(proxyResult.Data(), cpuResult.Data()) {
			t.Errorf("Add mismatch: proxy=%v cpu=%v", proxyResult.Data(), cpuResult.Data())
		}
	})

	t.Run("Softmax", func(t *testing.T) {
		a, _ := tensor.New[int]([]int{4}, []int{1, 2, 3, 4})
		proxyResult, _ := proxy.Softmax(ctx, a, 0)
		cpuResult, _ := cpu.Softmax(ctx, a, 0)
		if !reflect.DeepEqual(proxyResult.Data(), cpuResult.Data()) {
			t.Errorf("Softmax mismatch: proxy=%v cpu=%v", proxyResult.Data(), cpuResult.Data())
		}
	})
}
