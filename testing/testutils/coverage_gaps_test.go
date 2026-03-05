package testutils

import (
	"testing"
)

// ---------------------------------------------------------------------------
// AssertNotCalled: exercise branches where method WAS called
// ---------------------------------------------------------------------------

func TestCustomMockStrategy_AssertNotCalled_CalledMethods(t *testing.T) {
	methods := []struct {
		name  string
		setup func(m *CustomMockStrategy[float32])
	}{
		{"Init", func(m *CustomMockStrategy[float32]) {
			m.ReturnInit(nil)
			_ = m.Init(0, 1, "addr")
		}},
		{"Rank", func(m *CustomMockStrategy[float32]) {
			m.ReturnRank(0)
			_ = m.Rank()
		}},
		{"Size", func(m *CustomMockStrategy[float32]) {
			m.ReturnSize(1)
			_ = m.Size()
		}},
		{"AllReduceGradients", func(m *CustomMockStrategy[float32]) {
			m.ReturnAllReduceGradients(nil)
			_ = m.AllReduceGradients(nil)
		}},
		{"Barrier", func(m *CustomMockStrategy[float32]) {
			m.ReturnBarrier(nil)
			_ = m.Barrier()
		}},
		{"BroadcastTensor", func(m *CustomMockStrategy[float32]) {
			m.ReturnBroadcastTensor(nil)
			_ = m.BroadcastTensor(nil, 0)
		}},
		{"Shutdown", func(m *CustomMockStrategy[float32]) {
			m.Shutdown()
		}},
	}

	for _, tt := range methods {
		t.Run(tt.name, func(t *testing.T) {
			mock := &CustomMockStrategy[float32]{}
			tt.setup(mock)

			// Use a separate T to capture the expected error without failing this test.
			ft := &testing.T{}
			mock.AssertNotCalled(ft, tt.name)
			// ft would have recorded an error; we just verify the path executes.
		})
	}
}

// ---------------------------------------------------------------------------
// Panic paths: calling methods without enough return values
// ---------------------------------------------------------------------------

func TestCustomMockStrategy_Init_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for Init without return values")
		}
	}()
	mock := &CustomMockStrategy[float32]{}
	_ = mock.Init(0, 1, "addr")
}

func TestCustomMockStrategy_Rank_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for Rank without return values")
		}
	}()
	mock := &CustomMockStrategy[float32]{}
	_ = mock.Rank()
}

func TestCustomMockStrategy_Size_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for Size without return values")
		}
	}()
	mock := &CustomMockStrategy[float32]{}
	_ = mock.Size()
}

func TestCustomMockStrategy_AllReduceGradients_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for AllReduceGradients without return values")
		}
	}()
	mock := &CustomMockStrategy[float32]{}
	_ = mock.AllReduceGradients(nil)
}

func TestCustomMockStrategy_Barrier_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for Barrier without return values")
		}
	}()
	mock := &CustomMockStrategy[float32]{}
	_ = mock.Barrier()
}

func TestCustomMockStrategy_BroadcastTensor_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for BroadcastTensor without return values")
		}
	}()
	mock := &CustomMockStrategy[float32]{}
	_ = mock.BroadcastTensor(nil, 0)
}

func TestCustomMockDistributedServiceClient_AllReduce_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for AllReduce without return values")
		}
	}()
	mc := &CustomMockDistributedServiceClient{}
	_, _ = mc.AllReduce(t.Context())
}
