package serve

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBatchScheduler_BatchesRequests(t *testing.T) {
	var batchCount atomic.Int32

	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 4,
		BatchTimeout: 50 * time.Millisecond,
		Handler: func(_ context.Context, reqs []BatchRequest) []BatchResult {
			batchCount.Add(1)
			results := make([]BatchResult, len(reqs))
			for i, r := range reqs {
				results[i] = BatchResult{Value: "echo:" + r.Prompt}
			}
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	// Submit 4 requests concurrently — should be batched into 1 call.
	var wg sync.WaitGroup
	results := make([]BatchResult, 4)
	for i := range 4 {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			r, err := sched.Submit(context.Background(), BatchRequest{Prompt: "hello"})
			if err != nil {
				t.Errorf("submit %d: %v", idx, err)
				return
			}
			results[idx] = r
		}(i)
	}

	wg.Wait()

	for i, r := range results {
		if r.Value != "echo:hello" {
			t.Errorf("result[%d] = %q, want %q", i, r.Value, "echo:hello")
		}
	}

	if got := batchCount.Load(); got != 1 {
		t.Errorf("batch count = %d, want 1", got)
	}
}

func TestBatchScheduler_TimeoutFires(t *testing.T) {
	var batchSizes []int
	var mu sync.Mutex

	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 8,
		BatchTimeout: 20 * time.Millisecond,
		Handler: func(_ context.Context, reqs []BatchRequest) []BatchResult {
			mu.Lock()
			batchSizes = append(batchSizes, len(reqs))
			mu.Unlock()
			results := make([]BatchResult, len(reqs))
			for i := range reqs {
				results[i] = BatchResult{Value: "ok"}
			}
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	// Submit 2 requests (below max batch size). Timeout should fire.
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = sched.Submit(context.Background(), BatchRequest{Prompt: "test"})
		}()
	}

	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if len(batchSizes) != 1 || batchSizes[0] != 2 {
		t.Errorf("batch sizes = %v, want [2]", batchSizes)
	}
}

func TestBatchScheduler_MaxBatchSizeEnforced(t *testing.T) {
	var batchSizes []int
	var mu sync.Mutex

	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 2,
		BatchTimeout: 100 * time.Millisecond,
		Handler: func(_ context.Context, reqs []BatchRequest) []BatchResult {
			mu.Lock()
			batchSizes = append(batchSizes, len(reqs))
			mu.Unlock()
			results := make([]BatchResult, len(reqs))
			for i := range reqs {
				results[i] = BatchResult{Value: "ok"}
			}
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	// Submit 4 requests with max batch size 2 → should get 2 batches.
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = sched.Submit(context.Background(), BatchRequest{Prompt: "test"})
		}()
	}

	wg.Wait()

	mu.Lock()
	defer mu.Unlock()

	total := 0
	for _, s := range batchSizes {
		if s > 2 {
			t.Errorf("batch size %d exceeds max 2", s)
		}
		total += s
	}
	if total != 4 {
		t.Errorf("total requests processed = %d, want 4", total)
	}
}

func TestBatchScheduler_ContextCancellation(t *testing.T) {
	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 8,
		BatchTimeout: time.Second,
		Handler: func(_ context.Context, reqs []BatchRequest) []BatchResult {
			results := make([]BatchResult, len(reqs))
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err := sched.Submit(ctx, BatchRequest{Prompt: "test"})
	if err == nil {
		t.Error("expected error from canceled context")
	}
}
