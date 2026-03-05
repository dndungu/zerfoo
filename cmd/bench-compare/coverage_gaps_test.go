package main

import (
	"os"
	"path/filepath"
	"testing"
)

// ---------------------------------------------------------------------------
// parseBenchmarks: zero ns/op baseline entry (line 60-61)
// ---------------------------------------------------------------------------

func TestRun_ZeroBaselineNsOp(t *testing.T) {
	dir := t.TempDir()

	basePath := filepath.Join(dir, "baseline.txt")
	if err := os.WriteFile(basePath, []byte("BenchmarkZero-10    10000    0 ns/op\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	curPath := filepath.Join(dir, "current.txt")
	if err := os.WriteFile(curPath, []byte("BenchmarkZero-10    10000    500 ns/op\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	var buf = new(mockWriter)
	err := run(basePath, curPath, 10.0, buf)
	if err != nil {
		t.Errorf("expected no error for zero baseline, got: %v", err)
	}
}

type mockWriter struct{}

func (mockWriter) Write(p []byte) (int, error) { return len(p), nil }

// ---------------------------------------------------------------------------
// parseBenchmarks: malformed line with fewer than 3 fields (line 105-106)
// ---------------------------------------------------------------------------

func TestParseBenchmarks_MalformedLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "malformed.txt")

	// Mix of malformed and valid lines.
	content := "BenchmarkShort-10\nBenchmarkOK-10    10000    42000 ns/op\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseBenchmarks(path)
	if err != nil {
		t.Fatalf("parseBenchmarks: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("expected 1 result (malformed skipped), got %d", len(results))
	}
	if v, ok := results["BenchmarkOK-10"]; !ok || !approxEqual(v, 42000, 1) {
		t.Errorf("BenchmarkOK-10 = %v, want 42000", v)
	}
}
