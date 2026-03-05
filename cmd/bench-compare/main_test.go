package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseBenchmarks(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bench.txt")

	content := `goos: darwin
goarch: arm64
pkg: github.com/zerfoo/zerfoo/compute
BenchmarkMatMul-10       1000       500000 ns/op       1024 B/op       4 allocs/op
BenchmarkAdd-10         10000        15000 ns/op        256 B/op       2 allocs/op
BenchmarkSoftmax-10      5000        80000 ns/op        512 B/op       3 allocs/op
PASS
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseBenchmarks(path)
	if err != nil {
		t.Fatalf("parseBenchmarks: %v", err)
	}

	tests := []struct {
		name    string
		wantNS  float64
		epsilon float64
	}{
		{"BenchmarkMatMul-10", 500000, 1},
		{"BenchmarkAdd-10", 15000, 1},
		{"BenchmarkSoftmax-10", 80000, 1},
	}

	for _, tt := range tests {
		got, ok := results[tt.name]
		if !ok {
			t.Errorf("expected benchmark %s in results", tt.name)
			continue
		}
		if !approxEqual(got, tt.wantNS, tt.epsilon) {
			t.Errorf("%s = %.0f ns/op, want %.0f", tt.name, got, tt.wantNS)
		}
	}
}

func TestParseBenchmarks_MultipleRuns(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "multi.txt")

	content := `BenchmarkMatMul-10       1000       500000 ns/op
BenchmarkMatMul-10       1000       510000 ns/op
BenchmarkMatMul-10       1000       490000 ns/op
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseBenchmarks(path)
	if err != nil {
		t.Fatalf("parseBenchmarks: %v", err)
	}

	// Median of [490000, 500000, 510000] = 500000
	got := results["BenchmarkMatMul-10"]
	if !approxEqual(got, 500000, 1) {
		t.Errorf("median = %.0f, want 500000", got)
	}
}

func TestParseBenchmarks_FileNotFound(t *testing.T) {
	_, err := parseBenchmarks("/nonexistent/bench.txt")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestParseBenchmarks_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(path, []byte(""), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseBenchmarks(path)
	if err != nil {
		t.Fatalf("parseBenchmarks: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected empty results, got %d", len(results))
	}
}

func TestRun(t *testing.T) {
	dir := t.TempDir()

	baseContent := `BenchmarkAdd-10    10000    15000 ns/op
BenchmarkMul-10    10000    20000 ns/op
BenchmarkDiv-10    10000    25000 ns/op
`
	basePath := filepath.Join(dir, "baseline.txt")
	if err := os.WriteFile(basePath, []byte(baseContent), 0o600); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name        string
		baseline    string
		current     string
		curContent  string
		threshold   float64
		wantErr     bool
		errContains string
	}{
		{
			name:        "missing current flag",
			baseline:    basePath,
			current:     "",
			threshold:   10,
			wantErr:     true,
			errContains: "-current flag is required",
		},
		{
			name:        "missing baseline file",
			baseline:    "/nonexistent/baseline.txt",
			current:     basePath,
			threshold:   10,
			wantErr:     true,
			errContains: "error reading baseline",
		},
		{
			name:        "missing current file",
			baseline:    basePath,
			current:     "/nonexistent/current.txt",
			threshold:   10,
			wantErr:     true,
			errContains: "error reading current",
		},
		{
			name:     "no regression",
			baseline: basePath,
			curContent: `BenchmarkAdd-10    10000    15000 ns/op
BenchmarkMul-10    10000    20000 ns/op
`,
			threshold: 10,
			wantErr:   false,
		},
		{
			name:     "regression detected",
			baseline: basePath,
			curContent: `BenchmarkAdd-10    10000    30000 ns/op
`,
			threshold:   10,
			wantErr:     true,
			errContains: "regression detected",
		},
		{
			name:     "improvement",
			baseline: basePath,
			curContent: `BenchmarkAdd-10    10000    5000 ns/op
`,
			threshold: 10,
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			curPath := tt.current
			if tt.curContent != "" {
				curPath = filepath.Join(dir, tt.name+".txt")
				if err := os.WriteFile(curPath, []byte(tt.curContent), 0o600); err != nil {
					t.Fatal(err)
				}
			}

			var buf bytes.Buffer
			runErr := run(tt.baseline, curPath, tt.threshold, &buf)
			if tt.wantErr && runErr == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantErr && runErr != nil {
				t.Errorf("unexpected error: %v", runErr)
			}
			if tt.errContains != "" && runErr != nil {
				if !strings.Contains(runErr.Error(), tt.errContains) {
					t.Errorf("error %q does not contain %q", runErr.Error(), tt.errContains)
				}
			}
		})
	}
}

func TestMedian(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   float64
	}{
		{"empty", nil, 0},
		{"single", []float64{42}, 42},
		{"odd", []float64{3, 1, 2}, 2},
		{"even", []float64{4, 1, 3, 2}, 2.5},
	}

	for _, tt := range tests {
		got := median(tt.values)
		if !approxEqual(got, tt.want, 0.01) {
			t.Errorf("median(%s) = %f, want %f", tt.name, got, tt.want)
		}
	}
}
