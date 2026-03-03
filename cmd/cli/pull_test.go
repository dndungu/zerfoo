package cli

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/registry"
)

// mockPullRegistry is a test double for registry.ModelRegistry.
type mockPullRegistry struct {
	models map[string]*registry.ModelInfo
	pulled bool
}

func (r *mockPullRegistry) Get(id string) (*registry.ModelInfo, bool) {
	info, ok := r.models[id]
	return info, ok
}

func (r *mockPullRegistry) Pull(_ context.Context, id string) (*registry.ModelInfo, error) {
	r.pulled = true
	if info, ok := r.models[id]; ok {
		return info, nil
	}
	return &registry.ModelInfo{ID: id, Path: "/cache/" + id, Size: 1024}, nil
}

func (r *mockPullRegistry) List() []registry.ModelInfo { return nil }
func (r *mockPullRegistry) Delete(_ string) error      { return nil }

func TestPullCommand_Name(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if cmd.Name() != "pull" {
		t.Errorf("Name() = %q, want %q", cmd.Name(), "pull")
	}
}

func TestPullCommand_Description(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if cmd.Description() == "" {
		t.Error("Description() should not be empty")
	}
}

func TestPullCommand_Usage(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if !strings.Contains(cmd.Usage(), "pull") {
		t.Error("Usage() should contain 'pull'")
	}
}

func TestPullCommand_Examples(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if len(cmd.Examples()) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestPullCommand_MissingModelID(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), nil)
	if err == nil {
		t.Error("expected error for missing model ID")
	}
	if !strings.Contains(err.Error(), "model ID is required") {
		t.Errorf("error = %q, want 'model ID is required'", err.Error())
	}
}

func TestPullCommand_AlreadyCached(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{
		models: map[string]*registry.ModelInfo{
			"test-model": {ID: "test-model", Path: "/cache/test-model"},
		},
	}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"test-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(buf.String(), "Already up to date") {
		t.Errorf("output = %q, want 'Already up to date'", buf.String())
	}
}

func TestPullCommand_PullsNewModel(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"new-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !reg.pulled {
		t.Error("expected Pull to be called")
	}
	if !strings.Contains(buf.String(), "Model saved to") {
		t.Errorf("output = %q, want 'Model saved to'", buf.String())
	}
}

func TestPullCommand_CacheDirFlag(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"--cache-dir", "/tmp/cache", "test"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
}

func TestPullCommand_CacheDirMissingValue(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), []string{"--cache-dir"})
	if err == nil {
		t.Error("expected error for --cache-dir without value")
	}
}

func TestPullCommand_UnexpectedArgument(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), []string{"model1", "model2"})
	if err == nil {
		t.Error("expected error for extra argument")
	}
}

func TestPullCommand_NilRegistry(t *testing.T) {
	// When no registry is provided, it creates a default one.
	var buf bytes.Buffer
	cmd := NewPullCommand(nil, &buf)
	// This will fail to pull (no real model) but exercises the nil-registry code path.
	err := cmd.Run(context.Background(), []string{"nonexistent"})
	if err == nil {
		t.Error("expected error")
	}
}

// Ensure *PullCommand satisfies Command interface at compile time.
func TestPullCommand_Interface(t *testing.T) {
	var _ Command = (*PullCommand)(nil)
	_ = os.Stderr // use os to avoid unused import
}
