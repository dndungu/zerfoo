package cli

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/shutdown"
)

func TestServeCommand_Name(t *testing.T) {
	cmd := NewServeCommand(nil, nil)
	if cmd.Name() != "serve" {
		t.Errorf("Name() = %q, want %q", cmd.Name(), "serve")
	}
}

func TestServeCommand_Description(t *testing.T) {
	cmd := NewServeCommand(nil, nil)
	if cmd.Description() == "" {
		t.Error("Description() should not be empty")
	}
}

func TestServeCommand_Usage(t *testing.T) {
	cmd := NewServeCommand(nil, nil)
	if !strings.Contains(cmd.Usage(), "serve") {
		t.Error("Usage() should contain 'serve'")
	}
}

func TestServeCommand_Examples(t *testing.T) {
	cmd := NewServeCommand(nil, nil)
	if len(cmd.Examples()) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestServeCommand_MissingModelID(t *testing.T) {
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	err := cmd.Run(context.Background(), nil)
	if err == nil {
		t.Error("expected error for missing model ID")
	}
}

func TestServeCommand_LoadError(t *testing.T) {
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return nil, errors.New("load failed")
	}
	err := cmd.Run(context.Background(), []string{"test-model"})
	if err == nil {
		t.Error("expected error from load")
	}
}

func TestServeCommand_FlagParsing(t *testing.T) {
	tests := []struct {
		name string
		args []string
		err  string
	}{
		{"port missing value", []string{"--port"}, "--port requires a value"},
		{"cache-dir missing value", []string{"--cache-dir"}, "--cache-dir requires a value"},
		{"unexpected arg", []string{"model1", "model2"}, "unexpected argument"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := NewServeCommand(nil, &out)
			cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
				return nil, errors.New("should not be called")
			}
			err := cmd.Run(context.Background(), tc.args)
			if err == nil {
				t.Error("expected error")
			}
			if !strings.Contains(err.Error(), tc.err) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.err)
			}
		})
	}
}

func TestServeCommand_WithCoordinator(t *testing.T) {
	// Verify the command registers with the coordinator.
	coord := shutdown.New()
	var out bytes.Buffer
	cmd := NewServeCommand(coord, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return nil, errors.New("load failed")
	}
	// Fails at load, but exercises the coordinator path.
	_ = cmd.Run(context.Background(), []string{"test-model"})
}

func TestServeCommand_StartsAndStopsOnCancel(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	coord := shutdown.New()
	cmd := NewServeCommand(coord, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Run(ctx, []string{"--port", "0", "test-model"})
	}()

	// Give server a moment to start, then cancel.
	cancel()

	err := <-errCh
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(out.String(), "Serving test-model") {
		t.Errorf("output = %q, want 'Serving test-model'", out.String())
	}
}

func TestServeCommand_CustomPort(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Run(ctx, []string{"--port", "0", "test-model"})
	}()

	cancel()
	err := <-errCh
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
}

func TestShutdownAdapter_Close(t *testing.T) {
	// Test that shutdownAdapter properly delegates to http.Server.Shutdown.
	srv := &http.Server{}
	adapter := shutdownAdapter{srv}
	if err := adapter.Close(context.Background()); err != nil {
		t.Errorf("Close error: %v", err)
	}
}

func TestServeCommand_Interface(t *testing.T) {
	var _ Command = (*ServeCommand)(nil)
}
