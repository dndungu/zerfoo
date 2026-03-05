package model

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zmf"
	"google.golang.org/protobuf/proto"
)

func TestLoadZMFMmap(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.zmf")

	sample := &zmf.Model{
		ZmfVersion: "1.0.0",
		Metadata: &zmf.Metadata{
			ProducerName:    "mmap_test",
			ProducerVersion: "0.1.0",
			OpsetVersion:    1,
		},
		Graph: &zmf.Graph{
			Nodes: []*zmf.Node{
				{Name: "n1", OpType: "Add", Inputs: []string{"a", "b"}},
			},
		},
	}

	data, err := proto.Marshal(sample)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	model, r, err := LoadZMFMmap(path)
	if err != nil {
		t.Fatalf("LoadZMFMmap: %v", err)
	}
	defer func() { _ = r.Close() }()

	if model.ZmfVersion != "1.0.0" {
		t.Errorf("version = %q, want 1.0.0", model.ZmfVersion)
	}
	if model.Metadata.ProducerName != "mmap_test" {
		t.Errorf("producer = %q, want mmap_test", model.Metadata.ProducerName)
	}
	if len(model.Graph.Nodes) != 1 || model.Graph.Nodes[0].Name != "n1" {
		t.Error("graph node mismatch")
	}
}

func TestLoadZMFMmap_MissingFile(t *testing.T) {
	_, _, err := LoadZMFMmap("/nonexistent/model.zmf")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadZMFMmap_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.zmf")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	_, _, err := LoadZMFMmap(path)
	if err == nil {
		t.Fatal("expected error for empty file")
	}
}

func TestLoadZMFMmap_InvalidData(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.zmf")
	if err := os.WriteFile(path, []byte{0xFF, 0xFF, 0xFF}, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	_, _, err := LoadZMFMmap(path)
	if err == nil {
		t.Fatal("expected error for invalid protobuf data")
	}
}

func TestLoadZMFMmap_MatchesLoadZMF(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "parity.zmf")

	embedData := make([]byte, 4*8*4) // 4 vocab * 8 dim * 4 bytes
	sample := &zmf.Model{
		ZmfVersion: "0.5.0",
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Dtype: zmf.Tensor_FLOAT32, Shape: []int64{1, 8}},
			},
			Parameters: map[string]*zmf.Tensor{
				"embed_tokens": {Dtype: zmf.Tensor_FLOAT32, Shape: []int64{4, 8}, Data: embedData},
			},
		},
	}

	data, err := proto.Marshal(sample)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	// Load with both methods.
	ref, err := LoadZMF(path)
	if err != nil {
		t.Fatalf("LoadZMF: %v", err)
	}

	mmap, r, err := LoadZMFMmap(path)
	if err != nil {
		t.Fatalf("LoadZMFMmap: %v", err)
	}
	defer func() { _ = r.Close() }()

	// Compare key fields.
	if ref.ZmfVersion != mmap.ZmfVersion {
		t.Errorf("version mismatch: %q vs %q", ref.ZmfVersion, mmap.ZmfVersion)
	}
	if len(ref.Graph.Parameters) != len(mmap.Graph.Parameters) {
		t.Errorf("param count: %d vs %d", len(ref.Graph.Parameters), len(mmap.Graph.Parameters))
	}
}
