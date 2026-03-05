//go:build unix

package model

import (
	"fmt"

	"github.com/zerfoo/zmf"
	"google.golang.org/protobuf/proto"
)

// LoadZMFMmap memory-maps a ZMF file and unmarshals the protobuf directly
// from the mmap byte slice, avoiding the heap allocation of os.ReadFile.
// The returned MmapReader must be kept open for the lifetime of the model
// so tensor data backed by mmap pages remains accessible.
func LoadZMFMmap(filePath string) (*zmf.Model, *MmapReader, error) {
	r, err := NewMmapReader(filePath)
	if err != nil {
		return nil, nil, fmt.Errorf("mmap load %q: %w", filePath, err)
	}

	data := r.Bytes()
	if len(data) == 0 {
		_ = r.Close()
		return nil, nil, fmt.Errorf("mmap load %q: file is empty", filePath)
	}

	model := &zmf.Model{}
	if err := proto.Unmarshal(data, model); err != nil {
		_ = r.Close()
		return nil, nil, fmt.Errorf("mmap unmarshal %q: %w", filePath, err)
	}

	return model, r, nil
}
