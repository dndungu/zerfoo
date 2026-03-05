// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"fmt"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zmf"
	"google.golang.org/protobuf/proto"
)

// LoadZMF reads a Zerfoo Model Format (.zmf) file from the specified path,
// deserializes it, and returns the parsed Model object.
func LoadZMF(filePath string) (*zmf.Model, error) {
	// Read the entire file into a byte slice.
	//nolint:gosec // Reading a model file from a variable path is expected and validated by caller.
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read ZMF file '%s': %w", filePath, err)
	}

	// Create a new Model message to unmarshal into.
	model := &zmf.Model{}

	// Unmarshal the protobuf data.
	if err := proto.Unmarshal(data, model); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ZMF data: %w", err)
	}

	return model, nil
}

// LoadModelFromZMF loads a ZMF model from a file, builds the computation graph,
// and returns a complete Model object.
func LoadModelFromZMF[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	filePath string,
	buildOpts ...BuildOption,
) (*Model[T], error) {
	zmfModel, err := LoadZMF(filePath)
	if err != nil {
		return nil, err
	}

	g, err := BuildFromZMF(engine, ops, zmfModel, buildOpts...)
	if err != nil {
		return nil, err
	}

	// Search for embed_tokens parameter to create the embedding layer.
	var embedding *embeddings.TokenEmbedding[T]
	if zmfModel.Graph != nil && zmfModel.Graph.Parameters != nil {
		for name, tensorProto := range zmfModel.Graph.Parameters {
			if strings.Contains(name, "embed_tokens") {
				tv, err := DecodeTensor[T](tensorProto)
				if err != nil {
					break
				}
				param, err := graph.NewParameter[T](name, tv, tensor.New[T])
				if err != nil {
					break
				}
				embedding, _ = embeddings.NewTokenEmbeddingFromParam(engine, param)
				break
			}
		}
	}

	model := &Model[T]{
		Embedding:  embedding,
		Graph:      g,
		ZMFVersion: zmfModel.ZmfVersion,
	}

	return model, nil
}
