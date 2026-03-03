package inference

import (
	"testing"
)

func TestArchConfigRegistry_RegisterAndParse(t *testing.T) {
	reg := newArchConfigRegistry()

	called := false
	reg.Register("test_arch", func(raw map[string]interface{}) (*ModelMetadata, error) {
		called = true
		return &ModelMetadata{Architecture: "test_arch", VocabSize: 100}, nil
	})

	raw := map[string]interface{}{
		"model_type": "test_arch",
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if !called {
		t.Error("expected parser to be called")
	}
	if meta.Architecture != "test_arch" {
		t.Errorf("Architecture = %q, want %q", meta.Architecture, "test_arch")
	}
	if meta.VocabSize != 100 {
		t.Errorf("VocabSize = %d, want 100", meta.VocabSize)
	}
}

func TestArchConfigRegistry_FallbackForUnknown(t *testing.T) {
	reg := newArchConfigRegistry()

	raw := map[string]interface{}{
		"model_type":               "unknown_arch",
		"vocab_size":               float64(32000),
		"hidden_size":              float64(4096),
		"num_hidden_layers":        float64(32),
		"max_position_embeddings":  float64(8192),
		"eos_token_id":             float64(2),
		"bos_token_id":             float64(1),
		"intermediate_size":        float64(11008),
		"num_key_value_heads":      float64(8),
		"num_attention_heads":      float64(32),
		"rope_theta":               float64(500000),
		"tie_word_embeddings":      true,
		"sliding_window":           float64(4096),
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.Architecture != "unknown_arch" {
		t.Errorf("Architecture = %q, want %q", meta.Architecture, "unknown_arch")
	}
	if meta.VocabSize != 32000 {
		t.Errorf("VocabSize = %d, want 32000", meta.VocabSize)
	}
	if meta.HiddenSize != 4096 {
		t.Errorf("HiddenSize = %d, want 4096", meta.HiddenSize)
	}
	if meta.NumLayers != 32 {
		t.Errorf("NumLayers = %d, want 32", meta.NumLayers)
	}
	if meta.IntermediateSize != 11008 {
		t.Errorf("IntermediateSize = %d, want 11008", meta.IntermediateSize)
	}
	if meta.NumKeyValueHeads != 8 {
		t.Errorf("NumKeyValueHeads = %d, want 8", meta.NumKeyValueHeads)
	}
	if meta.RopeTheta != 500000 {
		t.Errorf("RopeTheta = %f, want 500000", meta.RopeTheta)
	}
	if !meta.TieWordEmbeddings {
		t.Error("TieWordEmbeddings = false, want true")
	}
	if meta.SlidingWindow != 4096 {
		t.Errorf("SlidingWindow = %d, want 4096", meta.SlidingWindow)
	}
}

func TestArchConfigRegistry_FallbackMissingFields(t *testing.T) {
	reg := newArchConfigRegistry()

	raw := map[string]interface{}{
		"model_type": "minimal",
		"vocab_size": float64(1000),
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.VocabSize != 1000 {
		t.Errorf("VocabSize = %d, want 1000", meta.VocabSize)
	}
	// Missing fields should be zero-valued.
	if meta.HiddenSize != 0 {
		t.Errorf("HiddenSize = %d, want 0", meta.HiddenSize)
	}
	if meta.NumLayers != 0 {
		t.Errorf("NumLayers = %d, want 0", meta.NumLayers)
	}
}

func TestArchConfigRegistry_NoModelType(t *testing.T) {
	reg := newArchConfigRegistry()

	raw := map[string]interface{}{
		"vocab_size": float64(1000),
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.Architecture != "" {
		t.Errorf("Architecture = %q, want empty", meta.Architecture)
	}
}

func TestGemmaConfigParser(t *testing.T) {
	tests := []struct {
		name string
		raw  map[string]interface{}
		want ModelMetadata
	}{
		{
			name: "gemma2 full config",
			raw: map[string]interface{}{
				"model_type":               "gemma2",
				"vocab_size":               float64(256000),
				"hidden_size":              float64(2304),
				"num_hidden_layers":        float64(26),
				"num_attention_heads":      float64(8),
				"num_key_value_heads":      float64(4),
				"intermediate_size":        float64(9216),
				"max_position_embeddings":  float64(8192),
				"eos_token_id":             float64(1),
				"bos_token_id":             float64(2),
			},
			want: ModelMetadata{
				Architecture:          "gemma2",
				VocabSize:             256000,
				HiddenSize:            2304,
				NumLayers:             26,
				NumQueryHeads:         8,
				NumKeyValueHeads:      4,
				IntermediateSize:      9216,
				MaxPositionEmbeddings: 8192,
				EOSTokenID:            1,
				BOSTokenID:            2,
				RopeTheta:             10000,
			},
		},
		{
			name: "gemma3 with rope_theta",
			raw: map[string]interface{}{
				"model_type":               "gemma3",
				"vocab_size":               float64(262144),
				"hidden_size":              float64(2048),
				"num_hidden_layers":        float64(26),
				"num_attention_heads":      float64(8),
				"num_key_value_heads":      float64(4),
				"intermediate_size":        float64(16384),
				"max_position_embeddings":  float64(8192),
				"eos_token_id":             float64(1),
				"bos_token_id":             float64(2),
				"rope_theta":               float64(10000),
			},
			want: ModelMetadata{
				Architecture:          "gemma3",
				VocabSize:             262144,
				HiddenSize:            2048,
				NumLayers:             26,
				NumQueryHeads:         8,
				NumKeyValueHeads:      4,
				IntermediateSize:      16384,
				MaxPositionEmbeddings: 8192,
				EOSTokenID:            1,
				BOSTokenID:            2,
				RopeTheta:             10000,
			},
		},
		{
			name: "gemma2 minimal",
			raw: map[string]interface{}{
				"model_type":          "gemma2",
				"vocab_size":          float64(256000),
				"num_hidden_layers":   float64(18),
				"num_attention_heads": float64(8),
			},
			want: ModelMetadata{
				Architecture:  "gemma2",
				VocabSize:     256000,
				NumLayers:     18,
				NumQueryHeads: 8,
				RopeTheta:     10000,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseGemmaConfig(tc.raw)
			if err != nil {
				t.Fatalf("parseGemmaConfig error: %v", err)
			}
			assertMetadataEqual(t, tc.want, *got)
		})
	}
}

func TestDefaultArchConfigRegistry_GemmaRegistered(t *testing.T) {
	reg := DefaultArchConfigRegistry()

	for _, modelType := range []string{"gemma", "gemma2", "gemma3"} {
		t.Run(modelType, func(t *testing.T) {
			raw := map[string]interface{}{
				"model_type":          modelType,
				"vocab_size":          float64(256000),
				"num_hidden_layers":   float64(26),
				"num_attention_heads": float64(8),
			}
			meta, err := reg.Parse(raw)
			if err != nil {
				t.Fatalf("Parse error: %v", err)
			}
			if meta.Architecture != modelType {
				t.Errorf("Architecture = %q, want %q", meta.Architecture, modelType)
			}
			if meta.VocabSize != 256000 {
				t.Errorf("VocabSize = %d, want 256000", meta.VocabSize)
			}
		})
	}
}

func TestRopeScalingConfig_FromRaw(t *testing.T) {
	raw := map[string]interface{}{
		"model_type": "unknown",
		"rope_scaling": map[string]interface{}{
			"type":                               "yarn",
			"factor":                             float64(4.0),
			"original_max_position_embeddings":   float64(32768),
		},
	}
	reg := newArchConfigRegistry()
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.RopeScaling == nil {
		t.Fatal("RopeScaling should not be nil")
	}
	if meta.RopeScaling.Type != "yarn" {
		t.Errorf("RopeScaling.Type = %q, want %q", meta.RopeScaling.Type, "yarn")
	}
	if meta.RopeScaling.Factor != 4.0 {
		t.Errorf("RopeScaling.Factor = %f, want 4.0", meta.RopeScaling.Factor)
	}
	if meta.RopeScaling.OriginalMaxPositionEmbeddings != 32768 {
		t.Errorf("RopeScaling.OriginalMaxPositionEmbeddings = %d, want 32768",
			meta.RopeScaling.OriginalMaxPositionEmbeddings)
	}
}

// assertMetadataEqual compares key fields of two ModelMetadata values.
func assertMetadataEqual(t *testing.T, want, got ModelMetadata) {
	t.Helper()
	if got.Architecture != want.Architecture {
		t.Errorf("Architecture = %q, want %q", got.Architecture, want.Architecture)
	}
	if got.VocabSize != want.VocabSize {
		t.Errorf("VocabSize = %d, want %d", got.VocabSize, want.VocabSize)
	}
	if got.HiddenSize != want.HiddenSize {
		t.Errorf("HiddenSize = %d, want %d", got.HiddenSize, want.HiddenSize)
	}
	if got.NumLayers != want.NumLayers {
		t.Errorf("NumLayers = %d, want %d", got.NumLayers, want.NumLayers)
	}
	if got.NumQueryHeads != want.NumQueryHeads {
		t.Errorf("NumQueryHeads = %d, want %d", got.NumQueryHeads, want.NumQueryHeads)
	}
	if got.NumKeyValueHeads != want.NumKeyValueHeads {
		t.Errorf("NumKeyValueHeads = %d, want %d", got.NumKeyValueHeads, want.NumKeyValueHeads)
	}
	if got.IntermediateSize != want.IntermediateSize {
		t.Errorf("IntermediateSize = %d, want %d", got.IntermediateSize, want.IntermediateSize)
	}
	if got.MaxPositionEmbeddings != want.MaxPositionEmbeddings {
		t.Errorf("MaxPositionEmbeddings = %d, want %d", got.MaxPositionEmbeddings, want.MaxPositionEmbeddings)
	}
	if got.EOSTokenID != want.EOSTokenID {
		t.Errorf("EOSTokenID = %d, want %d", got.EOSTokenID, want.EOSTokenID)
	}
	if got.BOSTokenID != want.BOSTokenID {
		t.Errorf("BOSTokenID = %d, want %d", got.BOSTokenID, want.BOSTokenID)
	}
	if got.RopeTheta != want.RopeTheta {
		t.Errorf("RopeTheta = %f, want %f", got.RopeTheta, want.RopeTheta)
	}
	if got.TieWordEmbeddings != want.TieWordEmbeddings {
		t.Errorf("TieWordEmbeddings = %v, want %v", got.TieWordEmbeddings, want.TieWordEmbeddings)
	}
	if got.SlidingWindow != want.SlidingWindow {
		t.Errorf("SlidingWindow = %d, want %d", got.SlidingWindow, want.SlidingWindow)
	}
	if got.AttentionBias != want.AttentionBias {
		t.Errorf("AttentionBias = %v, want %v", got.AttentionBias, want.AttentionBias)
	}
}
