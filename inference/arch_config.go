package inference

// RopeScalingConfig holds configuration for RoPE scaling methods (e.g., YaRN).
type RopeScalingConfig struct {
	Type                            string  `json:"type"`
	Factor                          float64 `json:"factor"`
	OriginalMaxPositionEmbeddings   int     `json:"original_max_position_embeddings"`
}

// ConfigParser parses a raw JSON map (from config.json) into ModelMetadata.
type ConfigParser func(raw map[string]interface{}) (*ModelMetadata, error)

// ArchConfigRegistry maps model_type strings to config parsers.
type ArchConfigRegistry struct {
	parsers map[string]ConfigParser
}

// newArchConfigRegistry creates an empty registry.
func newArchConfigRegistry() *ArchConfigRegistry {
	return &ArchConfigRegistry{
		parsers: make(map[string]ConfigParser),
	}
}

// Register adds a parser for the given model type.
func (r *ArchConfigRegistry) Register(modelType string, parser ConfigParser) {
	r.parsers[modelType] = parser
}

// Parse dispatches to the registered parser for the model_type in raw,
// or falls back to generic field extraction for unknown types.
func (r *ArchConfigRegistry) Parse(raw map[string]interface{}) (*ModelMetadata, error) {
	modelType, _ := raw["model_type"].(string)

	if parser, ok := r.parsers[modelType]; ok {
		return parser(raw)
	}
	return parseFallbackConfig(raw)
}

// DefaultArchConfigRegistry returns a registry with all built-in parsers registered.
func DefaultArchConfigRegistry() *ArchConfigRegistry {
	r := newArchConfigRegistry()
	r.Register("gemma", parseGemmaConfig)
	r.Register("gemma2", parseGemmaConfig)
	r.Register("gemma3", parseGemmaConfig)
	return r
}

// parseGemmaConfig parses Gemma-family config.json fields.
func parseGemmaConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		RopeScaling:           getRopeScaling(raw),
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 10000 // Gemma default
	}
	return meta, nil
}

// parseFallbackConfig extracts common fields using the most widespread
// HuggingFace naming conventions. Used for unknown model_type values.
func parseFallbackConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		TieWordEmbeddings:     getBool(raw, "tie_word_embeddings"),
		SlidingWindow:         getInt(raw, "sliding_window"),
		AttentionBias:         getBool(raw, "attention_bias"),
		RopeScaling:           getRopeScaling(raw),
	}

	// Try common alternative field names for num_layers.
	meta.NumLayers = getInt(raw, "num_hidden_layers")
	if meta.NumLayers == 0 {
		meta.NumLayers = getInt(raw, "num_layers")
	}

	return meta, nil
}

// --- Helper functions for extracting typed values from raw JSON maps ---

func getString(raw map[string]interface{}, key string) string {
	v, _ := raw[key].(string)
	return v
}

func getInt(raw map[string]interface{}, key string) int {
	switch v := raw[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return 0
	}
}

func getFloat(raw map[string]interface{}, key string) float64 {
	switch v := raw[key].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	default:
		return 0
	}
}

func getBool(raw map[string]interface{}, key string) bool {
	v, _ := raw[key].(bool)
	return v
}

func getRopeScaling(raw map[string]interface{}) *RopeScalingConfig {
	m, ok := raw["rope_scaling"].(map[string]interface{})
	if !ok {
		return nil
	}
	return &RopeScalingConfig{
		Type:                          getString(m, "type"),
		Factor:                        getFloat(m, "factor"),
		OriginalMaxPositionEmbeddings: getInt(m, "original_max_position_embeddings"),
	}
}
