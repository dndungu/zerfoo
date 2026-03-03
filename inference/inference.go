// Package inference provides a high-level API for loading models and
// generating text with minimal boilerplate.
package inference

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/pkg/tokenizer"
	"github.com/zerfoo/zerfoo/registry"
)

// Model is a loaded model ready for generation.
type Model struct {
	generator *generate.Generator[float32]
	tokenizer tokenizer.Tokenizer
	engine    compute.Engine[float32]
	config    ModelMetadata
	info      *registry.ModelInfo
}

// ModelMetadata holds model configuration loaded from config.json.
type ModelMetadata struct {
	Architecture          string `json:"architecture"`
	VocabSize             int    `json:"vocab_size"`
	HiddenSize            int    `json:"hidden_size"`
	NumLayers             int    `json:"num_layers"`
	MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	EOSTokenID            int    `json:"eos_token_id"`
	BOSTokenID            int    `json:"bos_token_id"`
	ChatTemplate          string `json:"chat_template"`

	// Extended fields for multi-architecture support.
	IntermediateSize  int                `json:"intermediate_size"`
	NumQueryHeads     int                `json:"num_attention_heads"`
	NumKeyValueHeads  int                `json:"num_key_value_heads"`
	RopeTheta         float64            `json:"rope_theta"`
	RopeScaling       *RopeScalingConfig `json:"rope_scaling,omitempty"`
	TieWordEmbeddings bool               `json:"tie_word_embeddings"`
	SlidingWindow     int                `json:"sliding_window"`
	AttentionBias     bool               `json:"attention_bias"`
}

// Option configures model loading.
type Option func(*loadOptions)

type loadOptions struct {
	cacheDir  string
	device    string
	maxSeqLen int
	registry  registry.ModelRegistry
}

// WithCacheDir sets the model cache directory.
func WithCacheDir(dir string) Option {
	return func(o *loadOptions) {
		o.cacheDir = dir
	}
}

// WithDevice sets the compute device ("cpu" or "cuda").
func WithDevice(device string) Option {
	return func(o *loadOptions) {
		o.device = device
	}
}

// WithMaxSeqLen overrides the model's default max sequence length.
func WithMaxSeqLen(n int) Option {
	return func(o *loadOptions) {
		o.maxSeqLen = n
	}
}

// WithRegistry provides a custom model registry.
func WithRegistry(r registry.ModelRegistry) Option {
	return func(o *loadOptions) {
		o.registry = r
	}
}

// Load loads a model by ID, pulling it if not cached.
func Load(modelID string, opts ...Option) (*Model, error) {
	o := &loadOptions{
		device: "cpu",
	}
	for _, opt := range opts {
		opt(o)
	}

	// Get or create registry.
	reg := o.registry
	if reg == nil {
		var err error
		if o.cacheDir != "" {
			reg, err = registry.NewLocalRegistry(o.cacheDir)
		} else {
			reg, err = registry.NewLocalRegistry("")
		}
		if err != nil {
			return nil, fmt.Errorf("create registry: %w", err)
		}
	}

	// Check cache first, pull if needed.
	info, ok := reg.Get(modelID)
	if !ok {
		var err error
		info, err = reg.Pull(context.Background(), modelID)
		if err != nil {
			return nil, fmt.Errorf("pull model %q: %w", modelID, err)
		}
	}

	// Load config.json.
	configPath := filepath.Join(info.Path, "config.json")
	meta, err := loadMetadata(configPath)
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	// Load tokenizer.
	tokPath := filepath.Join(info.Path, "tokenizer.json")
	tok, err := tokenizer.LoadFromJSON(tokPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	// Load ZMF model and build graph.
	zmfPath := filepath.Join(info.Path, "model.zmf")
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	mdl, err := model.LoadModelFromZMF[float32](eng, numeric.Float32Ops{}, zmfPath)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	return assembleModel(mdl.Graph, tok, eng, meta, info, o.maxSeqLen), nil
}

// assembleModel wires together loaded components into a ready-to-use Model.
func assembleModel(
	g *graph.Graph[float32],
	tok tokenizer.Tokenizer,
	eng compute.Engine[float32],
	meta *ModelMetadata,
	info *registry.ModelInfo,
	maxSeqLenOverride int,
) *Model {
	maxSeqLen := meta.MaxPositionEmbeddings
	if maxSeqLenOverride > 0 {
		maxSeqLen = maxSeqLenOverride
	}

	gen := generate.NewGenerator(g, tok, eng, generate.ModelConfig{
		VocabSize:  meta.VocabSize,
		MaxSeqLen:  maxSeqLen,
		EOSTokenID: meta.EOSTokenID,
		BOSTokenID: meta.BOSTokenID,
		NumLayers:  meta.NumLayers,
	})

	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    eng,
		config:    *meta,
		info:      info,
	}
}

// loadMetadata reads and parses config.json.
func loadMetadata(path string) (*ModelMetadata, error) {
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	var meta ModelMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, err
	}
	return &meta, nil
}

// GenerateOption configures a generation call.
type GenerateOption func(*generate.SamplingConfig)

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.Temperature = t
	}
}

// WithTopK sets the top-K sampling parameter.
func WithTopK(k int) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.TopK = k
	}
}

// WithTopP sets the top-P (nucleus) sampling parameter.
func WithTopP(p float64) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.TopP = p
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.MaxNewTokens = n
	}
}

// WithRepetitionPenalty sets the repetition penalty factor.
func WithRepetitionPenalty(p float64) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.RepetitionPenalty = p
	}
}

// WithStopStrings sets strings that stop generation.
func WithStopStrings(ss ...string) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.StopStrings = ss
	}
}

func buildSamplingConfig(opts []GenerateOption) generate.SamplingConfig {
	sc := generate.DefaultSamplingConfig()
	for _, opt := range opts {
		opt(&sc)
	}
	return sc
}

// Generate produces text from a prompt.
func (m *Model) Generate(ctx context.Context, prompt string, opts ...GenerateOption) (string, error) {
	sc := buildSamplingConfig(opts)
	return m.generator.Generate(ctx, prompt, sc)
}

// GenerateStream delivers tokens one at a time via a callback.
func (m *Model) GenerateStream(ctx context.Context, prompt string, handler generate.TokenStream, opts ...GenerateOption) error {
	sc := buildSamplingConfig(opts)
	return m.generator.GenerateStream(ctx, prompt, sc, handler)
}

// Message represents a chat message.
type Message struct {
	Role    string // "system", "user", or "assistant"
	Content string
}

// Response holds the result of a chat completion.
type Response struct {
	Content    string
	TokensUsed int
}

// Chat formats messages using the model's chat template and generates a response.
func (m *Model) Chat(ctx context.Context, messages []Message, opts ...GenerateOption) (Response, error) {
	prompt := m.formatMessages(messages)
	sc := buildSamplingConfig(opts)
	result, err := m.generator.Generate(ctx, prompt, sc)
	if err != nil {
		return Response{}, err
	}

	// Rough token count from the tokenizer.
	ids, _ := m.tokenizer.Encode(prompt + result)
	return Response{
		Content:    result,
		TokensUsed: len(ids),
	}, nil
}

// formatMessages converts messages to the model's chat template format.
func (m *Model) formatMessages(messages []Message) string {
	template := m.config.ChatTemplate
	if template == "" {
		// Default Gemma 3 format.
		template = "gemma"
	}

	var sb strings.Builder
	for _, msg := range messages {
		switch strings.ToLower(template) {
		case "gemma":
			sb.WriteString("<start_of_turn>")
			sb.WriteString(msg.Role)
			sb.WriteString("\n")
			sb.WriteString(msg.Content)
			sb.WriteString("<end_of_turn>\n")
		default:
			// Generic format: Role: Content\n
			sb.WriteString(msg.Role)
			sb.WriteString(": ")
			sb.WriteString(msg.Content)
			sb.WriteString("\n")
		}
	}

	// Add the assistant turn opening.
	switch strings.ToLower(template) {
	case "gemma":
		sb.WriteString("<start_of_turn>model\n")
	default:
		sb.WriteString("assistant: ")
	}

	return sb.String()
}

// Embed returns a float32 embedding vector for the given text.
// It runs the model forward and mean-pools the last layer's hidden states.
func (m *Model) Embed(ctx context.Context, text string) ([]float32, error) {
	ids, err := m.tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("text produced no tokens")
	}

	// Generate one token to get the last hidden state.
	// For embeddings, we use the output of a forward pass and mean-pool.
	result, err := m.generator.Generate(ctx, text, generate.SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("forward: %w", err)
	}

	// Since we don't have direct access to hidden states, we return
	// a hash of the result as a placeholder. Real implementation would
	// need access to intermediate layer outputs.
	_ = result
	return nil, fmt.Errorf("embeddings not yet supported: model does not expose hidden states")
}

// Config returns the model metadata.
func (m *Model) Config() ModelMetadata {
	return m.config
}

// Info returns the registry info for this model.
func (m *Model) Info() *registry.ModelInfo {
	return m.info
}

// NewTestModel constructs a Model from pre-built components.
// Intended for use in external test packages that need a Model
// without going through the full Load pipeline.
func NewTestModel(
	gen *generate.Generator[float32],
	tok tokenizer.Tokenizer,
	eng compute.Engine[float32],
	meta ModelMetadata,
	info *registry.ModelInfo,
) *Model {
	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    eng,
		config:    meta,
		info:      info,
	}
}
