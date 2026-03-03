package inference

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/pkg/tokenizer"
	"github.com/zerfoo/zerfoo/registry"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// fixedLogitsNode always returns logits where a specific token has the highest value.
type fixedLogitsNode struct {
	graph.NoParameters[float32]
	vocabSize     int
	tokenSequence []int
	callCount     int
}

func (n *fixedLogitsNode) OpType() string                     { return "FixedLogits" }
func (n *fixedLogitsNode) Attributes() map[string]interface{} { return nil }
func (n *fixedLogitsNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *fixedLogitsNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *fixedLogitsNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	seqLen := 1
	if len(inputs) > 0 {
		shape := inputs[0].Shape()
		if len(shape) >= 2 {
			seqLen = shape[1]
		}
	}

	data := make([]float32, seqLen*n.vocabSize)
	for pos := range seqLen {
		targetToken := n.tokenSequence[n.callCount%len(n.tokenSequence)]
		offset := pos * n.vocabSize
		for j := range n.vocabSize {
			data[offset+j] = -10.0
		}
		if targetToken >= 0 && targetToken < n.vocabSize {
			data[offset+targetToken] = 10.0
		}
		if pos == seqLen-1 {
			n.callCount++
		}
	}

	return tensor.New([]int{1, seqLen, n.vocabSize}, data)
}

func buildTestGraph(t *testing.T, vocabSize int, tokenSequence []int) *graph.Graph[float32] {
	t.Helper()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &fixedLogitsNode{
		vocabSize:     vocabSize,
		tokenSequence: tokenSequence,
	}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func buildTestTokenizer() tokenizer.Tokenizer {
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello") // 4
	tok.AddToken("world") // 5
	tok.AddToken("foo")   // 6
	tok.AddToken("bar")   // 7
	return tok
}

func buildTestModel(t *testing.T, vocabSize int, tokenSequence []int) *Model {
	t.Helper()
	tok := buildTestTokenizer()
	g := buildTestGraph(t, vocabSize, tokenSequence)
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gen := generate.NewGenerator(g, tok, eng, generate.ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})
	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    eng,
		config: ModelMetadata{
			Architecture:          "test",
			VocabSize:             vocabSize,
			HiddenSize:            64,
			NumLayers:             1,
			MaxPositionEmbeddings: 32,
			EOSTokenID:            2,
			BOSTokenID:            1,
			ChatTemplate:          "gemma",
		},
		info: &registry.ModelInfo{
			ID:   "test-model",
			Path: "/tmp/test",
		},
	}
}

// --- loadMetadata tests ---

func TestLoadMetadata(t *testing.T) {
	t.Run("valid config", func(t *testing.T) {
		dir := t.TempDir()
		cfg := ModelMetadata{
			Architecture:          "gemma3",
			VocabSize:             256000,
			HiddenSize:            2048,
			NumLayers:             26,
			MaxPositionEmbeddings: 8192,
			EOSTokenID:            1,
			BOSTokenID:            2,
			ChatTemplate:          "gemma",
		}
		data, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, "config.json")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}

		got, err := loadMetadata(path)
		if err != nil {
			t.Fatalf("loadMetadata error: %v", err)
		}
		if got.VocabSize != 256000 {
			t.Errorf("VocabSize = %d, want 256000", got.VocabSize)
		}
		if got.Architecture != "gemma3" {
			t.Errorf("Architecture = %q, want %q", got.Architecture, "gemma3")
		}
		if got.ChatTemplate != "gemma" {
			t.Errorf("ChatTemplate = %q, want %q", got.ChatTemplate, "gemma")
		}
	})

	t.Run("file not found", func(t *testing.T) {
		_, err := loadMetadata("/nonexistent/config.json")
		if err == nil {
			t.Error("expected error for nonexistent file")
		}
	})

	t.Run("invalid json", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "config.json")
		if err := os.WriteFile(path, []byte("not json"), 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := loadMetadata(path)
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})
}

// --- Option tests ---

func TestOptions(t *testing.T) {
	t.Run("WithCacheDir", func(t *testing.T) {
		o := &loadOptions{}
		WithCacheDir("/tmp/cache")(o)
		if o.cacheDir != "/tmp/cache" {
			t.Errorf("cacheDir = %q, want %q", o.cacheDir, "/tmp/cache")
		}
	})

	t.Run("WithDevice", func(t *testing.T) {
		o := &loadOptions{}
		WithDevice("cuda")(o)
		if o.device != "cuda" {
			t.Errorf("device = %q, want %q", o.device, "cuda")
		}
	})

	t.Run("WithMaxSeqLen", func(t *testing.T) {
		o := &loadOptions{}
		WithMaxSeqLen(4096)(o)
		if o.maxSeqLen != 4096 {
			t.Errorf("maxSeqLen = %d, want 4096", o.maxSeqLen)
		}
	})

	t.Run("WithRegistry", func(t *testing.T) {
		reg := &mockRegistry{models: map[string]*registry.ModelInfo{}}
		o := &loadOptions{}
		WithRegistry(reg)(o)
		if o.registry == nil {
			t.Error("registry not set")
		}
	})
}

// --- GenerateOption tests ---

func TestGenerateOptions(t *testing.T) {
	t.Run("WithTemperature", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithTemperature(0.7)(&sc)
		if sc.Temperature != 0.7 {
			t.Errorf("Temperature = %f, want 0.7", sc.Temperature)
		}
	})

	t.Run("WithTopK", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithTopK(50)(&sc)
		if sc.TopK != 50 {
			t.Errorf("TopK = %d, want 50", sc.TopK)
		}
	})

	t.Run("WithTopP", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithTopP(0.9)(&sc)
		if sc.TopP != 0.9 {
			t.Errorf("TopP = %f, want 0.9", sc.TopP)
		}
	})

	t.Run("WithMaxTokens", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithMaxTokens(100)(&sc)
		if sc.MaxNewTokens != 100 {
			t.Errorf("MaxNewTokens = %d, want 100", sc.MaxNewTokens)
		}
	})

	t.Run("WithRepetitionPenalty", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithRepetitionPenalty(1.2)(&sc)
		if sc.RepetitionPenalty != 1.2 {
			t.Errorf("RepetitionPenalty = %f, want 1.2", sc.RepetitionPenalty)
		}
	})

	t.Run("WithStopStrings", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithStopStrings("stop1", "stop2")(&sc)
		if len(sc.StopStrings) != 2 {
			t.Fatalf("StopStrings len = %d, want 2", len(sc.StopStrings))
		}
		if sc.StopStrings[0] != "stop1" || sc.StopStrings[1] != "stop2" {
			t.Errorf("StopStrings = %v, want [stop1 stop2]", sc.StopStrings)
		}
	})
}

func TestBuildSamplingConfig(t *testing.T) {
	t.Run("defaults with no options", func(t *testing.T) {
		sc := buildSamplingConfig(nil)
		def := generate.DefaultSamplingConfig()
		if sc.Temperature != def.Temperature {
			t.Errorf("Temperature = %f, want %f", sc.Temperature, def.Temperature)
		}
		if sc.MaxNewTokens != def.MaxNewTokens {
			t.Errorf("MaxNewTokens = %d, want %d", sc.MaxNewTokens, def.MaxNewTokens)
		}
	})

	t.Run("applies options in order", func(t *testing.T) {
		sc := buildSamplingConfig([]GenerateOption{
			WithTemperature(0.5),
			WithMaxTokens(10),
		})
		if sc.Temperature != 0.5 {
			t.Errorf("Temperature = %f, want 0.5", sc.Temperature)
		}
		if sc.MaxNewTokens != 10 {
			t.Errorf("MaxNewTokens = %d, want 10", sc.MaxNewTokens)
		}
	})
}

// --- Generate tests ---

func TestModel_Generate(t *testing.T) {
	vocabSize := 8
	// Produce tokens 6 (foo), 7 (bar), then EOS (2).
	m := buildTestModel(t, vocabSize, []int{6, 7, 2})

	result, err := m.Generate(context.Background(), "hello world",
		WithTemperature(0),
		WithMaxTokens(10),
	)
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if result != "foo bar" {
		t.Errorf("Generate = %q, want %q", result, "foo bar")
	}
}

func TestModel_Generate_MaxTokens(t *testing.T) {
	vocabSize := 8
	// Never produce EOS.
	m := buildTestModel(t, vocabSize, []int{6})

	result, err := m.Generate(context.Background(), "hello",
		WithTemperature(0),
		WithMaxTokens(3),
	)
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if result != "foo foo foo" {
		t.Errorf("Generate = %q, want %q", result, "foo foo foo")
	}
}

// --- GenerateStream tests ---

func TestModel_GenerateStream(t *testing.T) {
	vocabSize := 8
	m := buildTestModel(t, vocabSize, []int{6, 7, 2})

	var tokens []string
	err := m.GenerateStream(context.Background(), "hello",
		generate.TokenStreamFunc(func(token string, done bool) error {
			if !done {
				tokens = append(tokens, token)
			}
			return nil
		}),
		WithTemperature(0),
		WithMaxTokens(10),
	)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}
	got := strings.Join(tokens, "")
	if got != "foobar" && got != "foo bar" {
		t.Errorf("streamed tokens = %q, want foo+bar", got)
	}
}

// --- Chat tests ---

func TestModel_Chat(t *testing.T) {
	vocabSize := 8
	m := buildTestModel(t, vocabSize, []int{6, 7, 2})

	resp, err := m.Chat(context.Background(), []Message{
		{Role: "user", Content: "hello"},
	}, WithTemperature(0), WithMaxTokens(10))
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if resp.Content != "foo bar" {
		t.Errorf("Chat content = %q, want %q", resp.Content, "foo bar")
	}
	if resp.TokensUsed <= 0 {
		t.Errorf("TokensUsed = %d, want > 0", resp.TokensUsed)
	}
}

// --- formatMessages tests ---

func TestFormatMessages_Gemma(t *testing.T) {
	m := &Model{
		config: ModelMetadata{ChatTemplate: "gemma"},
	}
	messages := []Message{
		{Role: "user", Content: "Hello"},
		{Role: "model", Content: "Hi there"},
	}
	got := m.formatMessages(messages)
	want := "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Errorf("formatMessages =\n%q\nwant\n%q", got, want)
	}
}

func TestFormatMessages_DefaultTemplate(t *testing.T) {
	m := &Model{
		config: ModelMetadata{ChatTemplate: ""},
	}
	// Empty template defaults to "gemma".
	messages := []Message{
		{Role: "user", Content: "Hello"},
	}
	got := m.formatMessages(messages)
	if !strings.Contains(got, "<start_of_turn>") {
		t.Errorf("expected gemma template, got %q", got)
	}
}

func TestFormatMessages_GenericTemplate(t *testing.T) {
	m := &Model{
		config: ModelMetadata{ChatTemplate: "chatml"},
	}
	messages := []Message{
		{Role: "user", Content: "Hello"},
	}
	got := m.formatMessages(messages)
	want := "user: Hello\nassistant: "
	if got != want {
		t.Errorf("formatMessages =\n%q\nwant\n%q", got, want)
	}
}

func TestFormatMessages_MultipleMessages(t *testing.T) {
	m := &Model{
		config: ModelMetadata{ChatTemplate: "chatml"},
	}
	messages := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi!"},
		{Role: "user", Content: "How are you?"},
	}
	got := m.formatMessages(messages)
	want := "system: You are helpful.\nuser: Hello\nassistant: Hi!\nuser: How are you?\nassistant: "
	if got != want {
		t.Errorf("formatMessages =\n%q\nwant\n%q", got, want)
	}
}

// --- Embed tests ---

func TestModel_Embed_NotSupported(t *testing.T) {
	vocabSize := 8
	m := buildTestModel(t, vocabSize, []int{6, 2})

	_, err := m.Embed(context.Background(), "hello")
	if err == nil {
		t.Error("expected error from Embed")
	}
	if !strings.Contains(err.Error(), "not yet supported") {
		t.Errorf("error = %q, want 'not yet supported'", err.Error())
	}
}

func TestModel_Embed_EmptyText(t *testing.T) {
	vocabSize := 8
	m := buildTestModel(t, vocabSize, []int{6})

	_, err := m.Embed(context.Background(), "")
	if err == nil {
		t.Error("expected error for empty text")
	}
}

// --- Config and Info tests ---

func TestModel_Config(t *testing.T) {
	m := &Model{
		config: ModelMetadata{
			Architecture: "gemma3",
			VocabSize:    256000,
		},
	}
	cfg := m.Config()
	if cfg.Architecture != "gemma3" {
		t.Errorf("Architecture = %q, want %q", cfg.Architecture, "gemma3")
	}
	if cfg.VocabSize != 256000 {
		t.Errorf("VocabSize = %d, want 256000", cfg.VocabSize)
	}
}

func TestModel_Info(t *testing.T) {
	info := &registry.ModelInfo{ID: "test-model", Path: "/tmp/test"}
	m := &Model{info: info}
	got := m.Info()
	if got.ID != "test-model" {
		t.Errorf("Info().ID = %q, want %q", got.ID, "test-model")
	}
}

// --- mockRegistry for Load tests ---

type mockRegistry struct {
	models map[string]*registry.ModelInfo
}

func (r *mockRegistry) Get(id string) (*registry.ModelInfo, bool) {
	info, ok := r.models[id]
	return info, ok
}

func (r *mockRegistry) Pull(_ context.Context, id string) (*registry.ModelInfo, error) {
	info, ok := r.models[id]
	if !ok {
		return nil, os.ErrNotExist
	}
	return info, nil
}

func (r *mockRegistry) List() []registry.ModelInfo {
	var result []registry.ModelInfo
	for _, info := range r.models {
		result = append(result, *info)
	}
	return result
}

func (r *mockRegistry) Delete(id string) error {
	delete(r.models, id)
	return nil
}

// --- Load tests ---

func TestLoad_RegistryNotFound(t *testing.T) {
	reg := &mockRegistry{models: map[string]*registry.ModelInfo{}}
	_, err := Load("nonexistent-model", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for nonexistent model")
	}
}

func TestLoad_MissingConfig(t *testing.T) {
	dir := t.TempDir()
	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for missing config.json")
	}
	if !strings.Contains(err.Error(), "load config") {
		t.Errorf("error = %q, want 'load config' prefix", err.Error())
	}
}

func TestLoad_MissingTokenizer(t *testing.T) {
	dir := t.TempDir()
	// Write config.json but no tokenizer.json.
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1}
	data, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for missing tokenizer.json")
	}
	if !strings.Contains(err.Error(), "load tokenizer") {
		t.Errorf("error = %q, want 'load tokenizer' prefix", err.Error())
	}
}

func TestLoad_MissingZMF(t *testing.T) {
	dir := t.TempDir()
	// Write valid config.json and minimal tokenizer.json, but no model.zmf.
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1, MaxPositionEmbeddings: 128}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}

	tokJSON := `{"model":{"type":"BPE","vocab":{"hello":0},"merges":[]},"added_tokens":[]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokJSON), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for missing model.zmf")
	}
	if !strings.Contains(err.Error(), "load model") {
		t.Errorf("error = %q, want 'load model' prefix", err.Error())
	}
}

func TestLoad_InvalidZMF(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1, MaxPositionEmbeddings: 128}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}

	tokJSON := `{"model":{"type":"BPE","vocab":{"hello":0},"merges":[]},"added_tokens":[]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokJSON), 0o600); err != nil {
		t.Fatal(err)
	}

	// Write an invalid ZMF file (not a valid protobuf).
	if err := os.WriteFile(filepath.Join(dir, "model.zmf"), []byte("not a protobuf"), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for invalid model.zmf")
	}
	if !strings.Contains(err.Error(), "load model") {
		t.Errorf("error = %q, want 'load model' prefix", err.Error())
	}
}

func TestLoad_PullPath(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}

	// pullRegistry.Get always returns false to force the Pull path.
	reg := &pullRegistry{
		info: &registry.ModelInfo{ID: "test", Path: dir},
	}
	_, err := Load("test", WithRegistry(reg))
	// Will fail at tokenizer load but exercises the Pull path.
	if err == nil {
		t.Error("expected error")
	}
	if !reg.pulled {
		t.Error("expected Pull to be called")
	}
}

func TestLoad_DefaultCacheDir(t *testing.T) {
	// Load without WithCacheDir exercises the empty-cacheDir registry creation path.
	// Will fail immediately because the model doesn't exist in default registry.
	_, err := Load("nonexistent-model-12345")
	if err == nil {
		t.Error("expected error")
	}
}

func TestLoad_WithCacheDir(t *testing.T) {
	dir := t.TempDir()
	// Load with WithCacheDir exercises the cacheDir registry creation path.
	_, err := Load("nonexistent-model-12345", WithCacheDir(dir))
	if err == nil {
		t.Error("expected error")
	}
}

func TestLoad_MaxSeqLenOverride(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1, MaxPositionEmbeddings: 8192}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}

	tokJSON := `{"model":{"type":"BPE","vocab":{"hello":0},"merges":[]},"added_tokens":[]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokJSON), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.zmf"), []byte("bad"), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	// WithMaxSeqLen won't be reached because ZMF load fails first,
	// but it exercises the option path.
	_, err := Load("test", WithRegistry(reg), WithMaxSeqLen(2048))
	if err == nil {
		t.Error("expected error")
	}
}

// pullRegistry always forces the Pull path.
type pullRegistry struct {
	info   *registry.ModelInfo
	pulled bool
}

func (r *pullRegistry) Get(_ string) (*registry.ModelInfo, bool) {
	return nil, false
}

func (r *pullRegistry) Pull(_ context.Context, _ string) (*registry.ModelInfo, error) {
	r.pulled = true
	return r.info, nil
}

func (r *pullRegistry) List() []registry.ModelInfo { return nil }
func (r *pullRegistry) Delete(_ string) error      { return nil }

// --- Chat error path ---

// errorNode always returns an error on Forward.
type errorNode struct {
	graph.NoParameters[float32]
}

func (n *errorNode) OpType() string                     { return "Error" }
func (n *errorNode) Attributes() map[string]interface{} { return nil }
func (n *errorNode) OutputShape() []int                 { return []int{1, 1, 8} }
func (n *errorNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}
func (n *errorNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return nil, os.ErrInvalid
}

func buildErrorModel(t *testing.T) *Model {
	t.Helper()
	tok := buildTestTokenizer()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &errorNode{}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	gen := generate.NewGenerator(g, tok, engine, generate.ModelConfig{
		VocabSize:  8,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		NumLayers:  0,
	})
	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    engine,
		config:    ModelMetadata{ChatTemplate: "gemma"},
	}
}

func TestModel_Chat_GenerateError(t *testing.T) {
	m := buildErrorModel(t)
	_, err := m.Chat(context.Background(), []Message{
		{Role: "user", Content: "hello"},
	}, WithTemperature(0), WithMaxTokens(5))
	if err == nil {
		t.Error("expected error from Chat when Generate fails")
	}
}
