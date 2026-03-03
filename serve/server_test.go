package serve

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/inference"
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

func buildTestModel(t *testing.T) *inference.Model {
	t.Helper()
	vocabSize := 8
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello") // 4
	tok.AddToken("world") // 5
	tok.AddToken("foo")   // 6
	tok.AddToken("bar")   // 7

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &fixedLogitsNode{
		vocabSize:     vocabSize,
		tokenSequence: []int{6, 7, 2}, // foo, bar, EOS
	}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	gen := generate.NewGenerator(g, tok, engine, generate.ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})

	return inference.NewTestModel(gen, tok, engine,
		inference.ModelMetadata{
			VocabSize:  vocabSize,
			NumLayers:  1,
			EOSTokenID: 2,
			BOSTokenID: 1,
		},
		&registry.ModelInfo{ID: "test-model", Path: "/tmp/test"},
	)
}

// doPost sends a POST request with context.
func doPost(t *testing.T, url, contentType, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", contentType)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request error: %v", err)
	}
	return resp
}

// doGet sends a GET request with context.
func doGet(t *testing.T, url string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, http.NoBody)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request error: %v", err)
	}
	return resp
}

// errorNode always fails during Forward.
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
	return nil, errors.New("forward error")
}

func buildErrorModel(t *testing.T) *inference.Model {
	t.Helper()
	vocabSize := 8
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello")
	tok.AddToken("world")
	tok.AddToken("foo")
	tok.AddToken("bar")

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
		VocabSize:  vocabSize,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})

	return inference.NewTestModel(gen, tok, engine,
		inference.ModelMetadata{
			VocabSize:  vocabSize,
			NumLayers:  1,
			EOSTokenID: 2,
			BOSTokenID: 1,
		},
		&registry.ModelInfo{ID: "test-model", Path: "/tmp/test"},
	)
}

// --- Chat Completions ---

func TestHandleChatCompletions(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Object != "chat.completion" {
		t.Errorf("Object = %q, want %q", result.Object, "chat.completion")
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
	if result.Choices[0].Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", result.Choices[0].Message.Role, "assistant")
	}
	if result.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", result.Choices[0].FinishReason, "stop")
	}
}

func TestHandleChatCompletions_EmptyMessages(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[]}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleChatCompletions_InvalidJSON(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", "not json")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleChatCompletions_WithOptions(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"temperature":0.5,"top_p":0.9,"max_tokens":3}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

// --- Completions ---

func TestHandleCompletions(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Object != "text_completion" {
		t.Errorf("Object = %q, want %q", result.Object, "text_completion")
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
}

func TestHandleCompletions_EmptyPrompt(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":""}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleCompletions_InvalidJSON(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doPost(t, ts.URL+"/v1/completions", "application/json", "{bad")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleCompletions_WithOptions(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","temperature":0.5,"max_tokens":3}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

// --- Models ---

func TestHandleModels(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/v1/models")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result ModelListResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Object != "list" {
		t.Errorf("Object = %q, want %q", result.Object, "list")
	}
	if len(result.Data) != 1 {
		t.Fatalf("Data len = %d, want 1", len(result.Data))
	}
	if result.Data[0].ID != "test-model" {
		t.Errorf("model ID = %q, want %q", result.Data[0].ID, "test-model")
	}
}

// --- Streaming ---

func TestHandleChatCompletions_Stream(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want %q", ct, "text/event-stream")
	}
	// Drain the body to ensure no errors.
	_, _ = io.ReadAll(resp.Body)
}

func TestHandleCompletions_Stream(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	// Drain the body to ensure no errors.
	_, _ = io.ReadAll(resp.Body)
}

// --- Error paths ---

func TestHandleChatCompletions_GenerateError(t *testing.T) {
	mdl := buildErrorModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
}

func TestHandleCompletions_GenerateError(t *testing.T) {
	mdl := buildErrorModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
}

func TestHandleChatCompletions_StreamError(t *testing.T) {
	mdl := buildErrorModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	// Streaming starts with 200 but body contains the error.
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	data, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(data), "error") {
		t.Errorf("body should contain error, got %q", string(data))
	}
}

func TestHandleCompletions_StreamError(t *testing.T) {
	mdl := buildErrorModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	data, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(data), "error") {
		t.Errorf("body should contain error, got %q", string(data))
	}
}

// --- Close ---

func TestServer_Close(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	if err := srv.Close(context.Background()); err != nil {
		t.Errorf("Close error: %v", err)
	}
}
