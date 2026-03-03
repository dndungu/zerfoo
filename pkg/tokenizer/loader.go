package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// tokenizerJSON represents the HuggingFace tokenizer.json schema.
type tokenizerJSON struct {
	Model        modelJSON        `json:"model"`
	AddedTokens  []addedTokenJSON `json:"added_tokens"`
	PreTokenizer *preTokenizerJSON `json:"pre_tokenizer"`
	Normalizer   *normalizerJSON  `json:"normalizer"`
}

type modelJSON struct {
	Type   string         `json:"type"`
	Vocab  map[string]int `json:"vocab"`
	Merges []string       `json:"merges"`
}

type addedTokenJSON struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type preTokenizerJSON struct {
	Type          string             `json:"type"`
	PreTokenizers []preTokenizerJSON `json:"pretokenizers"`
}

type normalizerJSON struct {
	Type        string           `json:"type"`
	Normalizers []normalizerJSON `json:"normalizers"`
}

// LoadFromJSON reads a HuggingFace tokenizer.json file and returns a BPETokenizer.
func LoadFromJSON(path string) (*BPETokenizer, error) {
	data, err := os.ReadFile(path) //nolint:gosec // user-provided path
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}

	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}

	if tj.Model.Type != "" && tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("unsupported model type: %q (only BPE supported)", tj.Model.Type)
	}

	// Parse merges.
	merges := make([]MergePair, 0, len(tj.Model.Merges))
	for i, m := range tj.Model.Merges {
		left, right, ok := strings.Cut(m, " ")
		if !ok {
			return nil, fmt.Errorf("invalid merge at index %d: %q", i, m)
		}
		merges = append(merges, MergePair{Left: left, Right: right})
	}

	// Detect byte-level BPE from pre-tokenizer config.
	byteLevelBPE := isByteLevelPreTokenizer(tj.PreTokenizer)

	// Extract special tokens.
	special := extractSpecialTokens(tj.AddedTokens)

	// Build normalizer function if present.
	normalizer := buildNormalizer(tj.Normalizer)

	tok := NewBPETokenizer(tj.Model.Vocab, merges, special, byteLevelBPE)
	tok.normalizer = normalizer
	return tok, nil
}

// isByteLevelPreTokenizer returns true if the pre-tokenizer config uses ByteLevel.
func isByteLevelPreTokenizer(pt *preTokenizerJSON) bool {
	if pt == nil {
		return false
	}
	if pt.Type == "ByteLevel" {
		return true
	}
	if pt.Type == "Sequence" {
		for _, child := range pt.PreTokenizers {
			if child.Type == "ByteLevel" {
				return true
			}
		}
	}
	return false
}

// extractSpecialTokens finds BOS, EOS, PAD, UNK from added_tokens.
func extractSpecialTokens(tokens []addedTokenJSON) SpecialTokens {
	special := SpecialTokens{}
	for _, t := range tokens {
		if !t.Special {
			continue
		}
		switch {
		case strings.Contains(t.Content, "bos") || t.Content == "<s>":
			special.BOS = t.ID
		case strings.Contains(t.Content, "eos") || t.Content == "</s>":
			special.EOS = t.ID
		case strings.Contains(t.Content, "pad") || t.Content == "<pad>":
			special.PAD = t.ID
		case strings.Contains(t.Content, "unk") || t.Content == "<unk>":
			special.UNK = t.ID
		}
	}
	return special
}

// NormalizerFunc transforms text before tokenization.
type NormalizerFunc func(string) string

// buildNormalizer creates a normalizer function from the JSON config.
func buildNormalizer(n *normalizerJSON) NormalizerFunc {
	if n == nil {
		return nil
	}
	switch n.Type {
	case "NFC":
		return func(s string) string { return norm.NFC.String(s) }
	case "NFD":
		return func(s string) string { return norm.NFD.String(s) }
	case "Lowercase":
		return strings.ToLower
	case "Strip":
		return func(s string) string { return strings.TrimFunc(s, unicode.IsSpace) }
	case "Sequence":
		var chain []NormalizerFunc
		for i := range n.Normalizers {
			if fn := buildNormalizer(&n.Normalizers[i]); fn != nil {
				chain = append(chain, fn)
			}
		}
		if len(chain) == 0 {
			return nil
		}
		return func(s string) string {
			for _, fn := range chain {
				s = fn(s)
			}
			return s
		}
	default:
		return nil
	}
}
