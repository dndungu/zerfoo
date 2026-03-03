package registry

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const (
	defaultHFAPIURL = "https://huggingface.co/api/models"
	defaultHFCDNURL = "https://huggingface.co"
)

// ProgressFunc reports download progress. total may be -1 if unknown.
type ProgressFunc func(downloaded, total int64)

// HFPullOptions configures HuggingFace model downloads.
type HFPullOptions struct {
	// APIURL overrides the HuggingFace API endpoint.
	APIURL string
	// CDNURL overrides the HuggingFace CDN endpoint.
	CDNURL string
	// Token is an optional HuggingFace API token for gated models.
	Token string
	// OnProgress is called during file downloads.
	OnProgress ProgressFunc
	// Client overrides the HTTP client used for downloads.
	Client *http.Client
}

// HFSibling represents a file entry in a HuggingFace model listing.
type HFSibling struct {
	Filename string `json:"rfilename"`
}

// HFModelInfo represents the model metadata returned by the HuggingFace API.
type HFModelInfo struct {
	ID       string      `json:"id"`
	Siblings []HFSibling `json:"siblings"`
}

// NewHFPullFunc creates a PullFunc that downloads models from HuggingFace Hub.
func NewHFPullFunc(opts HFPullOptions) PullFunc {
	if opts.APIURL == "" {
		opts.APIURL = getEnvOr("HUGGINGFACE_API_URL", defaultHFAPIURL)
	}
	if opts.CDNURL == "" {
		opts.CDNURL = getEnvOr("HUGGINGFACE_CDN_URL", defaultHFCDNURL)
	}
	if opts.Token == "" {
		opts.Token = os.Getenv("HF_TOKEN")
	}
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}

	return func(ctx context.Context, modelID string, targetDir string) (*ModelInfo, error) {
		return pullFromHF(ctx, opts, modelID, targetDir)
	}
}

func pullFromHF(ctx context.Context, opts HFPullOptions, modelID string, targetDir string) (*ModelInfo, error) {
	// 1. List model files.
	files, err := listModelFiles(ctx, opts, modelID)
	if err != nil {
		return nil, fmt.Errorf("list files: %w", err)
	}

	// 2. Download relevant files.
	var totalSize int64
	for _, f := range files {
		if !shouldDownload(f.Filename) {
			continue
		}
		size, dlErr := downloadFile(ctx, opts, modelID, f.Filename, targetDir)
		if dlErr != nil {
			return nil, fmt.Errorf("download %s: %w", f.Filename, dlErr)
		}
		totalSize += size
	}

	return &ModelInfo{
		ID:   modelID,
		Path: targetDir,
		Size: totalSize,
	}, nil
}

// listModelFiles queries the HuggingFace API for model file listing.
func listModelFiles(ctx context.Context, opts HFPullOptions, modelID string) ([]HFSibling, error) {
	url := fmt.Sprintf("%s/%s", opts.APIURL, modelID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	addAuthHeader(req, opts.Token)

	resp, err := opts.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	var info HFModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return info.Siblings, nil
}

// downloadFile downloads a single file from HuggingFace CDN.
func downloadFile(ctx context.Context, opts HFPullOptions, modelID, filename, targetDir string) (int64, error) {
	url := fmt.Sprintf("%s/%s/resolve/main/%s", opts.CDNURL, modelID, filename)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	addAuthHeader(req, opts.Token)

	resp, err := opts.Client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("download returned status %d", resp.StatusCode)
	}

	// Create subdirectories if needed (e.g., onnx/model.onnx).
	destPath := filepath.Join(targetDir, filename)
	if err := os.MkdirAll(filepath.Dir(destPath), 0o750); err != nil {
		return 0, err
	}

	f, err := os.Create(destPath) //nolint:gosec // user-controlled target dir
	if err != nil {
		return 0, err
	}
	defer f.Close() //nolint:errcheck

	var reader io.Reader = resp.Body
	if opts.OnProgress != nil {
		reader = &progressReader{
			reader:   resp.Body,
			total:    resp.ContentLength,
			callback: opts.OnProgress,
		}
	}

	n, err := io.Copy(f, reader)
	if err != nil {
		return 0, err
	}
	return n, nil
}

// shouldDownload returns true if the file is relevant for model caching.
func shouldDownload(filename string) bool {
	lower := strings.ToLower(filename)
	// Download ONNX models, tokenizer files, and config files.
	switch {
	case strings.HasSuffix(lower, ".onnx"):
		return true
	case strings.Contains(lower, "tokenizer") && (strings.HasSuffix(lower, ".json") || strings.HasSuffix(lower, ".model")):
		return true
	case lower == "config.json" || lower == "generation_config.json":
		return true
	case strings.HasSuffix(lower, ".onnx_data"):
		return true
	default:
		return false
	}
}

// progressReader wraps an io.Reader to report download progress.
type progressReader struct {
	reader     io.Reader
	total      int64
	downloaded int64
	callback   ProgressFunc
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.reader.Read(p)
	pr.downloaded += int64(n)
	if pr.callback != nil {
		pr.callback(pr.downloaded, pr.total)
	}
	return n, err
}

func addAuthHeader(req *http.Request, token string) {
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}

func getEnvOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
