# Quality Gates

## Test Coverage (2026-03-05)

Minimum threshold: 75% (enforced by coverage-gate CI)

### 100% Coverage
- config, data, device, internal/xblas, layers/components, layers/registry,
  layers/tokenizers, metrics, shutdown

### 98-99%
- compute (98.0%), distributed/coordinator (98.3%), features (99.0%),
  model/hrm (98.1%), numeric (98.5%), tensor (97.9%),
  testing/testutils (99.3%), tests/internal/testutil (98.5%)

### 96-97%
- graph (97.3%), inference (96.3%), layers/activations (97.4%),
  layers/attention (96.5%), layers/recurrent (97.0%),
  layers/regularization (97.6%), layers/transformer (96.4%),
  layers/transpose (97.6%), log (97.7%), metrics/runtime (96.5%),
  pkg/tokenizer (96.2%), serve (96.4%), training/loss (97.4%),
  training/optimizer (96.6%)

### 95-96%
- distributed (95.8%), generate (95.0%), layers/core (95.9%),
  layers/hrm (95.5%), layers/normalization (95.7%),
  layers/reducesum (95.9%), model (95.1%), training (95.9%)

### 90-94%
- cmd/cli (93.6%), cmd/bench-compare (89.7%), health (90.0%),
  layers/embeddings (92.9%), layers/features (93.8%),
  layers/gather (93.5%), layers/sequence (94.0%), registry (93.2%)

### Below 90%
- cmd/coverage-gate (84.9%), cmd/zerfoo-predict (76.6%), cmd/zerfoo-tokenize (74.1%)

### Known Untestable Gaps
- health: EngineCheck takes concrete *CPUEngine type, preventing mock testing
- Most remaining gaps are tensor.New unreachable error paths or engine error
  paths that require mock infrastructure
- layers/attention: dupl linter blocks MLA Forward engine error test
- cmd/zerfoo-predict: main(), runNewCLI (requires cli framework), os.Exit paths
- cmd/zerfoo-tokenize: main() and os.Exit paths

## Linting

- golangci-lint v2 with project .golangci.yml
- go vet: clean
- gosec: G704/G705 excluded (taint analysis false positives)

## Build

- `go build ./...`: clean
- `-race` flag enabled for all tests
