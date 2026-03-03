package generate

import (
	"context"

	"github.com/zerfoo/zerfoo/tensor"
)

type kvCacheKey struct{}

// WithKVCache returns a new context that carries the given KVCache.
func WithKVCache[T tensor.Numeric](ctx context.Context, cache *KVCache[T]) context.Context {
	return context.WithValue(ctx, kvCacheKey{}, cache)
}

// GetKVCache extracts the KVCache from the context, if present.
func GetKVCache[T tensor.Numeric](ctx context.Context) (*KVCache[T], bool) {
	cache, ok := ctx.Value(kvCacheKey{}).(*KVCache[T])
	if !ok || cache == nil {
		return nil, false
	}
	return cache, true
}
