package generate

import "context"

type kvCacheKey struct{}

// WithKVCache returns a new context that carries the given KVCache.
func WithKVCache(ctx context.Context, cache *KVCache) context.Context {
	return context.WithValue(ctx, kvCacheKey{}, cache)
}

// GetKVCache extracts the KVCache from the context, if present.
func GetKVCache(ctx context.Context) (*KVCache, bool) {
	cache, ok := ctx.Value(kvCacheKey{}).(*KVCache)
	if !ok || cache == nil {
		return nil, false
	}
	return cache, true
}
