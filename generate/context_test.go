package generate

import (
	"context"
	"testing"
)

func TestWithKVCache_GetKVCache(t *testing.T) {
	cache := NewKVCache(4)
	ctx := WithKVCache(context.Background(), cache)

	got, ok := GetKVCache(ctx)
	if !ok {
		t.Fatal("GetKVCache should return true")
	}
	if got != cache {
		t.Error("GetKVCache returned different cache instance")
	}
}

func TestGetKVCache_Missing(t *testing.T) {
	_, ok := GetKVCache(context.Background())
	if ok {
		t.Error("GetKVCache on plain context should return false")
	}
}

func TestGetKVCache_NilValue(t *testing.T) {
	ctx := WithKVCache(context.Background(), nil)
	cache, ok := GetKVCache(ctx)
	if ok {
		t.Errorf("GetKVCache with nil cache should return false, got %v", cache)
	}
}
