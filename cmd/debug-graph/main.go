package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"sort"

	"github.com/zerfoo/zerfoo/compute"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/pkg/tokenizer"
	"github.com/zerfoo/zerfoo/tensor"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: debug-graph <model-dir> <prompt>\n")
		fmt.Fprintf(os.Stderr, "  model-dir: path to model directory (contains model.zmf, tokenizer.json)\n")
		fmt.Fprintf(os.Stderr, "  prompt: text prompt to test\n")
		os.Exit(1)
	}

	layerreg.RegisterAll()

	modelDir := os.Args[1]
	prompt := os.Args[2]

	zmfPath := modelDir + "/model.zmf"
	tokPath := modelDir + "/tokenizer.json"

	log.Printf("Loading tokenizer from %s", tokPath)
	tok, err := tokenizer.LoadFromJSON(tokPath)
	if err != nil {
		log.Fatalf("load tokenizer: %v", err)
	}

	log.Printf("Encoding prompt: %q", prompt)
	tokenIDs, err := tok.Encode(prompt)
	if err != nil {
		log.Fatalf("encode: %v", err)
	}
	log.Printf("Token IDs: %v (len=%d)", tokenIDs, len(tokenIDs))

	log.Printf("Loading model from %s", zmfPath)
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	mdl, err := model.LoadModelFromZMF[float32](eng, numeric.Float32Ops{}, zmfPath)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	g := mdl.Graph

	// Print graph summary.
	nodes := g.Nodes()
	opCounts := make(map[string]int)
	for _, n := range nodes {
		opCounts[n.OpType()]++
	}
	log.Printf("Graph has %d nodes:", len(nodes))
	type kv struct {
		op    string
		count int
	}
	var sorted []kv
	for op, c := range opCounts {
		sorted = append(sorted, kv{op, c})
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].count > sorted[j].count })
	for _, s := range sorted {
		log.Printf("  %-30s %d", s.op, s.count)
	}

	// Create input tensor [1, seqLen].
	data := make([]float32, len(tokenIDs))
	for i, id := range tokenIDs {
		data[i] = float32(id)
	}
	inputTensor, err := tensor.New([]int{1, len(tokenIDs)}, data)
	if err != nil {
		log.Fatalf("create input tensor: %v", err)
	}

	log.Printf("Running DebugForward...")
	snapshots, output, err := g.DebugForward(context.Background(), inputTensor)
	if err != nil {
		// Print the last few snapshots before the error.
		start := len(snapshots) - 5
		if start < 0 {
			start = 0
		}
		for _, s := range snapshots[start:] {
			log.Printf("  node[%d] %-25s shape=%v data=%v", s.Index, s.OpType, s.Shape, s.Data)
		}
		log.Fatalf("DebugForward failed: %v", err)
	}

	log.Printf("DebugForward completed: %d node outputs recorded", len(snapshots))

	// Print all non-trivial node outputs up to first Softmax (first attention layer).
	log.Printf("\n=== Nodes up to first Softmax (attention layer 0) ===")
	softmaxCount := 0
	for _, s := range snapshots {
		if s.OpType == "Parameter" || s.OpType == "Constant" || s.OpType == "AutoPositionIds" || s.OpType == "AutoZeroKVCache" {
			continue
		}
		log.Printf("  node[%d] %-25s shape=%-20v first4=%v", s.Index, s.OpType, s.Shape, truncate(s.Data, 4))
		if s.OpType == "Softmax" {
			softmaxCount++
			if softmaxCount >= 1 {
				break
			}
		}
	}

	// Print nodes around first Add after Softmax*V matmul (residual connection).
	log.Printf("\n=== After first attention layer (output projection + residual) ===")
	pastSoftmax := false
	afterCount := 0
	for _, s := range snapshots {
		if s.OpType == "Softmax" {
			pastSoftmax = true
			continue
		}
		if !pastSoftmax {
			continue
		}
		if s.OpType == "Parameter" || s.OpType == "Constant" {
			continue
		}
		log.Printf("  node[%d] %-25s shape=%-20v first4=%v", s.Index, s.OpType, s.Shape, truncate(s.Data, 4))
		afterCount++
		if afterCount >= 30 {
			break
		}
	}

	// Print nodes around each Softmax (attention stats per layer).
	log.Printf("\n=== Softmax outputs per layer ===")
	for _, s := range snapshots {
		if s.OpType == "Softmax" {
			log.Printf("  node[%d] Softmax shape=%-20v first8=%v", s.Index, s.Shape, truncate(s.Data, 8))
		}
	}

	// Find Add nodes with shape [1, 5, 1152] (residual connections).
	log.Printf("\n=== Residual Add outputs (shape [1,5,1152]) - first value magnitude ===")
	addCount := 0
	for _, s := range snapshots {
		if s.OpType != "Add" {
			continue
		}
		if len(s.Shape) == 3 && s.Shape[0] == 1 && s.Shape[1] == 5 && s.Shape[2] == 1152 {
			mag := float64(0)
			for _, v := range s.Data {
				if float64(v) > mag {
					mag = float64(v)
				}
				if float64(-v) > mag {
					mag = float64(-v)
				}
			}
			log.Printf("  node[%d] Add [1,5,1152] max_abs=%.4f first4=%v", s.Index, mag, truncate(s.Data, 4))
			addCount++
		}
	}
	log.Printf("  Total residual adds: %d", addCount)

	// Print last 10 node outputs.
	log.Printf("\n=== Last 10 node outputs ===")
	start := len(snapshots) - 10
	if start < 0 {
		start = 0
	}
	for _, s := range snapshots[start:] {
		log.Printf("  node[%d] %-25s shape=%-20v first4=%v", s.Index, s.OpType, s.Shape, truncate(s.Data, 4))
	}

	// Analyze final logits.
	if output != nil {
		analyzeLogits(output, tok)
	}
}

func truncate(data []float32, n int) []float32 {
	if len(data) <= n {
		return data
	}
	return data[:n]
}

func analyzeLogits(logits *tensor.TensorNumeric[float32], tok tokenizer.Tokenizer) {
	shape := logits.Shape()
	log.Printf("\n=== Logits Analysis ===")
	log.Printf("Shape: %v", shape)

	if len(shape) != 3 {
		log.Printf("Expected 3D logits, got %dD", len(shape))
		return
	}

	vocabSize := shape[2]
	seqLen := shape[1]
	data := logits.Data()

	// Get logits for the last position.
	lastStart := (seqLen - 1) * vocabSize
	lastLogits := data[lastStart : lastStart+vocabSize]

	// Find top 10 tokens.
	type tokenScore struct {
		id    int
		score float32
	}
	scores := make([]tokenScore, vocabSize)
	for i := range vocabSize {
		scores[i] = tokenScore{i, lastLogits[i]}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })

	log.Printf("\nTop-10 tokens (last position):")
	for i := range 10 {
		s := scores[i]
		text, _ := tok.Decode([]int{s.id})
		log.Printf("  %d. id=%-6d %-15q logit=%.4f", i+1, s.id, text, s.score)
	}

	// Check specific tokens.
	checkTokens := []int{818, 1437, 2217, 64753, 50429} // The, the, not, Capital, Paris
	log.Printf("\nReference tokens:")
	for _, id := range checkTokens {
		if id < vocabSize {
			text, _ := tok.Decode([]int{id})
			log.Printf("  id=%-6d %-15q logit=%.4f", id, text, lastLogits[id])
		}
	}

	// Check logits at each position for token 818.
	log.Printf("\nToken 818 logits at each position:")
	for pos := range seqLen {
		posStart := pos * vocabSize
		if posStart+818 < len(data) {
			log.Printf("  pos %d: %.4f", pos, data[posStart+818])
		}
	}

	// Basic stats.
	var sum, sumSq float64
	minVal, maxVal := float64(math.MaxFloat64), -float64(math.MaxFloat64)
	for _, v := range lastLogits {
		fv := float64(v)
		sum += fv
		sumSq += fv * fv
		if fv < minVal {
			minVal = fv
		}
		if fv > maxVal {
			maxVal = fv
		}
	}
	mean := sum / float64(vocabSize)
	variance := sumSq/float64(vocabSize) - mean*mean
	log.Printf("\nLogits stats: min=%.4f max=%.4f mean=%.4f std=%.4f", minVal, maxVal, mean, math.Sqrt(variance))
}
