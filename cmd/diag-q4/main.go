// diag-q4 compares F32 and Q4 model execution to find where divergence occurs.
//
// Usage: diag-q4 -f32 /path/to/f32/model -q4 /path/to/q4/model -prompt "text"
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/zerfoo/zerfoo/compute"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/pkg/tokenizer"
	"github.com/zerfoo/zerfoo/tensor"
)

func main() {
	f32Dir := flag.String("f32", "", "path to F32 model directory")
	q4Dir := flag.String("q4", "", "path to Q4 model directory")
	prompt := flag.String("prompt", "The capital of France is", "prompt text")
	flag.Parse()

	if *f32Dir == "" || *q4Dir == "" {
		fmt.Fprintf(os.Stderr, "Usage: diag-q4 -f32 <f32-model-dir> -q4 <q4-model-dir>\n")
		os.Exit(1)
	}

	layerreg.RegisterAll()

	tok, err := tokenizer.LoadFromJSON(*f32Dir + "/tokenizer.json")
	if err != nil {
		log.Fatalf("load tokenizer: %v", err)
	}

	tokenIDs, err := tok.Encode(*prompt)
	if err != nil {
		log.Fatalf("encode: %v", err)
	}
	tokenIDs = append([]int{2}, tokenIDs...) // BOS
	log.Printf("Token IDs: %v (len=%d)", tokenIDs, len(tokenIDs))

	// Create input tensors.
	data := make([]float32, len(tokenIDs))
	for i, id := range tokenIDs {
		data[i] = float32(id)
	}
	inputF32, _ := tensor.New([]int{1, len(tokenIDs)}, append([]float32(nil), data...))
	inputQ4, _ := tensor.New([]int{1, len(tokenIDs)}, append([]float32(nil), data...))

	// Load F32 model.
	log.Printf("Loading F32 model from %s", *f32Dir)
	engF32 := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	mdlF32, err := model.LoadModelFromZMF(engF32, numeric.Float32Ops{}, *f32Dir+"/model.zmf")
	if err != nil {
		log.Fatalf("load F32: %v", err)
	}

	// Load Q4 model.
	log.Printf("Loading Q4 model from %s", *q4Dir)
	engQ4 := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	mdlQ4, err := model.LoadModelFromZMF(engQ4, numeric.Float32Ops{}, *q4Dir+"/model.zmf")
	if err != nil {
		log.Fatalf("load Q4: %v", err)
	}

	// --- Phase 1: Q4 Scale Health Check ---
	log.Printf("\n=== Phase 1: Q4 Scale Health Check ===")

	ctx := context.Background()
	q4Nodes := mdlQ4.Graph.Nodes()
	log.Printf("Q4 graph: %d nodes", len(q4Nodes))

	var q4Count, f32Count int
	var totalBlocks int
	var zeroScales, nanScales, infScales int
	var minScale, maxScale float32 = math.MaxFloat32, 0

	for _, n := range q4Nodes {
		if n.OpType() != "Parameter" {
			continue
		}
		t, fErr := n.Forward(ctx)
		if fErr != nil || t == nil {
			continue
		}
		q4stor, ok := t.GetStorage().(*tensor.Q4Storage)
		if ok {
			q4Count++
			nb := q4stor.NumBlocks()
			totalBlocks += nb
			for bi := range nb {
				s := q4stor.BlockScaleF32(bi)
				if s == 0 {
					zeroScales++
				}
				if math.IsNaN(float64(s)) {
					nanScales++
				}
				if math.IsInf(float64(s), 0) {
					infScales++
				}
				abs := float32(math.Abs(float64(s)))
				if abs > 0 && abs < minScale {
					minScale = abs
				}
				if abs > maxScale {
					maxScale = abs
				}
			}
		} else {
			f32Count++
		}
	}
	log.Printf("Q4 tensors: %d, F32 tensors: %d", q4Count, f32Count)
	log.Printf("Total Q4 blocks: %d", totalBlocks)
	log.Printf("Scale stats: zero=%d NaN=%d Inf=%d min=%.6e max=%.6e",
		zeroScales, nanScales, infScales, minScale, maxScale)

	// --- Phase 2: Compare first few Parameter tensors ---
	log.Printf("\n=== Phase 2: Weight Comparison (first 20) ===")
	f32Nodes := mdlF32.Graph.Nodes()
	compareCount := 0
	for i := range min(len(q4Nodes), len(f32Nodes)) {
		if q4Nodes[i].OpType() != "Parameter" || f32Nodes[i].OpType() != "Parameter" {
			continue
		}
		q4t, _ := q4Nodes[i].Forward(ctx)
		f32t, _ := f32Nodes[i].Forward(ctx)
		if q4t == nil || f32t == nil {
			continue
		}

		q4Data := q4t.Data()
		f32Data := f32t.Data()

		if len(q4Data) != len(f32Data) || len(q4Data) < 32 {
			if len(q4Data) != len(f32Data) {
				log.Printf("  node[%d] SIZE MISMATCH q4=%d f32=%d shape_q4=%v shape_f32=%v",
					i, len(q4Data), len(f32Data), q4t.Shape(), f32t.Shape())
			}
			continue
		}

		_, isQ4 := q4t.GetStorage().(*tensor.Q4Storage)
		var maxDiff float64
		for j := range q4Data {
			d := math.Abs(float64(q4Data[j] - f32Data[j]))
			if d > maxDiff {
				maxDiff = d
			}
		}

		label := "F32"
		if isQ4 {
			label = "Q4 "
		}

		if compareCount < 20 || maxDiff > 0.1 {
			log.Printf("  node[%d] [%s] shape=%-20v maxDiff=%.6f first4_q4=%v first4_f32=%v",
				i, label, q4t.Shape(), maxDiff,
				truncF(q4Data, 4), truncF(f32Data, 4))
		}
		compareCount++
		if compareCount >= 30 && maxDiff < 0.1 {
			break
		}
	}
	log.Printf("Compared %d parameter tensors", compareCount)

	// --- Phase 3: DebugForward Comparison ---
	log.Printf("\n=== Phase 3: DebugForward Comparison ===")

	log.Printf("Running F32 DebugForward...")
	f32Snaps, f32Out, err := mdlF32.Graph.DebugForward(ctx, inputF32)
	if err != nil {
		log.Printf("F32 DebugForward failed at node %d: %v", len(f32Snaps), err)
	}

	log.Printf("Running Q4 DebugForward...")
	q4Snaps, q4Out, err := mdlQ4.Graph.DebugForward(ctx, inputQ4)
	if err != nil {
		log.Printf("Q4 DebugForward failed at node %d: %v", len(q4Snaps), err)
	}

	log.Printf("F32 snapshots: %d, Q4 snapshots: %d", len(f32Snaps), len(q4Snaps))

	// Compare snapshots: print early nodes, all MatMul/Softmax, and large divergences.
	minSnaps := min(len(f32Snaps), len(q4Snaps))
	divergeAt := -1
	matmulIdx := 0
	softmaxIdx := 0

	for i := range minSnaps {
		f32s := f32Snaps[i]
		q4s := q4Snaps[i]

		if f32s.OpType != q4s.OpType {
			log.Printf("  node[%d] OP MISMATCH: f32=%s q4=%s", i, f32s.OpType, q4s.OpType)
			continue
		}

		skip := f32s.OpType == "Parameter" || f32s.OpType == "Constant" ||
			f32s.OpType == "AutoPositionIds" || f32s.OpType == "AutoZeroKVCache"
		if skip || len(f32s.Data) == 0 || len(q4s.Data) == 0 {
			continue
		}

		nCmp := min(len(f32s.Data), len(q4s.Data))
		var maxD float32
		for j := range nCmp {
			d := float32(math.Abs(float64(f32s.Data[j] - q4s.Data[j])))
			if d > maxD {
				maxD = d
			}
		}

		isMatMul := f32s.OpType == "MatMul"
		isSoftmax := f32s.OpType == "Softmax"

		if isMatMul {
			matmulIdx++
		}
		if isSoftmax {
			softmaxIdx++
		}

		// Print: first 30 compute nodes, every MatMul (first 20), every Softmax, large diffs.
		printThis := i < 30 || maxD > 1.0 ||
			(isMatMul && matmulIdx <= 20) ||
			(isSoftmax && softmaxIdx <= 5)

		if printThis {
			log.Printf("  node[%d] %-25s shape=%-15v maxDiff=%.4f f32=%v q4=%v",
				i, f32s.OpType, f32s.Shape, maxD,
				truncF(f32s.Data, 4), truncF(q4s.Data, 4))
		}

		if divergeAt < 0 && maxD > 10.0 {
			divergeAt = i
		}
	}

	if divergeAt >= 0 {
		log.Printf("\nFirst large divergence (maxDiff > 10) at node[%d] %s", divergeAt, f32Snaps[divergeAt].OpType)
		start := max(0, divergeAt-5)
		end := min(minSnaps, divergeAt+6)
		for i := start; i < end; i++ {
			f32s := f32Snaps[i]
			q4s := q4Snaps[i]
			if len(f32s.Data) == 0 || len(q4s.Data) == 0 {
				continue
			}
			nCmp := min(len(f32s.Data), len(q4s.Data))
			var maxD float32
			for j := range nCmp {
				d := float32(math.Abs(float64(f32s.Data[j] - q4s.Data[j])))
				if d > maxD {
					maxD = d
				}
			}
			log.Printf("  node[%d] %-25s shape=%-15v maxDiff=%.4f f32=%v q4=%v",
				i, f32s.OpType, f32s.Shape, maxD,
				truncF(f32s.Data, 4), truncF(q4s.Data, 4))
		}
	}

	// --- Phase 4: Final Logits ---
	log.Printf("\n=== Phase 4: Final Logits ===")
	if f32Out != nil && q4Out != nil {
		analyzeAndCompare(f32Out, q4Out, tok)
	}
}

func truncF(data []float32, n int) []float32 {
	if len(data) <= n {
		return data
	}
	return data[:n]
}

func analyzeAndCompare(f32Out, q4Out *tensor.TensorNumeric[float32], tok tokenizer.Tokenizer) {
	f32Shape := f32Out.Shape()
	q4Shape := q4Out.Shape()
	log.Printf("F32 logits shape: %v, Q4 logits shape: %v", f32Shape, q4Shape)

	if len(f32Shape) != 3 || len(q4Shape) != 3 {
		return
	}

	vocabSize := f32Shape[2]
	seqLen := f32Shape[1]

	f32Data := f32Out.Data()
	q4Data := q4Out.Data()

	f32Last := f32Data[(seqLen-1)*vocabSize : seqLen*vocabSize]
	q4Last := q4Data[(seqLen-1)*vocabSize : seqLen*vocabSize]

	log.Printf("F32 top-5:")
	printTopK(f32Last, tok, 5)
	log.Printf("Q4 top-5:")
	printTopK(q4Last, tok, 5)

	var f32Sum, q4Sum float64
	var maxDiff float32
	for i := range vocabSize {
		f32Sum += float64(f32Last[i])
		q4Sum += float64(q4Last[i])
		d := float32(math.Abs(float64(f32Last[i] - q4Last[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	log.Printf("Logit maxDiff=%.4f f32Mean=%.4f q4Mean=%.4f", maxDiff, f32Sum/float64(vocabSize), q4Sum/float64(vocabSize))
}

func printTopK(logits []float32, tok tokenizer.Tokenizer, k int) {
	type ts struct {
		id    int
		score float32
	}
	n := len(logits)
	top := make([]ts, k)
	for i := range k {
		top[i] = ts{-1, -math.MaxFloat32}
	}
	for i := range n {
		if logits[i] > top[k-1].score {
			top[k-1] = ts{i, logits[i]}
			for j := k - 1; j > 0 && top[j].score > top[j-1].score; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		}
	}
	for _, t := range top {
		text, _ := tok.Decode([]int{t.id})
		log.Printf("  id=%-6d %-15q logit=%.4f", t.id, text, t.score)
	}
}
