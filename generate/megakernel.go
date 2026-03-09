package generate

import (
	"context"
	"log"
	"os"
	"sync/atomic"
	"unsafe"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/internal/codegen"
	"github.com/zerfoo/zerfoo/tensor"
)

// tryCompileMegakernel attempts to compile and load a megakernel for the
// given ExecutionPlan. On success, it sets the megakernel function on the
// plan so that subsequent Run() calls use the fused kernel. On failure, it
// silently falls back to the per-instruction execution path.
//
// This function is safe to call from a goroutine. The megakernel function
// is set atomically via SetMegakernelFn.
func tryCompileMegakernel[T tensor.Numeric](plan *graph.ExecutionPlan[T], ready *atomic.Bool) {
	instructions := plan.Instructions()
	if len(instructions) == 0 {
		return
	}

	// Check if all ops are supported by the code generator.
	unsupported := codegen.CheckSupport(instructions)
	if len(unsupported) > 0 {
		log.Printf("megakernel: %d unsupported ops: %v", len(unsupported), unsupported)
		return
	}

	// Build megakernel config from the plan.
	// Detect Q4 frozen slots to upload raw Q4 bytes instead of dequantizing.
	frozenSlots := plan.FrozenSlots()
	frozenMeta := make([]codegen.FrozenSlotMeta, len(frozenSlots))
	for i, f := range frozenSlots {
		isQ4 := false
		if f.Data != nil {
			if _, ok := any(f.Data.GetStorage()).(*tensor.Q4Storage); ok {
				isQ4 = true
			}
		}
		frozenMeta[i] = codegen.FrozenSlotMeta{SlotIdx: f.SlotIdx, IsQ4: isQ4}
	}

	slotShapes := plan.SlotShapes()

	// Discover constant workspace slots: slots used as inputs but never
	// produced as outputs by any instruction. These hold data like
	// dequantized weight copies that the tracer didn't mark as frozen.
	// We must patch their shapes BEFORE emitting CUDA so the workspace
	// layout allocates enough space and offsets are correct in the code.
	frozenSet := make(map[int]bool, len(frozenSlots))
	for _, f := range frozenSlots {
		frozenSet[f.SlotIdx] = true
	}
	producedSlots := make(map[int]bool)
	for _, inst := range instructions {
		producedSlots[inst.OutputIdx] = true
	}

	type constSlot struct {
		idx  int
		data []float32
	}
	var constants []constSlot
	seen := make(map[int]bool)
	for _, inst := range instructions {
		for _, idx := range inst.InputIdx {
			if frozenSet[idx] || producedSlots[idx] || seen[idx] {
				continue
			}
			seen[idx] = true
			td := plan.SlotData(idx)
			if td == nil {
				continue
			}
			raw := td.Data()
			f32 := make([]float32, len(raw))
			for j, v := range raw {
				f32[j] = float32(v)
			}
			constants = append(constants, constSlot{idx: idx, data: f32})
			// Patch the shape so workspace layout allocates enough space.
			if idx < len(slotShapes) {
				slotShapes[idx] = []int{len(f32)}
			}
		}
	}

	cfg := codegen.MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		FrozenSlots:  frozenMeta,
		InputSlots:   plan.InputSlots(),
		OutputSlot:   plan.OutputSlot(),
	}

	// Emit CUDA source.
	source, err := codegen.EmitMegakernel(cfg)
	if err != nil {
		log.Printf("megakernel: emit failed: %v", err)
		return
	}

	// Compile to .so (cached by source hash).
	cacheDir := os.TempDir()
	soPath, err := codegen.CachedCompile(source, cacheDir, "megakernel")
	if err != nil {
		log.Printf("megakernel: compile failed: %v", err)
		return
	}

	// Load the compiled .so.
	runner, err := codegen.LoadMegakernel(soPath)
	if err != nil {
		log.Printf("megakernel: load failed: %v", err)
		return
	}

	// Extract frozen slot data for GPU upload.
	// Q4 frozen slots: upload raw Q4 bytes reinterpreted as []float32.
	// Float32 frozen slots: dequantize and upload as before.
	frozenData := make([][]float32, len(frozenSlots))
	var totalFrozenBytes int64
	for i, f := range frozenSlots {
		if f.Data == nil {
			continue
		}
		if qs, ok := any(f.Data.GetStorage()).(*tensor.Q4Storage); ok {
			// Upload raw Q4 bytes. Reinterpret as []float32 for the
			// PrepareWorkspace API -- the GPU just sees raw bytes.
			rawBytes := qs.RawBytes()
			nFloats := (len(rawBytes) + 3) / 4 // round up to float32 boundary
			f32 := make([]float32, nFloats)
			copy(
				unsafe.Slice((*byte)(unsafe.Pointer(&f32[0])), nFloats*4),
				rawBytes,
			)
			frozenData[i] = f32
			totalFrozenBytes += int64(len(rawBytes))
		} else {
			raw := f.Data.Data()
			f32 := make([]float32, len(raw))
			for j, v := range raw {
				f32[j] = float32(v)
			}
			frozenData[i] = f32
			totalFrozenBytes += int64(len(f32)) * 4
		}
	}

	// Allocate GPU workspace and upload weights.
	if err := runner.PrepareWorkspace(cfg, frozenData); err != nil {
		log.Printf("megakernel: prepare workspace failed: %v", err)
		_ = runner.Close()
		return
	}

	// Upload constant slot data to the workspace.
	layout := codegen.ComputeWorkspaceLayout(cfg)
	var wsInitBytes int64
	for _, c := range constants {
		offset, ok := layout.SlotOffsets[c.idx]
		if !ok {
			continue
		}
		if err := runner.InitWorkspaceSlot(offset, c.data); err != nil {
			log.Printf("megakernel: init workspace slot %d failed: %v", c.idx, err)
			_ = runner.Close()
			return
		}
		wsInitBytes += int64(len(c.data)) * 4
	}

	outputShape := runner.OutputShape()

	// Set the megakernel function on the plan.
	plan.SetMegakernelFn(func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
		if len(inputs) == 0 {
			return nil, nil
		}

		// Extract input data as float32.
		inputRaw := inputs[0].Data()
		inputF32 := make([]float32, len(inputRaw))
		for i, v := range inputRaw {
			inputF32[i] = float32(v)
		}

		// Launch kernel.
		pos := 0 // position for rotary embeddings (TODO: wire from KV cache)
		outputF32, err := runner.Launch(inputF32, pos)
		if err != nil {
			return nil, err
		}

		// Convert output back to T and wrap in tensor.
		outputT := make([]T, len(outputF32))
		for i, v := range outputF32 {
			outputT[i] = T(v)
		}

		shape := outputShape
		if shape == nil {
			shape = []int{1, 1, len(outputF32)}
		}

		return tensor.New(shape, outputT)
	})

	if ready != nil {
		ready.Store(true)
	}
	log.Printf("megakernel: compiled and loaded (%d instructions, %d frozen slots, %d MB frozen, %d constants, %d MB workspace init, %s)",
		len(instructions), len(frozenSlots), totalFrozenBytes/(1024*1024), len(constants), wsInitBytes/(1024*1024), soPath)
}
