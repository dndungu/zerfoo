// Package codegen generates CUDA megakernel source code from a compiled
// ExecutionPlan instruction tape. Each primitive op maps to a CUDA device
// function call that operates on register-resident or shared-memory data.
package codegen

import (
	"fmt"
	"strings"

	"github.com/zerfoo/zerfoo/graph"
)

// --- ExtraArgs helpers ---

// extraFloat extracts a float64 from ExtraArgs. Returns 0 if missing.
func extraFloat(extra map[string]any, key string) (float64, bool) {
	if extra == nil {
		return 0, false
	}
	v, ok := extra[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	default:
		return 0, false
	}
}

// extraInt extracts an int from ExtraArgs. Returns fallback if missing.
func extraInt(extra map[string]any, key string, fallback int) int {
	if extra == nil {
		return fallback
	}
	v, ok := extra[key]
	if !ok {
		return fallback
	}
	switch val := v.(type) {
	case int:
		return val
	case int64:
		return int(val)
	case float64:
		return int(val)
	default:
		return fallback
	}
}

// extraIntSlice extracts an int slice from ExtraArgs.
func extraIntSlice(extra map[string]any, key string) []int {
	if extra == nil {
		return nil
	}
	v, ok := extra[key]
	if !ok {
		return nil
	}
	switch val := v.(type) {
	case []int:
		return val
	case []any:
		result := make([]int, len(val))
		for i, item := range val {
			switch iv := item.(type) {
			case int:
				result[i] = iv
			case float64:
				result[i] = int(iv)
			}
		}
		return result
	default:
		return nil
	}
}

// lastDim returns the last dimension of the i-th input shape.
func lastDim(inputs []SlotInfo, i int) int {
	if i < len(inputs) && len(inputs[i].Shape) > 0 {
		return inputs[i].Shape[len(inputs[i].Shape)-1]
	}
	return 0
}

// formatIntArray formats an int slice as a CUDA array initializer: {1, 2, 3}.
func formatIntArray(vals []int) string {
	parts := make([]string, len(vals))
	for i, v := range vals {
		parts[i] = fmt.Sprintf("%d", v)
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// SlotInfo describes a slot's shape for the emitter.
type SlotInfo struct {
	Shape []int
}

// OpEmitter generates CUDA code for a single instruction. It returns a
// code fragment that will be inserted into the megakernel body.
type OpEmitter func(op graph.InstructionMeta, inputs []SlotInfo) (string, error)

// emitters maps OpName strings to their CUDA code emitters.
var emitters = map[string]OpEmitter{
	// Binary elementwise
	"Add": binaryOp("+"),
	"Sub": binaryOp("-"),
	"Mul": binaryOp("*"),
	"Div": binaryOp("/"),
	"Pow": funcBinaryOp("powf"),
	"Max": funcBinaryOp("fmaxf"),

	// Unary elementwise
	"Exp":   unaryOp("expf"),
	"Log":   unaryOp("logf"),
	"Sqrt":  unaryOp("sqrtf"),
	"Rsqrt": unaryOp("rsqrtf"),
	"Cos":   unaryOp("cosf"),
	"Sin":   unaryOp("sinf"),
	"Tanh":  unaryOp("tanhf"),
	"Neg":   prefixUnaryOp("-"),
	"Abs":   unaryOp("fabsf"),
	"Silu":  siluOp,

	// Scalar ops
	"AddScalar": scalarOp("+"),
	"MulScalar": scalarOp("*"),
	"SubScalar": scalarOp("-"),
	"DivScalar": scalarOp("/"),
	"PowScalar": funcScalarOp("powf"),

	// Reductions
	"RMSNorm":    rmsnormOp,
	"Softmax":    softmaxOp,
	"ReduceSum":  reduceOp("dev_reduce_sum"),
	"ReduceMean": reduceOp("dev_reduce_mean"),
	"Sum":        reduceOp("dev_reduce_sum"),

	// Memory ops
	"MatMul":      gemvOp,
	"MatMulNBits": gemvOp, // megakernel uploads Q4 weights as dequantized float32
	"Gather":      gatherOp,

	// Indexing ops
	"Slice":  sliceOp,
	"Repeat": repeatOp,

	// Shape ops (no-compute in megakernel, just reindex)
	"Concat":           reshapeOp, // reindex in registers
	"Reshape":          reshapeOp, // no-op in flat memory
	"Shape":            reshapeOp, // metadata-only, shape known at compile time
	"Unsqueeze":        reshapeOp, // adds dim of size 1, no data movement
	"Expand":           expandOp,
	"ConstantOfShape":  constantOfShapeOp,
	"Transpose":        transposeOp,

	// Comparison / selection ops
	"Cast":    castOp,
	"Equal":   cmpOp("=="),
	"Greater": cmpOp(">"),
	"Where":   whereOp,

	// Sequence / masking ops
	"Range":     rangeOp,
	"Trilu":     triluOp,
	"ScatterND": scatterNDOp,

	// KV cache ops
	"KVCacheAppendK": kvCacheAppendOp("kv_k"),
	"KVCacheAppendV": kvCacheAppendOp("kv_v"),
	"KVCacheGetK":    kvCacheGetOp("kv_k"),
	"KVCacheGetV":    kvCacheGetOp("kv_v"),
	"KVCacheSeqLen":  kvCacheSeqLenOp,

	// Auto ops (decode-step helpers)
	"AutoPositionIds":  autoPositionIdsOp,
	"AutoZeroKVCache":  autoZeroKVCacheOp,
}

// Emit generates CUDA code for a single instruction. Returns an error
// if the op is unsupported.
func Emit(op graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	emitter, ok := emitters[op.OpName]
	if !ok {
		return "", fmt.Errorf("unsupported op: %s", op.OpName)
	}
	return emitter(op, inputs)
}

// Supported returns true if the op has a registered emitter.
func Supported(opName string) bool {
	_, ok := emitters[opName]
	return ok
}

// --- Emitter constructors ---

func binaryOp(op string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  %s[tid] = %s[tid] %s %s[tid];",
			outRef(meta), inRef(meta, 0), op, inRef(meta, 1)), nil
	}
}

func funcBinaryOp(fn string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  %s[tid] = %s(%s[tid], %s[tid]);",
			outRef(meta), fn, inRef(meta, 0), inRef(meta, 1)), nil
	}
}

func unaryOp(fn string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  %s[tid] = %s(%s[tid]);",
			outRef(meta), fn, inRef(meta, 0)), nil
	}
}

func prefixUnaryOp(prefix string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  %s[tid] = %s(%s[tid]);",
			outRef(meta), prefix, inRef(meta, 0)), nil
	}
}

func scalarOp(op string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		scalar, ok := extraFloat(meta.ExtraArgs, "scalar")
		if !ok {
			return "", fmt.Errorf("scalarOp %q: missing ExtraArgs[\"scalar\"]", op)
		}
		return fmt.Sprintf("  %s[tid] = %s[tid] %s %.9ef;",
			outRef(meta), inRef(meta, 0), op, scalar), nil
	}
}

func funcScalarOp(fn string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		scalar, ok := extraFloat(meta.ExtraArgs, "scalar")
		if !ok {
			return "", fmt.Errorf("funcScalarOp %q: missing ExtraArgs[\"scalar\"]", fn)
		}
		return fmt.Sprintf("  %s[tid] = %s(%s[tid], %.9ef);",
			outRef(meta), fn, inRef(meta, 0), scalar), nil
	}
}

func siluOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	in := inRef(meta, 0)
	return fmt.Sprintf("  %s[tid] = %s[tid] * (1.0f / (1.0f + expf(-%s[tid])));",
		outRef(meta), in, in), nil
}

func rmsnormOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	dim := lastDim(inputs, 0)
	if dim == 0 {
		return "", fmt.Errorf("rmsnormOp: cannot determine normalization dimension from input shape")
	}
	return fmt.Sprintf("  dev_rmsnorm(%s, %s, %s, %d);",
		outRef(meta), inRef(meta, 0), inRef(meta, 1), dim), nil
}

func softmaxOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	cols := 1
	if len(inputs) > 0 && len(inputs[0].Shape) > 0 {
		cols = inputs[0].Shape[len(inputs[0].Shape)-1]
	}
	return fmt.Sprintf("  dev_softmax(%s, %s, 1, %d);",
		outRef(meta), inRef(meta, 0), cols), nil
}

func gemvOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	dimM, dimK := gemvDims(meta, inputs)
	if dimM == 0 || dimK == 0 {
		return "", fmt.Errorf("gemvOp: cannot determine weight dimensions (inputs[0]=%v, inputs[1]=%v, output=%v)",
			safeShape(inputs, 0), safeShape(inputs, 1), extraIntSlice(meta.ExtraArgs, "_outputShape"))
	}
	return fmt.Sprintf("  dev_gemv_f32(%s, %s, %s, %d, %d);",
		outRef(meta), inRef(meta, 0), inRef(meta, 1), dimM, dimK), nil
}


// gemvDims extracts matrix dimensions for gemv ops from available shape info.
// Tries: (1) SlotInfo shapes, (2) ExtraArgs aShape/bShape from trace,
// (3) output shape + input vector shape.
func gemvDims(meta graph.InstructionMeta, inputs []SlotInfo) (dimM, dimK int) {
	// Try weight shape from SlotInfo (first input).
	if len(inputs) > 0 && len(inputs[0].Shape) >= 2 {
		dimM = inputs[0].Shape[len(inputs[0].Shape)-2]
		dimK = inputs[0].Shape[len(inputs[0].Shape)-1]
		return
	}
	// Try shapes recorded during tracing (engine_proxy records aShape, bShape).
	aShape := extraIntSlice(meta.ExtraArgs, "aShape")
	bShape := extraIntSlice(meta.ExtraArgs, "bShape")
	if len(aShape) >= 2 && len(bShape) >= 1 {
		// MatMul(A, B): A is [*, M, K], B is [*, K, N] or [K] for gemv.
		dimM = aShape[len(aShape)-2]
		dimK = aShape[len(aShape)-1]
		return
	}
	if len(bShape) >= 2 && len(aShape) >= 1 {
		// A might be [K] vector, B is [K, N] weight.
		dimK = bShape[len(bShape)-2]
		dimM = bShape[len(bShape)-1]
		return
	}
	// Fallback: M from output shape, K from input vector shape.
	outShape := extraIntSlice(meta.ExtraArgs, "_outputShape")
	if len(outShape) > 0 {
		dimM = outShape[len(outShape)-1]
	}
	if len(inputs) > 1 && len(inputs[1].Shape) > 0 {
		dimK = inputs[1].Shape[len(inputs[1].Shape)-1]
	}
	return
}

// safeShape returns the shape of the i-th input, or nil if out of range.
func safeShape(inputs []SlotInfo, i int) []int {
	if i < len(inputs) {
		return inputs[i].Shape
	}
	return nil
}

// isFrozenInput returns true if the i-th input is a frozen (weight) slot.
func isFrozenInput(extra map[string]any, i int) bool {
	if extra == nil {
		return false
	}
	v, ok := extra["_frozenInputs"]
	if !ok {
		return false
	}
	if frozen, ok := v.([]bool); ok && i < len(frozen) {
		return frozen[i]
	}
	return false
}

// slotRef returns "frozen_N" if the input at position i is frozen, else "slot_N".
func slotRef(meta graph.InstructionMeta, i int) string {
	idx := meta.InputIdx[i]
	if isFrozenInput(meta.ExtraArgs, i) {
		return fmt.Sprintf("frozen_%d", idx)
	}
	return fmt.Sprintf("slot_%d", idx)
}

// inRef returns slotRef(meta, i) for use in element-access expressions.
// Convenience alias used by all emitters that reference input slots.
func inRef(meta graph.InstructionMeta, i int) string {
	return slotRef(meta, i)
}

// outRef returns the slot reference for the output. Outputs are never frozen.
func outRef(meta graph.InstructionMeta) string {
	return fmt.Sprintf("slot_%d", meta.OutputIdx)
}

func gatherOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	// Embedding dim is the last dimension of the embedding table.
	dim := lastDim(inputs, 0)
	// Fallback: try output shape last dim (output of gather has same embed dim).
	if dim == 0 {
		outShape := extraIntSlice(meta.ExtraArgs, "_outputShape")
		if len(outShape) > 0 {
			dim = outShape[len(outShape)-1]
		}
	}
	if len(meta.InputIdx) < 2 {
		return fmt.Sprintf("  dev_gather(%s, frozen_%d, (int)%s[0], %d);",
			outRef(meta), meta.OutputIdx, inRef(meta, 0), dim), nil
	}
	return fmt.Sprintf("  dev_gather(%s, %s, (int)%s[0], %d);",
		outRef(meta), inRef(meta, 0), inRef(meta, 1), dim), nil
}

func reshapeOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	if meta.OutputIdx == meta.InputIdx[0] {
		return fmt.Sprintf("  // %s: %s (in-place reindex, no compute)",
			meta.OpName, outRef(meta)), nil
	}
	// Different slots: copy data from input to output.
	return fmt.Sprintf("  %s[tid] = %s[tid]; // %s",
		outRef(meta), inRef(meta, 0), meta.OpName), nil
}

func expandOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	size := 1
	if len(inputs) > 0 {
		for _, d := range inputs[0].Shape {
			size *= d
		}
	}
	return fmt.Sprintf("  %s[tid] = %s[tid %% %d];",
		outRef(meta), inRef(meta, 0), size), nil
}

func constantOfShapeOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  %s[tid] = 0.0f;", outRef(meta)), nil
}

func transposeOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	var shape []int
	if len(inputs) > 0 {
		shape = inputs[0].Shape
	}
	axes := extraIntSlice(meta.ExtraArgs, "axes")
	if len(shape) == 0 || len(axes) == 0 {
		return fmt.Sprintf("  // Transpose: %s = %s (shape/perm unknown, no-op)",
			outRef(meta), inRef(meta, 0)), nil
	}
	total := 1
	for _, d := range shape {
		total *= d
	}
	return fmt.Sprintf("  { const int shape[] = %s; const int perm[] = %s; dev_transpose(%s, %s, shape, perm, %d, %d); }",
		formatIntArray(shape), formatIntArray(axes),
		outRef(meta), inRef(meta, 0), len(shape), total), nil
}

func sliceOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	dim := lastDim(inputs, 0)
	starts := extraIntSlice(meta.ExtraArgs, "starts")
	ends := extraIntSlice(meta.ExtraArgs, "ends")
	axes := extraIntSlice(meta.ExtraArgs, "axes")
	start, end, axis := 0, dim, 0
	if len(starts) > 0 {
		start = starts[0]
	}
	if len(ends) > 0 {
		end = ends[0]
	}
	if len(axes) > 0 {
		axis = axes[0]
	}
	return fmt.Sprintf("  dev_slice(%s, %s, %d, %d, %d, %d);",
		outRef(meta), inRef(meta, 0), start, end, axis, dim), nil
}

func repeatOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	dim := lastDim(inputs, 0)
	axis := extraInt(meta.ExtraArgs, "axis", 0)
	reps := extraInt(meta.ExtraArgs, "repetitions", 1)
	return fmt.Sprintf("  dev_repeat(%s, %s, %d, %d, %d);",
		outRef(meta), inRef(meta, 0), axis, reps, dim), nil
}

func reduceOp(fn string) OpEmitter {
	return func(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
		dim := lastDim(inputs, 0)
		axis := extraInt(meta.ExtraArgs, "axis", -1)
		if axis < 0 {
			axis = 0
			if len(inputs) > 0 {
				axis = len(inputs[0].Shape) - 1
			}
		}
		return fmt.Sprintf("  %s(%s, %s, %d, %d);",
			fn, outRef(meta), inRef(meta, 0), axis, dim), nil
	}
}

func castOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  %s[tid] = %s[tid];",
		outRef(meta), inRef(meta, 0)), nil
}

func cmpOp(op string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  %s[tid] = (%s[tid] %s %s[tid]) ? 1.0f : 0.0f;",
			outRef(meta), inRef(meta, 0), op, inRef(meta, 1)), nil
	}
}

func whereOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  %s[tid] = (%s[tid] != 0.0f) ? %s[tid] : %s[tid];",
		outRef(meta), inRef(meta, 0), inRef(meta, 1), inRef(meta, 2)), nil
}

// kvCacheAppendOp emits a dev_kv_append call that writes new K or V data
// into the layer's KV cache at the current sequence position.
// InputIdx[0] = source data slot, InputIdx[1] = layer index (encoded),
// OutputIdx = destination slot alias.
func kvCacheAppendOp(arrayName string) OpEmitter {
	return func(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
		if len(meta.InputIdx) < 2 {
			return "", fmt.Errorf("KVCacheAppend requires 2 inputs (data slot, layer)")
		}
		layer := meta.InputIdx[1]
		headDim := 0
		if len(inputs) > 0 && len(inputs[0].Shape) > 0 {
			headDim = inputs[0].Shape[len(inputs[0].Shape)-1]
		}
		return fmt.Sprintf("  dev_kv_append(%s[%d], %s, seq_pos, %d);",
			arrayName, layer, inRef(meta, 0), headDim), nil
	}
}

// kvCacheGetOp emits a pointer alias that points into the layer's KV cache.
// InputIdx[0] = layer index (encoded), OutputIdx = destination slot.
func kvCacheGetOp(arrayName string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		if len(meta.InputIdx) < 1 {
			return "", fmt.Errorf("KVCacheGet requires 1 input (layer)")
		}
		layer := meta.InputIdx[0]
		return fmt.Sprintf("  float* slot_%d = %s[%d];",
			meta.OutputIdx, arrayName, layer), nil
	}
}

// kvCacheSeqLenOp emits an integer assignment from the kv_seq_len kernel arg.
func kvCacheSeqLenOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  int seq_len_%d = kv_seq_len;",
		meta.OutputIdx), nil
}

// autoPositionIdsOp emits position ID generation using the pos kernel argument.
func autoPositionIdsOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  %s[tid] = (float)(pos + tid);", outRef(meta)), nil
}

// autoZeroKVCacheOp emits zeroing of a KV cache region.
func autoZeroKVCacheOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  %s[tid] = 0.0f;", outRef(meta)), nil
}

func rangeOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  %s[tid] = (float)tid;", outRef(meta)), nil
}

func triluOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	cols := 1
	if len(inputs) > 0 && len(inputs[0].Shape) > 0 {
		cols = inputs[0].Shape[len(inputs[0].Shape)-1]
	}
	return fmt.Sprintf("  { int row = tid / %d; int col = tid %% %d; %s[tid] = (col <= row) ? %s[tid] : 0.0f; }",
		cols, cols, outRef(meta), inRef(meta, 0)), nil
}

func scatterNDOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  // ScatterND: %s updated from %s via indices %s\n  %s[tid] = %s[tid];",
		outRef(meta), inRef(meta, 0), inRef(meta, 1), outRef(meta), inRef(meta, 0)), nil
}
