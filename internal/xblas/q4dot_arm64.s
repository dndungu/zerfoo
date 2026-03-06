#include "textflag.h"

// func q4DotBlockSIMD(packed *byte, scale float32, x *float32) float32
//
// Computes the dot product of one Q4 block (16 packed bytes = 32 nibbles)
// with 32 float32 activation values, using NEON.
//
// Algorithm:
//   1. Load 16 packed bytes, extract 32 nibbles (AND 0x0F, USHR 4)
//   2. Interleave low/high nibbles to match activation layout
//   3. Widen uint8 → uint16 → subtract 8 → int16 → int32 → float32
//   4. FMLA with activation float32x4 vectors
//   5. Horizontal reduce, multiply by scale
//
// Layout: packed=0(FP), scale=8(FP), x=16(FP), ret=24(FP)
TEXT ·q4DotBlockSIMD(SB), NOSPLIT, $0-28
	MOVD	packed+0(FP), R0
	FMOVS	scale+8(FP), F7
	MOVD	x+16(FP), R1

	// Load 16 packed bytes.
	VLD1	(R0), [V0.B16]

	// Create 0x0F mask: load 64-bit constant, duplicate.
	MOVD	$0x0F0F0F0F0F0F0F0F, R3
	VMOV	R3, V16.D[0]
	VMOV	R3, V16.D[1]

	// Extract nibbles.
	VAND	V0.B16, V16.B16, V1.B16		// V1 = low nibbles (16 × uint8)
	// USHR V0.16B, #4 → V2.16B (high nibbles)
	WORD	$0x6F0C0402

	// Interleave: [lo0, hi0, lo1, hi1, ...]
	// ZIP1 V3.16B, V1.16B, V2.16B (positions 0-15)
	WORD	$0x4E023823
	// ZIP2 V4.16B, V1.16B, V2.16B (positions 16-31)
	WORD	$0x4E027824

	// Create uint16 vector of 8s for subtraction.
	MOVD	$0x0008000800080008, R4
	VMOV	R4, V17.D[0]
	VMOV	R4, V17.D[1]

	// Zero accumulators.
	VEOR	V30.B16, V30.B16, V30.B16
	VEOR	V31.B16, V31.B16, V31.B16

	// --- Process V3: positions 0-15 ---

	// USHLL V5.8H, V3.8B, #0 (widen lower 8 bytes to uint16)
	WORD	$0x2F08A465
	// USHLL2 V6.8H, V3.16B, #0 (widen upper 8 bytes to uint16)
	WORD	$0x6F08A466

	// SUB V5.8H, V5.8H, V17.8H (subtract 8, now int16)
	WORD	$0x6E7184A5
	// SUB V6.8H, V6.8H, V17.8H
	WORD	$0x6E7184C6

	// SSHLL V10.4S, V5.4H, #0 (widen lower 4 int16 to int32)
	WORD	$0x0F10A4AA
	// SSHLL2 V11.4S, V5.8H, #0 (widen upper 4 int16 to int32)
	WORD	$0x4F10A4AB
	// SSHLL V12.4S, V6.4H, #0
	WORD	$0x0F10A4CC
	// SSHLL2 V13.4S, V6.8H, #0
	WORD	$0x4F10A4CD

	// SCVTF V10.4S, V10.4S (int32 → float32)
	WORD	$0x4E21D94A
	// SCVTF V11.4S, V11.4S
	WORD	$0x4E21D96B
	// SCVTF V12.4S, V12.4S
	WORD	$0x4E21D98C
	// SCVTF V13.4S, V13.4S
	WORD	$0x4E21D9AD

	// Load activations x[0:16] (4 × Q-register = 64 bytes).
	VLD1.P	64(R1), [V20.S4, V21.S4, V22.S4, V23.S4]

	// FMLA: acc += dequant × activation
	VFMLA	V10.S4, V20.S4, V30.S4		// positions 0-3
	VFMLA	V11.S4, V21.S4, V31.S4		// positions 4-7
	VFMLA	V12.S4, V22.S4, V30.S4		// positions 8-11
	VFMLA	V13.S4, V23.S4, V31.S4		// positions 12-15

	// --- Process V4: positions 16-31 ---

	// USHLL V5.8H, V4.8B, #0
	WORD	$0x2F08A485
	// USHLL2 V6.8H, V4.16B, #0
	WORD	$0x6F08A486

	// SUB V5.8H, V5.8H, V17.8H
	WORD	$0x6E7184A5
	// SUB V6.8H, V6.8H, V17.8H
	WORD	$0x6E7184C6

	// SSHLL V10.4S, V5.4H, #0
	WORD	$0x0F10A4AA
	// SSHLL2 V11.4S, V5.8H, #0
	WORD	$0x4F10A4AB
	// SSHLL V12.4S, V6.4H, #0
	WORD	$0x0F10A4CC
	// SSHLL2 V13.4S, V6.8H, #0
	WORD	$0x4F10A4CD

	// SCVTF
	WORD	$0x4E21D94A
	WORD	$0x4E21D96B
	WORD	$0x4E21D98C
	WORD	$0x4E21D9AD

	// Load activations x[16:32].
	VLD1	(R1), [V20.S4, V21.S4, V22.S4, V23.S4]

	// FMLA: accumulate
	VFMLA	V10.S4, V20.S4, V30.S4		// positions 16-19
	VFMLA	V11.S4, V21.S4, V31.S4		// positions 20-23
	VFMLA	V12.S4, V22.S4, V30.S4		// positions 24-27
	VFMLA	V13.S4, V23.S4, V31.S4		// positions 28-31

	// --- Horizontal reduction ---

	// FADD V30.4S, V30.4S, V31.4S
	WORD	$0x4E3FD7DE
	// FADDP V30.4S, V30.4S, V30.4S
	WORD	$0x6E3ED7DE
	// FADDP V30.4S, V30.4S, V30.4S (now V30.S[0] = total sum)
	WORD	$0x6E3ED7DE

	// Multiply by scale: result = sum * scale
	FMULS	F30, F7, F0
	FMOVS	F0, ret+24(FP)
	RET
