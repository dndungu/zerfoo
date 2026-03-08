#include "textflag.h"

// func RoPEF32(out, in, cos, sin *float32, halfDim, headDim int)
//
// out    = +0(FP)
// in     = +8(FP)
// cos    = +16(FP)
// sin    = +24(FP)
// halfDim = +32(FP)
// headDim = +40(FP)
//
// Algorithm (per position):
//   For i in [0, halfDim):
//     out[i]         = in[i]*cos[i] - in[i+halfDim]*sin[i]
//     out[i+halfDim] = in[i+halfDim]*cos[i] + in[i]*sin[i]
//   For j in [2*halfDim, headDim):
//     out[j] = in[j]

TEXT ·RoPEF32(SB), NOSPLIT, $0-48
	MOVD out+0(FP), R0       // R0 = out ptr
	MOVD in+8(FP), R1        // R1 = in ptr
	MOVD cos+16(FP), R2      // R2 = cos ptr
	MOVD sin+24(FP), R3      // R3 = sin ptr
	MOVD halfDim+32(FP), R4  // R4 = halfDim
	MOVD headDim+40(FP), R5  // R5 = headDim

	// Compute byte offset for the second-half pointers
	// halfDim * 4 bytes per float32
	LSL $2, R4, R6           // R6 = halfDim * 4 (byte offset)

	// R7 = &in[halfDim], R8 = &out[halfDim]
	ADD R6, R1, R7           // R7 = in + halfDim*4  (second half input)
	ADD R6, R0, R8           // R8 = out + halfDim*4 (second half output)

	// Set up loop counters
	// R9 = number of 4-wide iterations
	// R10 = remaining scalar elements
	LSR $2, R4, R9           // R9 = halfDim / 4
	AND $3, R4, R10          // R10 = halfDim % 4

	// Pointers for the NEON loop:
	// R11 = in first half cursor
	// R12 = in second half cursor
	// R13 = out first half cursor
	// R14 = out second half cursor
	// R15 = cos cursor
	// R16 = sin cursor
	MOVD R1, R11
	MOVD R7, R12
	MOVD R0, R13
	MOVD R8, R14
	MOVD R2, R15
	MOVD R3, R16

	CBZ R9, tail_scalar

neon_loop:
	// Load cos[i:i+4] and sin[i:i+4]
	VLD1.P 16(R15), [V0.S4]  // V0 = cos[i..i+3]
	VLD1.P 16(R16), [V1.S4]  // V1 = sin[i..i+3]

	// Load in[i:i+4] (first half) and in[i+halfDim:i+halfDim+4] (second half)
	VLD1.P 16(R11), [V2.S4]  // V2 = in[i..i+3] (first half)
	VLD1.P 16(R12), [V3.S4]  // V3 = in[i+hd..i+hd+3] (second half)

	// Compute out[i] = in[i]*cos[i] - in[i+halfDim]*sin[i]
	// V4 = V2 * V0  (in_first * cos)
	// FMUL V4.4S, V2.4S, V0.4S  => encoding: 0x6E20DC00 | (Rm<<16) | (Rn<<5) | Rd
	// V2=2, V0=0, V4=4: 0x6E20DC00 | (0<<16) | (2<<5) | 4 = 0x6E20DC44
	WORD $0x6E20DC44          // FMUL V4.4S, V2.4S, V0.4S
	// V4 = V4 - V3*V1  (V4 -= in_second * sin)
	// FMLS V4.4S, V3.4S, V1.4S => encoding: 0x4EA0CC00 | (Rm<<16) | (Rn<<5) | Rd
	// V1=1, V3=3, V4=4: 0x4EA0CC00 | (1<<16) | (3<<5) | 4 = 0x4EB1CC64
	WORD $0x4EB1CC64          // FMLS V4.4S, V3.4S, V1.4S

	// Compute out[i+halfDim] = in[i+halfDim]*cos[i] + in[i]*sin[i]
	// V5 = V3 * V0  (in_second * cos)
	// FMUL V5.4S, V3.4S, V0.4S => 0x6E20DC00 | (0<<16) | (3<<5) | 5 = 0x6E20DC65
	WORD $0x6E20DC65          // FMUL V5.4S, V3.4S, V0.4S
	// V5 += V2 * V1  (V5 += in_first * sin)
	VFMLA V2.S4, V1.S4, V5.S4

	// Store results
	VST1.P [V4.S4], 16(R13)  // out[i..i+3]
	VST1.P [V5.S4], 16(R14)  // out[i+hd..i+hd+3]

	SUB $1, R9, R9
	CBNZ R9, neon_loop

tail_scalar:
	CBZ R10, passthrough

scalar_loop:
	// Load scalar values
	FMOVS (R15), F0           // cos[i]
	FMOVS (R16), F1           // sin[i]
	FMOVS (R11), F2           // in[i] (first half)
	FMOVS (R12), F3           // in[i+halfDim] (second half)

	// out[i] = in[i]*cos[i] - in[i+halfDim]*sin[i]
	FMULS F0, F2, F4          // F4 = in_first * cos
	FMULS F1, F3, F5          // F5 = in_second * sin
	FSUBS F5, F4, F4           // F4 = F4 - F5

	// out[i+halfDim] = in[i+halfDim]*cos[i] + in[i]*sin[i]
	FMULS F0, F3, F5          // F5 = in_second * cos
	FMULS F1, F2, F6          // F6 = in_first * sin
	FADDS F6, F5, F5           // F5 = F5 + F6

	FMOVS F4, (R13)           // store out[i]
	FMOVS F5, (R14)           // store out[i+halfDim]

	ADD $4, R11, R11
	ADD $4, R12, R12
	ADD $4, R13, R13
	ADD $4, R14, R14
	ADD $4, R15, R15
	ADD $4, R16, R16

	SUB $1, R10, R10
	CBNZ R10, scalar_loop

passthrough:
	// Copy in[2*halfDim .. headDim) to out[2*halfDim .. headDim)
	// R17 = headDim - 2*halfDim
	LSL $1, R4, R17           // R17 = 2 * halfDim
	SUB R17, R5, R17          // R17 = headDim - 2*halfDim
	CBZ R17, done

	// R11 = &in[2*halfDim], R13 = &out[2*halfDim]
	LSL $2, R4, R6            // R6 = halfDim * 4
	LSL $1, R6, R6            // R6 = halfDim * 8 = 2*halfDim*4
	ADD R6, R1, R11            // R11 = in + 2*halfDim*4
	ADD R6, R0, R13            // R13 = out + 2*halfDim*4

	// Try 4-wide NEON copy
	LSR $2, R17, R9           // R9 = remaining / 4
	AND $3, R17, R10          // R10 = remaining % 4
	CBZ R9, passthrough_scalar

passthrough_neon:
	VLD1.P 16(R11), [V0.S4]
	VST1.P [V0.S4], 16(R13)
	SUB $1, R9, R9
	CBNZ R9, passthrough_neon

passthrough_scalar:
	CBZ R10, done

passthrough_scalar_loop:
	FMOVS (R11), F0
	FMOVS F0, (R13)
	ADD $4, R11, R11
	ADD $4, R13, R13
	SUB $1, R10, R10
	CBNZ R10, passthrough_scalar_loop

done:
	RET
