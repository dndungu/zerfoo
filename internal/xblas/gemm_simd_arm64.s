#include "textflag.h"

// func vdotf32(a, b unsafe.Pointer, n int) float32
//
// Computes dot product of two float32 vectors using NEON FMLA.
TEXT ·vdotf32(SB), NOSPLIT, $0-28
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R1
	MOVD	n+16(FP), R2

	VEOR	V0.B16, V0.B16, V0.B16
	VEOR	V1.B16, V1.B16, V1.B16

	CMP	$8, R2
	BLT	dot_tail4

dot_loop8:
	VLD1.P	32(R0), [V2.S4, V3.S4]
	VLD1.P	32(R1), [V4.S4, V5.S4]
	VFMLA	V2.S4, V4.S4, V0.S4
	VFMLA	V3.S4, V5.S4, V1.S4
	SUB	$8, R2, R2
	CMP	$8, R2
	BGE	dot_loop8

	// fadd v0.4s, v0.4s, v1.4s
	WORD	$0x4e21d400

dot_tail4:
	CMP	$4, R2
	BLT	dot_tail1

	VLD1.P	16(R0), [V2.S4]
	VLD1.P	16(R1), [V3.S4]
	VFMLA	V2.S4, V3.S4, V0.S4
	SUB	$4, R2, R2

dot_tail1:
	CBZ	R2, dot_reduce

	VEOR	V2.B16, V2.B16, V2.B16

dot_scalar:
	FMOVS	(R0), F3
	FMOVS	(R1), F4
	FMULS	F3, F4, F3
	FADDS	F3, F2, F2
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	SUB	$1, R2, R2
	CBNZ	R2, dot_scalar

	FADDS	F2, F0, F0

dot_reduce:
	// faddp v0.4s, v0.4s, v0.4s
	WORD	$0x6e20d400
	// faddp v0.4s, v0.4s, v0.4s
	WORD	$0x6e20d400

	FMOVS	F0, ret+24(FP)
	RET

// func sgemmAccRowNeon(c, b unsafe.Pointer, aVal float32, n int)
//
// Computes c[j] += aVal * b[j] for j = 0..n-1 using NEON FMLA.
// Layout: c=0(FP), b=8(FP), aVal=16(FP), n=24(FP)
TEXT ·sgemmAccRowNeon(SB), NOSPLIT, $0-32
	MOVD	c+0(FP), R0
	MOVD	b+8(FP), R1
	FMOVS	aVal+16(FP), F6
	MOVD	n+24(FP), R2

	// Broadcast F6 to all 4 lanes of V6
	VDUP	V6.S[0], V6.S4

	CMP	$8, R2
	BLT	acc_tail4

acc_loop8:
	VLD1	(R0), [V0.S4, V1.S4]	// load c[0:8]
	VLD1	(R1), [V2.S4, V3.S4]	// load b[0:8]
	VFMLA	V6.S4, V2.S4, V0.S4	// c[0:4] += aVal * b[0:4]
	VFMLA	V6.S4, V3.S4, V1.S4	// c[4:8] += aVal * b[4:8]
	VST1	[V0.S4, V1.S4], (R0)
	ADD	$32, R0, R0
	ADD	$32, R1, R1
	SUB	$8, R2, R2
	CMP	$8, R2
	BGE	acc_loop8

acc_tail4:
	CMP	$4, R2
	BLT	acc_tail1

	VLD1	(R0), [V0.S4]
	VLD1	(R1), [V2.S4]
	VFMLA	V6.S4, V2.S4, V0.S4
	VST1	[V0.S4], (R0)
	ADD	$16, R0, R0
	ADD	$16, R1, R1
	SUB	$4, R2, R2

acc_tail1:
	CBZ	R2, acc_done

acc_scalar:
	FMOVS	(R1), F1
	FMULS	F6, F1, F1
	FMOVS	(R0), F2
	FADDS	F1, F2, F2
	FMOVS	F2, (R0)
	ADD	$4, R0, R0
	ADD	$4, R1, R1
	SUB	$1, R2, R2
	CBNZ	R2, acc_scalar

acc_done:
	RET
