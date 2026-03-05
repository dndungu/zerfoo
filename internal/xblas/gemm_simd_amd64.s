#include "textflag.h"

// func vdotf32(a, b unsafe.Pointer, n int) float32
//
// Computes dot product of two float32 vectors using AVX2 FMA.
TEXT ·vdotf32(SB), NOSPLIT, $0-28
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	MOVQ	n+16(FP), CX

	VXORPS	Y0, Y0, Y0
	VXORPS	Y1, Y1, Y1

	CMPQ	CX, $16
	JLT	dot_tail8

dot_loop16:
	VMOVUPS	(SI), Y2
	VMOVUPS	32(SI), Y3
	VMOVUPS	(DI), Y4
	VMOVUPS	32(DI), Y5
	VFMADD231PS	Y2, Y4, Y0
	VFMADD231PS	Y3, Y5, Y1
	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$16, CX
	CMPQ	CX, $16
	JGE	dot_loop16

	VADDPS	Y1, Y0, Y0

dot_tail8:
	CMPQ	CX, $8
	JLT	dot_tail4

	VMOVUPS	(SI), Y2
	VMOVUPS	(DI), Y3
	VFMADD231PS	Y2, Y3, Y0
	ADDQ	$32, SI
	ADDQ	$32, DI
	SUBQ	$8, CX

dot_tail4:
	VEXTRACTF128	$1, Y0, X1
	VADDPS	X1, X0, X0

	CMPQ	CX, $4
	JLT	dot_tail1

	VMOVUPS	(SI), X2
	VMOVUPS	(DI), X3
	VFMADD231PS	X2, X3, X0
	ADDQ	$16, SI
	ADDQ	$16, DI
	SUBQ	$4, CX

dot_tail1:
	VHADDPS	X0, X0, X0
	VHADDPS	X0, X0, X0

	CMPQ	CX, $0
	JEQ	dot_done

dot_scalar:
	MOVSS	(SI), X1
	MOVSS	(DI), X2
	MULSS	X1, X2
	ADDSS	X2, X0
	ADDQ	$4, SI
	ADDQ	$4, DI
	SUBQ	$1, CX
	JNZ	dot_scalar

dot_done:
	VZEROUPPER
	MOVSS	X0, ret+24(FP)
	RET

// func sgemmAccRow(c, b unsafe.Pointer, aVal float32, n int)
//
// Computes c[j] += aVal * b[j] for j = 0..n-1 using AVX2 FMA.
// Layout: c=0(FP), b=8(FP), aVal=16(FP), n=24(FP)
// Total frame args: 32 bytes.
TEXT ·sgemmAccRow(SB), NOSPLIT, $0-32
	MOVQ	c+0(FP), DI
	MOVQ	b+8(FP), SI
	MOVSS	aVal+16(FP), X6
	MOVQ	n+24(FP), CX

	// Broadcast aVal to all 8 lanes of Y6
	VBROADCASTSS	X6, Y6

	CMPQ	CX, $16
	JLT	acc_tail8

acc_loop16:
	VMOVUPS	(DI), Y2
	VMOVUPS	32(DI), Y3
	VMOVUPS	(SI), Y4
	VMOVUPS	32(SI), Y5
	VFMADD231PS	Y6, Y4, Y2
	VFMADD231PS	Y6, Y5, Y3
	VMOVUPS	Y2, (DI)
	VMOVUPS	Y3, 32(DI)
	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$16, CX
	CMPQ	CX, $16
	JGE	acc_loop16

acc_tail8:
	CMPQ	CX, $8
	JLT	acc_tail4

	VMOVUPS	(DI), Y2
	VMOVUPS	(SI), Y3
	VFMADD231PS	Y6, Y3, Y2
	VMOVUPS	Y2, (DI)
	ADDQ	$32, SI
	ADDQ	$32, DI
	SUBQ	$8, CX

acc_tail4:
	CMPQ	CX, $4
	JLT	acc_tail1

	VMOVUPS	(DI), X2
	VMOVUPS	(SI), X3
	VFMADD231PS	X6, X3, X2
	VMOVUPS	X2, (DI)
	ADDQ	$16, SI
	ADDQ	$16, DI
	SUBQ	$4, CX

acc_tail1:
	CMPQ	CX, $0
	JEQ	acc_done

acc_scalar:
	MOVSS	(SI), X1
	MULSS	X6, X1
	ADDSS	(DI), X1
	MOVSS	X1, (DI)
	ADDQ	$4, SI
	ADDQ	$4, DI
	SUBQ	$1, CX
	JNZ	acc_scalar

acc_done:
	VZEROUPPER
	RET
