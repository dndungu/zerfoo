#include "textflag.h"

// Assembly trampolines for dynamically imported C library functions
// on Linux arm64. Each trampoline is a branch to the symbol resolved
// by //go:cgo_import_dynamic at load time.

TEXT ·libc_dlopen_trampoline(SB),NOSPLIT,$0-0
	JMP libc_dlopen(SB)

TEXT ·libc_dlsym_trampoline(SB),NOSPLIT,$0-0
	JMP libc_dlsym(SB)

TEXT ·libc_dlclose_trampoline(SB),NOSPLIT,$0-0
	JMP libc_dlclose(SB)

TEXT ·libc_dlerror_trampoline(SB),NOSPLIT,$0-0
	JMP libc_dlerror(SB)

// ccallTrampoline is called by asmcgocall(ccallTrampoline, &ccallArgs).
// It runs on the system stack (g0). R0 points to a ccallArgs struct:
//
//   struct ccallArgs {
//       fn   uintptr    // offset 0
//       args [12]uintptr // offset 8 (args[0] at 8, args[1] at 16, ... args[11] at 96)
//       ret  uintptr    // offset 104
//   }
//
// AAPCS64 calling convention: R0-R7 hold the first 8 args, stack holds the rest.
// R9 is used as a scratch register (caller-saved).
// R19 is callee-saved in AAPCS64, safe to use for saving the args pointer.
TEXT ·ccallTrampoline(SB),NOSPLIT,$32
	// Save args struct pointer in callee-saved register
	MOVD R0, R19

	// Load C function pointer
	MOVD 0(R19), R9

	// Load register arguments (AAPCS64: R0-R7)
	MOVD 8(R19), R0
	MOVD 16(R19), R1
	MOVD 24(R19), R2
	MOVD 32(R19), R3
	MOVD 40(R19), R4
	MOVD 48(R19), R5
	MOVD 56(R19), R6
	MOVD 64(R19), R7

	// Load stack arguments (args[8] through args[11])
	// AAPCS64: stack args start at SP+0, growing upward.
	MOVD 72(R19), R10
	MOVD R10, 0(RSP)
	MOVD 80(R19), R10
	MOVD R10, 8(RSP)
	MOVD 88(R19), R10
	MOVD R10, 16(RSP)
	MOVD 96(R19), R10
	MOVD R10, 24(RSP)

	// Call the C function (BLR Xn in ARM64 = CALL (Rn) in Go asm)
	CALL (R9)

	// Store return value
	MOVD R0, 104(R19)

	RET
