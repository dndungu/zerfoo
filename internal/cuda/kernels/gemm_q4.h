/* Q4_0 dequant-GEMM kernel interface for Zerfoo's Q4Storage format.
 *
 * Block format (18 bytes per 32 values):
 *   - 2 bytes: float16 scale (little-endian IEEE 754 half)
 *   - 16 bytes: 32 x 4-bit unsigned values packed (2 per byte)
 *     Low nibble = first value, high nibble = second value.
 *     Dequant: float_val = (nibble - 8) * scale
 *
 * Computes: C[m,n] = sum_k( dequant(A[m,k]) * B[k,n] )
 * A is [M * num_blocks_per_row * 18] packed Q4_0 blocks.
 * B is [K, N] row-major FP32. C is [M, N] row-major FP32.
 */
#ifndef GEMM_Q4_H
#define GEMM_Q4_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* gemm_q4_f32 performs Q4_0 dequant-GEMM:
 *   C[m,n] = sum_k( dequant(A_q4[m,k]) * B[k,n] )
 *
 * A_q4:   device pointer to packed Q4_0 blocks for matrix A [M, K].
 *         M * ceil(K/32) blocks, each 18 bytes. Row-major block layout.
 * B:      device pointer to [K * N] float array (row-major).
 * C:      device pointer to [M * N] float array (row-major output).
 * M, K, N: matrix dimensions. K must be a multiple of 32.
 * stream: CUDA stream.
 */
cudaError_t gemm_q4_f32(
    const void* A_q4, const float* B, float* C,
    int M, int K, int N,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_Q4_H */
