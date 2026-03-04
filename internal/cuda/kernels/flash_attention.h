/* Flash attention forward kernel interface.
 * Computes: O = softmax(Q * K^T / sqrt(head_dim)) * V
 * with optional causal masking.
 *
 * Layout: All tensors are [batch, heads, seq_len, head_dim] in row-major order.
 */
#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* flash_attention_forward_f32 computes scaled dot-product attention in a single
 * fused pass using tiled computation with shared memory staging.
 *
 * Q, K, V: device pointers to [batch * heads * seq_len * head_dim] float32 arrays.
 * O:       device pointer to output [batch * heads * seq_len * head_dim].
 * batch:   number of sequences in the batch.
 * heads:   number of attention heads.
 * seq_len: sequence length (same for Q, K, V).
 * head_dim: dimension per head.
 * causal:  if nonzero, apply causal (upper-triangular) mask.
 * stream:  CUDA stream for async execution.
 */
cudaError_t flash_attention_forward_f32(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int heads, int seq_len, int head_dim,
    int causal, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_ATTENTION_H */
