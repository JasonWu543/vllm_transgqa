# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for fused sparse GQA attention.

Given per-token top-k global cache slot indices, this kernel computes
attention by gathering K/V on-the-fly from paged KV cache, avoiding
materializing the full gathered KV tensor. Uses online softmax for
numerical stability.

Grid: (num_tokens, num_heads)
Each program processes one (token, head) pair and iterates over the
top-k positions in tiles of BLOCK_TOPK.

For a TileLang-style variant that uses grid (num_tokens, num_kv_heads) and
reuses K/V loads within each GQA group, see ``sparse_gqa_kv_grouped.py``.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _sparse_gqa_attention_kernel(
    Q,       # [num_tokens, num_heads, head_dim]
    KC,      # [total_slots, num_kv_heads, head_dim] (flattened paged K cache)
    VC,      # [total_slots, num_kv_heads, head_dim] (flattened paged V cache)
    O,       # [num_tokens, num_heads, head_dim]
    IDX,     # [num_tokens, topk_k] int32 global cache slot indices
    softmax_scale,
    topk_k,
    head_dim,
    gqa_group_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    stride_qt, stride_qh, stride_qd,
    stride_ks, stride_kh, stride_kd,
    stride_vs, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_it, stride_ik,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // gqa_group_size

    d = tl.arange(0, HEAD_DIM)
    d_mask = d < head_dim

    q = tl.load(
        Q + pid_t * stride_qt + pid_h * stride_qh + d * stride_qd,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    m = tl.zeros([1], dtype=tl.float32) + float('-inf')
    l = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start in range(0, topk_k, BLOCK_TOPK):
        offs = start + tl.arange(0, BLOCK_TOPK)
        k_mask = offs < topk_k

        idx = tl.load(
            IDX + pid_t * stride_it + offs * stride_ik,
            mask=k_mask, other=-1,
        )
        valid = (idx >= 0) & k_mask

        kp = (idx[:, None] * stride_ks
              + kv_h * stride_kh
              + d[None, :] * stride_kd)
        kt = tl.load(
            KC + kp,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        s = tl.sum(kt * q[None, :], axis=1) * softmax_scale
        s = tl.where(valid, s, float('-inf'))

        mt = tl.max(s)
        mn = tl.maximum(m, mt)
        # If every position in this tile is invalid, mt is -inf and m may be
        # -inf on the first tile; exp(s - mn) then hits (-inf) - (-inf) -> NaN.
        # Skip the softmax update for this tile (same as reference: no mass).
        tile_skip = mt <= float('-inf')
        es = tl.where(
            tile_skip,
            0.0,
            tl.where(valid, tl.exp(s - mn), 0.0),
        )
        lt = tl.sum(es)
        al = tl.where(tile_skip, 1.0, tl.exp(m - mn))
        ln = l * al + lt

        vp = (idx[:, None] * stride_vs
              + kv_h * stride_vh
              + d[None, :] * stride_vd)
        vt = tl.load(
            VC + vp,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc = acc * al + tl.sum(es[:, None] * vt, axis=0)
        m = mn
        l = ln

    out = acc / tl.maximum(l, 1e-6)
    tl.store(
        O + pid_t * stride_ot + pid_h * stride_oh + d * stride_od,
        out.to(tl.bfloat16),
        mask=d_mask,
    )


def sparse_gqa_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    global_indices: torch.Tensor,
    softmax_scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute sparse GQA attention using global cache slot indices.

    Args:
        q: [num_tokens, num_heads, head_dim] in bf16
        key_cache: [total_slots, num_kv_heads, head_dim] flattened paged K
        value_cache: [total_slots, num_kv_heads, head_dim] flattened paged V
        global_indices: [num_tokens, topk_k] int32, -1 = invalid
        softmax_scale: 1/sqrt(head_dim) or similar
        num_heads: query head count
        num_kv_heads: KV head count (for GQA grouping)
        head_dim: dimension per head

    Returns:
        [num_tokens, num_heads, head_dim] in bf16
    """
    num_tokens = q.shape[0]
    topk_k = global_indices.shape[1]
    gqa_group_size = num_heads // num_kv_heads

    output = torch.empty(
        num_tokens, num_heads, head_dim,
        dtype=q.dtype, device=q.device,
    )

    HEAD_DIM = triton.next_power_of_2(head_dim)
    BLOCK_TOPK = min(128, triton.next_power_of_2(topk_k))

    grid = (num_tokens, num_heads)

    _sparse_gqa_attention_kernel[grid](
        q, key_cache, value_cache, output, global_indices,
        softmax_scale,
        topk_k,
        head_dim,
        gqa_group_size,
        HEAD_DIM,
        BLOCK_TOPK,
        q.stride(0), q.stride(1), q.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        global_indices.stride(0), global_indices.stride(1),
    )

    return output
