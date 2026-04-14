# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse GQA attention — KV-head grouped kernel (TileLang-style).

This is a **copy for comparison** with ``sparse_gqa.py``. The default vLLM
entry point remains ``sparse_gqa.sparse_gqa_attention``.

Differences vs. ``sparse_gqa.py``:
- Grid ``(num_tokens, num_kv_heads)`` instead of ``(num_tokens, num_heads)``.
- Each program loads each top-k K/V tile once and reuses it for all query
  heads in the GQA group.
- Running max initialized to ``-(2**30)`` instead of ``-inf``.

To try this implementation in isolation::

    from vllm.v1.attention.ops.sparse_gqa_kv_grouped import sparse_gqa_attention
"""

import torch

from vllm.triton_utils import tl, triton

_NEG_INF_REPR = -(2**30)


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
    kv_h = tl.program_id(1)

    d = tl.arange(0, HEAD_DIM)
    d_mask = d < head_dim

    m = tl.full([gqa_group_size], _NEG_INF_REPR, dtype=tl.float32)
    l = tl.zeros([gqa_group_size], dtype=tl.float32)
    acc = tl.zeros([gqa_group_size, HEAD_DIM], dtype=tl.float32)

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

        vp = (idx[:, None] * stride_vs
              + kv_h * stride_vh
              + d[None, :] * stride_vd)
        vt = tl.load(
            VC + vp,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for g in tl.static_range(gqa_group_size):
            qh = kv_h * gqa_group_size + g
            q = tl.load(
                Q + pid_t * stride_qt + qh * stride_qh + d * stride_qd,
                mask=d_mask, other=0.0,
            ).to(tl.float32)

            s = tl.sum(kt * q[None, :], axis=1) * softmax_scale
            s = tl.where(valid, s, float('-inf'))

            mt = tl.max(s)
            row_mask = tl.arange(0, gqa_group_size) == g
            mg = tl.sum(m * row_mask.to(tl.float32))
            mn = tl.maximum(mg, mt)

            al = tl.exp(mg - mn)
            es = tl.exp(s - mn)
            es = tl.where(valid, es, 0.0)
            lt = tl.sum(es)

            lg = tl.sum(l * row_mask.to(tl.float32))
            ln = lg * al + lt

            contrib = tl.sum(es[:, None] * vt, axis=0)

            acc_g = tl.sum(acc * row_mask[:, None].to(tl.float32), axis=0)
            new_row = acc_g * al + contrib

            m = tl.where(row_mask, mn, m)
            l = tl.where(row_mask, ln, l)

            row_mask_2d = row_mask[:, None]
            new_acc_tile = tl.broadcast_to(new_row[None, :],
                                           (gqa_group_size, HEAD_DIM))
            acc = tl.where(row_mask_2d, new_acc_tile, acc)

    inv_l = tl.maximum(l, 1e-6)
    inv_l = 1.0 / inv_l
    out_heads = acc * inv_l[:, None]

    for g in tl.static_range(gqa_group_size):
        qh = kv_h * gqa_group_size + g
        out_row = out_heads[g, :]
        tl.store(
            O + pid_t * stride_ot + qh * stride_oh + d * stride_od,
            out_row.to(tl.bfloat16),
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
    """Same signature as ``sparse_gqa.sparse_gqa_attention`` (KV-grouped impl)."""
    num_tokens = q.shape[0]
    topk_k = global_indices.shape[1]
    gqa_group_size = num_heads // num_kv_heads

    output = torch.empty(
        num_tokens, num_heads, head_dim,
        dtype=q.dtype, device=q.device,
    )

    HEAD_DIM = triton.next_power_of_2(head_dim)
    BLOCK_TOPK = min(128, triton.next_power_of_2(topk_k))

    grid = (num_tokens, num_kv_heads)

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
