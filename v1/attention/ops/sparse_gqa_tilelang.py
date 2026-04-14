# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse GQA attention via TileLang (KV-head / group tiled path).

This mirrors the math of ``sparse_gqa.sparse_gqa_attention`` (global cache-slot
gather + online softmax) but uses a **2-D grid** ``(num_tokens, num_kv_heads)``:
each program loads each top-*k* tile of K/V once and shares it across all
query heads in the GQA group—same grouping idea as the reference TileLang
``sparse_gqa_fwd`` sketch.

**TransGQA / ``transgqa_rope`` call shape** (same as ``transgqa_sparse_gqa_forward``)::

    torch.ops.vllm.transgqa_sparse_gqa_forward_tilelang(
        q_3d, k_3d, v_3d, topk_indices,
        layer_name,
        softmax_scale,
        num_local_heads,
        num_local_kv_heads,
        head_dim,
        block_n,  # e.g. 128; top-*k* is padded to a multiple of this
    )

Requires the optional ``tilelang`` package (``pip install tilelang``).

Constraints (checked in Python before compile):
    * ``head_dim`` must be a power of two (pad tensors externally if needed).
    * ``topk`` (after padding) must be divisible by ``block_I`` (default 32).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# TileLang-only path (``sparse_gqa_attention_tilelang``) must not import
# ``vllm.forward_context`` / ``vllm.platforms`` at module load: those pull in
# ``vllm._C``, which is absent in source checkouts until ``pip install -e .``.
# Unit tests import this module with only ``torch`` + ``tilelang`` available.
def _try_tilelang():
    try:
        import tilelang as tl_mod  # type: ignore[import-untyped]
        from tilelang import language as T_mod  # type: ignore[import-untyped]
        return True, tl_mod, T_mod
    except ImportError:
        return False, None, None


_HAS_TL, tilelang, T = _try_tilelang()
if not _HAS_TL:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]

_NEG_INIT = -(2**30)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_req_id_per_token(
    query_start_loc: torch.Tensor,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    num_reqs = query_start_loc.shape[0] - 1
    lens = query_start_loc[1:num_reqs + 1] - query_start_loc[:num_reqs]
    return torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device=device), lens)


def _pad_topk_indices(
    topk_indices: torch.Tensor,
    block_n: int,
) -> torch.Tensor:
    topk_k = topk_indices.shape[1]
    remainder = topk_k % block_n
    if remainder != 0:
        pad_size = block_n - remainder
        topk_indices = torch.nn.functional.pad(
            topk_indices, (0, pad_size), value=-1)
    return topk_indices


if _HAS_TL:
    assert T is not None and tilelang is not None

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _sparse_gqa_global_gather_fwd(
        heads: int,
        dim: int,
        topk: int,
        heads_kv: int,
        sm_scale: float,
        block_I: int,
        num_stages: int,
        threads: int,
    ):
        """Compile sparse GQA gather + online softmax (grouped by KV head)."""
        assert topk % block_I == 0
        G = heads // heads_kv
        padded_G = max(tilelang.math.next_power_of_2(G), 16)
        BI = block_I
        NI = (topk + BI - 1) // BI
        D = dim
        dtype = "bfloat16"
        accum_dtype = "float"
        index_dtype = "int32"

        seq_len = T.symbolic("seq_len")
        total_slots = T.symbolic("total_slots")

        q_shape = [seq_len, heads, D]
        k_shape = [total_slots, heads_kv, D]
        v_shape = [total_slots, heads_kv, D]
        indices_shape = [seq_len, 1, topk]
        o_shape = [seq_len, heads, D]

        H_per_block = padded_G

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore[arg-type]
            K: T.Tensor(k_shape, dtype),  # type: ignore[arg-type]
            V: T.Tensor(v_shape, dtype),  # type: ignore[arg-type]
            Indices: T.Tensor(indices_shape, index_dtype),  # type: ignore[arg-type]
            Output: T.Tensor(o_shape, dtype),  # type: ignore[arg-type]
        ):
            with T.Kernel(seq_len, heads_kv, threads=threads) as (bx, by):
                Q_shared = T.alloc_shared([H_per_block, D], dtype)
                K_shared = T.alloc_shared([BI, D], dtype)
                V_shared = T.alloc_shared([BI, D], dtype)
                mask = T.alloc_fragment([BI], "bool")

                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, _NEG_INIT)

                hkv_i = by
                H_start = hkv_i * G

                for h_i, d_i in T.Parallel(H_per_block, D):
                    if h_i < G:
                        Q_shared[h_i, d_i] = Q[bx, H_start + h_i, d_i]
                    else:
                        Q_shared[h_i, d_i] = 0

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        tok = Indices[bx, 0, i_i * BI + bi_i]
                        mask[bi_i] = tok >= 0

                    for bi_i, d_i in T.Parallel(BI, D):
                        tok = Indices[bx, 0, i_i * BI + bi_i]
                        slot = T.if_then_else(tok >= 0, tok, 0)
                        K_shared[bi_i, d_i] = K[slot, hkv_i, d_i]
                        V_shared[bi_i, d_i] = V[slot, hkv_i, d_i]

                    # Pre-fill acc_s with mask: 0 for valid, -inf for invalid/padding
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype))

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)

                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp(
                            (m_i_prev[h_i] - m_i[h_i]) * sm_scale)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp(
                            acc_s[h_i, bi_i] * sm_scale
                            - m_i[h_i] * sm_scale)

                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]

                    for h_i, d_i in T.Parallel(H_per_block, D):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(
                        S_shared,
                        V_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for h_i, d_i in T.Parallel(G, D):
                    acc_o[h_i, d_i] /= sumexp[h_i]

                for h_i, d_i in T.Parallel(G, D):
                    Output[bx, H_start + h_i, d_i] = T.cast(
                        acc_o[h_i, d_i], dtype)

        return main

else:
    _sparse_gqa_global_gather_fwd = None  # type: ignore[misc, assignment]


_kernel_cache: dict[tuple[Any, ...], Any] = {}


def _get_cached_kernel(
    *,
    heads: int,
    dim: int,
    topk: int,
    heads_kv: int,
    sm_scale: float,
    block_I: int,
    num_stages: int,
    threads: int,
) -> Any:
    key = (heads, dim, topk, heads_kv, sm_scale, block_I, num_stages, threads)
    if key not in _kernel_cache:
        assert _sparse_gqa_global_gather_fwd is not None
        _kernel_cache[key] = _sparse_gqa_global_gather_fwd(
            heads,
            dim,
            topk,
            heads_kv,
            sm_scale,
            block_I,
            num_stages,
            threads,
        )
    return _kernel_cache[key]


def sparse_gqa_attention_tilelang(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    global_indices: torch.Tensor,
    softmax_scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    block_I: int = 32,
    num_stages: int = 2,
    threads: int = 128,
) -> torch.Tensor:
    """Same contract as ``sparse_gqa.sparse_gqa_attention`` (TileLang backend).

    Args:
        q: [num_tokens, num_heads, head_dim] bf16
        key_cache / value_cache: [total_slots, num_kv_heads, head_dim] bf16
        global_indices: [num_tokens, topk_k] int32 (-1 invalid)
        softmax_scale: typically ``head_dim ** -0.5``
        block_I: tile size along top-k; ``topk_k`` must be divisible by this
            (pad with -1 before calling if needed).
    """
    if not _HAS_TL:
        raise RuntimeError(
            "sparse_gqa_attention_tilelang requires package `tilelang` "
            "(pip install tilelang).")

    num_tokens = q.shape[0]
    topk_k = global_indices.shape[1]
    if topk_k % block_I != 0:
        raise ValueError(
            f"topk_k ({topk_k}) must be divisible by block_I ({block_I}); "
            "pad global_indices with -1 columns first.")

    hd = head_dim
    if hd != _next_pow2(hd):
        raise ValueError(
            f"head_dim ({hd}) must be a power of 2 for the TileLang kernel.")

    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
    if key_cache.dtype != torch.bfloat16:
        key_cache = key_cache.to(torch.bfloat16)
    if value_cache.dtype != torch.bfloat16:
        value_cache = value_cache.to(torch.bfloat16)

    indices_3d = global_indices.unsqueeze(1).contiguous().to(torch.int32)

    out = torch.empty(
        num_tokens, num_heads, hd, dtype=torch.bfloat16, device=q.device)

    kernel = _get_cached_kernel(
        heads=num_heads,
        dim=hd,
        topk=topk_k,
        heads_kv=num_kv_heads,
        sm_scale=float(softmax_scale),
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    kernel(
        q.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        indices_3d,
        out,
    )
    return out.to(q.dtype) if q.dtype != torch.bfloat16 else out


def transgqa_sparse_gqa_forward_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_indices: torch.Tensor,
    layer_name: str,
    softmax_scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_n: int,
) -> torch.Tensor:
    """Drop-in sibling of ``transgqa_sparse_gqa_forward`` using TileLang.

    Same argument list as ``torch.ops.vllm.transgqa_sparse_gqa_forward`` so
    ``transgqa_rope.py`` can switch op name only, e.g.::

        torch.ops.vllm.transgqa_sparse_gqa_forward_tilelang(...)
    """
    if not _HAS_TL:
        raise RuntimeError(
            "transgqa_sparse_gqa_forward_tilelang requires package `tilelang`.")

    from vllm.forward_context import get_forward_context
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        triton_convert_req_index_to_global_index)

    forward_ctx = get_forward_context()
    attn_metadata = forward_ctx.attn_metadata
    if not isinstance(attn_metadata, dict):
        return torch.empty(
            q.shape[0], num_heads * head_dim,
            dtype=q.dtype, device=q.device)

    attn_meta = attn_metadata[layer_name]
    attn_layer = forward_ctx.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_ctx.virtual_engine]

    key_cache, value_cache = kv_cache.unbind(0)

    ops.reshape_and_cache_flash(
        k, v, key_cache, value_cache,
        attn_meta.slot_mapping,
        attn_layer.kv_cache_dtype,
        attn_layer._k_scale,
        attn_layer._v_scale,
    )

    _, block_size_cache, nkv, hd = key_cache.shape
    key_flat = key_cache.reshape(-1, nkv, hd)
    value_flat = value_cache.reshape(-1, nkv, hd)

    num_actual = attn_meta.num_actual_tokens
    req_id = _build_req_id_per_token(
        attn_meta.query_start_loc, num_actual, q.device)
    block_table = attn_meta.block_table

    ti = topk_indices[:num_actual].to(torch.int32)
    ti = _pad_topk_indices(ti, block_n)

    global_indices = triton_convert_req_index_to_global_index(
        req_id, block_table, ti,
        BLOCK_SIZE=block_size_cache,
        NUM_TOPK_TOKENS=ti.shape[1],
        BLOCK_N=block_n,
    )

    output = sparse_gqa_attention_tilelang(
        q[:num_actual], key_flat, value_flat, global_indices,
        softmax_scale, num_heads, num_kv_heads, head_dim,
        block_I=32,
    )
    return output.view(-1, num_heads * head_dim)


def transgqa_sparse_gqa_forward_tilelang_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_indices: torch.Tensor,
    layer_name: str,
    softmax_scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_n: int,
) -> torch.Tensor:
    return torch.empty(
        q.shape[0], num_heads * head_dim,
        dtype=q.dtype, device=q.device)


def _register_transgqa_sparse_gqa_forward_tilelang_op() -> None:
    """Register ``torch.ops.vllm.transgqa_sparse_gqa_forward_tilelang`` (needs full vLLM)."""
    from vllm.platforms import current_platform
    from vllm.utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="transgqa_sparse_gqa_forward_tilelang",
        op_func=transgqa_sparse_gqa_forward_tilelang,
        mutates_args=[],
        fake_impl=transgqa_sparse_gqa_forward_tilelang_fake,
        dispatch_key=current_platform.dispatch_key,
    )


try:
    _register_transgqa_sparse_gqa_forward_tilelang_op()
except Exception as e:
    logger.debug(
        "transgqa_sparse_gqa_forward_tilelang op not registered: %s", e)
