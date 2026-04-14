# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from Qwen3 model implementation with sparse attention indexer
# inspired by hisa.py (DeepseekV3.2 hierarchical sparse attention)
#
# This file implements a two-step sparse GQA attention mechanism:
# Step 1: An independent MLA-based Indexer selects top-k important key tokens
# Step 2: The main GQA attention only attends to those selected tokens

"""Inference-only Qwen3 model with TransGQA sparse attention."""
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention import Attention
from vllm.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils import cdiv, direct_register_custom_op
from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
from vllm.utils.custom_ops import (fp8_hierarchy_mqa_logits,
                                   fp8_hierarchy_paged_mqa_logits)
from vllm.v1.attention.backends.mla.indexer import (DeepseekV32IndexerBackend,
                                                    DeepseekV32IndexerMetadata)

from .interfaces import SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

logger = init_logger(__name__)


# =============================================================================
# MLP (same as Qwen3)
# =============================================================================

class TransGQAMLP(nn.Module):
    """Standard Qwen3 MLP with SwiGLU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# =============================================================================
# Indexer FP8 K Cache
# =============================================================================

class TransGQAIndexerCache(DeepseekV32IndexerCache):
    """FP8 quantized K cache for the sparse attention indexer.

    Inherits from DeepseekV32IndexerCache so that gpu_model_runner's
    get_kv_cache_spec() discovery (which uses isinstance check against
    DeepseekV32IndexerCache) can find this layer automatically.
    """
    pass


# =============================================================================
# Gather indexer K cache (CPU reference)
# =============================================================================

@torch.inference_mode()
def cp_gather_indexer_k_quant_cache(
    kv_cache,      # [num_blocks, block_size, head_dim + 1]
    dst_value,     # [cu_seq_lens[-1], head_dim]
    dst_scale,     # [cu_seq_lens[-1], 4]
    block_table,   # [batch_size, num_blocks]
    cu_seq_lens,   # [batch_size + 1, ]
    batch_size,
):
    num_blocks, block_size, _ = kv_cache.shape
    head_dim = dst_value.shape[-1]
    kv_cache = kv_cache.view(num_blocks, -1)

    expected_value = []
    expected_scale = []
    for b in range(batch_size):
        s = cu_seq_lens[b + 1] - cu_seq_lens[b]
        if s == 0:
            continue
        tot = cdiv(s, block_size)
        blocks = block_table[b, :tot]

        full_block = torch.arange(tot - 1,
                                  device=kv_cache.device,
                                  dtype=torch.int32)
        non_remaining_value = kv_cache[blocks[full_block], :block_size *
                                       head_dim].view(-1, head_dim)
        non_remaining_scale = kv_cache[blocks[full_block],
                                       block_size * head_dim:].view(-1, 4)

        remaining = s - (tot - 1) * block_size

        value = torch.cat([
            non_remaining_value,
            kv_cache[blocks[-1], :remaining * head_dim].view(-1, head_dim)
        ], dim=0)
        scale = torch.cat([
            non_remaining_scale,
            kv_cache[blocks[-1], block_size * head_dim:block_size * head_dim +
                     remaining * 4].view(-1, 4)
        ], dim=0)

        expected_value.append(value)
        expected_scale.append(scale)

    gather_value = torch.cat(expected_value, dim=0).view(-1, head_dim)
    gather_scale = torch.cat(expected_scale, dim=0).view(-1, 4)
    gather_value = gather_value.view(torch.float8_e4m3fn)
    gather_scale = gather_scale.view(torch.float32)
    dst_value.copy_(gather_value)
    dst_scale.copy_(gather_scale)


# =============================================================================
# Flat sparse attention indexer (top-k over all tokens)
# =============================================================================

def transgqa_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: Optional[torch.Tensor],
) -> torch.Tensor:

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return transgqa_sparse_attn_indexer_fake(
            hidden_states, k_cache_prefix, kv_cache, q_fp8, k, weights,
            quant_block_size, scale_fmt, topk_tokens, head_dim,
            max_model_len, total_seq_lens, topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    ops.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt,
    )

    topk_indices_buffer[:hidden_states.shape[0]] = -1

    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        for chunk in prefill_metadata.chunks:
            k_fp8 = torch.empty([chunk.total_seq_lens, head_dim],
                                device=k.device,
                                dtype=torch.float8_e4m3fn)
            k_scale = torch.empty([chunk.total_seq_lens, 1],
                                  device=k.device,
                                  dtype=torch.float32)
            cp_gather_indexer_k_quant_cache(
                kv_cache, k_fp8, k_scale, chunk.block_table,
                chunk.cu_seq_lens, chunk.num_reqs,
            )
            logits = fp8_mqa_logits(
                q_fp8[chunk.token_start:chunk.token_end],
                (k_fp8, k_scale),
                weights[chunk.token_start:chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            topk_indices = logits.topk(
                min(topk_tokens, logits.shape[-1]), dim=-1)[1]
            topk_indices -= chunk.cu_seqlen_ks[:, None]
            mask_lo = topk_indices >= 0
            mask_hi = topk_indices - (
                chunk.cu_seqlen_ke - chunk.cu_seqlen_ks)[:, None] < 0
            mask = torch.full_like(topk_indices, False, dtype=torch.bool,
                                   device=topk_indices.device)
            mask = mask_lo & mask_hi
            topk_indices = topk_indices.masked_fill(~mask, -1)
            topk_indices_buffer[
                chunk.token_start:chunk.token_end,
                :topk_indices.shape[-1]] = topk_indices.to(dtype=torch.int32)

    if has_decode:
        decode_metadata = attn_metadata.decode
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens)
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:])
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n
        logits = fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )
        current_device = padded_q_fp8_decode_tokens.device
        padded_num_tokens = batch_size * next_n
        positions = torch.arange(
            max_model_len, device=current_device
        ).unsqueeze(0).expand(batch_size * next_n, -1)
        row_indices = torch.arange(
            padded_num_tokens, device=current_device) // next_n
        next_n_offset = torch.arange(
            padded_num_tokens, device=current_device) % next_n
        index_end_pos = (
            decode_metadata.seq_lens[row_indices] - next_n + next_n_offset
        ).unsqueeze(1)
        mask = positions <= index_end_pos
        logits = logits.masked_fill(~mask, float('-inf'))
        topk_indices = logits.topk(
            topk_tokens, dim=-1)[1].to(torch.int32)
        topk_indices[topk_indices > index_end_pos] = -1
        if decode_metadata.requires_padding:
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(
                    batch_size, -1, topk_indices.shape[-1]),
                decode_lens)
        topk_indices_buffer[
            :num_decode_tokens,
            :topk_indices.shape[-1]] = topk_indices.to(dtype=torch.int32)

    return topk_indices_buffer


def transgqa_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: Optional[torch.Tensor],
) -> torch.Tensor:
    # profile run: allocate max possible flattened_kv for memory estimation
    _flattened_kv = torch.empty([total_seq_lens, head_dim + 4],
                                device=k.device,
                                dtype=torch.uint8)
    _k_fp8 = _flattened_kv[..., :head_dim].view(
        torch.float8_e4m3fn).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(
        torch.float32).contiguous()
    return topk_indices_buffer


direct_register_custom_op(
    op_name="transgqa_sparse_attn_indexer",
    op_func=transgqa_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=transgqa_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


# =============================================================================
# Hierarchical sparse attention indexer (block-level then token-level)
# =============================================================================

def transgqa_hierarchy_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    k_block_size: int,
    block_topk: int,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: Optional[torch.Tensor],
) -> torch.Tensor:

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return transgqa_sparse_attn_indexer_fake(
            hidden_states, k_cache_prefix, kv_cache, q_fp8, k, weights,
            quant_block_size, scale_fmt, topk_tokens, head_dim,
            max_model_len, total_seq_lens, topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    ops.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt,
    )

    topk_indices_buffer[:hidden_states.shape[0]] = -1

    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        for chunk in prefill_metadata.chunks:
            k_fp8 = torch.empty([chunk.total_seq_lens, head_dim],
                                device=k.device,
                                dtype=torch.float8_e4m3fn)
            k_scale = torch.empty([chunk.total_seq_lens, 1],
                                  device=k.device,
                                  dtype=torch.float32)
            cp_gather_indexer_k_quant_cache(
                kv_cache, k_fp8, k_scale, chunk.block_table,
                chunk.cu_seq_lens, chunk.num_reqs,
            )
            block_sparse_logits, topk_block_indices = \
                fp8_hierarchy_mqa_logits(
                    q_fp8[chunk.token_start:chunk.token_end],
                    (k_fp8, k_scale),
                    weights[chunk.token_start:chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    k_block_size,
                    block_topk,
                )
            relevant_topk_indices = block_sparse_logits.topk(
                min(topk_tokens, block_sparse_logits.shape[-1]),
                dim=-1)[1]
            absolute_topk_block_indices = torch.gather(
                topk_block_indices, dim=-1,
                index=(relevant_topk_indices // k_block_size))
            topk_indices = (absolute_topk_block_indices * k_block_size +
                            (relevant_topk_indices % k_block_size))

            topk_indices -= chunk.cu_seqlen_ks[:, None]
            mask_lo = topk_indices >= 0
            mask_hi = topk_indices - (
                chunk.cu_seqlen_ke - chunk.cu_seqlen_ks)[:, None] < 0
            mask = torch.full_like(topk_indices, False, dtype=torch.bool,
                                   device=topk_indices.device)
            mask = mask_lo & mask_hi
            topk_indices = topk_indices.masked_fill(~mask, -1)
            topk_indices_buffer[
                chunk.token_start:chunk.token_end,
                :topk_indices.shape[-1]] = topk_indices.to(dtype=torch.int32)

    if has_decode:
        decode_metadata = attn_metadata.decode
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens)
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:])
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        current_device = padded_q_fp8_decode_tokens.device
        padded_num_tokens = batch_size * next_n
        row_indices = torch.arange(
            padded_num_tokens, device=current_device) // next_n
        next_n_offset = torch.arange(
            padded_num_tokens, device=current_device) % next_n
        index_end_pos = (
            decode_metadata.seq_lens[row_indices] - next_n + next_n_offset
        ).unsqueeze(1)

        assert next_n == 1, (
            "Hierarchical sparse attention indexer only supports "
            "decoding 1 token at a time for now")

        block_sparse_logits, topk_block_indices = \
            fp8_hierarchy_paged_mqa_logits(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                decode_metadata.seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                k_block_size=k_block_size,
                block_topk=block_topk,
            )
        relevant_topk_indices = block_sparse_logits.topk(
            min(topk_tokens, block_sparse_logits.shape[-1]),
            dim=-1)[1]
        absolute_topk_block_indices = torch.gather(
            topk_block_indices, dim=-1,
            index=(relevant_topk_indices // k_block_size))
        topk_indices = (absolute_topk_block_indices * k_block_size +
                        (relevant_topk_indices % k_block_size))

        topk_indices[topk_indices > index_end_pos] = -1
        if decode_metadata.requires_padding:
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(
                    batch_size, -1, topk_indices.shape[-1]),
                decode_lens)
        topk_indices_buffer[
            :num_decode_tokens,
            :topk_indices.shape[-1]] = topk_indices.to(dtype=torch.int32)

    return topk_indices_buffer


def transgqa_hierarchy_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    k_block_size: int,
    block_topk: int,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: Optional[torch.Tensor],
) -> torch.Tensor:
    _flattened_kv = torch.empty([total_seq_lens, head_dim + 4],
                                device=k.device,
                                dtype=torch.uint8)
    _k_fp8 = _flattened_kv[..., :head_dim].view(
        torch.float8_e4m3fn).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(
        torch.float32).contiguous()
    return topk_indices_buffer


direct_register_custom_op(
    op_name="transgqa_hierarchy_sparse_attn_indexer",
    op_func=transgqa_hierarchy_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=transgqa_hierarchy_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


# =============================================================================
# Sparse GQA Forward (custom op: KV cache update + sparse attention kernel)
# =============================================================================

def _build_req_id_per_token(
    query_start_loc: torch.Tensor,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Map each token position to its request index."""
    num_reqs = query_start_loc.shape[0] - 1
    lens = query_start_loc[1:num_reqs + 1] - query_start_loc[:num_reqs]
    return torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device=device), lens)


def _pad_topk_indices(
    topk_indices: torch.Tensor,
    block_n: int,
) -> torch.Tensor:
    """Pad topk_indices columns to a multiple of block_n."""
    topk_k = topk_indices.shape[1]
    remainder = topk_k % block_n
    if remainder != 0:
        pad_size = block_n - remainder
        topk_indices = torch.nn.functional.pad(
            topk_indices, (0, pad_size), value=-1)
    return topk_indices


def transgqa_sparse_gqa_forward(
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
    """Custom op: update KV cache then run sparse GQA attention kernel.

    In vLLM 0.11, Attention.forward() bundles KV cache update and full
    attention together. For our sparse path we need to:
    1. Write K/V into the paged cache ourselves (reshape_and_cache_flash)
    2. Convert request-local topk_indices to global cache slot indices
    3. Run the fused sparse GQA Triton kernel that gathers K/V on the fly
    """
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

    num_blocks, block_size_cache, nkv, hd = key_cache.shape
    key_flat = key_cache.reshape(-1, nkv, hd)
    value_flat = value_cache.reshape(-1, nkv, hd)

    num_tokens = attn_meta.num_actual_tokens
    req_id = _build_req_id_per_token(
        attn_meta.query_start_loc, num_tokens, q.device)
    block_table = attn_meta.block_table

    ti = topk_indices[:num_tokens].to(torch.int32)
    ti = _pad_topk_indices(ti, block_n)

    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        triton_convert_req_index_to_global_index)
    global_indices = triton_convert_req_index_to_global_index(
        req_id, block_table, ti,
        BLOCK_SIZE=block_size_cache,
        NUM_TOPK_TOKENS=ti.shape[1],
        BLOCK_N=block_n,
    )

    from vllm.v1.attention.ops.sparse_gqa import sparse_gqa_attention
    output = sparse_gqa_attention(
        q[:num_tokens], key_flat, value_flat, global_indices,
        softmax_scale, num_heads, num_kv_heads, head_dim,
    )
    return output.view(-1, num_heads * head_dim)


def transgqa_sparse_gqa_forward_fake(
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


direct_register_custom_op(
    op_name="transgqa_sparse_gqa_forward",
    op_func=transgqa_sparse_gqa_forward,
    mutates_args=[],
    fake_impl=transgqa_sparse_gqa_forward_fake,
    dispatch_key=current_platform.dispatch_key,
)

# Optional TileLang sparse GQA path (``transgqa_sparse_gqa_forward_tilelang``).
try:
    import vllm.v1.attention.ops.sparse_gqa_tilelang  # noqa: F401
except Exception as e:
    logger.debug("sparse_gqa_tilelang not registered: %s", e)


# =============================================================================
# Indexer Module (independent weights, MLA-style FP8 top-k path)
# =============================================================================

class TransGQAIndexer(nn.Module):
    """Independent MLA-based indexer for sparse GQA attention.

    Uses the same FP8 K cache and DSA top-k kernels as before.  Index Q/K and
    per-head weights are produced with the TranSgqa-style projection on the
    main attention Q/K/V, via ``qk_index_proj``, ``qk_nope_proj``, and
    ``v_transform``.

    The projection has two paths (matching the PyTorch reference):
    * **Rope path** — post-RoPE ``q``, ``k`` → freq-fold → einsum with
      ``qk_index_proj`` → take first ``ff//2`` output columns.
    * **Nope path** — pre-RoPE ``q_orig``, ``k_orig`` → freq-fold → einsum
      with the same ``qk_index_proj`` → take remaining output columns → pass
      through ``qk_nope_proj`` (linear compression).
    * Final ``index_q``, ``index_k`` are ``cat(rope, nope)`` with head_dim=D.

    **TP strategy**: for ``tp > 1``, all-gather ``Q, K, V, Q_orig, K_orig``
    along the head dimension, collapse KV replicas if needed, then run the
    full monolithic TranSgqa formula on every rank (identical results).
    Indexer weights (``qk_index_proj``, ``qk_nope_proj``, ``v_transform``)
    remain **replicated** on all TP ranks.
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 config: Qwen3Config,
                 hidden_size: int,
                 quant_config: Optional[QuantizationConfig],
                 cache_config: Optional[CacheConfig],
                 topk_indices_buffer: Optional[torch.Tensor],
                 prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        # FP8 MQA path uses one "query head" per GQA group (same as TranSgqa).
        self.n_head = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(
            config, "head_dim",
            hidden_size // config.num_attention_heads)
        self.freqfold = getattr(config, "freqfold", 2)
        self.lora_rank = getattr(config, "rank_of_wo", 4)

        nband = self.head_dim // (2 * self.freqfold)
        gff = self.num_kv_heads * self.freqfold
        self.qk_index_proj = nn.Parameter(
            torch.empty(nband, gff, gff))
        self.v_transform = nn.Parameter(
            torch.empty(
                self.num_heads,
                self.head_dim,
                self.lora_rank,
            ))
        nope_in = (self.num_kv_heads * self.head_dim
                    - self.head_dim // 2)
        self.qk_nope_proj = nn.Linear(nope_in, self.head_dim // 2,
                                       bias=False)
        nn.init.normal_(self.qk_index_proj, std=0.02)
        nn.init.normal_(self.v_transform, std=0.02)

        # Hierarchical sparse attention config
        self.use_hisa = getattr(config, "use_hisa", False)
        if self.use_hisa:
            self.k_block_size = config.hisa_k_block_size
            self.block_topk = config.hisa_block_topk

        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        # FP8 K cache for the indexer
        self.k_cache = TransGQAIndexerCache(
            head_dim=self.head_dim +
            self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config)

        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size)
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = (get_tensor_model_parallel_rank()
                        if self.tp_size > 1 else 0)
        if self.tp_size > 1:
            assert self.num_heads % self.tp_size == 0, (
                "num_attention_heads must be divisible by tensor_parallel_size")

    # ---- frequency-fold helpers (shared by monolithic & gather paths) ----

    def _fold_k(self, t: torch.Tensor, G: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """[N, G, D] → real/imag each [N, nband, G*ff]."""
        D = t.shape[-1]
        Df = D // 2
        ff = self.freqfold
        nband = Df // ff
        t = t.transpose(-1, -2).contiguous()                   # [N, D, G]
        real = (t[:, :Df, :].reshape(-1, nband, ff, G)
                .transpose(1, 2).reshape(-1, nband, G * ff))
        imag = (t[:, Df:, :].reshape(-1, nband, ff, G)
                .transpose(1, 2).reshape(-1, nband, G * ff))
        return real, imag

    def _fold_q(self, t: torch.Tensor, G: int, groups: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """[N, H, D] → real/imag each [N, groups, nband, G*ff]."""
        D = t.shape[-1]
        Df = D // 2
        ff = self.freqfold
        nband = Df // ff
        t = (t.view(-1, G, groups, D).permute(0, 2, 1, 3)
             .transpose(-1, -2).contiguous())                   # [N, groups, D, G]
        real = (t[:, :, :Df, :].reshape(-1, groups, nband, ff, G)
                .transpose(2, 3).reshape(-1, groups, nband, G * ff))
        imag = (t[:, :, Df:, :].reshape(-1, groups, nband, ff, G)
                .transpose(2, 3).reshape(-1, groups, nband, G * ff))
        return real, imag

    # ---- core TranSgqa computation (rope + nope two-path) ----

    def _index_q_k_weights_monolithic(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_orig: torch.Tensor,
        k_orig: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full TranSgqa indexer on global-shaped tensors (single-rank view).

        Args:
            q, k:       post-RoPE  [N, H, D], [N, G, D]
            v:          [N, G, D]
            q_orig, k_orig: pre-RoPE (after QK-norm)  [N, H, D], [N, G, D]
        """
        G = self.num_kv_heads
        groups = self.n_head
        ff = self.freqfold
        half_ff = ff // 2
        D = q.shape[-1]
        nband = D // (2 * ff)

        qk_w = self.qk_index_proj.to(dtype=q.dtype, device=q.device)
        v_w = self.v_transform.to(dtype=q.dtype, device=q.device)

        # ---- Rope part (post-RoPE q, k) ----
        kr_re, kr_im = self._fold_k(k, G)
        kr_re_p = torch.einsum("bnf,nfr->bnr", kr_re, qk_w)
        kr_im_p = torch.einsum("bnf,nfr->bnr", kr_im, qk_w)
        index_k_rope = torch.cat([
            kr_re_p[:, :, :half_ff].reshape(-1, nband * half_ff),
            kr_im_p[:, :, :half_ff].reshape(-1, nband * half_ff),
        ], dim=-1).to(k.dtype)

        qr_re, qr_im = self._fold_q(q, G, groups)
        qr_re_p = torch.einsum("tgnf,nfr->tgnr", qr_re, qk_w)
        qr_im_p = torch.einsum("tgnf,nfr->tgnr", qr_im, qk_w)
        index_q_rope = torch.cat([
            qr_re_p[:, :, :, :half_ff].reshape(-1, groups, nband * half_ff),
            qr_im_p[:, :, :, :half_ff].reshape(-1, groups, nband * half_ff),
        ], dim=-1).to(q.dtype)

        # ---- Nope part (pre-RoPE q_orig, k_orig) ----
        nope_cols = G * ff - half_ff

        kn_re, kn_im = self._fold_k(k_orig, G)
        kn_re_p = torch.einsum("bnf,nfr->bnr", kn_re, qk_w)
        kn_im_p = torch.einsum("bnf,nfr->bnr", kn_im, qk_w)
        k_nope_rotary = torch.cat([
            kn_re_p[:, :, half_ff:].reshape(-1, nband * nope_cols),
            kn_im_p[:, :, half_ff:].reshape(-1, nband * nope_cols),
        ], dim=-1).to(k.dtype)
        k_nope_lr = self.qk_nope_proj(k_nope_rotary).to(k.dtype)
        index_k = torch.cat([index_k_rope, k_nope_lr], dim=-1)

        qn_re, qn_im = self._fold_q(q_orig, G, groups)
        qn_re_p = torch.einsum("tgnf,nfr->tgnr", qn_re, qk_w)
        qn_im_p = torch.einsum("tgnf,nfr->tgnr", qn_im, qk_w)
        q_nope_rotary = torch.cat([
            qn_re_p[:, :, :, half_ff:].reshape(-1, groups, nband * nope_cols),
            qn_im_p[:, :, :, half_ff:].reshape(-1, groups, nband * nope_cols),
        ], dim=-1).to(q.dtype)
        q_nope_lr = self.qk_nope_proj(q_nope_rotary).to(q.dtype)
        index_q = torch.cat([index_q_rope, q_nope_lr], dim=-1)

        # ---- Weights (from V) ----
        v_h = v.repeat_interleave(groups, dim=1)
        w = torch.einsum("nhd,hdr->nhr", v_h, v_w)
        w = w.view(-1, G, groups, v_w.shape[-1])
        w = w.permute(0, 2, 1, 3).reshape(
            -1, groups, G * v_w.shape[-1])
        weights = torch.norm(w.to(torch.float32), p=2, dim=-1)

        return index_q.contiguous(), index_k.contiguous(), weights.contiguous()

    def _index_q_k_weights_gather_full(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_orig: torch.Tensor,
        k_orig: torch.Tensor,
        num_kv_head_replicas: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All-gather head dims, collapse KV replicas, run monolithic."""
        q_f = tensor_model_parallel_all_gather(q.contiguous(), dim=1)
        k_f = tensor_model_parallel_all_gather(k.contiguous(), dim=1)
        v_f = tensor_model_parallel_all_gather(v.contiguous(), dim=1)
        qo_f = tensor_model_parallel_all_gather(q_orig.contiguous(), dim=1)
        ko_f = tensor_model_parallel_all_gather(k_orig.contiguous(), dim=1)

        G = self.num_kv_heads

        if num_kv_head_replicas > 1:
            idxs = [gi * num_kv_head_replicas for gi in range(G)]
            k_f = torch.stack([k_f[:, i, :] for i in idxs], dim=1)
            v_f = torch.stack([v_f[:, i, :] for i in idxs], dim=1)
            ko_f = torch.stack([ko_f[:, i, :] for i in idxs], dim=1)

        assert q_f.shape[1] == self.num_heads
        assert k_f.shape[1] == G
        return self._index_q_k_weights_monolithic(
            q_f, k_f, v_f, qo_f, ko_f)

    def _index_q_k_weights_from_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_orig: torch.Tensor,
        k_orig: torch.Tensor,
        num_kv_head_replicas: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to monolithic (tp==1) or all-gather (tp>1)."""
        if self.tp_size <= 1:
            return self._index_q_k_weights_monolithic(
                q, k, v, q_orig, k_orig)
        return self._index_q_k_weights_gather_full(
            q, k, v, q_orig, k_orig, num_kv_head_replicas)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_orig: torch.Tensor,
        k_orig: torch.Tensor,
        num_kv_head_replicas: int = 1,
    ) -> torch.Tensor:
        """Compute top-k important key indices.

        Args:
            hidden_states: [num_tokens, hidden_size]
            positions: [num_tokens] (unused; kept for vLLM call pattern)
            rotary_emb: shared rope module (unused; kept for vLLM call pattern)
            q, k, v: main attention Q/K/V **after** QK-norm and RoPE,
                [N, num_local_heads, D], [N, num_local_kv_heads, D],
                [N, num_local_kv_heads, D]
            q_orig, k_orig: Q/K **after** QK-norm but **before** RoPE,
                same shapes as q, k.  Used for the nope (non-positional)
                path of the TranSgqa projection.
            num_kv_head_replicas: from ``QKVParallelLinear`` when
                ``tensor_parallel_size >= num_key_value_heads`` (KV replicated).

        Returns:
            topk_indices_buffer: [max_tokens, topk_tokens] with selected
                                 key indices per token, -1 for invalid
        """
        _ = (positions, rotary_emb)
        index_q, index_k, weights = self._index_q_k_weights_from_qkv(
            q, k, v, q_orig, k_orig,
            num_kv_head_replicas=num_kv_head_replicas)

        q_flat = index_q.reshape(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q_flat, self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None)
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = weights.to(torch.float32)
        weights = (weights.unsqueeze(-1) * q_scale * self.softmax_scale *
                   (self.n_head**-0.5))
        weights = weights.squeeze(-1)

        if not self.use_hisa:
            result = torch.ops.vllm.transgqa_sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8,
                index_k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            result = torch.ops.vllm.transgqa_hierarchy_sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8,
                index_k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.k_block_size,
                self.block_topk,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )

        return result


# =============================================================================
# Sparse GQA Attention (Qwen3-style with indexer)
# =============================================================================

class TransGQAAttention(nn.Module):
    """Qwen3-style GQA attention with sparse attention via an indexer.

    Two-step process:
    1. The Indexer (independent weights) selects top-k important key positions
       using lightweight FP8 approximate attention.
    2. The main GQA attention uses standard Q/K/V projections but only
       attends to the selected key positions, reducing computation for
       long sequences.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Qwen3Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 131072,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict[str, Any]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        topk_indices_buffer: Optional[torch.Tensor] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        assert num_kv_heads % tp_size == 0
        self.num_local_kv_heads = num_kv_heads // tp_size
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # QKV projections (standard GQA)
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj")

        # Qwen3 uses QK-Norm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Output projection
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj")

        # Rotary embedding (Qwen3 uses neox style)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        # Sparse attention indexer (independent weights)
        self.is_sparse = hasattr(config, "index_topk")
        self.topk_indices_buffer = topk_indices_buffer
        if self.is_sparse:
            self.indexer = TransGQAIndexer(
                vllm_config,
                config,
                hidden_size,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer = None

        # Standard attention layer
        self.attn = Attention(
            self.num_local_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                             dim=-1)

        q = q.view(-1, self.num_local_heads, self.head_dim)
        k = k.view(-1, self.num_local_kv_heads, self.head_dim)
        v = v.view(-1, self.num_local_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q_orig = q                                   # pre-RoPE (after QK-norm)
        k_orig = k
        q, k = self.rotary_emb(positions, q, k)      # post-RoPE

        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                positions,
                self.rotary_emb,
                q, k, v,
                q_orig, k_orig,
                num_kv_head_replicas=getattr(
                    self.qkv_proj, "num_kv_head_replicas", 1),
            )
        else:
            topk_indices = None

        q = q.reshape(-1, self.num_local_heads * self.head_dim)
        k = k.reshape(-1, self.num_local_kv_heads * self.head_dim)
        v = v.reshape(-1, self.num_local_kv_heads * self.head_dim)

        if topk_indices is not None:
            q_3d = q.view(-1, self.num_local_heads, self.head_dim)
            k_3d = k.view(-1, self.num_local_kv_heads, self.head_dim)
            v_3d = v.view(-1, self.num_local_kv_heads, self.head_dim)
            attn_output = torch.ops.vllm.transgqa_sparse_gqa_forward(
                q_3d, k_3d, v_3d, topk_indices,
                self.attn.layer_name,
                self.scaling,
                self.num_local_heads,
                self.num_local_kv_heads,
                self.head_dim,
                128,
            )
        else:
            attn_output = self.attn(q, k, v)

        output, _ = self.o_proj(attn_output)
        return output


# =============================================================================
# Decoder Layer
# =============================================================================

class TransGQADecoderLayer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(
            config, "max_position_embeddings", 131072)

        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx

        self.self_attn = TransGQAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim",
                             config.hidden_size // config.num_attention_heads),
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            cache_config=cache_config,
            quant_config=quant_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = TransGQAMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# =============================================================================
# Model
# =============================================================================

@support_torch_compile
class TransGQAModel(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        # Sparse attention config
        self.is_sparse = hasattr(config, "index_topk")
        if self.is_sparse:
            vllm_config.cache_config.block_size = max(
                vllm_config.cache_config.block_size, 64)
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device="cuda")
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: TransGQADecoderLayer(
                vllm_config, prefix, topk_indices_buffer),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# =============================================================================
# Causal LM
# =============================================================================

class TransGQAForCausalLM(nn.Module, SupportsPP):
    """Qwen3 with TransGQA sparse attention for causal language modeling.

    This model uses a two-step sparse attention mechanism:
    1. An independent MLA-based indexer selects top-k important key positions
    2. The main GQA attention only attends to those positions

    Config requirements (in addition to standard Qwen3Config):
        - index_topk: int, number of top-k tokens to select
        - freqfold: int (optional, default 2), TranSgqa indexer folding factor
        - rank_of_wo: int (optional, default 4), TranSgqa ``v_transform`` rank
        - use_hisa: bool (optional), hierarchical sparse indexer (HISA)
        - hisa_k_block_size / hisa_block_topk: HISA options when use_hisa=True

    Weights include ``indexer.qk_index_proj`` and ``indexer.v_transform`` (TranSgqa
    indexer).  Tensor parallelism: ``num_attention_heads`` must divide
    ``tensor_parallel_size``.  If ``tensor_parallel_size >= num_key_value_heads``,
    vLLM replicates KV heads (``QKVParallelLinear.num_kv_head_replicas``); the
    indexer then all-gathers Q/K/V and collapses replicas so TranSgqa + MLA FP8
    still see ``H/G`` index query groups (e.g. 16 for 32 Q / 2 KV).
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = TransGQAModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
