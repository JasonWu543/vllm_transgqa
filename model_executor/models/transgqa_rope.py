# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only TransGQA model with rope-only indexer and MoE MLP.

Differences from ``transgqa.py``:
* **Indexer** uses only the rope (post-RoPE) path.  ``qk_index_proj`` has
  shape ``[nband, G*ff, ff]`` — no ``qk_nope_proj``.
* **MLP** is Mixture-of-Experts (FusedMoE) to match the checkpoint that
  contains ``mlp.gate.weight`` and ``mlp.experts.{0-N}.{gate,up,down}_proj``.
* **QKV bias** is read from ``config.attention_bias`` (default False).
* Custom sparse-attention ops are shared with the base ``transgqa`` module
  (imported at module level to trigger ``direct_register_custom_op``).
"""
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (get_ep_group, get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
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
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

# Importing from the base transgqa module triggers registration of the shared
# custom ops (transgqa_sparse_attn_indexer, transgqa_hierarchy_sparse_attn_indexer,
# transgqa_sparse_gqa_forward, transgqa_sparse_gqa_forward_tilelang) via
# ``transgqa`` and ``sparse_gqa_tilelang`` so they are available via torch.ops.vllm.*.
from .transgqa import TransGQAIndexerCache

from .interfaces import SupportsPP
from .utils import (PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

logger = init_logger(__name__)


# =============================================================================
# Dense MLP (fallback for non-MoE layers)
# =============================================================================

class TransGQARopeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
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
            reduce_results=reduce_results,
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
# MoE MLP (matches Qwen3MoeSparseMoeBlock pattern)
# =============================================================================

class TransGQARopeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()

        num_experts = getattr(config, "num_experts", 0)
        self.n_routed_experts = num_experts

        if self.tp_size > num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {num_experts}.")

        vllm_cfg = get_current_vllm_config()
        parallel_config = vllm_cfg.parallel_config
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        self.n_logical_experts = num_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = num_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = (self.ep_rank *
                                      self.n_local_physical_experts)
        self.physical_expert_end = (self.physical_expert_start +
                                    self.n_local_physical_experts)

        moe_intermediate_size = getattr(config, "moe_intermediate_size",
                                        config.intermediate_size)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 8)
        norm_topk_prob = getattr(config, "norm_topk_prob", True)

        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=moe_intermediate_size,
            reduce_results=True,
            renormalize=norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            from vllm.model_executor.models.utils import (
                sequence_parallel_chunk)
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.squeeze(0) if is_input_1d else \
            final_hidden_states


# =============================================================================
# Rope-only Indexer
# =============================================================================

class TransGQARopeIndexer(nn.Module):
    """Rope-only TransGQA indexer (no nope path, no ``qk_nope_proj``).

    Uses post-RoPE Q/K → freq-fold → ``qk_index_proj`` projection.
    ``qk_index_proj`` shape: ``[nband, G*ff, ff]`` where ``ff = freqfold``.
    Output ``index_q`` / ``index_k`` dimension = ``head_dim``.

    TP strategy (same as full TransGQAIndexer): for ``tp > 1``, all-gather
    Q/K/V along the head dimension, collapse KV replicas if needed, then run
    the full monolithic computation on every rank.
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
        self.n_head = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(
            config, "head_dim",
            hidden_size // config.num_attention_heads)
        self.freqfold = getattr(config, "freqfold", 2)
        self.lora_rank = getattr(config, "rank_of_wo", 4)

        nband = self.head_dim // (2 * self.freqfold)
        gff = self.num_kv_heads * self.freqfold
        ff = self.freqfold
        self.qk_index_proj = nn.Parameter(
            torch.empty(nband, gff, ff))
        self.v_transform = nn.Parameter(
            torch.empty(
                self.num_heads,
                self.head_dim,
                self.lora_rank,
            ))
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

    # ---- frequency-fold helpers ----

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

    # ---- rope-only TranSgqa computation ----

    def _index_q_k_weights_monolithic(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rope-only indexer on global-shaped tensors.

        Args:
            q: post-RoPE  [N, H, D]
            k: post-RoPE  [N, G, D]
            v: [N, G, D]
        """
        G = self.num_kv_heads
        groups = self.n_head
        ff = self.freqfold
        D = q.shape[-1]
        nband = D // (2 * ff)

        qk_w = self.qk_index_proj.to(dtype=q.dtype, device=q.device)
        v_w = self.v_transform.to(dtype=q.dtype, device=q.device)

        # ---- K projection ----
        kr_re, kr_im = self._fold_k(k, G)
        kr_re_p = torch.einsum("bnf,nfr->bnr", kr_re, qk_w)
        kr_im_p = torch.einsum("bnf,nfr->bnr", kr_im, qk_w)
        index_k = torch.cat([
            kr_re_p.reshape(-1, nband * ff),
            kr_im_p.reshape(-1, nband * ff),
        ], dim=-1).to(k.dtype)

        # ---- Q projection ----
        qr_re, qr_im = self._fold_q(q, G, groups)
        qr_re_p = torch.einsum("tgnf,nfr->tgnr", qr_re, qk_w)
        qr_im_p = torch.einsum("tgnf,nfr->tgnr", qr_im, qk_w)
        index_q = torch.cat([
            qr_re_p.reshape(-1, groups, nband * ff),
            qr_im_p.reshape(-1, groups, nband * ff),
        ], dim=-1).to(q.dtype)

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
        num_kv_head_replicas: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All-gather head dims, collapse KV replicas, run monolithic."""
        q_f = tensor_model_parallel_all_gather(q.contiguous(), dim=1)
        k_f = tensor_model_parallel_all_gather(k.contiguous(), dim=1)
        v_f = tensor_model_parallel_all_gather(v.contiguous(), dim=1)

        G = self.num_kv_heads

        if num_kv_head_replicas > 1:
            idxs = [gi * num_kv_head_replicas for gi in range(G)]
            k_f = torch.stack([k_f[:, i, :] for i in idxs], dim=1)
            v_f = torch.stack([v_f[:, i, :] for i in idxs], dim=1)

        assert q_f.shape[1] == self.num_heads
        assert k_f.shape[1] == G
        return self._index_q_k_weights_monolithic(q_f, k_f, v_f)

    def _index_q_k_weights_from_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_kv_head_replicas: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to monolithic (tp==1) or all-gather (tp>1)."""
        if self.tp_size <= 1:
            return self._index_q_k_weights_monolithic(q, k, v)
        return self._index_q_k_weights_gather_full(
            q, k, v, num_kv_head_replicas)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_kv_head_replicas: int = 1,
    ) -> torch.Tensor:
        """Compute top-k important key indices (rope-only path).

        Args:
            hidden_states: [num_tokens, hidden_size]
            positions: unused (kept for call-pattern compatibility)
            rotary_emb: unused (kept for call-pattern compatibility)
            q, k, v: post-RoPE Q/K and V, each 3-D
                [N, num_local_heads, D], [N, num_local_kv_heads, D],
                [N, num_local_kv_heads, D]
            num_kv_head_replicas: from QKVParallelLinear when tp >= num_kv_heads

        Returns:
            topk_indices_buffer: [max_tokens, topk_tokens]
        """
        _ = (positions, rotary_emb)
        index_q, index_k, weights = self._index_q_k_weights_from_qkv(
            q, k, v,
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
# Attention (GQA with sparse indexer)
# =============================================================================

class TransGQARopeAttention(nn.Module):
    """Qwen3-style GQA attention with rope-only sparse indexer.

    Two-step process:
    1. The Indexer selects top-k important key positions using FP8 MQA.
    2. The main GQA attention only attends to those selected positions.
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
        if num_kv_heads >= tp_size:
            assert num_kv_heads % tp_size == 0
            self.num_local_kv_heads = num_kv_heads // tp_size
        else:
            assert tp_size % num_kv_heads == 0
            self.num_local_kv_heads = 1
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        qkv_bias = getattr(config, 'attention_bias', False)
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj")

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj")

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        self.is_sparse = hasattr(config, "index_topk")
        self.topk_indices_buffer = topk_indices_buffer
        if self.is_sparse:
            self.indexer = TransGQARopeIndexer(
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

        q, k = self.rotary_emb(positions, q, k)      # post-RoPE

        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                positions,
                self.rotary_emb,
                q, k, v,
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
            # Triton: ``transgqa_sparse_gqa_forward``; TileLang (optional):
            # ``transgqa_sparse_gqa_forward_tilelang`` — same argument list.
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
# Decoder Layer (MoE or dense MLP, selected per config)
# =============================================================================

class TransGQARopeDecoderLayer(nn.Module):

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

        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx

        self.self_attn = TransGQARopeAttention(
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

        num_experts = getattr(config, "num_experts", 0)
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)

        if (layer_idx not in mlp_only_layers
                and num_experts > 0
                and (layer_idx + 1) % decoder_sparse_step == 0):
            self.mlp = TransGQARopeSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp")
        else:
            self.mlp = TransGQARopeMLP(
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
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# =============================================================================
# Model
# =============================================================================

@support_torch_compile
class TransGQARopeModel(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

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
            lambda prefix: TransGQARopeDecoderLayer(
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

class TransGQARopeForCausalLM(nn.Module, SupportsPP):
    """TransGQA (rope-only indexer) with MoE MLP for causal language modeling.

    Config requirements (in addition to standard Qwen3Config):
        - index_topk: int — number of top-k tokens to select
        - freqfold: int (default 2) — indexer folding factor
        - rank_of_wo: int (default 4) — ``v_transform`` rank
        - num_experts: int — number of MoE experts (0 → dense MLP)
        - num_experts_per_tok: int — MoE top-k routing
        - moe_intermediate_size: int — MoE expert intermediate size
        - decoder_sparse_step: int (default 1) — which layers are MoE
        - norm_topk_prob: bool (default True) — renormalize routing probs
        - attention_bias: bool (default False) — QKV bias
        - use_hisa / hisa_k_block_size / hisa_block_topk: HISA options
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

        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.model = TransGQARopeModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head.weight = self.model.embed_tokens.weight
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

        num_experts = getattr(self.config, "num_experts", 0)
        if num_experts > 0:
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
                num_redundant_experts=self.num_redundant_experts)
        else:
            expert_params_mapping = []

        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # --- stacked QKV / dense gate_up_proj ---
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # --- MoE expert weights ---
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if (name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict):
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(Callable[..., bool],
                                                param.weight_loader)
                    success = weight_loader(param,
                                            loaded_weight,
                                            name_mapped,
                                            shard_id=shard_id,
                                            expert_id=expert_id,
                                            return_success=True)
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue

                    # --- standard weights ---
                    if (name.endswith(ignore_suffixes)
                            and name not in params_dict):
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
