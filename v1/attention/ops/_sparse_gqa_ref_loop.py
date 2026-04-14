"""Pure-PyTorch loop-based reference for sparse GQA attention.

Extracted from test_sparse_gqa.py so that other scripts can import it
without pulling in the Triton kernel or any vllm package dependencies.
"""

import torch


def sparse_gqa_attention_ref(
    q: torch.Tensor,            # [num_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,    # [total_slots, num_kv_heads, head_dim]
    value_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim]
    global_indices: torch.Tensor,  # [num_tokens, topk_k] int32, -1=invalid
    softmax_scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Exact reference: loop over tokens x heads, gather, masked softmax."""
    num_tokens = q.shape[0]
    topk_k = global_indices.shape[1]
    gqa_group_size = num_heads // num_kv_heads

    q_f32 = q.float()
    kc_f32 = key_cache.float()
    vc_f32 = value_cache.float()

    output = torch.zeros(num_tokens, num_heads, head_dim,
                         dtype=torch.float32, device=q.device)

    for t in range(num_tokens):
        idx = global_indices[t]                       # [topk_k]
        valid = idx >= 0                              # [topk_k]

        for h in range(num_heads):
            kv_h = h // gqa_group_size

            safe_idx = idx.clamp(min=0)
            k_gathered = kc_f32[safe_idx, kv_h, :]    # [topk_k, head_dim]
            v_gathered = vc_f32[safe_idx, kv_h, :]    # [topk_k, head_dim]

            scores = (k_gathered * q_f32[t, h, :]).sum(dim=-1)
            scores = scores * softmax_scale
            scores = scores.masked_fill(~valid, float('-inf'))

            scores_max = scores.max()
            if scores_max == float('-inf'):
                continue
            exp_scores = torch.exp(scores - scores_max)
            exp_scores = exp_scores.masked_fill(~valid, 0.0)
            denom = exp_scores.sum().clamp(min=1e-6)
            weights = exp_scores / denom

            output[t, h, :] = (weights[:, None] * v_gathered).sum(dim=0)

    return output.to(q.dtype)
