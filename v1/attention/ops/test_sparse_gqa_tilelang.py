"""Phase 2: compare TileLang sparse_gqa_attention_tilelang vs PyTorch reference.

Requires CUDA, optional ``tilelang`` (skip with message if missing).

Run:
  python v1/attention/ops/test_sparse_gqa_tilelang.py
"""
from __future__ import annotations

import math
import sys

import torch


def sparse_gqa_attention_ref(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    global_indices: torch.Tensor,
    softmax_scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Same reference as test_sparse_gqa.py (keep in sync)."""
    num_tokens = q.shape[0]
    topk_k = global_indices.shape[1]
    gqa_group_size = num_heads // num_kv_heads
    q_f32 = q.float()
    kc_f32 = key_cache.float()
    vc_f32 = value_cache.float()
    output = torch.zeros(
        num_tokens, num_heads, head_dim, dtype=torch.float32, device=q.device)
    for t in range(num_tokens):
        idx = global_indices[t]
        valid = idx >= 0
        for h in range(num_heads):
            kv_h = h // gqa_group_size
            safe_idx = idx.clamp(min=0)
            k_gathered = kc_f32[safe_idx, kv_h, :]
            v_gathered = vc_f32[safe_idx, kv_h, :]
            scores = (k_gathered * q_f32[t, h, :]).sum(dim=-1) * softmax_scale
            scores = scores.masked_fill(~valid, float("-inf"))
            scores_max = scores.max()
            if scores_max == float("-inf"):
                continue
            exp_scores = torch.exp(scores - scores_max)
            exp_scores = exp_scores.masked_fill(~valid, 0.0)
            denom = exp_scores.sum().clamp(min=1e-6)
            weights = exp_scores / denom
            output[t, h, :] = (weights[:, None] * v_gathered).sum(dim=0)
    return output.to(q.dtype)


def _check_close(out_tl, out_ref, label: str, tol: float = 2e-2) -> None:
    diff = (out_tl.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    ok = max_diff < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}  (max_diff={max_diff:.6f})")
    if not ok:
        raise AssertionError(f"{label}: max abs diff {max_diff:.6f} exceeds {tol}")


def _make_data(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    topk_k: int,
    total_slots: int,
    *,
    invalid_ratio: float = 0.0,
    device: str = "cuda",
):
    torch.manual_seed(0)
    q = torch.randn(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        total_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    value_cache = torch.randn(
        total_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    global_indices = torch.randint(
        0, total_slots, (num_tokens, topk_k), dtype=torch.int32, device=device)
    if invalid_ratio > 0:
        mask = torch.rand(num_tokens, topk_k, device=device) < invalid_ratio
        global_indices[mask] = -1
    scale = 1.0 / math.sqrt(head_dim)
    return q, key_cache, value_cache, global_indices, scale


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        sys.exit(0)

    try:
        import tilelang  # noqa: F401
    except ImportError:
        print("SKIP: tilelang not installed (pip install tilelang)")
        sys.exit(0)

    # Import after tilelang check; ``sparse_gqa_tilelang`` avoids ``vllm._C`` at load.
    from vllm.v1.attention.ops.sparse_gqa_tilelang import sparse_gqa_attention_tilelang

    print("TileLang vs PyTorch reference (constraints: head_dim power of 2, topk % 32 == 0)")
    cases = [
        (8, 8, 2, 128, 64, "GQA 8h/2kv dim=128 topk=64"),
        (4, 8, 2, 128, 256, "GQA 8h/2kv dim=128 topk=256"),
        (8, 16, 1, 64, 32, "MQA 16h/1kv dim=64 topk=32"),
    ]
    for nt, nh, nkv, hd, tk, label in cases:
        q, kc, vc, idx, sc = _make_data(nt, nh, nkv, hd, tk, 2048)
        out_tl = sparse_gqa_attention_tilelang(
            q, kc, vc, idx, sc, nh, nkv, hd, block_I=32)
        out_ref = sparse_gqa_attention_ref(q, kc, vc, idx, sc, nh, nkv, hd)
        _check_close(out_tl, out_ref, label)

    q, kc, vc, idx, sc = _make_data(
        8, 8, 2, 128, 64, 1024, invalid_ratio=0.3)
    out_tl = sparse_gqa_attention_tilelang(
        q, kc, vc, idx, sc, 8, 2, 128, block_I=32)
    out_ref = sparse_gqa_attention_ref(q, kc, vc, idx, sc, 8, 2, 128)
    _check_close(out_tl, out_ref, "30% invalid indices")

    print("All TileLang tests passed.")


if __name__ == "__main__":
    main()
