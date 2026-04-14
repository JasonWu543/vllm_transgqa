"""Unit tests for sparse_gqa_attention Triton kernel.

Compares the Triton kernel output against a pure-PyTorch reference
implementation that computes the exact same math (gather K/V by indices,
masked softmax, weighted sum) in float32 for maximum precision.

Run:
    python test_sparse_gqa.py

Requires CUDA (exits 0 with SKIP message if unavailable).
"""
import math
import sys

import torch

from vllm.v1.attention.ops.sparse_gqa import sparse_gqa_attention


# ============================================================================
# Pure-PyTorch reference (float32, no approximation)
# ============================================================================

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


# ============================================================================
# Helper
# ============================================================================

def _make_test_data(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    topk_k: int,
    total_slots: int,
    invalid_ratio: float = 0.0,
    device: str = "cuda",
):
    torch.manual_seed(42)

    q = torch.randn(num_tokens, num_heads, head_dim,
                     dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(total_slots, num_kv_heads, head_dim,
                            dtype=torch.bfloat16, device=device)
    value_cache = torch.randn(total_slots, num_kv_heads, head_dim,
                              dtype=torch.bfloat16, device=device)

    global_indices = torch.randint(
        0, total_slots, (num_tokens, topk_k),
        dtype=torch.int32, device=device)

    if invalid_ratio > 0:
        mask = torch.rand(num_tokens, topk_k, device=device) < invalid_ratio
        global_indices[mask] = -1

    softmax_scale = 1.0 / math.sqrt(head_dim)

    return q, key_cache, value_cache, global_indices, softmax_scale


def _check_close(out_triton, out_ref, label):
    diff = (out_triton.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    passed = max_diff < 2e-2
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}  (max_diff={max_diff:.6f})")
    if not passed:
        raise AssertionError(
            f"{label}: max absolute diff {max_diff:.6f} exceeds 2e-2")


# ============================================================================
# Tests
# ============================================================================

def test_basic_correctness(num_tokens, num_heads, num_kv_heads,
                           head_dim, topk_k, label):
    total_slots = 2048
    q, kc, vc, idx, scale = _make_test_data(
        num_tokens, num_heads, num_kv_heads, head_dim, topk_k, total_slots)

    out_triton = sparse_gqa_attention(
        q, kc, vc, idx, scale, num_heads, num_kv_heads, head_dim)
    out_ref = sparse_gqa_attention_ref(
        q, kc, vc, idx, scale, num_heads, num_kv_heads, head_dim)

    _check_close(out_triton, out_ref, label)


def test_with_invalid_indices():
    q, kc, vc, idx, scale = _make_test_data(
        num_tokens=8, num_heads=8, num_kv_heads=2,
        head_dim=128, topk_k=64, total_slots=1024,
        invalid_ratio=0.3)

    out_triton = sparse_gqa_attention(q, kc, vc, idx, scale, 8, 2, 128)
    out_ref = sparse_gqa_attention_ref(q, kc, vc, idx, scale, 8, 2, 128)
    _check_close(out_triton, out_ref, "Invalid indices (30% masked)")


def test_all_invalid_indices():
    q, kc, vc, _, scale = _make_test_data(
        num_tokens=4, num_heads=8, num_kv_heads=2,
        head_dim=64, topk_k=32, total_slots=512)

    idx_all_invalid = torch.full(
        (4, 32), -1, dtype=torch.int32, device="cuda")

    out_triton = sparse_gqa_attention(
        q, kc, vc, idx_all_invalid, scale, 8, 2, 64)

    max_val = out_triton.abs().max().item()
    passed = max_val < 1e-2
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] All invalid -> near-zero  (max_abs={max_val:.6f})")
    if not passed:
        raise AssertionError(f"All-invalid: max abs {max_val:.6f} >= 1e-2")


def test_single_valid_index():
    num_tokens, num_heads, num_kv_heads, head_dim = 2, 4, 2, 64
    topk_k, total_slots = 16, 256

    q, kc, vc, _, scale = _make_test_data(
        num_tokens, num_heads, num_kv_heads, head_dim, topk_k, total_slots)

    idx = torch.full((num_tokens, topk_k), -1,
                     dtype=torch.int32, device="cuda")
    chosen_slots = torch.randint(0, total_slots, (num_tokens,),
                                 device="cuda", dtype=torch.int32)
    idx[:, 0] = chosen_slots

    out_triton = sparse_gqa_attention(
        q, kc, vc, idx, scale, num_heads, num_kv_heads, head_dim)
    out_ref = sparse_gqa_attention_ref(
        q, kc, vc, idx, scale, num_heads, num_kv_heads, head_dim)
    _check_close(out_triton, out_ref, "Single valid index")


def test_topk_not_power_of_2():
    q, kc, vc, idx, scale = _make_test_data(
        num_tokens=4, num_heads=8, num_kv_heads=4,
        head_dim=128, topk_k=37, total_slots=512)

    out_triton = sparse_gqa_attention(q, kc, vc, idx, scale, 8, 4, 128)
    out_ref = sparse_gqa_attention_ref(q, kc, vc, idx, scale, 8, 4, 128)
    _check_close(out_triton, out_ref, "Non-power-of-2 topk_k=37")


def test_head_dim_not_power_of_2():
    q, kc, vc, idx, scale = _make_test_data(
        num_tokens=4, num_heads=8, num_kv_heads=2,
        head_dim=96, topk_k=32, total_slots=512)

    out_triton = sparse_gqa_attention(q, kc, vc, idx, scale, 8, 2, 96)
    out_ref = sparse_gqa_attention_ref(q, kc, vc, idx, scale, 8, 2, 96)
    _check_close(out_triton, out_ref, "Non-power-of-2 head_dim=96")


def test_large_topk():
    q, kc, vc, idx, scale = _make_test_data(
        num_tokens=2, num_heads=8, num_kv_heads=2,
        head_dim=128, topk_k=512, total_slots=4096)

    out_triton = sparse_gqa_attention(q, kc, vc, idx, scale, 8, 2, 128)
    out_ref = sparse_gqa_attention_ref(q, kc, vc, idx, scale, 8, 2, 128)
    _check_close(out_triton, out_ref, "Large topk_k=512 (multi-block)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: test_sparse_gqa.py requires CUDA (no GPU in this environment).")
        sys.exit(0)

    print("=" * 60)
    print("Sparse GQA Attention — Triton vs PyTorch Reference")
    print("=" * 60)

    configs = [
        dict(num_tokens=8, num_heads=8, num_kv_heads=8,
             head_dim=128, topk_k=64,
             label="MHA  8h/8kv  dim=128 topk=64"),
        dict(num_tokens=8, num_heads=8, num_kv_heads=2,
             head_dim=128, topk_k=64,
             label="GQA  8h/2kv  dim=128 topk=64"),
        dict(num_tokens=8, num_heads=16, num_kv_heads=1,
             head_dim=64, topk_k=32,
             label="MQA 16h/1kv  dim=64  topk=32"),
        dict(num_tokens=4, num_heads=8, num_kv_heads=2,
             head_dim=128, topk_k=256,
             label="GQA  8h/2kv  dim=128 topk=256"),
        dict(num_tokens=1, num_heads=8, num_kv_heads=2,
             head_dim=64, topk_k=16,
             label="GQA  8h/2kv  dim=64  topk=16  (single token)"),
        dict(num_tokens=32, num_heads=16, num_kv_heads=1,
             head_dim=128, topk_k=256,
             label="MQA 16h/1kv  dim=128 topk=256 (batch=32)"),
    ]

    for cfg in configs:
        test_basic_correctness(**cfg)

    test_with_invalid_indices()
    test_all_invalid_indices()
    test_single_valid_index()
    test_topk_not_power_of_2()
    test_head_dim_not_power_of_2()
    test_large_topk()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
