"""Vectorized PyTorch reference for sparse GQA attention.

Same math as the loop-based ``sparse_gqa_attention_ref`` in
``test_sparse_gqa.py``, but fully batched over tokens and heads —
no Python for-loops.  This is NOT a production kernel (still pure
PyTorch), but gives a fairer timing baseline vs Triton.

Run standalone for a quick correctness + speed check:

    python v1/attention/ops/sparse_gqa_ref_vectorized.py
"""
from __future__ import annotations

import math
import sys
import time

import torch


def sparse_gqa_attention_ref_vectorized(
    q: torch.Tensor,              # [num_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,      # [total_slots, num_kv_heads, head_dim]
    value_cache: torch.Tensor,    # [total_slots, num_kv_heads, head_dim]
    global_indices: torch.Tensor, # [num_tokens, topk_k] int32, -1=invalid
    softmax_scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Vectorized (no Python loops) sparse GQA attention reference.

    Semantics identical to the loop reference:
      for each (token, query_head):
        gather K/V at global_indices from the correct kv_head,
        masked softmax, weighted sum over V.
    """
    num_tokens = q.shape[0]
    topk_k = global_indices.shape[1]
    gqa_group_size = num_heads // num_kv_heads

    # ---- safe indices & validity mask ----
    valid = global_indices >= 0                          # [T, topk]
    safe_idx = global_indices.clamp(min=0).long()        # [T, topk]

    # ---- build per-head kv_head mapping ----
    # kv_head_ids: [num_heads]
    kv_head_ids = torch.arange(num_heads, device=q.device) // gqa_group_size

    # ---- gather K and V for every (token, kv_head, candidate) ----
    # key_cache is [total_slots, num_kv_heads, head_dim].
    # We want k_gathered[t, kv_h, i, :] = key_cache[safe_idx[t,i], kv_h, :].
    #
    # Strategy: first gather along dim-0 to get [T, topk, num_kv_heads, hd],
    # then select the right kv_head per query head later.
    flat_idx = safe_idx.reshape(-1)                      # [T*topk]
    kc_gathered = key_cache[flat_idx]                     # [T*topk, nkv, hd]
    vc_gathered = value_cache[flat_idx]                   # [T*topk, nkv, hd]

    kc_gathered = kc_gathered.reshape(num_tokens, topk_k, num_kv_heads, head_dim)
    vc_gathered = vc_gathered.reshape(num_tokens, topk_k, num_kv_heads, head_dim)

    # ---- select the right kv_head for each query head ----
    # kv_head_ids: [H]  ->  index into dim=2 of [T, topk, nkv, hd]
    # Result: [T, topk, H, hd]
    kc_per_head = kc_gathered[:, :, kv_head_ids, :]      # [T, topk, H, hd]
    vc_per_head = vc_gathered[:, :, kv_head_ids, :]      # [T, topk, H, hd]

    # Transpose to [T, H, topk, hd] for batched dot-product
    kc_per_head = kc_per_head.permute(0, 2, 1, 3).float()
    vc_per_head = vc_per_head.permute(0, 2, 1, 3).float()

    # q: [T, H, hd] -> [T, H, hd, 1] for batched matmul against K
    q_f32 = q.float().unsqueeze(-1)                      # [T, H, hd, 1]

    # scores[t, h, i] = sum_d( K[t,h,i,d] * q[t,h,d] ) * scale
    scores = torch.matmul(kc_per_head, q_f32).squeeze(-1) * softmax_scale
    # scores: [T, H, topk]

    # ---- mask invalid positions ----
    # valid: [T, topk] -> [T, 1, topk] broadcast over H
    scores = scores.masked_fill(~valid.unsqueeze(1), float('-inf'))

    # ---- numerically-stable softmax ----
    scores_max, _ = scores.max(dim=-1, keepdim=True)     # [T, H, 1]
    all_invalid = (scores_max == float('-inf'))
    scores_max = scores_max.masked_fill(all_invalid, 0.0)

    exp_scores = torch.exp(scores - scores_max)
    exp_scores = exp_scores.masked_fill(~valid.unsqueeze(1), 0.0)

    denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [T,H,1]
    weights = exp_scores / denom                                   # [T,H,topk]

    # ---- weighted sum over V ----
    # weights: [T, H, 1, topk]  x  vc_per_head: [T, H, topk, hd]
    output = torch.matmul(weights.unsqueeze(-2), vc_per_head).squeeze(-2)
    # output: [T, H, hd]

    # all-invalid heads -> zero
    output = output.masked_fill(all_invalid, 0.0)

    return output.to(q.dtype)


# ======================================================================
# Quick self-test & timing when run directly
# ======================================================================

def _make_data(nt, nh, nkv, hd, topk, total_slots, invalid_ratio=0.0,
               device="cuda"):
    torch.manual_seed(42)
    q = torch.randn(nt, nh, hd, dtype=torch.bfloat16, device=device)
    kc = torch.randn(total_slots, nkv, hd, dtype=torch.bfloat16, device=device)
    vc = torch.randn(total_slots, nkv, hd, dtype=torch.bfloat16, device=device)
    idx = torch.randint(0, total_slots, (nt, topk), dtype=torch.int32,
                        device=device)
    if invalid_ratio > 0:
        mask = torch.rand(nt, topk, device=device) < invalid_ratio
        idx[mask] = -1
    scale = 1.0 / math.sqrt(hd)
    return q, kc, vc, idx, scale


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available"); sys.exit(0)

    # ---- correctness vs loop ref ----
    from vllm.v1.attention.ops.test_sparse_gqa import sparse_gqa_attention_ref

    print("=" * 60)
    print("Vectorized PyTorch ref — correctness & timing")
    print("=" * 60)

    cases = [
        (8,  8,  2, 128, 64,  2048, 0.0, "GQA 8h/2kv topk=64"),
        (8, 16,  1,  64, 32,  2048, 0.0, "MQA 16h/1kv topk=32"),
        (8,  8,  2, 128, 64,  1024, 0.3, "GQA 30% invalid"),
    ]
    for nt, nh, nkv, hd, topk, ts, inv, label in cases:
        q, kc, vc, idx, sc = _make_data(nt, nh, nkv, hd, topk, ts, inv)
        out_vec = sparse_gqa_attention_ref_vectorized(
            q, kc, vc, idx, sc, nh, nkv, hd)
        out_ref = sparse_gqa_attention_ref(
            q, kc, vc, idx, sc, nh, nkv, hd)
        diff = (out_vec.float() - out_ref.float()).abs().max().item()
        ok = diff < 1e-5
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  (max_diff={diff:.7f})")
        if not ok:
            raise AssertionError(f"{label}: diff {diff:.7f}")

    # ---- timing ----
    print()
    nt, nh, nkv, hd, topk, ts = 64, 32, 8, 128, 128, 8192
    q, kc, vc, idx, sc = _make_data(nt, nh, nkv, hd, topk, ts)

    warmup, repeat = 10, 50

    def run_vec():
        sparse_gqa_attention_ref_vectorized(q, kc, vc, idx, sc, nh, nkv, hd)

    for _ in range(warmup):
        run_vec()
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        starter.record()
        run_vec()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    print(f"Vectorized ref:  {mean:.4f} ms ± {std:.4f}  "
          f"(tokens={nt} heads={nh} kv={nkv} dim={hd} topk={topk})")

    # Compare with loop ref (fewer repeats)
    from vllm.v1.attention.ops.test_sparse_gqa import sparse_gqa_attention_ref
    def run_loop():
        sparse_gqa_attention_ref(q, kc, vc, idx, sc, nh, nkv, hd)

    for _ in range(2):
        run_loop()
    torch.cuda.synchronize()

    loop_times = []
    for _ in range(5):
        starter.record()
        run_loop()
        ender.record()
        torch.cuda.synchronize()
        loop_times.append(starter.elapsed_time(ender))
    loop_mean = sum(loop_times) / len(loop_times)
    print(f"Loop ref:        {loop_mean:.4f} ms  (5 samples)")
    print(f"Vectorized speedup over loop: {loop_mean / mean:.1f}x")
    print()
    print("(Triton will still be faster — this just shows how much was "
          "Python-loop overhead vs actual compute.)")


if __name__ == "__main__":
    main()
