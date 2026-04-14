"""Benchmark & correctness comparison: Triton kernel vs Vectorized PyTorch ref.

Compares:
  1. Triton sparse GQA kernel  (sparse_gqa.py)
  2. Vectorized PyTorch ref    (sparse_gqa_ref_vectorized.py)
  3. Loop-based PyTorch ref    (_sparse_gqa_ref_loop.py)  [correctness only]

Covers:
  - Extensive corner-case correctness tests (~30 configs)
  - Performance sweeps across 5 dimensions:
      A. Batch size (num_tokens)
      B. Top-k (sparse attention width)
      C. Total slots (KV cache size)
      D. Head configuration (MHA / GQA / MQA)
      E. Head dimension

Run:
    python v1/attention/ops/bench_triton_vs_vectorized.py

Requires CUDA.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Make sure we can import sibling modules from the same directory
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sparse_gqa import sparse_gqa_attention                       # Triton
from sparse_gqa_ref_vectorized import sparse_gqa_attention_ref_vectorized  # Vectorized
from _sparse_gqa_ref_loop import sparse_gqa_attention_ref         # Loop ref


# ============================================================================
# Helpers
# ============================================================================

def make_test_data(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    topk_k: int,
    total_slots: int,
    invalid_ratio: float = 0.0,
    all_invalid: bool = False,
    device: str = "cuda",
    seed: int = 42,
):
    """Generate random test data for sparse GQA attention.

    Args:
        all_invalid: if True, set ALL indices to -1 (edge case).
    """
    torch.manual_seed(seed)
    q = torch.randn(num_tokens, num_heads, head_dim,
                     dtype=torch.bfloat16, device=device)
    kc = torch.randn(total_slots, num_kv_heads, head_dim,
                      dtype=torch.bfloat16, device=device)
    vc = torch.randn(total_slots, num_kv_heads, head_dim,
                      dtype=torch.bfloat16, device=device)
    idx = torch.randint(0, total_slots, (num_tokens, topk_k),
                        dtype=torch.int32, device=device)
    if all_invalid:
        idx.fill_(-1)
    elif invalid_ratio > 0:
        mask = torch.rand(num_tokens, topk_k, device=device) < invalid_ratio
        idx[mask] = -1
    scale = 1.0 / math.sqrt(head_dim)
    return q, kc, vc, idx, scale


def bench_fn(fn: Callable, warmup: int = 20, repeat: int = 100) -> Tuple[float, float]:
    """Benchmark *fn* using CUDA events. Returns (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times: List[float] = []
    for _ in range(repeat):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std


def _print_section(title: str):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _print_bench_header():
    print(f"  {'Config':<55} {'Triton(ms)':>12} {'VecRef(ms)':>12} {'Speedup':>9}")
    print("  " + "-" * 90)


def _bench_one(label: str, nt, nh, nkv, hd, topk, ts,
               warmup_t=20, repeat_t=100, warmup_v=10, repeat_v=50):
    """Run one benchmark config and print the result line."""
    q, kc, vc, idx, sc = make_test_data(nt, nh, nkv, hd, topk, ts)

    # Use default args to capture current loop variables
    def run_triton(_q=q, _kc=kc, _vc=vc, _idx=idx, _sc=sc, _nh=nh, _nkv=nkv, _hd=hd):
        return sparse_gqa_attention(_q, _kc, _vc, _idx, _sc, _nh, _nkv, _hd)

    def run_vec(_q=q, _kc=kc, _vc=vc, _idx=idx, _sc=sc, _nh=nh, _nkv=nkv, _hd=hd):
        return sparse_gqa_attention_ref_vectorized(_q, _kc, _vc, _idx, _sc, _nh, _nkv, _hd)

    t_mean, t_std = bench_fn(run_triton, warmup=warmup_t, repeat=repeat_t)
    v_mean, v_std = bench_fn(run_vec,    warmup=warmup_v, repeat=repeat_v)

    speedup = v_mean / t_mean if t_mean > 0 else float('inf')
    print(f"  {label:<55} {t_mean:>6.4f}±{t_std:.3f} {v_mean:>6.4f}±{v_std:.3f} {speedup:>7.1f}x")
    return t_mean, v_mean


# ============================================================================
# Phase 1: Exhaustive correctness tests
# ============================================================================

def check_correctness():
    _print_section("Phase 1: Correctness — Triton vs Vectorized vs Loop ref")
    print()

    # Each entry: (nt, nh, nkv, hd, topk, total_slots, inv_ratio, all_invalid, label)
    configs = [
        # --- Basic configurations ---
        (8,   8,  8, 128,  64, 2048, 0.0, False, "MHA  8h/8kv   dim=128 topk=64"),
        (8,   8,  2, 128,  64, 2048, 0.0, False, "GQA  8h/2kv   dim=128 topk=64"),
        (8,  16,  1,  64,  32, 2048, 0.0, False, "MQA 16h/1kv   dim=64  topk=32"),
        (8,  32,  4, 128, 128, 4096, 0.0, False, "GQA 32h/4kv   dim=128 topk=128"),

        # --- Single token (decode-like) ---
        (1,   8,  2, 128,  64, 2048, 0.0, False, "Single token, GQA 8h/2kv topk=64"),
        (1,  32,  8, 128, 128, 8192, 0.0, False, "Single token, GQA 32h/8kv topk=128"),
        (1,   1,  1,  64,  16,  512, 0.0, False, "Single token, single head, dim=64"),

        # --- Large batch ---
        (64,  8,  2, 128,  64, 4096, 0.0, False, "Batch=64, GQA 8h/2kv topk=64"),
        (128, 32,  8, 128, 128, 8192, 0.0, False, "Batch=128, GQA 32h/8kv topk=128"),
        (256,  8,  2, 128,  64, 4096, 0.0, False, "Batch=256, GQA 8h/2kv topk=64"),

        # --- Different head dimensions ---
        (8,   8,  2,  32,  64, 2048, 0.0, False, "dim=32  (small head)"),
        (8,   8,  2,  64,  64, 2048, 0.0, False, "dim=64  (Llama-like)"),
        (8,   8,  2, 128,  64, 2048, 0.0, False, "dim=128 (standard)"),
        (8,   8,  2, 256,  64, 2048, 0.0, False, "dim=256 (large head)"),

        # --- Different topk values ---
        (8,   8,  2, 128,  16, 2048, 0.0, False, "topk=16  (very sparse)"),
        (8,   8,  2, 128,  32, 2048, 0.0, False, "topk=32  (sparse)"),
        (8,   8,  2, 128, 128, 4096, 0.0, False, "topk=128 (moderate)"),
        (8,   8,  2, 128, 256, 4096, 0.0, False, "topk=256 (dense)"),
        (4,   8,  2, 128, 512, 8192, 0.0, False, "topk=512 (very dense)"),

        # --- Extreme GQA ratios ---
        (8,  64,  1, 128,  64, 2048, 0.0, False, "MQA 64h/1kv  (extreme ratio=64)"),
        (8,  32,  2, 128,  64, 2048, 0.0, False, "GQA 32h/2kv  (ratio=16)"),
        (8,   4,  4, 128,  64, 2048, 0.0, False, "MHA  4h/4kv  (ratio=1)"),
        (8,   2,  1, 128,  64, 2048, 0.0, False, "GQA  2h/1kv  (minimal heads)"),

        # --- Invalid index handling ---
        (8,   8,  2, 128,  64, 1024, 0.1, False, "10% invalid indices"),
        (8,   8,  2, 128,  64, 1024, 0.3, False, "30% invalid indices"),
        (8,   8,  2, 128,  64, 1024, 0.5, False, "50% invalid indices"),
        (8,   8,  2, 128,  64, 1024, 0.9, False, "90% invalid indices"),
        (8,   8,  2, 128,  64, 1024, 0.0, True,  "ALL invalid (100%) -> zero output"),

        # --- Small KV cache (total_slots barely > topk) ---
        (4,   8,  2, 128,  32,   64, 0.0, False, "Tiny cache: 64 slots, topk=32"),
        (4,   8,  2, 128,  16,   16, 0.0, False, "Cache == topk (16 slots, topk=16)"),

        # --- Large KV cache ---
        (8,   8,  2, 128,  64, 32768, 0.0, False, "Large cache: 32K slots"),
        (4,   8,  2, 128, 128, 65536, 0.0, False, "Large cache: 64K slots"),
    ]

    all_pass = True
    pass_count = 0
    fail_count = 0

    for nt, nh, nkv, hd, topk, ts, inv, all_inv, label in configs:
        q, kc, vc, idx, sc = make_test_data(nt, nh, nkv, hd, topk, ts, inv, all_inv)

        out_triton = sparse_gqa_attention(q, kc, vc, idx, sc, nh, nkv, hd)
        out_vec    = sparse_gqa_attention_ref_vectorized(q, kc, vc, idx, sc, nh, nkv, hd)
        out_loop   = sparse_gqa_attention_ref(q, kc, vc, idx, sc, nh, nkv, hd)

        diff_tv = (out_triton.float() - out_vec.float()).abs().max().item()
        diff_tl = (out_triton.float() - out_loop.float()).abs().max().item()
        diff_vl = (out_vec.float()    - out_loop.float()).abs().max().item()

        # bfloat16 tolerance: 2e-2 for Triton vs ref, 2e-3 for vec vs loop
        # (large batches can accumulate slightly more FP rounding differences)
        ok_tv = diff_tv < 2e-2
        ok_tl = diff_tl < 2e-2
        ok_vl = diff_vl < 2e-3

        ok = ok_tv and ok_tl and ok_vl
        status = "PASS" if ok else "FAIL"
        if ok:
            pass_count += 1
        else:
            all_pass = False
            fail_count += 1

        print(f"  [{status}] {label}")
        if not ok:
            print(f"          Triton vs Vec:  {diff_tv:.6f}  {'✓' if ok_tv else '✗'}")
            print(f"          Triton vs Loop: {diff_tl:.6f}  {'✓' if ok_tl else '✗'}")
            print(f"          Vec    vs Loop: {diff_vl:.6f}  {'✓' if ok_vl else '✗'}")

    print()
    print(f"  Results: {pass_count} passed, {fail_count} failed, {pass_count + fail_count} total")
    if all_pass:
        print("  ✅ All correctness checks passed!")
    else:
        print("  ❌ Some correctness checks failed!")
    print()
    return all_pass


# ============================================================================
# Phase 2: Performance sweeps
# ============================================================================

def sweep_batch_size():
    """Sweep A: vary num_tokens (batch size) with fixed head config."""
    _print_section("Sweep A: Batch Size (num_tokens)")
    print("  Fixed: 32 heads, 8 kv_heads, dim=128, topk=128, 8192 slots")
    print()
    _print_bench_header()

    nh, nkv, hd, topk, ts = 32, 8, 128, 128, 8192
    for nt in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        label = f"batch={nt}"
        _bench_one(label, nt, nh, nkv, hd, topk, ts)
    print()


def sweep_topk():
    """Sweep B: vary topk (sparse attention width)."""
    _print_section("Sweep B: Top-K (sparse attention width)")
    print("  Fixed: batch=32, 32 heads, 8 kv_heads, dim=128, 16384 slots")
    print()
    _print_bench_header()

    nt, nh, nkv, hd, ts = 32, 32, 8, 128, 16384
    for topk in [16, 32, 64, 128, 256, 512, 1024]:
        label = f"topk={topk}"
        _bench_one(label, nt, nh, nkv, hd, topk, ts)
    print()


def sweep_total_slots():
    """Sweep C: vary total_slots (KV cache size)."""
    _print_section("Sweep C: Total Slots (KV cache size)")
    print("  Fixed: batch=32, 32 heads, 8 kv_heads, dim=128, topk=128")
    print()
    _print_bench_header()

    nt, nh, nkv, hd, topk = 32, 32, 8, 128, 128
    for ts in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        label = f"slots={ts}"
        _bench_one(label, nt, nh, nkv, hd, topk, ts)
    print()


def sweep_head_config():
    """Sweep D: vary head configuration (MHA / GQA / MQA)."""
    _print_section("Sweep D: Head Configuration (MHA / GQA / MQA)")
    print("  Fixed: batch=32, dim=128, topk=128, 8192 slots")
    print()
    _print_bench_header()

    nt, hd, topk, ts = 32, 128, 128, 8192
    configs = [
        # (nh, nkv, label)
        ( 1,  1, "MHA  1h/1kv   (ratio=1)"),
        ( 4,  4, "MHA  4h/4kv   (ratio=1)"),
        ( 8,  8, "MHA  8h/8kv   (ratio=1)"),
        ( 8,  4, "GQA  8h/4kv   (ratio=2)"),
        ( 8,  2, "GQA  8h/2kv   (ratio=4)"),
        ( 8,  1, "MQA  8h/1kv   (ratio=8)"),
        (16,  4, "GQA 16h/4kv   (ratio=4)"),
        (32,  8, "GQA 32h/8kv   (ratio=4)"),
        (32,  4, "GQA 32h/4kv   (ratio=8)"),
        (32,  2, "GQA 32h/2kv   (ratio=16)"),
        (32,  1, "MQA 32h/1kv   (ratio=32)"),
        (64,  8, "GQA 64h/8kv   (ratio=8)"),
        (64,  1, "MQA 64h/1kv   (ratio=64)"),
    ]
    for nh, nkv, label in configs:
        _bench_one(label, nt, nh, nkv, hd, topk, ts)
    print()


def sweep_head_dim():
    """Sweep E: vary head_dim."""
    _print_section("Sweep E: Head Dimension")
    print("  Fixed: batch=32, 32 heads, 8 kv_heads, topk=128, 8192 slots")
    print()
    _print_bench_header()

    nt, nh, nkv, topk, ts = 32, 32, 8, 128, 8192
    for hd in [32, 64, 128, 256]:
        label = f"head_dim={hd}"
        _bench_one(label, nt, nh, nkv, hd, topk, ts)
    print()


def sweep_invalid_ratio():
    """Sweep F: vary invalid index ratio."""
    _print_section("Sweep F: Invalid Index Ratio")
    print("  Fixed: batch=32, 32 heads, 8 kv_heads, dim=128, topk=128, 8192 slots")
    print()
    _print_bench_header()

    nt, nh, nkv, hd, topk, ts = 32, 32, 8, 128, 128, 8192
    for inv in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        q, kc, vc, idx, sc = make_test_data(nt, nh, nkv, hd, topk, ts, invalid_ratio=inv)
        label = f"invalid={inv*100:.0f}%"

        def run_triton(_q=q, _kc=kc, _vc=vc, _idx=idx, _sc=sc):
            return sparse_gqa_attention(_q, _kc, _vc, _idx, _sc, nh, nkv, hd)

        def run_vec(_q=q, _kc=kc, _vc=vc, _idx=idx, _sc=sc):
            return sparse_gqa_attention_ref_vectorized(_q, _kc, _vc, _idx, _sc, nh, nkv, hd)

        t_mean, t_std = bench_fn(run_triton, warmup=20, repeat=100)
        v_mean, v_std = bench_fn(run_vec,    warmup=10, repeat=50)
        speedup = v_mean / t_mean if t_mean > 0 else float('inf')
        print(f"  {label:<55} {t_mean:>6.4f}±{t_std:.3f} {v_mean:>6.4f}±{v_std:.3f} {speedup:>7.1f}x")
    print()


def run_realistic_scenarios():
    """Realistic model configurations (Llama-like, Mistral-like, etc.)."""
    _print_section("Realistic Model Scenarios")
    print()
    _print_bench_header()

    scenarios = [
        # (nt, nh, nkv, hd, topk, ts, label)
        # Llama-2 7B style: 32 heads, 32 kv_heads (MHA), dim=128
        (1,   32, 32, 128,  64,  4096, "Llama2-7B decode  (MHA, topk=64, 4K ctx)"),
        (32,  32, 32, 128,  64,  4096, "Llama2-7B batch32 (MHA, topk=64, 4K ctx)"),
        (1,   32, 32, 128, 128, 16384, "Llama2-7B decode  (MHA, topk=128, 16K ctx)"),

        # Llama-2 70B style: 64 heads, 8 kv_heads (GQA), dim=128
        (1,   64,  8, 128,  64,  4096, "Llama2-70B decode (GQA 64/8, topk=64, 4K)"),
        (32,  64,  8, 128,  64,  4096, "Llama2-70B batch32(GQA 64/8, topk=64, 4K)"),
        (1,   64,  8, 128, 256, 32768, "Llama2-70B decode (GQA 64/8, topk=256, 32K)"),
        (64,  64,  8, 128, 256, 32768, "Llama2-70B batch64(GQA 64/8, topk=256, 32K)"),

        # Mistral-7B style: 32 heads, 8 kv_heads (GQA), dim=128
        (1,   32,  8, 128,  64,  8192, "Mistral-7B decode (GQA 32/8, topk=64, 8K)"),
        (32,  32,  8, 128, 128,  8192, "Mistral-7B batch32(GQA 32/8, topk=128, 8K)"),
        (64,  32,  8, 128, 256, 32768, "Mistral-7B batch64(GQA 32/8, topk=256, 32K)"),

        # Qwen-2 style: 28 heads, 4 kv_heads (GQA), dim=128
        (1,   28,  4, 128,  64,  8192, "Qwen2-7B decode   (GQA 28/4, topk=64, 8K)"),
        (32,  28,  4, 128, 128, 32768, "Qwen2-7B batch32  (GQA 28/4, topk=128, 32K)"),

        # DeepSeek-V2 style: many heads, few kv_heads, dim=128
        (1,   64,  2, 128,  64, 16384, "DeepSeek-V2 decode(GQA 64/2, topk=64, 16K)"),
        (32,  64,  2, 128, 128, 32768, "DeepSeek-V2 batch (GQA 64/2, topk=128, 32K)"),
    ]

    for nt, nh, nkv, hd, topk, ts, label in scenarios:
        _bench_one(label, nt, nh, nkv, hd, topk, ts)
    print()


def loop_ref_comparison():
    """Compare all three implementations for a few configs (loop ref is slow)."""
    _print_section("Three-Way Comparison (incl. Loop ref — slow)")
    print()

    configs = [
        (8,   8,  2, 128,  64, 4096, "Small:  batch=8,  GQA 8/2,  topk=64"),
        (16, 32,  8, 128, 128, 8192, "Medium: batch=16, GQA 32/8, topk=128"),
    ]

    for nt, nh, nkv, hd, topk, ts, label in configs:
        q, kc, vc, idx, sc = make_test_data(nt, nh, nkv, hd, topk, ts)

        def run_triton(_q=q, _kc=kc, _vc=vc, _idx=idx, _sc=sc):
            return sparse_gqa_attention(_q, _kc, _vc, _idx, _sc, nh, nkv, hd)

        def run_vec(_q=q, _kc=kc, _vc=vc, _idx=idx, _sc=sc):
            return sparse_gqa_attention_ref_vectorized(_q, _kc, _vc, _idx, _sc, nh, nkv, hd)

        def run_loop(_q=q, _kc=kc, _vc=vc, _idx=idx, _sc=sc):
            return sparse_gqa_attention_ref(_q, _kc, _vc, _idx, _sc, nh, nkv, hd)

        t_mean, _ = bench_fn(run_triton, warmup=10, repeat=50)
        v_mean, _ = bench_fn(run_vec,    warmup=5,  repeat=20)
        l_mean, _ = bench_fn(run_loop,   warmup=2,  repeat=5)

        print(f"  {label}")
        print(f"    {'Triton kernel':<25} {t_mean:>10.4f} ms  (1.0x)")
        print(f"    {'Vectorized ref':<25} {v_mean:>10.4f} ms  ({v_mean/t_mean:.1f}x slower)")
        print(f"    {'Loop ref':<25} {l_mean:>10.4f} ms  ({l_mean/t_mean:.1f}x slower)")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        sys.exit(0)

    device_name = torch.cuda.get_device_name(0)
    print()
    print("╔" + "═" * 78 + "╗")
    print(f"║  Sparse GQA Attention — Triton vs Vectorized PyTorch Benchmark{' ' * 14}║")
    print(f"║  Device: {device_name:<68}║")
    print(f"║  PyTorch: {torch.__version__:<67}║")
    print("╚" + "═" * 78 + "╝")

    # ---- Phase 1: Correctness ----
    passed = check_correctness()
    if not passed:
        print("  ⚠️  Correctness issues detected — benchmark results may not be meaningful.\n")

    # ---- Phase 2: Performance sweeps ----
    sweep_batch_size()
    sweep_topk()
    sweep_total_slots()
    sweep_head_config()
    sweep_head_dim()
    sweep_invalid_ratio()

    # ---- Phase 3: Realistic scenarios ----
    run_realistic_scenarios()

    # ---- Phase 4: Three-way comparison ----
    loop_ref_comparison()

    print("=" * 80)
    print("  All benchmarks complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
