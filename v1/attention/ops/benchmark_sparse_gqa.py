"""Benchmark: sparse GQA — Triton kernel vs PyTorch reference (timing only).

The reference in ``test_sparse_gqa.py`` uses Python nested loops and is meant
for correctness; it can be orders of magnitude slower than Triton. For large
shapes this script skips the ref by default unless ``--force-ref`` is set.

Run (from repo root with PYTHONPATH so ``import vllm`` works):

  python v1/attention/ops/benchmark_sparse_gqa.py

  python v1/attention/ops/benchmark_sparse_gqa.py --num-tokens 512 --topk 256 --repeat 100
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from vllm.v1.attention.ops.sparse_gqa import sparse_gqa_attention
from vllm.v1.attention.ops.test_sparse_gqa import (
    _make_test_data,
    sparse_gqa_attention_ref,
)


def _cuda_ms(fn, warmup: int, repeat: int) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for ``fn()`` using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(repeat):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    std = var**0.5
    return mean, std


def _ref_work(nt: int, nh: int, topk: int, hd: int) -> int:
    """Rough scalar work estimate for the Python reference."""
    return nt * nh * topk * hd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # Defaults chosen so nt*nh*topk*hd < ref_threshold (both Triton + ref run).
    p.add_argument("--num-tokens", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--total-slots", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument(
        "--force-ref",
        action="store_true",
        help="Run PyTorch reference timing even for large shapes (can be very slow).",
    )
    p.add_argument(
        "--skip-ref",
        action="store_true",
        help="Only benchmark Triton (no PyTorch reference).",
    )
    p.add_argument(
        "--ref-threshold",
        type=int,
        default=50_000_000,
        help="Skip ref if _ref_work(nt,nh,topk,hd) exceeds this (unless --force-ref).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    nt, nh, nkv, hd, topk = (
        args.num_tokens,
        args.num_heads,
        args.num_kv_heads,
        args.head_dim,
        args.topk,
    )

    q, kc, vc, idx, scale = _make_test_data(
        nt,
        nh,
        nkv,
        hd,
        topk,
        args.total_slots,
        invalid_ratio=0.0,
        device=device,
    )

    work = _ref_work(nt, nh, topk, hd)
    skip_ref = args.skip_ref or (
        not args.force_ref and work > args.ref_threshold
    )

    print("Sparse GQA benchmark (same tensors for both paths)")
    print(
        f"  shape: tokens={nt} heads={nh} kv_heads={nkv} "
        f"head_dim={hd} topk={topk} total_slots={args.total_slots}"
    )
    print(f"  warmup={args.warmup} repeat={args.repeat}")
    print(f"  ref_work_estimate (nt*nh*topk*hd)={work}")
    print()

    def run_triton():
        sparse_gqa_attention(q, kc, vc, idx, scale, nh, nkv, hd)

    t_mean, t_std = _cuda_ms(run_triton, args.warmup, args.repeat)
    print(f"Triton:  {t_mean:.4f} ms ± {t_std:.4f}  (per forward)")

    if skip_ref:
        if not args.skip_ref and work > args.ref_threshold:
            print(
                f"PyTorch ref: SKIPPED (work > {args.ref_threshold}; "
                "use --force-ref to run, or reduce shape)"
            )
        else:
            print("PyTorch ref: SKIPPED (--skip-ref)")
        print()
        print(f"speedup vs ref: N/A")
        return

    def run_ref():
        sparse_gqa_attention_ref(q, kc, vc, idx, scale, nh, nkv, hd)

    # Reference is CPU-Python heavy; use fewer repeats if one iter is slow.
    ref_repeat = min(args.repeat, 20)
    ref_warmup = min(args.warmup, 5)
    r_mean, r_std = _cuda_ms(run_ref, ref_warmup, ref_repeat)
    print(f"PyTorch ref: {r_mean:.4f} ms ± {r_std:.4f}  (per forward, {ref_repeat} samples)")

    speedup = r_mean / t_mean if t_mean > 0 else float("inf")
    print()
    print(f"speedup (ref / Triton): {speedup:.2f}x")
    print()
    print(
        "Note: PyTorch reference uses nested Python loops in test_sparse_gqa.py; "
        "large shapes can take minutes. Triton is the intended production path."
    )


if __name__ == "__main__":
    main()
