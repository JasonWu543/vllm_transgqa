"""Phase 3: TransGQA indexer Q/K/weights path vs inline reference.

1) ``test_fold_shapes`` — only PyTorch; checks frequency-fold output shapes.
2) ``test_monolithic_matches_reference`` — lazy-imports ``TransGQAIndexer``;
   skips with exit 0 if vLLM deps are not installed (e.g. minimal CI).

Run:
  PYTHONPATH=<parent-of-vllm-package> python v1/attention/ops/test_indexer_math.py

Keep ``index_q_k_weights_reference`` in sync with
``model_executor/models/transgqa.py`` ``TransGQAIndexer._index_q_k_weights_monolithic``.
"""
from __future__ import annotations

import math
import sys

import torch
from torch import nn


def _fold_k_standalone(t: torch.Tensor, G: int, freqfold: int):
    """Same as TransGQAIndexer._fold_k (keep in sync)."""
    D = t.shape[-1]
    Df = D // 2
    ff = freqfold
    nband = Df // ff
    t = t.transpose(-1, -2).contiguous()
    real = (t[:, :Df, :].reshape(-1, nband, ff, G)
            .transpose(1, 2).reshape(-1, nband, G * ff))
    imag = (t[:, Df:, :].reshape(-1, nband, ff, G)
            .transpose(1, 2).reshape(-1, nband, G * ff))
    return real, imag


def _fold_q_standalone(t: torch.Tensor, G: int, groups: int, freqfold: int):
    """Same as TransGQAIndexer._fold_q (keep in sync)."""
    D = t.shape[-1]
    Df = D // 2
    ff = freqfold
    nband = Df // ff
    t = (t.view(-1, G, groups, D).permute(0, 2, 1, 3)
         .transpose(-1, -2).contiguous())
    real = (t[:, :, :Df, :].reshape(-1, groups, nband, ff, G)
            .transpose(2, 3).reshape(-1, groups, nband, G * ff))
    imag = (t[:, :, Df:, :].reshape(-1, groups, nband, ff, G)
            .transpose(2, 3).reshape(-1, groups, nband, G * ff))
    return real, imag


def index_q_k_weights_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    qk_index_proj: torch.Tensor,
    v_transform: torch.Tensor,
    qk_nope_proj_weight: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    freqfold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference copy of indexer monolithic path (sync with transgqa.py)."""
    G = num_kv_heads
    groups = num_heads // G
    ff = freqfold
    half_ff = ff // 2
    D = q.shape[-1]
    nband = D // (2 * ff)
    qk_w = qk_index_proj.to(dtype=q.dtype, device=q.device)
    v_w = v_transform.to(dtype=q.dtype, device=q.device)

    kr_re, kr_im = _fold_k_standalone(k, G, ff)
    kr_re_p = torch.einsum("bnf,nfr->bnr", kr_re, qk_w)
    kr_im_p = torch.einsum("bnf,nfr->bnr", kr_im, qk_w)
    index_k_rope = torch.cat([
        kr_re_p[:, :, :half_ff].reshape(-1, nband * half_ff),
        kr_im_p[:, :, :half_ff].reshape(-1, nband * half_ff),
    ], dim=-1).to(k.dtype)

    qr_re, qr_im = _fold_q_standalone(q, G, groups, ff)
    qr_re_p = torch.einsum("tgnf,nfr->tgnr", qr_re, qk_w)
    qr_im_p = torch.einsum("tgnf,nfr->tgnr", qr_im, qk_w)
    index_q_rope = torch.cat([
        qr_re_p[:, :, :, :half_ff].reshape(-1, groups, nband * half_ff),
        qr_im_p[:, :, :, :half_ff].reshape(-1, groups, nband * half_ff),
    ], dim=-1).to(q.dtype)

    nope_cols = G * ff - half_ff
    kn_re, kn_im = _fold_k_standalone(k_orig, G, ff)
    kn_re_p = torch.einsum("bnf,nfr->bnr", kn_re, qk_w)
    kn_im_p = torch.einsum("bnf,nfr->bnr", kn_im, qk_w)
    k_nope_rotary = torch.cat([
        kn_re_p[:, :, half_ff:].reshape(-1, nband * nope_cols),
        kn_im_p[:, :, half_ff:].reshape(-1, nband * nope_cols),
    ], dim=-1).to(k.dtype)
    k_nope_lr = torch.nn.functional.linear(
        k_nope_rotary, qk_nope_proj_weight).to(k.dtype)
    index_k = torch.cat([index_k_rope, k_nope_lr], dim=-1)

    qn_re, qn_im = _fold_q_standalone(q_orig, G, groups, ff)
    qn_re_p = torch.einsum("tgnf,nfr->tgnr", qn_re, qk_w)
    qn_im_p = torch.einsum("tgnf,nfr->tgnr", qn_im, qk_w)
    q_nope_rotary = torch.cat([
        qn_re_p[:, :, :, half_ff:].reshape(-1, groups, nband * nope_cols),
        qn_im_p[:, :, :, half_ff:].reshape(-1, groups, nband * nope_cols),
    ], dim=-1).to(q.dtype)
    q_nope_lr = torch.nn.functional.linear(
        q_nope_rotary, qk_nope_proj_weight).to(q.dtype)
    index_q = torch.cat([index_q_rope, q_nope_lr], dim=-1)

    v_h = v.repeat_interleave(groups, dim=1)
    w = torch.einsum("nhd,hdr->nhr", v_h, v_w)
    w = w.view(-1, G, groups, v_w.shape[-1])
    w = w.permute(0, 2, 1, 3).reshape(-1, groups, G * v_w.shape[-1])
    weights = torch.norm(w.to(torch.float32), p=2, dim=-1)

    return index_q.contiguous(), index_k.contiguous(), weights.contiguous()


def test_fold_shapes() -> None:
    N, G, D, ff = 3, 8, 128, 2
    nband = (D // 2) // ff
    gff = G * ff
    k = torch.randn(N, G, D)
    re, im = _fold_k_standalone(k, G, ff)
    assert re.shape == (N, nband, gff)
    assert im.shape == (N, nband, gff)
    H, groups = 32, 32 // G
    q = torch.randn(N, H, D)
    qre, qim = _fold_q_standalone(q, G, groups, ff)
    assert qre.shape == (N, groups, nband, gff)
    print("  [PASS] fold output shapes")


def test_monolithic_matches_reference() -> None:
    try:
        from vllm.model_executor.models.transgqa import TransGQAIndexer
    except ImportError as e:
        print(f"  [SKIP] TransGQAIndexer import ({e})")
        return

    torch.manual_seed(1)
    N, H, G, D = 5, 32, 8, 128
    ff = 2
    groups = H // G
    nband = D // (2 * ff)
    gff = G * ff
    nope_in = G * D - D // 2

    idx = TransGQAIndexer.__new__(TransGQAIndexer)
    idx.num_kv_heads = G
    idx.num_heads = H
    idx.n_head = groups
    idx.freqfold = ff
    idx.head_dim = D
    idx.qk_index_proj = nn.Parameter(torch.randn(nband, gff, gff))
    idx.v_transform = nn.Parameter(torch.randn(H, D, 4))
    idx.qk_nope_proj = nn.Linear(nope_in, D // 2, bias=False)

    q = torch.randn(N, H, D)
    k = torch.randn(N, G, D)
    v = torch.randn(N, G, D)
    q_orig = torch.randn(N, H, D)
    k_orig = torch.randn(N, G, D)

    impl = TransGQAIndexer._index_q_k_weights_monolithic(
        idx, q, k, v, q_orig, k_orig)

    ref = index_q_k_weights_reference(
        q, k, v, q_orig, k_orig,
        idx.qk_index_proj.data,
        idx.v_transform.data,
        idx.qk_nope_proj.weight.data,
        G, H, ff,
    )

    for name, a, b in zip(
        ("index_q", "index_k", "weights"),
        impl,
        ref,
        strict=True,
    ):
        d = (a.float() - b.float()).abs().max().item()
        ok = d < 1e-5
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} max_abs_diff={d:.3e}")
        if not ok:
            raise AssertionError(f"{name} mismatch max_abs_diff={d}")

    iq, ik, w = impl
    assert iq.shape == (N, groups, D)
    assert ik.shape == (N, D)
    assert w.shape == (N, groups)
    print("  [PASS] monolithic vs reference + shape checks")


def main() -> None:
    print("Phase 3: indexer math")
    test_fold_shapes()
    test_monolithic_matches_reference()
    print("Phase 3 indexer math: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
