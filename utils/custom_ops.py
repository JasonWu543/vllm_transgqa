import torch
import tilelang
from tilelang import language as T

@tilelang.jit(
        pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def block_mqa_attn_return_logits(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
    dtype="bfloat16",
):
    if block_Q is None:
        block_Q = 128 // heads
    accum_dtype = "float32"
    index_dtype = "int32"

    seq_len = T.dynamic("seq_len")
    seq_len_blocked_kv = T.dynamic("seq_len_blocked_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_blocked_kv, index_dim]
    index_k_scale_shape = [seq_len_blocked_kv]
    logits_shape = [seq_len, seq_len_blocked_kv]

    @T.prim_func
    def block_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexBlockedK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenBlockedKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenBlockedKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenBlockedKS[seq_len_i + bq_i], seq_len_blocked_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenBlockedKE[seq_len_i + bq_i], seq_len_blocked_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                T.copy(IndexBlockedK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]) 

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]

    return block_mqa_attn_return_logits_kernel


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel

@tilelang.jit
def force_maintain_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def force_maintain_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx == cu_k_s or idx == cu_k_e - 1:
                        Logits[bx, idx] = T.infinity(dtype)

    return force_maintain_logits_kernel

@tilelang.jit
def clean_and_maintain_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_and_maintain_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx == cu_k_s or idx == cu_k_e - 1:
                        Logits[bx, idx] = T.infinity(dtype)
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_and_maintain_logits_kernel

def block_mqa_attn_return_logits_interface(q, blocked_kv, kv_block_size, weights, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke, clean_logits=True, force_maintain=True, dtype="bfloat16"):
    seq_len, heads, index_dim = q.shape
    seq_len_blocked_kv = blocked_kv.shape[0]

    block_mqa_attn_return_logits_kernel = block_mqa_attn_return_logits(heads=heads, index_dim=index_dim, dtype=dtype)
    logits = torch.empty([seq_len, seq_len_blocked_kv], device=q.device, dtype=torch.float32)
    block_mqa_attn_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        blocked_kv,
        logits,
        weights,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
    )
    if clean_logits and force_maintain:
        clean_and_maintain_logits_kernel = clean_and_maintain_logits_()
        clean_and_maintain_logits_kernel(logits, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)
    else:
        clean_logits_kernel = clean_logits_()
        force_maintain_logits_kernel = force_maintain_logits_()
        if clean_logits:
            clean_logits_kernel(logits, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)
        if force_maintain:
            force_maintain_logits_kernel(logits, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)
    return logits

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def block_sparse_mqa_attn_return_logits(
    kv_block_size,
    topk,
    heads,
    index_dim,
    block_N=128,
    num_stages=1,
    threads=256,
    dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    index_k_scale_shape = [seq_len_kv]
    logits_shape = [seq_len, topk * kv_block_size]

    # TODO check padded H in sparse_mla_fwd
    # does it matter here?
    H_per_block = heads
    block_N = T.min(block_N, kv_block_size)

    @T.prim_func
    def block_sparse_mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            index_q_shared = T.alloc_shared([H_per_block, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, H_per_block], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, H_per_block // heads, heads))
            logits = T.alloc_fragment([block_N, H_per_block // heads], accum_dtype)
            weights = T.alloc_fragment([H_per_block // heads, heads], accum_dtype)

            seq_len_i = bx

            cu_k_s_min = CuSeqLenKS[seq_len_i]
            cu_k_e_max = CuSeqLenKE[seq_len_i]

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for n_i in T.serial(topk):
                topk_block_id = TopKBlockIndex[seq_len_i, n_i]
                block_s = topk_block_id * kv_block_size
                for b_i in T.Pipelined(kv_block_size // block_N, num_stages=num_stages):
                    block_s_i = block_s + b_i * block_N
                    T.copy(IndexK[block_s_i, 0], index_k_shared)

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, H_per_block, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i])
                    
                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for i_i in T.Parallel(block_N):
                        k_i = block_s_i + i_i
                        if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                            logits[i_i, 0] = -T.infinity(accum_dtype)

                    for bn_i in T.Parallel(block_N):
                        Logits[seq_len_i, n_i * kv_block_size + b_i * block_N + bn_i] = logits[bn_i, 0] 

    return block_sparse_mqa_attn_return_logits_kernel

def block_sparse_mqa_attn_return_logits_interface(q, kv, topk_block_index, kv_block_size, weights, cu_seqlen_ks, cu_seqlen_ke, dtype="bfloat16"):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    topk = topk_block_index.shape[1]

    block_sparse_mqa_attn_return_logits_kernel = block_sparse_mqa_attn_return_logits(heads=heads, index_dim=index_dim, kv_block_size=kv_block_size, topk=topk)
    logits = torch.empty([seq_len, topk * kv_block_size], device=q.device, dtype=torch.float32)
    block_sparse_mqa_attn_return_logits_kernel(
        q.view(seq_len * heads, index_dim),
        kv,
        topk_block_index,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    return logits

def fp8_hierarchy_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    k_block_size: int, 
    block_topk: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    q = q.float()  # [M, H, D]
    k_fp8, k_scales = kv
    if k_scales.ndim == 1:
        k_scales = k_scales.unsqueeze(-1)  # [N, 1]
    k = k_fp8.float() * k_scales  # [N, D]
    q = q.bfloat16()
    k = k.bfloat16()
    weights = weights.bfloat16()

    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

    seq_len_k = k.shape[0]
    num_k_blocks = (seq_len_k + k_block_size - 1) // k_block_size
    blocked_k_mean = []

    for i in range(num_k_blocks):
        start_idx = i * k_block_size
        end_idx = min((i + 1) * k_block_size, seq_len_k)
        block_kv = k[start_idx:end_idx, :]
        block_mean = block_kv.mean(dim=0, keepdim=True)  # [block_size, D] -> [1, D]
        blocked_k_mean.append(block_mean)
    blocked_k = torch.cat(blocked_k_mean, dim=0)  # [num_block, D]

    block_k_indexer_score = block_mqa_attn_return_logits_interface(q=q, blocked_kv=blocked_k, kv_block_size=k_block_size, weights=weights, cu_seqlen_blocked_ks=cu_seqlen_blocked_ks, cu_seqlen_blocked_ke=cu_seqlen_blocked_ke)

    topk_block_indices = torch.topk(block_k_indexer_score, k=min(block_topk, block_k_indexer_score.shape[-1]), dim=-1).indices  # [M, topk]
    topk_block_indices = topk_block_indices.to(torch.int32)

    block_sparse_logits = block_sparse_mqa_attn_return_logits_interface(q=q, kv=k, topk_block_index=topk_block_indices, kv_block_size=k_block_size, weights=weights, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke)

    return block_sparse_logits, topk_block_indices

# def batch_block_mqa_attn_return_logits_interface(q, blocked_kv, kv_block_size, weights, context_lens, clean_logits=True, force_maintain=True, dtype="bfloat16"):
#     batch, seq_len, heads, index_dim = q.shape
#     seq_len_blocked_kv = blocked_kv.shape[1]



# def paged_block_sparse_mqa_attn_return_logits_interface(q, kv_cache, topk_block_index, kv_block_size, weights, context_lens, block_tables, dtype="bfloat16"):


# TODO gen by ai, check need
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def batch_decode_block_mqa_attn_return_logits(
    heads: int,
    index_dim: int,
    block_N: int = 64,
    block_H: int = 64,
    num_stages: int = 3,
    threads: int = 256,
    dtype: str = "bfloat16",
):
    """
    Decode 专用：q_len 固定为 1，不在 Q 维分块，只在 H 和 Nb 上分块。

    Shapes:
      Q:          [B, 1, H, D]
      BlockedK:   [B, Nb, D]
      Logits:     [B, 1, Nb] fp32
      Weights:    [B, 1, H]
      ContextLens:[B]  (有效 Nb)
    """
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    nb = T.dynamic("seq_len_blocked_kv")

    q_shape = [batch, heads, index_dim]
    k_shape = [batch, nb, index_dim]
    logits_shape = [batch, nb]
    w_shape = [batch, heads]

    # padding 到 16 对齐，避免 gemm 列维过小/不合法
    block_H_pad = T.ceildiv(block_H, 16) * 16
    assert block_H_pad == heads

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        BlockedK: T.Tensor(k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(w_shape, dtype),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch, 1, threads=threads) as (bx, by):
            # shared tiles
            k_shared = T.alloc_shared([block_N, index_dim], dtype)
            q_shared = T.alloc_shared([block_H_pad, index_dim], dtype)

            # fragments
            s = T.alloc_fragment([block_N, block_H_pad], accum_dtype)
            w = T.alloc_fragment([block_H_pad], accum_dtype)
            logits_accum = T.alloc_fragment([block_N], accum_dtype)

            # valid kv range
            k_e = T.min(ContextLens[bx], nb)
            T.copy(Q[bx, 0, 0], q_shared)
            T.copy(Weights[bx, 0], w)

            for k_i in T.Pipelined(T.ceildiv(nb, block_N), num_stages=num_stages):
                k_start = k_i * block_N
                T.copy(BlockedK[bx, k_start, 0], k_shared)

                T.gemm(
                    k_shared,
                    q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for kn_i, hn_i in T.Parallel(block_N, block_H_pad):
                    s[kn_i, hn_i] = T.max(s[kn_i, hn_i], 0) * w[hn_i]

                T.reduce_sum(s, logits_accum, dim=1, clear=True)

                for kn_i in T.Parallel(block_N):
                    k_col = k_start + kn_i
                    if k_col < k_e:
                        Logits[bx, k_col] = logits_accum[kn_i]
                    else:
                        Logits[bx, k_col] = float("-inf")

    return kernel


def batch_block_mqa_attn_return_logits_interface(
    q: torch.Tensor,
    blocked_kv: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    kv_block_size: int,
    clean_logits: bool = True,
    force_maintain: bool = True,
    dtype: str = "bfloat16",
    block_N: int = 64,
):
    """
    Decode 接口：
      q:          [B, 1, H, D]
      blocked_kv: [B, Nb, D]
      weights:    [B,1,H]
      context_lens:[B] (有效 Nb)
    Return:
      logits: [B, Nb] fp32
    """

    assert len(q.shape) == 4
    B, seq_len_q, H, D = q.shape
    B, seq_len_kv, D = blocked_kv.shape

    assert seq_len_q == 1, "decode expects q_len=1"

    q = q.squeeze(1)
    weights = weights.squeeze(1)

    logits = torch.empty((B, seq_len_kv), device=q.device, dtype=torch.float32)

    kernel = batch_decode_block_mqa_attn_return_logits(
        heads=H,
        index_dim=D,
        block_N=block_N,
        block_H=H,
        dtype=dtype,
    )
    kernel(
        q,
        blocked_kv,
        logits,
        weights,
        context_lens.to(torch.int32),
    )

    if clean_logits:
        n = torch.arange(seq_len_kv, device=q.device)[None, :]
        valid = n < context_lens.to(torch.int64)[:, None]
        logits = logits.masked_fill(~valid[:, :], float("-inf"))

    if force_maintain:
        ctx = context_lens.to(torch.int64).clamp(min=0, max=seq_len_kv)
        b_idx = torch.arange(B, device=q.device)
        logits[b_idx, 0] = float("inf")
        last = (ctx - 1).clamp(min=0, max=seq_len_kv - 1)
        logits[b_idx, last] = float("inf")

    logits = logits.unsqueeze(1)  # [B,1,Nb]

    return logits

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def paged_block_sparse_mqa_attn_return_logits(
        paged_block_size,
        kv_block_size,
        topk,
        heads,
        index_dim,
        block_N=64,
        num_stages=1,
        threads=128,
        dtype="bfloat16",
):
    accum_dtype = T.float32
    index_dtype = T.int32

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    max_blocks = T.dynamic("max_blocks")
    num_phys_blocks = T.dynamic("num_phys_blocks")

    index_q_shape = [batch, seq_len, heads, index_dim]
    kv_cache_shape = [num_phys_blocks, paged_block_size, 1, index_dim]
    logits_shape = [batch, seq_len, topk * kv_block_size]
    weights_shape = [batch, seq_len, heads]

    H_per_block = heads
    block_N = T.min(block_N, kv_block_size)

    assert kv_block_size % block_N == 0, "block_N must divide kv_block_size"
    assert paged_block_size % block_N == 0, "block_N must divide paged_block_size"

    @T.prim_func
    def paged_block_sparse_mqa_attn_return_logits_kernel(
            IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
            KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
            TopKBlockIndex: T.Tensor([batch, seq_len, topk], index_dtype),  # type: ignore
            Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
            Weights: T.Tensor(weights_shape, dtype),  # type: ignore
            ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
            BlockTables: T.Tensor([batch, max_blocks], index_dtype),  # type: ignore
    ):
        total_q = batch * seq_len

        with T.Kernel(total_q, threads=threads) as bx:
            b = bx // seq_len
            m = bx - b * seq_len

            index_q_shared = T.alloc_shared([H_per_block, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([H_per_block, block_N], accum_dtype)
            logits = T.alloc_fragment([block_N], accum_dtype)
            weights = T.alloc_fragment([H_per_block], accum_dtype)

            seq_len_i = m
            ctx_len = ContextLens[b]

            # -----------------------------
            # 先处理无效 query：直接整行写 -inf
            # -----------------------------
            if seq_len_i >= ctx_len:
                for out_i in T.Parallel(topk * kv_block_size):
                    Logits[b, seq_len_i, out_i] = -T.infinity(accum_dtype)
            else:
                cu_k_s_min = T.cast(0, index_dtype)
                cu_k_e_max = ctx_len

                # 显式搬运 Q，避免 T.copy 切片语义不稳定
                for h_i, d_i in T.Parallel(H_per_block, index_dim):
                    index_q_shared[h_i, d_i] = IndexQ[b, seq_len_i, h_i, d_i]

                # 搬运 weights
                for h_i in T.Parallel(H_per_block):
                    weights[h_i] = T.cast(Weights[b, seq_len_i, h_i], accum_dtype)

                for n_i in T.serial(topk):
                    topk_block_id = TopKBlockIndex[b, seq_len_i, n_i]

                    # topk block 非法，整块输出 -inf
                    if topk_block_id < 0:
                        for out_i in T.Parallel(kv_block_size):
                            Logits[b, seq_len_i, n_i * kv_block_size + out_i] = -T.infinity(accum_dtype)
                    else:
                        block_s = topk_block_id * kv_block_size

                        for b_i in T.serial(kv_block_size // block_N):
                            block_s_i = block_s + b_i * block_N

                            # -----------------------------
                            # 搬运 K block
                            # -----------------------------
                            for bn_i, d_i in T.Parallel(block_N, index_dim):
                                k_i = block_s_i + bn_i

                                if k_i >= cu_k_s_min and k_i < cu_k_e_max:
                                    p = k_i // paged_block_size
                                    o = k_i - p * paged_block_size

                                    if p >= 0 and p < max_blocks:
                                        phys = BlockTables[b, p]
                                        if phys >= 0 and phys < num_phys_blocks:
                                            index_k_shared[bn_i, d_i] = KvCache[phys, o, 0, d_i]
                                        else:
                                            index_k_shared[bn_i, d_i] = T.cast(0, dtype)
                                    else:
                                        index_k_shared[bn_i, d_i] = T.cast(0, dtype)
                                else:
                                    index_k_shared[bn_i, d_i] = T.cast(0, dtype)

                            # -----------------------------
                            # GEMM: [H, D] x [N, D]^T -> [H, N]
                            # -----------------------------
                            T.gemm(
                                index_q_shared,
                                index_k_shared,
                                s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=True,
                            )

                            # ReLU + weight (match ref: score_h = relu(q@k); out = sum(score_h * w))
                            for h_i, bn_i in T.Parallel(H_per_block, block_N):
                                s[h_i, bn_i] = T.max(s[h_i, bn_i], T.cast(0, accum_dtype)) * weights[h_i]

                            # 按 head 求和 => [N]
                            T.reduce_sum(s, logits, dim=0, clear=True)

                            # -----------------------------
                            # 无效 token 置 -inf
                            # -----------------------------
                            for i_i in T.Parallel(block_N):
                                k_i = block_s_i + i_i

                                if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                                    logits[i_i] = -T.infinity(accum_dtype)
                                else:
                                    p = k_i // paged_block_size
                                    if p < 0 or p >= max_blocks:
                                        logits[i_i] = -T.infinity(accum_dtype)
                                    else:
                                        phys = BlockTables[b, p]
                                        if phys < 0 or phys >= num_phys_blocks:
                                            logits[i_i] = -T.infinity(accum_dtype)

                            # 写回
                            for bn_i in T.Parallel(block_N):
                                Logits[
                                    b,
                                    seq_len_i,
                                    n_i * kv_block_size + b_i * block_N + bn_i,
                                ] = logits[bn_i]

    return paged_block_sparse_mqa_attn_return_logits_kernel


def paged_block_sparse_mqa_attn_return_logits_interface(
        q,
        kv_cache,
        topk_block_index,
        kv_block_size,
        weights,
        context_lens,
        block_tables,
        dtype="bfloat16",
):
    batch, seq_len, heads, index_dim = q.shape
    topk = int(topk_block_index.shape[-1])
    paged_block_size = int(kv_cache.shape[1])

    if weights.ndim == 2:
        weights = weights.view(batch, seq_len, heads)

    logits = torch.full(
        (batch, seq_len, topk * kv_block_size),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    kernel = paged_block_sparse_mqa_attn_return_logits(
        paged_block_size=paged_block_size,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        dtype=dtype,
    )
    kernel(
        q,
        kv_cache,
        topk_block_index.to(torch.int32),
        logits,
        weights,
        context_lens.to(torch.int32),
        block_tables.to(torch.int32),
    )

    return logits

def fp8_hierarchy_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    k_block_size: int, 
    block_topk: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """

    # TODO 
    # 1. convert q, kv and weights to bf16
    # 2. call the paged version of tilelang kernels

    q = q_fp8.float()
    kv_cache_quant = kv_cache_fp8[..., :-4].view(torch.float8_e4m3fn)
    kv_cache_scales = kv_cache_fp8[..., -4:].view(torch.float32)
    kv_cache = kv_cache_quant.float() * kv_cache_scales  # [num_blocks, block_size, 1, D]

    q = q.bfloat16()
    kv_cache = kv_cache.bfloat16()
    weights = weights.bfloat16()
    
    # paged block size = 64 is usually smaller than hisa's k_block_size

    # paged kv cache -> continuous blocked kv for block indexer
    # def paged_mean_pooling(
    #     kv_cache: torch.Tensor,
    #     context_lens: torch.Tensor,
    #     block_tables: torch
    # ):
    #     """
    #     Args:
    #         kv_cache: [num_blocks, block_size, 1, D]
    #         context_lens: Tensor of shape [B], dtype int32; effective context length
    #             for each batch element.
    #         block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
    #             block indices to physical blocks in the paged cache.
    #     Returns:
    #         blocked_k: [B, num_blocks_for_mean_pooling, D]
    #     """

    #     batch = context_lens.shape[0]
    #     blocked_k_list = []
    #     for b in range(batch):
    #         seqlen = context_lens[b].item()
    #         num_pooling_blocks = (seqlen + k_block_size - 1) // k_block_size

    #         pooled_block_list = []
    #         for n in range(num_pooling_blocks):
    #             pooling_block_start = n * k_block_size
    #             pooling_block_end = min((n + 1) * k_block_size, seqlen)
    #             pooling_block_len = pooling_block_end - pooling_block_start

    #             num_paged_blocks = (pooling_block_len + block_tables.shape[1] - 1) // block_tables.shape[1]
    #             paged_block_start = pooling_block_start // block_tables.shape[1]
    #             paged_block_end = paged_block_start + num_paged_blocks

    #             paged_block_indices = block_tables[b, paged_block_start:paged_block_end].to(torch.long)  # [num_paged_blocks]
    #             paged_blocks = kv_cache[paged_block_indices]  # [num_paged_blocks, block_size, 1, D]
    #             paged_blocks = paged_blocks.view(-1, kv_cache.shape[-1])  # [total_tokens_in_pooling_block, D]
    #             pooled_blocks = paged_blocks.mean(dim=0)  # [D]
    #             pooled_block_list.append(pooled_blocks)

    #         blocked_k_list.append(torch.stack(pooled_block_list, dim=0))  # [num_pooling_blocks, D]
    #     blocked_k = torch.stack(blocked_k_list, dim=0)  # [B, num_pooling_blocks, D]
            
    #     return blocked_k
    
    def paged_mean_pooling(
        kv_cache: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
    ):
        """
        Args:
            kv_cache: [num_blocks, block_size, 1, D]
            context_lens: [B]
            block_tables: [B, max_blocks]  (逻辑 paged block -> 物理 block)
        Returns:
            blocked_k: [B, max_num_pooling_blocks, D]  (已 padding)
            num_pooling_blocks: [B]  (每个样本真实的 pooling block 数，用于 mask)
        """
        device = kv_cache.device
        B = int(context_lens.shape[0])
        block_size = int(kv_cache.shape[1])        # paged block size，通常 64（修复原来的错误）
        D = int(kv_cache.shape[-1])

        # 每个样本需要的 k-block 数不同，先 pad 到 batch 内最大值
        max_seqlen = int(context_lens.max().item()) if B > 0 else 0
        max_num_pooling_blocks = (max_seqlen + k_block_size - 1) // k_block_size

        blocked_k = torch.zeros(
            (B, max_num_pooling_blocks, D),
            device=device,
            dtype=kv_cache.dtype,
        )
        num_pooling_blocks = torch.empty((B,), device=device, dtype=torch.int32)

        for b in range(B):
            seqlen = int(context_lens[b].item())
            nblocks = (seqlen + k_block_size - 1) // k_block_size
            num_pooling_blocks[b] = nblocks

            for n in range(nblocks):
                pooling_block_start = n * k_block_size
                pooling_block_end = min((n + 1) * k_block_size, seqlen)
                pooling_block_len = pooling_block_end - pooling_block_start
                if pooling_block_len <= 0:
                    continue

                # 计算覆盖该 pooling block 所需的 paged blocks（注意：用 block_size，不是 block_tables.shape[1]）
                paged_block_start = pooling_block_start // block_size
                paged_block_end = (pooling_block_end + block_size - 1) // block_size  # exclusive
                paged_block_indices = block_tables[b, paged_block_start:paged_block_end].to(torch.long)

                # [num_paged_blocks, block_size, 1, D] -> [num_paged_blocks*block_size, D]
                paged_blocks = kv_cache.index_select(0, paged_block_indices)
                tokens = paged_blocks.reshape(-1, 1, D).reshape(-1, D)

                # pooling_block_start 可能落在 paged block 中间，切片到准确的 token 范围
                offset = pooling_block_start - paged_block_start * block_size  # == pooling_block_start % block_size
                tokens = tokens[offset : offset + pooling_block_len]           # [pooling_block_len, D]

                blocked_k[b, n] = tokens.mean(dim=0)

        return blocked_k, num_pooling_blocks
    
    blocked_k, num_pooling_blocks = paged_mean_pooling(kv_cache, context_lens, block_tables)  # [B, num_pooling_blocks, D], [B]

    block_k_indexer_score = batch_block_mqa_attn_return_logits_interface(q=q, blocked_kv=blocked_k, kv_block_size=k_block_size, weights=weights, context_lens=num_pooling_blocks)  # [B, next_n, num_pooling_blocks]
    topk_block_indices = torch.topk(block_k_indexer_score, k=min(block_topk, block_k_indexer_score.shape[-1]), dim=-1).indices  # [B, next_n, topk]
    topk_block_indices = topk_block_indices.to(torch.int32)

    block_sparse_k_indexer_score = paged_block_sparse_mqa_attn_return_logits_interface(q=q, kv_cache=kv_cache, topk_block_index=topk_block_indices, kv_block_size=k_block_size, weights=weights, context_lens=context_lens, block_tables=block_tables)  # [B, next_n, topk*kv_block_size]

    relevant_topk_indices = torch.topk(block_sparse_k_indexer_score, k=min(block_topk * k_block_size, block_sparse_k_indexer_score.shape[-1]), dim=-1).indices  # [B, next_n, topk*kv_block_size]
    relevant_topk_indices = relevant_topk_indices.to(torch.int32)
    absolute_topk_block_indices = torch.gather(topk_block_indices, dim=-1, index=(relevant_topk_indices // k_block_size))  # [B, next_n, topk*kv_block_size]

    topk_indices = absolute_topk_block_indices * k_block_size + (relevant_topk_indices % k_block_size)
    topk_indices = torch.where(topk_indices < context_lens.unsqueeze(1).unsqueeze(2), topk_indices, -1)

    return topk_indices


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    q = q.float()  # [M, H, D]
    k_fp8, k_scales = kv
    if k_scales.ndim == 1:
        k_scales = k_scales.unsqueeze(-1)  # [N, 1]
    k = k_fp8.float() * k_scales  # [N, D]

    seqlen_kv = k.shape[0]
    mask_lo = torch.arange(0, seqlen_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seqlen_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """

    device = q_fp8.device
    B, next_n, H, D = q_fp8.shape
    num_blocks, block_size = kv_cache_fp8.shape[0], kv_cache_fp8.shape[1]

    # Output logits for the full max_model_len (masked beyond each context_len)
    out = torch.full(
        (B * next_n, max_model_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    # Dequantize q once
    q = q_fp8.float()  # [B, next_n, H, D]

    # Helper: gather logical [0:seqlen) tokens' packed (fp8+scale) bytes for one sequence
    def _gather_flattened_kv_for_seq(b: int, seqlen: int) -> torch.Tensor:
        if seqlen <= 0:
            # Return empty [0, 1, D+4]
            return kv_cache_fp8.new_empty((0, 1, D + 4))
        needed_blocks = (seqlen + block_size - 1) // block_size
        phys = block_tables[b, :needed_blocks].to(torch.long)  # [needed_blocks]
        # [needed_blocks, block_size, 1, D+4] -> [needed_blocks*block_size, 1, D+4]
        flat = kv_cache_fp8.index_select(0, phys).reshape(-1, 1, D + 4)
        return flat[:seqlen]  # [seqlen, 1, D+4]

    for b in range(B):
        seqlen = int(context_lens[b].item())
        seqlen = min(seqlen, max_model_len)
        if seqlen <= 0:
            continue

        _flattened_kv = _gather_flattened_kv_for_seq(b, seqlen)  # [N, 1, D+4] uint8

        # ---- 你的示例：从 cache bytes 转成 k_fp8 与 k_scale ----
        # k_fp8: [N, D] float8_e4m3fn
        # k_scale: [N, 1] float32
        k_fp8 = _flattened_kv[..., :D].view(torch.float8_e4m3fn).contiguous().view(seqlen, D)
        k_scale = _flattened_kv[..., D:].view(torch.float32).contiguous().view(seqlen, 1)
        k = k_fp8.float() * k_scale  # [N, D] float32

        # 对该 batch 的每个 decode 位置计算 logits
        for t in range(next_n):
            row = b * next_n + t
            qb = q[b, t]  # [H, D] float32
            w = weights[row]  # [H] float32

            # score: [H, N]
            score = torch.einsum("hd,nd->hn", qb, k)
            # logits: [N]
            logits = (score.relu() * w[:, None]).sum(dim=0)

            out[row, :seqlen] = logits

    return out

