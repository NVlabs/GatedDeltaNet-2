# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

r"""
Chunkwise Triton kernels for GDN-2 (Gated DeltaNet 2).

GDN-2 extends the gated delta rule with two independent channel-wise gates.
The per-token recurrence on the matrix state ``S_t`` in ``R^{d_k x d_v}`` is

    S_t = (I - k_t (b_t * k_t)^T) Diag(alpha_t) S_{t-1} + k_t (w_t * v_t)^T

where ``*`` is the elementwise (Hadamard) product, ``b_t`` in ``[0,1]^{d_k}``
is the channel-wise erase gate on the key axis, ``w_t`` in ``[0,1]^{d_v}`` is
the channel-wise write gate on the value axis, and ``alpha_t`` is the
channel-wise decay. Setting ``b_t`` and ``w_t`` to a shared scalar broadcast
recovers KDA; further collapsing ``alpha_t`` to a scalar recovers Gated
DeltaNet.

Training uses a chunkwise schedule: the sequence is split into chunks of size
``C = 64``, the recurrence runs between chunks, and all intra-chunk token
interactions are expressed as dense matmuls via a WY representation. This
keeps complexity linear in sequence length and maps onto tensor cores.

Pipeline
--------
Forward (see ``chunk_gdn2_fwd``):
  1. ``chunk_gdn2_fwd_kernel_intra_token_parallel``  - intra-chunk Q-K and
     gated K-K score matrices (Aqk, Akk).
  2. ``chunk_gdn2_fwd_kernel_intra_sub_chunk``       - sub-chunk refinement
     of the score matrices.
  3. ``chunk_gdn2_fwd_kernel_inter_solve_fused``     - WY triangular solve
     A = (I + T)^{-1} and construction of the WY auxiliaries.
  4. ``recompute_w_u_fwd_gdn2_kernel``               - gate-aware pseudo-key
     / pseudo-value blocks (w_wy, u_wy).
  The inter-chunk state recurrence and the output kernel are shared with the
  gated delta rule / KDA path and imported from ``chunk_kda``.

Backward (see ``chunk_gdn2_bwd``):
  - ``chunk_gdn2_bwd_kernel_wy_dqkg_fused`` - gate-aware vector-Jacobian
    product through the WY inverse. The channel-wise gates are baked into the
    dA accumulation directly; a scalar post-scale (valid for scalar-beta
    models) cannot reconstruct two independent gates living on different axes.
  - ``chunk_gdn2_bwd_kernel_intra``         - intra-chunk gradients for Q, K,
    the erase gate, and the cumulative decay.
  The dAv and dhu backward kernels are shared with KDA.

Public entry point: ``chunk_gdn2``. Autograd wrapper: ``ChunkGDN2Function``.
All other names are internal orchestration or Triton kernels.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_fwd, l2norm_bwd
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2, gather
from fla.utils import (
    IS_GATHER_SUPPORTED,
    IS_NVIDIA_HOPPER,
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    check_shared_mem,
    input_guard,
)

NUM_WARPS_WY = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]
NUM_WARPS_INTRA = [1, 2, 4] if IS_NVIDIA_HOPPER else [1, 2, 4, 8]
NUM_WARPS_GENERIC = [1, 2, 4] if IS_NVIDIA_HOPPER else [1, 2, 4, 8]

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum

# =============================================================================
# FORWARD KERNELS
# =============================================================================
# Kernel 1: chunk_gdn2_fwd_kernel_intra_token_parallel
# -----------------------------------------------------------------------------
# Builds the two intra-chunk score matrices used by the WY solve:
#   Aqk - causal query-key scores, decay-weighted, for the output path.
#   Akk - gated key-key scores (the strictly-lower matrix T whose inverse
#         (I + T)^{-1} defines the WY representation). The erase gate b is
#         folded into the key tile before the dot product, which is the only
#         GDN-2-specific change relative to the gated delta rule.
# Token-parallel: each program handles a block of tokens within one chunk.
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in NUM_WARPS_GENERIC
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'N'])
def chunk_gdn2_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    b,          
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    q += bos * H * K
    k += bos * H * K
    g += bos * H * K
    Aqk += bos * H * BT
    Akk += bos * H * BC
    b += bos * H * K

    BK: tl.constexpr = triton.next_power_of_2(K)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K

    p_q = tl.make_block_ptr(q + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_b = tl.make_block_ptr(b + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_b = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)

    b_k = b_k * b_b

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kj = tl.make_block_ptr(k + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        p_gj = tl.make_block_ptr(g + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        # [BH, BK]
        b_kj = tl.load(p_kj, boundary_check=(0, 1)).to(tl.float32)
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        b_kgj = b_kj * exp2(b_g - b_gj)

        b_kgj = tl.where(m_k[None, :], b_kgj, 0.0)
        # [BH] -- scalar output per head
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        tl.store(Aqk + i_t * H * BT + (i_hg * BH + o_h) * BT + j % BT,
                 b_Aqk.to(Aqk.dtype.element_ty), mask=m_h)
        tl.store(Akk + i_t * H * BC + (i_hg * BH + o_h) * BC + j - i_ts,
                 b_Akk.to(Akk.dtype.element_ty), mask=m_h)


def chunk_gdn2_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    b: torch.Tensor,            
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
):
    """Token-parallel intra-chunk Aqk + diagonal-Akk builder for GDN-2."""
    B, T, H, K = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    def grid(meta):
        return (B * T, triton.cdiv(H, meta['BH']))

    chunk_gdn2_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        b=b,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
    return Aqk, Akk


# =============================================================================
# Kernel 2: chunk_gdn2_fwd_kernel_intra_sub_chunk
# -----------------------------------------------------------------------------
# Refines the score matrices at sub-chunk granularity. The chunk is split into
# smaller sub-chunks; this kernel fills the off-diagonal sub-chunk blocks of
# Aqk and Akk that the token-parallel kernel above leaves to a second pass.
# Together kernels 1 and 2 produce the complete intra-chunk score matrices.
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_GENERIC
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gdn2_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    b,          
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * H + i_h) * K
    b = b + (bos * H + i_h) * K
    Aqk = Aqk + (bos * H + i_h) * BT
    Akk = Akk + (bos * H + i_h) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))

    if USE_GATHER:
        b_gn = gather(b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0)
    else:
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

    b_kgt = tl.trans(b_k * b_gk)

    b_bk = (b_b.to(tl.float32) * b_k.to(tl.float32)).to(b_k.dtype)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_bk * b_gq, b_kgt)

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * H * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32' if check_shared_mem() else 'ieee')


# =============================================================================
# Kernel 3: chunk_gdn2_fwd_kernel_inter_solve_fused
# -----------------------------------------------------------------------------
# Solves the WY representation. Given the gated key-key matrix T (= Akk), this
# computes A = (I + T)^{-1} by blocked forward substitution. The triangular
# solve is the most precision-sensitive step in the chunk, so its matmuls run
# at the precision picked by SOLVE_TRIL_DOT_PRECISION above: fp32 (ieee) when
# shared memory allows it, tf32 otherwise. The resulting A is consumed by the
# pseudo-key / pseudo-value construction in kernel 4.
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gdn2_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    b,          
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_b1 = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            b_b1 = tl.load(p_b1, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            # row side; for Aqk we use q (no absorption).
            b_bk1 = b_b1 * b_k1
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_bk1 * b_gqn, b_kgt)

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_b2 = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                b_b2 = tl.load(p_b2, boundary_check=(0, 1)).to(tl.float32)
                # [BK]
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_bkg2 = (b_b2 * b_k2) * b_gqn2
                # [BK, BC]
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_bkg2, b_kgt)
                # [BC, BC]
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_bkg2, b_kgt)

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_b3 = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    b_b3 = tl.load(p_b3, boundary_check=(0, 1)).to(tl.float32)
                    # [BK]
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_bkg3 = (b_b3 * b_k3) * b_gqn3
                    # [BK, BC]
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_bkg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_bkg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_bkg3, b_kgt)

    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ############################################################################
    # compute merged inverse using off-diagonals (UNCHANGED vs KDA).
    ############################################################################
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ############################################################################
    # store full Akk_inv (UNCHANGED vs KDA).
    ############################################################################
    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
    p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# Kernel 4: recompute_w_u_fwd_gdn2_kernel
# -----------------------------------------------------------------------------
# Produce the WY auxiliaries w (for the P-term) and u (for the H-term) from
# the solved Akk_inv matrix and the gated inputs.
#
# Also (optionally) produces qg = q * exp(gk) and kg = k * exp(gn - gk), which
# are tail-decayed variants of q and k used by the downstream recurrence.
#
# GDN-2 changes vs KDA:
#   KDA:
#     u = A @ (beta * v)           # beta scalar broadcast to V
#     w = A @ (beta * exp(gk) * k) # beta scalar broadcast to K
#   GDN-2:
#     u = A @ (wg  * v)            # wg is [B,T,H,V]  -- channel-wise write gate
#     w = A @ (b   * exp(gk) * k)  # b  is [B,T,H,K]  -- channel-wise erase gate
#   So wg (write gate) is loaded per BV block, and b (erase gate) is loaded
#   per BK block. kg and qg are unchanged (beta-free).
# =============================================================================
@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_WY
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_gdn2_kernel(
    q,
    k,
    qg,
    kg,
    v,
    b,          
    wg,         
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1),
                            (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_wg = tl.make_block_ptr(wg + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                 (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_wg = tl.load(p_wg, boundary_check=(0, 1))
        b_vb = (b_v * b_wg).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_b = tl.make_block_ptr(b + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_kb = b_k * b_b

        p_gk = tl.make_block_ptr(gk + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                 (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb *= exp2(b_gk)

        if STORE_QG:
            p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                    (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                     (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k,
                           mask=m_k, other=0.).to(tl.float32)
            b_kg = b_k * tl.where(
                (i_t * BT + tl.arange(0, BT) < T)[:, None],
                exp2(b_gn[None, :] - b_gk),
                0,
            )
            p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                     (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# FORWARD ORCHESTRATION
# -----------------------------------------------------------------------------
# Python-level wrappers that launch the Triton kernels above, allocate output
# buffers, and chain the forward pipeline together.
# =============================================================================
def chunk_gdn2_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    b: torch.Tensor,        
    wg: torch.Tensor,       
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
):
    """Intra-chunk forward: build Aqk, Akk_inv, and the WY auxiliaries w, u.

    Mirrors `chunk_kda_fwd_intra` from chunk.py but wires in the GDN-2 kernels
    that use channel-wise b and wg instead of scalar beta.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - kernel only writes lower triangular.
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    # Separate fp32 buffer for diagonal BC x BC blocks.
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)

    # Step 1: build the diagonal Akk sub-blocks (and full Aqk) in fp32.
    if safe_gate:
        grid = (NT, triton.cdiv(BT, BC), B * H)
        BK = triton.next_power_of_2(K)
        chunk_gdn2_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            g=gk,
            b=b,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=IS_GATHER_SUPPORTED,
        )
    else:
        chunk_gdn2_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            b=b,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Step 2: Fused inter-subchunk Akk blocks + solve_tril -> full Akk_inv.
    grid = (NT, B * H)
    chunk_gdn2_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        b=b,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        USE_SAFE_GATE=safe_gate,
    )

    # Step 3: Build w = A @ (b * exp(gk) * k)  and  u = A @ (wg * v).
    w, u, qg, kg = recompute_w_u_fwd_gdn2(
        k=k,
        v=v,
        b=b,
        wg=wg,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk


def recompute_w_u_fwd_gdn2(
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,         
    wg: torch.Tensor,        
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Produce WY auxiliaries w, u from the solved Akk and channel-wise gates.

    Mirrors `recompute_w_u_fwd` from chunk.py but dispatches to the GDN-2
    kernel that expects channel-wise `b` and `wg` tensors.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if q is not None else None
    kg = torch.empty_like(k) if gk is not None else None
    recompute_w_u_fwd_gdn2_kernel[(NT, B * H)](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=v,
        b=b,
        wg=wg,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u, qg, kg


def chunk_gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,         
    wg: torch.Tensor,        
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    transpose_state_layout: bool = False,
):
    """Top-level GDN-2 forward pipeline.

    Returns (o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h,
    initial_state) -- same tuple shape as `chunk_kda_fwd` so downstream code
    can be wired up symmetrically.
    """
    if use_gate_in_kernel:
        g = kda_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        g = chunk_local_cumsum(
            g=g,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    w, u, qg, kg, Aqk, Akk = chunk_gdn2_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        b=b,
        wg=wg,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=transpose_state_layout,
    )

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=transpose_state_layout,
    )

    if disable_recompute is False:
        # Free memory we don't need to retain for the current path.
        w, u, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            h = None
        if use_gate_in_kernel:
            g = None
    return o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state


@torch.compiler.disable
def chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    transpose_state_layout: bool = False,
    **kwargs,
):
    r"""
    Chunkwise forward for GDN-2 (Gated DeltaNet 2).

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H, K]`.
            If `use_gate_in_kernel=True`, this is the raw pre-activation and
            the kernel computes `-exp(A_log) * softplus(g + dt_bias)`
            (or the bounded variant if `safe_gate` + `lower_bound` is set).
        b (torch.Tensor):
            channel-wise ERASE gate of shape `[B, T, H, K]`. Replaces KDA's
            scalar beta. Typical range: [0, 2].
        w (torch.Tensor):
            channel-wise WRITE gate of shape `[B, T, H, V]`. New for GDN-2.
            Typical range: [0, 1].
        scale (Optional[float]):
            Scale factor. Defaults to `1 / sqrt(K)`.
        initial_state (Optional[torch.Tensor]):
            Initial recurrent state, shape `[N, H, K, V]`, dtype float32.
        output_final_state (bool):
            Whether to return the final recurrent state.
        use_qk_l2norm_in_kernel (bool):
            If True, L2-normalize q and k before attention.
        use_gate_in_kernel (bool):
            If True, compute the gate activation from raw g using A_log
            (required) and dt_bias (optional).
        cu_seqlens (torch.LongTensor, optional):
            Packed-sequence cumulative lengths, shape `[N+1]`. When provided
            the batch dim of q/k/v/... must be 1.
        cu_seqlens_cpu (torch.LongTensor, optional):
            CPU mirror of cu_seqlens (forwarded to the state-recurrence kernel).
        safe_gate (bool):
            Enable the safe-gate intra kernel variant (exploits M=16 TensorCores;
            requires gate values in [-5, 0)).
        lower_bound (Optional[float]):
            When safe_gate=True and use_gate_in_kernel=True, use the bounded
            gate activation `lower_bound * sigmoid(exp(A_log) * g)`. Must be
            in `[-5, 0)`.
        disable_recompute (bool):
            If True, retain forward-pass intermediates (qg, kg, v_new, h) for
            a faster backward at the cost of memory. Default: False.
        return_intermediate_states (bool):
            If True, also return the per-chunk pre-states `h` (shape
            [B, NT, H, K, V]). Must be used within `torch.inference_mode()`.
        transpose_state_layout (bool):
            Use the transposed state layout in the recurrence kernel.

    Returns:
        Normal:  (o, final_state)
        Intermediate (when return_intermediate_states=True):
                 (o, final_state, h)
    """
    # ------------------ argument validation --------------------------------
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                f"when using `cu_seqlens`. Please flatten variable-length "
                f"inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the "
                f"number of input sequences, i.e., {len(cu_seqlens) - 1} rather "
                f"than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError(
                "`lower_bound` must be specified when `safe_gate=True` and "
                "`use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert k.shape[-1] <= 256, "Currently we only support key headdim <=256 for GDN-2 :-("
    assert b.shape == q.shape, (
        "b (channel-wise erase gate) must have shape [B, T, H, K] matching q; "
        f"got {tuple(b.shape)} vs q {tuple(q.shape)}."
    )
    assert w.shape == v.shape, (
        "w (channel-wise write gate) must have shape [B, T, H, V] matching v; "
        f"got {tuple(w.shape)} vs v {tuple(v.shape)}."
    )
    assert v.shape == (*q.shape[:3], v.shape[-1]), (
        "v must be of shape (batch size, seq len, num of head, head dim)."
    )

    if scale is None:
        scale = k.shape[-1] ** -0.5


    return ChunkGDN2Function.apply(
        q,
        k,
        v,
        g,
        b,
        w,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        lower_bound,
        disable_recompute,
        return_intermediate_states,
        transpose_state_layout,
    )


# =============================================================================
# BACKWARD KERNELS
# =============================================================================
# Kernel: chunk_gdn2_bwd_kernel_wy_dqkg_fused
# -----------------------------------------------------------------------------
# Gate-aware vector-Jacobian product through the WY inverse. This is the
# central GDN-2-specific backward kernel.
#
# For a scalar write strength beta, the contribution of u to dA factors as
# dU @ (beta * V)^T = beta * (dU @ V^T), so beta can be applied as a scalar
# post-scale after a gate-free matmul. GDN-2 replaces beta with the
# channel-wise gates b and w, which live on different axes and act as a
# different per-row diagonal at every row. No scalar post-scale can recover
# them, so the gates are baked directly into the dA accumulation here:
#   dA += dU @ (w * V)^T        (write gate, value axis)
#   dA += dW @ (b * exp(gk) * K)^T   (erase gate, key axis)
# The kernel also emits dq, dk, dg, db, and dw.
#
# Hopper note: the (BK=32, num_warps=4) config is filtered out in the autotune
# list above to avoid a WGMMA layout assertion (see NUM_WARPS_WY).
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in NUM_WARPS_WY
        for num_stages in [2, 3, 4]
        if not (IS_NVIDIA_HOPPER and BK == 32 and num_warps == 4)
    ],
    key=['BT', 'TRANSPOSE_STATE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gdn2_bwd_kernel_wy_dqkg_fused(
    q,
    k,
    v,
    v_new,
    g,
    b,          
    wg,         
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,         
    dw,         
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    wg += (bos * H + i_h) * V
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K * V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    db += (bos * H + i_h) * K
    dw += (bos * H + i_h) * V
    dA += (bos * H + i_h) * BT

    p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_b = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_b = tl.load(p_b, boundary_check=(0, 1))

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw_flow = tl.zeros([BT, BK], dtype=tl.float32)  
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            if TRANSPOSE_STATE:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            # [BT, BV]
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            # [BT, BV]
            b_dv = tl.load(p_dv, boundary_check=(0, 1))

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            b_dw_flow += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))
            tl.debug_barrier()

            if i_k == 0:
                p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_wg = tl.make_block_ptr(wg, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_dw_gate = tl.make_block_ptr(dw, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_wg = tl.load(p_wg, boundary_check=(0, 1))
                b_dA += tl.dot(b_dv, tl.trans(b_v * b_wg))

                b_dvb = tl.dot(b_A, b_dv)                      
                b_dv2 = b_dvb * b_wg
                b_dw_gate = b_dvb * b_v

                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dw_gate, b_dw_gate.to(p_dw_gate.dtype.element_ty), boundary_check=(0, 1))

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_b
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)

        b_kg = b_k * b_gk_exp

        b_dw_flow = -b_dw_flow.to(b_A.dtype)
        b_dA += tl.dot(b_dw_flow, tl.trans((b_kg * b_b).to(b_A.dtype)))

        b_dkgb = tl.dot(b_A, b_dw_flow)                       
        p_db = tl.make_block_ptr(db, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_db_partial = b_dkgb * b_kg
        tl.store(p_db, b_db_partial.to(p_db.dtype.element_ty), boundary_check=(0, 1))

        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_b
        b_dk = b_dk + b_dkgb * b_gb

        p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)
    p_dA = tl.make_block_ptr(dA, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# Kernel: chunk_gdn2_bwd_kernel_intra
# -----------------------------------------------------------------------------
# Intra-chunk backward. Given the gradients dAqk and dAkk on the two score
# matrices, this accumulates the within-chunk contributions to dq, dk, the
# erase gate db, and the cumulative decay dg. The decay gradient is reduced by
# a reverse cumulative sum across the chunk.
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_INTRA
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_gdn2_bwd_kernel_intra(
    q,
    k,
    g,
    b,                  
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,                 
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K

    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K
    db += (i_k * all + bos) * H * BK + i_h * BK

    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    p_b = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_b = tl.load(p_b, boundary_check=(0, 1))

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

    if i_i > 0:
        p_gn = g + i_ti * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_kj = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_kj * exp2(b_gn - b_gk)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            # [BC, BK]
            b_dq2 += tl.dot(b_dAqk, b_kg)
            b_dk2 += tl.dot(b_dAkk, b_kg)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn


    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * H * BT + i_i * BC
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]

        p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        b_dAqk_diag_qk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
        exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
        exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

        b_k_exp_diag_qk = b_k * exp_neg_b_g_diag_qk
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC]
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            # [BK]
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] >= j
            # [BC, BK]
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)

            p_kj += H * K
            p_gkj += H * K


    b_db_tile = b_dk2 * b_k
    b_dk2 = b_dk2 * b_b

    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T, BK), (H * BK, 1), (i_ti, 0), (BC, BK), (1, 0))

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db_tile.to(p_db.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * H * K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_bj = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            # [BC, BK]
            b_bj = tl.load(p_bj, boundary_check=(0, 1))
            b_qj = tl.load(p_q, boundary_check=(0, 1))
            b_kbj = tl.load(p_k, boundary_check=(0, 1)) * b_bj
            b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kbj * tl.where(m_j[:, None], b_gkn, 0)
            # [BC, BK]
            b_dkt += tl.dot(b_dAqk, b_qg)
            b_dkt += tl.dot(b_dAkk, b_kbg)
        b_dkt *= exp2(b_gn - b_g)

    o_dA = i_ti * H * BT + i_i * BC + o_i
    p_qj = q + i_ti * H * K + o_k
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k
    p_bj_ptr = b + i_ti * H * K + o_k

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H * BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H * BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        b_dAqk_diag_kk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_kk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_kk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)
        b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.)
        exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.)
        exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.)

        b_q_exp = b_q * exp_b_g_diag_kk
        b_kb_exp = b_k * b_b * exp_b_g_diag_kk

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp) * exp_neg_b_g_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp) * exp_neg_b_g_diag_kk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC,]
            b_dAqk = tl.load(dAqk + o_dA + j * H * BT)
            b_dAkk = tl.load(dAkk + o_dA + j * H * BT)
            # [BK,]
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_bj = tl.load(p_bj_ptr, mask=m_k, other=0).to(tl.float32)
            b_kbj = b_kj * b_bj
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.)

            p_qj += H * K
            p_kj += H * K
            p_gkj += H * K
            p_bj_ptr += H * K

    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(dg2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# BACKWARD ORCHESTRATION
# -----------------------------------------------------------------------------
# Python-level wrappers that launch the backward Triton kernels and assemble
# the full gradient set for the chunkwise backward pass.
# =============================================================================
def chunk_gdn2_bwd_wy_dqkg_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,        
    wg: torch.Tensor,       
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
):
    """Fused backward for WY auxiliaries + dq/dk/dg/db/dw.

    Returns:
        dq, dk, dv, db, dw, dg, dA
        with db shape [B, T, H, K]  (channel-wise, GDN-2-specific)
        and dw shape [B, T, H, V]   (channel-wise write gate, new in GDN-2).
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float32)
    db = torch.empty_like(b, dtype=torch.float32)
    dw = torch.empty_like(wg, dtype=torch.float32)
    dA = torch.empty_like(A, dtype=torch.float32)

    grid = (NT, B * H)
    chunk_gdn2_bwd_kernel_wy_dqkg_fused[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        b=b,
        wg=wg,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dw=dw,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    dv = dv2
    return dq, dk, dv, db, dw, dg, dA


def chunk_gdn2_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,                
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,               # shape [B, T, H, K]  (accumulator)
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    """Intra-chunk backward: q, k, g, b contributions from dAqk, dAkk."""
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)

    db2 = q.new_empty(NK, B, T, H, BK, dtype=torch.float32)
    dg2 = torch.empty_like(dg, dtype=torch.float32)
    grid = (NK * NC, NT, B * H)
    chunk_gdn2_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        b=b,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg,
        dg2=dg2,
        db=db2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        SAFE_GATE=safe_gate,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    dq = dq2
    dk = dk2
    db2_combined = db2.permute(1, 2, 3, 0, 4).contiguous().reshape(B, T, H, NK * BK)[..., :K]
    db = db.add_(db2_combined)
    dg = dg2
    return dq, dk, db, dg


def chunk_gdn2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    wg: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    g: torch.Tensor | None = None,
    g_org: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    transpose_state_layout: bool = False,
    w_wy: torch.Tensor | None = None,
    u_wy: torch.Tensor | None = None,
    qg: torch.Tensor | None = None,
    kg: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
    h: torch.Tensor | None = None,
    disable_recompute: bool = False,
):
    """End-to-end GDN-2 backward.

    Returns:
        dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias_grad
        where db is [B,T,H,K] and dw is [B,T,H,V] (the new GDN-2 gradients).
    """
    if not disable_recompute:
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
        w_wy, u_wy, qg, kg = recompute_w_u_fwd_gdn2(
            k=k,
            v=v,
            b=b,
            wg=wg,
            A=Akk,
            q=q,
            gk=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w_wy,
            u=u_wy,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=transpose_state_layout,
        )

    dAqk, dv = chunk_kda_bwd_dAv(
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w_wy,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=transpose_state_layout,
    )

    dq, dk, dv, db, dw, dg, dAkk = chunk_gdn2_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        b=b,
        wg=wg,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )

    dq, dk, db, dg = chunk_gdn2_bwd_intra(
        q=q,
        k=k,
        g=g,
        b=b,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )

    dA_log, dt_bias_grad = None, None
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if use_gate_in_kernel:
        dg, dA_log, dt_bias_grad = kda_gate_bwd(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dg,
            lower_bound=lower_bound,
        )

    return dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias_grad


# =============================================================================
# AUTOGRAD AND PUBLIC API
# -----------------------------------------------------------------------------
# `ChunkGDN2Function` is the torch.autograd.Function that ties the forward and
# backward orchestration together. `chunk_gdn2` (defined earlier) is the public
# entry point that callers should use; it normalizes arguments and dispatches
# through this Function so that gradients are tracked.
# =============================================================================
class ChunkGDN2Function(torch.autograd.Function):
    """Autograd-compatible wrapper around GDN-2 forward/backward.

    Forward signature mirrors the public `chunk_gdn2` positional args but
    adds A_log, dt_bias as first-class tensor args so autograd can track
    their gradients through the gate-in-kernel path.
    """

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        use_gate_in_kernel: bool,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_cpu: torch.LongTensor | None,
        safe_gate: bool,
        lower_bound: float | None,
        disable_recompute: bool,
        return_intermediate_states: bool,
        transpose_state_layout: bool,
    ):
        chunk_size = 64

        # L2-norm (save rstd for backward).
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, chunk_size)
            if cu_seqlens is not None else None
        )

        g_input = g

        (o, final_state, g_cumsum, Aqk, Akk,
         w_wy, u_wy, qg, kg, v_new, h, initial_state) = chunk_gdn2_fwd(
            q=q,
            k=k,
            v=v,
            g=g_input,
            b=b,
            wg=w,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            transpose_state_layout=transpose_state_layout,
        )

        if return_intermediate_states:
            assert torch.is_inference_mode_enabled(), (
                "return_intermediate_states is only allowed in inference mode"
            )
            assert disable_recompute is False, (
                "return_intermediate_states must be used with disable_recompute=False"
            )
            return o.type_as(q), final_state, h

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g_cumsum, g_input, b, w, A_log, dt_bias,
            Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h,
            initial_state, cu_seqlens, chunk_indices,
        )
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.disable_recompute = disable_recompute
        ctx.transpose_state_layout = transpose_state_layout
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        (q, q_rstd, k, k_rstd, v, g_cumsum, g_input, b, w, A_log, dt_bias,
         Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h,
         initial_state, cu_seqlens, chunk_indices) = ctx.saved_tensors

        dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias_grad = chunk_gdn2_bwd(
            q=q,
            k=k,
            v=v,
            b=b,
            wg=w,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            g=g_cumsum,
            g_org=g_input if ctx.use_gate_in_kernel else None,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            transpose_state_layout=ctx.transpose_state_layout,
            w_wy=w_wy, u_wy=u_wy, qg=qg, kg=kg, v_new=v_new, h=h,
            disable_recompute=ctx.disable_recompute,
        )

        # Backprop through the optional l2norm_fwd on q, k.
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # Map grads back to the forward() argument list. The None slots
        # correspond to non-tensor arguments (scale, bools, cu_seqlens, etc.).
        return (
            dq.to(q.dtype),          # q
            dk.to(k.dtype),          # k
            dv.to(v.dtype),          # v
            dg.to(g_input.dtype),    # g
            db.to(b.dtype),          # b
            dw.to(w.dtype),          # w
            dA_log,                  # A_log
            dt_bias_grad,            # dt_bias
            None,                    # scale
            dh0,                     # initial_state
            None,                    # output_final_state
            None,                    # use_qk_l2norm_in_kernel
            None,                    # use_gate_in_kernel
            None,                    # cu_seqlens
            None,                    # cu_seqlens_cpu
            None,                    # safe_gate
            None,                    # lower_bound
            None,                    # disable_recompute
            None,                    # return_intermediate_states
            None,                    # transpose_state_layout
        )


__all__ = [
    "chunk_gdn2",
    "ChunkGDN2Function",
    "chunk_gdn2_fwd",
    "chunk_gdn2_fwd_intra",
    "recompute_w_u_fwd_gdn2",
    "chunk_gdn2_bwd",
    "chunk_gdn2_bwd_wy_dqkg_fused",
    "chunk_gdn2_bwd_intra",
]