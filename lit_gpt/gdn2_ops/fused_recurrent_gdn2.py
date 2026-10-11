# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

r"""
Token-by-token recurrent kernel for GDN-2 (Gated DeltaNet 2).

This is the inference-time counterpart of the chunkwise training kernels. It
runs the GDN-2 recurrence one token at a time, with no chunk padding, which
makes it the preferred path for autoregressive decoding at short sequence
lengths. It is forward-only and does not track gradients; training always uses
the chunkwise kernels.

Per token, the matrix state ``S`` in ``R^{d_k x d_v}`` is updated by

    S <- Diag(alpha) * S                  # channel-wise decay
    v_new = (w * v) - (b * k)^T S          # gated write minus gated read
    S <- S + k (v_new)^T                   # rank-one write
    o = S^T q                              # output read

where ``*`` is the elementwise product, ``b`` is the channel-wise erase gate
on the key axis, ``w`` is the channel-wise write gate on the value axis, and
``alpha`` is the channel-wise decay. This is the same recurrence the chunkwise
path implements, just unrolled token-serially.

The kernel supports the inference features expected by a serving stack:
packed variable-length sequences (``cu_seqlens``), continuous batching with a
paged state pool (``ssm_state_indices``), and speculative decoding
(``num_accepted_tokens``). It can also fuse the decay-gate activation
(``use_gate_in_kernel``) so the caller can pass raw pre-activations.

The conditional beta-contrast safeguard of ``chunk_gdn2`` is fused into the
token loop (``beta_guard``, on by default). It reads the exact normalized key
and activated decay consumed by each update, leaves safe erase gates
unchanged, and shrinks only unsafe channel contrast, so decode consumes the
same effective ``b`` as the chunkwise path. Decay and write gate are never
modified.

Public entry point: ``fused_recurrent_gdn2``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import input_guard

from .chunk_gdn2 import GUARD_MARGIN, GUARD_QUANT_TOL_MULT


# =============================================================================
# RECURRENT FORWARD KERNEL
# -----------------------------------------------------------------------------
# fused_recurrent_gdn2_fwd_kernel
#
# Each program owns one (sequence, value-head, K-block, V-block) tile and
# walks the tokens of that sequence serially, carrying the state tile b_h in
# registers. The per-token body is the four-line recurrence from the module
# docstring: decay, gated read/write, rank-one update, output read.
#
# State layout is selectable via TRANSPOSE_STATE: the default is [K, V];
# transposed is [V, K]. Both branches appear throughout because the serving
# stack may request either layout.
#
# Continuous batching: when ssm_state_indices is given, the per-sequence state
# is fetched from (and written back to) a paged pool indexed by those indices,
# rather than a contiguous [N, HV, K, V] buffer. Speculative decoding uses
# num_accepted_tokens to pick the correct rolled-back state slot.
#
# Beta guard: with BETA_GUARD set, the erase-gate tile goes through the same
# key-weighted contrast test and projection as chunk_gdn2.py before it is used,
# on the normalized key b_k and activated decay b_gk of this very step. The
# reductions over K rely on one program covering the full K axis
# (BK = next_power_of_2(K), as set by the launcher).
# =============================================================================
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gdn2_fwd_kernel(
    q,
    k,
    v,
    g,
    b,           
    w,           
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    lower_bound,
    quant_eps,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    BETA_GUARD: tl.constexpr,
    MARGIN: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Decompose the flat program id into (K-block, V-block, sequence,
    # value-head). i_h maps the value-head back to its key-head for GVA.
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    NK = tl.cdiv(K, BK)
    i_k = pid % NK
    pid_rest = pid // NK

    i_v = pid_rest % NV
    i_nh = pid_rest // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + (bos * HV + i_hv) * K + o_k
    p_w = w + (bos * HV + i_hv) * V + o_v
    p_g = g + (bos * HV + i_hv) * K + o_k
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if TRANSPOSE_STATE:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = (
                h0
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                * stride_init_state_token
            )
            if TRANSPOSE_STATE:
                p_h0 = p_h0 + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        else:
            if TRANSPOSE_STATE:
                p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in tl.range(0, T, num_stages=num_stages):
        b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy='evict_last').to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy='evict_last').to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy='evict_first').to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_g = tl.load(p_g, eviction_policy='evict_last').to(tl.float32)

        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_h).to(tl.float32)

            if HAS_DT_BIAS:
                b_bias = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
                b_g = b_g + b_bias

            if USE_LOWER_BOUND:
                b_gk = lower_bound * tl.sigmoid(exp(b_A) * b_g)
            else:
                b_gk = -exp(b_A) * softplus(b_g)
        else:
            b_gk = b_g

        # Apply per-channel decay to the running state.
        if TRANSPOSE_STATE:
            b_h *= exp(b_gk[None, :])
        else:
            b_h *= exp(b_gk[:, None])

        b_b_tile = tl.load(p_b, mask=mask_k, other=0, eviction_policy='evict_last').to(tl.float32)

        if BETA_GUARD:
            # Exact rank-one erase geometry, matching chunk_gdn2.py:
            # n = |k|^2, a = sum(b k^2), nu = sum(b^2 k^2), r2 = n*nu - a^2 >= 0.
            b_k2 = b_k * b_k
            n = tl.sum(b_k2, axis=0)
            a = tl.sum(b_b_tile * b_k2, axis=0)
            nu = tl.sum(b_b_tile * b_b_tile * b_k2, axis=0)
            r2 = tl.maximum(n * nu - a * a, 0.0)

            # ||Diag(exp(g))||_2 = exp(g_max). The full update is certified
            # non-expansive when the beta-contrast budget below is respected.
            g_max = tl.max(
                tl.where(mask_k, b_gk, -float("inf")),
                axis=0,
            )
            c2 = tl.exp(-2.0 * g_max)
            r2_crit = (c2 - 1.0) * (
                1.0 - ((1.0 - a) * (1.0 - a)) / c2
            )
            r2_crit = tl.maximum(r2_crit, 0.0)

            has_key = n > 1.0e-20
            unsafe = has_key & (r2 > r2_crit)

            # Preserve the key-weighted mean erase strength and reduce only
            # the unsafe channel contrast.
            mu = a / tl.maximum(n, 1.0e-20)
            eta = tl.sqrt((1.0 - MARGIN) * r2_crit / tl.maximum(r2, 1.0e-30))
            eta = tl.minimum(tl.maximum(eta, 0.0), 1.0)
            projected = mu + eta * (b_b_tile - mu)
            candidate = tl.where(unsafe, projected, b_b_tile)

            # Match the chunk path exactly: quantize to beta's storage dtype,
            # verify the quantized candidate, and use a scalar fallback only
            # if quantization crosses the analytical boundary.
            candidate_q = candidate.to(b.dtype.element_ty).to(tl.float32)
            a_q = tl.sum(candidate_q * b_k2, axis=0)
            nu_q = tl.sum(candidate_q * candidate_q * b_k2, axis=0)
            r2_q = tl.maximum(n * nu_q - a_q * a_q, 0.0)
            r2_crit_q = (c2 - 1.0) * (
                1.0 - ((1.0 - a_q) * (1.0 - a_q)) / c2
            )
            r2_crit_q = tl.maximum(r2_crit_q, 0.0)

            # Exact projection lands ON the boundary; tolerate out-dtype
            # rounding noise on the r2 = n*nu - a^2 cancellation.
            quant_tol = quant_eps * (n * nu_q + a_q * a_q)
            needs_scalar_fallback = unsafe & (r2_q > r2_crit_q + quant_tol)
            mu_q = a_q / tl.maximum(n, 1.0e-20)
            final_b = tl.where(needs_scalar_fallback, mu_q, candidate_q)

            # The chunk path stores final_b to beta's dtype before recurrence.
            # Re-quantize here so decode consumes the same effective values.
            b_b_tile = final_b.to(b.dtype.element_ty).to(tl.float32)

        b_bk = b_b_tile * b_k

        # b_v_new = (w ⊙ v) - (b ⊙ k)^T @ S
        # Project the state onto (b ⊙ k) to get the erase contribution [BV].
        if TRANSPOSE_STATE:
            erase_d = tl.sum(b_h * b_bk[None, :], 1)   # [BV]
        else:
            erase_d = tl.sum(b_h * b_bk[:, None], 0)   # [BV]

        b_w_tile = tl.load(p_w, mask=mask_v, other=0, eviction_policy='evict_first').to(tl.float32)
        b_v_new = b_w_tile * b_v - erase_d

        # State update: S += k ⊗ v_new
        if TRANSPOSE_STATE:
            b_h += b_v_new[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_h += b_k[:, None] * b_v_new[None, :]
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v, eviction_policy='evict_first')

        if IS_CONTINUOUS_BATCHING:
            if INPLACE_FINAL_STATE:
                p_ht = (
                    ht
                    + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                        tl.int64
                    )
                    * stride_final_state_token
                )
            else:
                p_ht = ht + (bos + i_t) * stride_final_state_token
            if TRANSPOSE_STATE:
                p_ht = p_ht + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV * K
        p_b += HV * K
        p_w += HV * V

    if not IS_CONTINUOUS_BATCHING:
        if STORE_FINAL_STATE:
            if TRANSPOSE_STATE:
                p_ht = ht + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_ht = ht + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


# =============================================================================
# ORCHESTRATION AND PUBLIC API
# =============================================================================
@torch.compiler.disable
def fused_recurrent_gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    out: torch.Tensor | None = None,
    transpose_state_layout: bool = False,
    beta_guard: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate buffers and launch the recurrent forward kernel.

    Internal launcher. Most callers should use `fused_recurrent_gdn2`, which
    adds argument validation. This function resolves the output and
    final-state buffers, computes the state-pool strides needed for continuous
    batching, builds the launch grid, and invokes the kernel.

    The `inplace_final_state` path writes the final state back into
    `initial_state` (used by serving stacks that own a persistent state pool);
    `output_final_state` instead allocates a fresh final-state buffer.

    Args mirror `fused_recurrent_gdn2`, plus `out` (optional preallocated
    output buffer) and `inplace_final_state`. Returns `(out, final_state)`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = 32
    if beta_guard:
        # Same head-dim range as `_GDN2BetaGuard.project` in chunk_gdn2.py.
        assert 2 <= K <= 256, f"beta_guard supports 2 <= K <= 256; got K={K}."

    if out is None:
        out = torch.zeros_like(v)
    else:
        assert out.shape == v.shape
    if inplace_final_state:
        assert initial_state is not None, (
            "inplace_final_state=True requires an initial_state"
        )
        final_state = initial_state
    elif output_final_state:
        if transpose_state_layout:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    stride_init_state_token = initial_state.stride(0) if initial_state is not None else 1
    stride_final_state_token = final_state.stride(0) if final_state is not None else 1

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (triton.cdiv(V, BV) * N * HV, )
    fused_recurrent_gdn2_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        lower_bound=lower_bound,
        quant_eps=GUARD_QUANT_TOL_MULT * torch.finfo(b.dtype).eps,
        scale=scale,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        TRANSPOSE_STATE=transpose_state_layout,
        BETA_GUARD=beta_guard,
        MARGIN=GUARD_MARGIN,
        num_warps=4,
        num_stages=2,
    )

    return out, final_state


@input_guard
def fused_recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    beta_guard: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Token-by-token forward for GDN-2. Inference-only (no gradients).

    Args:
        q (torch.Tensor): queries of shape `[B, T, H, K]`.
        k (torch.Tensor): keys of shape `[B, T, H, K]`.
        v (torch.Tensor): values of shape `[B, T, HV, V]` (GVA if HV > H).
        g (torch.Tensor): log-space decay of shape `[B, T, HV, K]`. If
            `use_gate_in_kernel=True`, this is the raw pre-activation.
        b (torch.Tensor): channel-wise erase gate of shape `[B, T, HV, K]`
            (replaces KDA's scalar beta).
        w (torch.Tensor): channel-wise write gate of shape `[B, T, HV, V]`
            (new in GDN-2).
        scale (Optional[float]): attention scale, defaults to 1/sqrt(K).
        initial_state (Optional[torch.Tensor]): `[N, HV, K, V]`, dtype fp32.
        output_final_state (bool): whether to output the final state.
        use_qk_l2norm_in_kernel (bool): L2-normalize q and k inside the kernel.
        use_gate_in_kernel (bool): compute gate activation from raw g via A_log
            (and optional dt_bias / lower_bound).
        lower_bound (Optional[float]): when set and `use_gate_in_kernel=True`,
            use the bounded activation `lower_bound * sigmoid(exp(A_log) * g)`.
        cu_seqlens (Optional[torch.LongTensor]): `[N+1]` packed-sequence offsets.
        transpose_state_layout (bool): store state as `[V, K]` instead of `[K, V]`.
        beta_guard (bool): fuse the conditional key-weighted beta-contrast
            safeguard of `chunk_gdn2` into the token loop. Default: `True`.
            Safe erase gates are used unchanged; only unsafe channel contrast
            is reduced. Decay and write gate are never modified. Requires
            `use_qk_l2norm_in_kernel=True`, as in the chunkwise path.

    Returns:
        o (torch.Tensor): outputs of shape `[B, T, HV, V]`.
        final_state (Optional[torch.Tensor]): final state of shape `[N, HV, K, V]`
            (or `[N, HV, V, K]` if `transpose_state_layout=True`) if
            `output_final_state=True`, else `None`.
    """
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
    # Shape checks (differ from KDA because b is K-dim, w is V-dim).
    assert b.shape == (*q.shape[:3], k.shape[-1]), (
        f"b must have shape [B, T, HV, K]; got {tuple(b.shape)} "
        f"vs expected {(*q.shape[:3], k.shape[-1])}."
    )
    assert w.shape == v.shape, (
        f"w must have shape [B, T, HV, V] matching v; got {tuple(w.shape)} "
        f"vs v {tuple(v.shape)}."
    )
    # Same contract as `chunk_gdn2`: the guard must see the normalized key.
    if beta_guard and not use_qk_l2norm_in_kernel:
        raise ValueError(
            "`beta_guard=True` requires `use_qk_l2norm_in_kernel=True` so the "
            "sensor sees the exact normalized key used by the recurrence."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = fused_recurrent_gdn2_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=False,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=transpose_state_layout,
        beta_guard=beta_guard,
    )
    return o, final_state


__all__ = [
    "fused_recurrent_gdn2",
    "fused_recurrent_gdn2_fwd",
    "fused_recurrent_gdn2_fwd_kernel",
]