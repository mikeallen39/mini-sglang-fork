from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is expected on CUDA setups.
    triton = None
    tl = None


SUPPORTED_LINEAR_ATTN_BACKENDS = {"torch", "sglang"}

_CURRENT_LINEAR_ATTN_BACKEND = "torch"


def set_linear_attn_backend(backend: str) -> None:
    if backend not in SUPPORTED_LINEAR_ATTN_BACKENDS:
        raise ValueError(f"Unsupported linear attention backend: {backend}")
    global _CURRENT_LINEAR_ATTN_BACKEND
    _CURRENT_LINEAR_ATTN_BACKEND = backend


def get_linear_attn_backend() -> str:
    return _CURRENT_LINEAR_ATTN_BACKEND


def has_sglang_linear_attn_kernel() -> bool:
    return triton is not None and tl is not None


if triton is not None:

    @triton.jit
    def _fused_gdn_gating_kernel(
        gate,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        stride_gate_tok: tl.constexpr,
        stride_beta_tok: tl.constexpr,
        stride_a_tok: tl.constexpr,
        stride_b_tok: tl.constexpr,
        T: tl.constexpr,
        HV: tl.constexpr,
        BH: tl.constexpr,
    ):
        i_t = tl.program_id(0)
        i_h_blk = tl.program_id(1)
        head_off = i_h_blk * BH + tl.arange(0, BH)
        mask = (i_t < T) & (head_off < HV)
        blk_A_log = tl.load(A_log + head_off, mask=head_off < HV, other=0).to(tl.float32)
        blk_a = tl.load(a + i_t * stride_a_tok + head_off, mask=mask, other=0).to(tl.float32)
        blk_b = tl.load(b + i_t * stride_b_tok + head_off, mask=mask, other=0).to(tl.float32)
        blk_bias = tl.load(dt_bias + head_off, mask=head_off < HV, other=0).to(tl.float32)
        x = blk_a + blk_bias
        softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        blk_g = -tl.exp(blk_A_log) * softplus_x
        blk_beta = tl.sigmoid(blk_b)
        tl.store(gate + i_t * stride_gate_tok + head_off, blk_g, mask=mask)
        tl.store(beta_output + i_t * stride_beta_tok + head_off, blk_beta, mask=mask)

    @triton.jit
    def _fused_linear_attn_decode_kv_kernel(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        output,
        state,
        state_indices,
        scale,
        stride_mixed_qkv_tok: tl.constexpr,
        stride_a_tok: tl.constexpr,
        stride_b_tok: tl.constexpr,
        stride_output_tok: tl.constexpr,
        stride_state_token: tl.constexpr,
        stride_state_head: tl.constexpr,
        stride_state_k: tl.constexpr,
        stride_indices_tok: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        SOFTPLUS_THRESHOLD: tl.constexpr,
        USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        state_idx = tl.load(state_indices + i_n * stride_indices_tok).to(tl.int64)
        if state_idx < 0:
            return

        p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
        q_off = i_h * K + o_k
        k_off = (H * K) + i_h * K + o_k
        v_off = (2 * H * K) + i_hv * V + o_v
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
        b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
        A_log_val = tl.load(A_log + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(
            x <= SOFTPLUS_THRESHOLD,
            tl.log(1.0 + tl.exp(x)),
            x,
        )
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val)

        p_state = (
            state
            + state_idx * stride_state_token
            + i_hv * stride_state_head
            + o_k[:, None] * stride_state_k
            + o_v[None, :]
        )
        b_h = tl.load(p_state, mask=mask_h, other=0).to(tl.float32)
        b_h *= tl.exp(g_val)
        b_v -= tl.sum(b_h * b_k[:, None], axis=0)
        b_v *= beta_val
        b_h += b_k[:, None] * b_v[None, :]

        p_out = output + i_n * stride_output_tok + i_hv * V + o_v
        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_out, b_o.to(output.dtype.element_ty), mask=mask_v)
        tl.store(p_state, b_h.to(state.dtype.element_ty), mask=mask_h)

    @triton.jit
    def _fused_linear_attn_prefill_kv_kernel(
        q,
        k,
        v,
        gate,
        beta,
        output,
        state,
        scale,
        stride_q_tok: tl.constexpr,
        stride_q_head: tl.constexpr,
        stride_k_tok: tl.constexpr,
        stride_k_head: tl.constexpr,
        stride_v_tok: tl.constexpr,
        stride_v_head: tl.constexpr,
        stride_gate_tok: tl.constexpr,
        stride_beta_tok: tl.constexpr,
        stride_output_tok: tl.constexpr,
        stride_output_head: tl.constexpr,
        stride_state_head: tl.constexpr,
        stride_state_k: tl.constexpr,
        T: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_h = tl.program_id(1)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        p_state = (
            state
            + i_h * stride_state_head
            + o_k[:, None] * stride_state_k
            + o_v[None, :]
        )
        b_h = tl.load(p_state, mask=mask_h, other=0).to(tl.float32)

        for i_t in range(T):
            p_q = q + i_t * stride_q_tok + i_h * stride_q_head + o_k
            p_k = k + i_t * stride_k_tok + i_h * stride_k_head + o_k
            p_v = v + i_t * stride_v_tok + i_h * stride_v_head + o_v
            p_gate = gate + i_t * stride_gate_tok + i_h
            p_beta = beta + i_t * stride_beta_tok + i_h

            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_gate).to(tl.float32)
            b_beta = tl.load(p_beta).to(tl.float32)

            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale

            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], axis=0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            p_out = output + i_t * stride_output_tok + i_h * stride_output_head + o_v
            b_o = tl.sum(b_h * b_q[:, None], axis=0)
            tl.store(p_out, b_o.to(output.dtype.element_ty), mask=mask_v)

        tl.store(p_state, b_h.to(state.dtype.element_ty), mask=mask_h)


def fused_gdn_gating_sglang(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for the sglang linear attention backend")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Expected a and b to be 2D, got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    if a.shape != b.shape:
        raise ValueError(f"Expected a and b to have same shape, got {tuple(a.shape)} and {tuple(b.shape)}")
    t, hv = a.shape
    if A_log.numel() != hv or dt_bias.numel() != hv:
        raise ValueError(
            f"Expected A_log/dt_bias to have {hv} elements, got {A_log.numel()} and {dt_bias.numel()}"
        )
    gate = torch.empty((t, hv), dtype=torch.float32, device=a.device)
    beta = torch.empty((t, hv), dtype=torch.float32, device=a.device)
    bh = 32
    grid = (t, triton.cdiv(hv, bh))
    _fused_gdn_gating_kernel[grid](
        gate,
        beta,
        A_log,
        a,
        b,
        dt_bias,
        stride_gate_tok=gate.stride(0),
        stride_beta_tok=beta.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        T=t,
        HV=hv,
        BH=bh,
        num_warps=1,
        num_stages=1,
    )
    return gate, beta


def fused_linear_attn_decode_sglang(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    *,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for the sglang linear attention backend")
    if mixed_qkv.ndim != 2:
        raise ValueError(f"Expected mixed_qkv to be 2D, got shape={tuple(mixed_qkv.shape)}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Expected a and b to be 2D, got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    if state.ndim != 4:
        raise ValueError(f"Expected state to be 4D, got shape={tuple(state.shape)}")

    batch_size = mixed_qkv.shape[0]
    hv, k_dim, v_dim = state.shape[-3:]
    if a.shape != (batch_size, hv) or b.shape != (batch_size, hv):
        raise ValueError(
            f"Expected a/b shape {(batch_size, hv)}, got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if A_log.numel() != hv or dt_bias.numel() != hv:
        raise ValueError(
            f"Expected A_log/dt_bias to have {hv} elements, got {A_log.numel()} and {dt_bias.numel()}"
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - hv * v_dim
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(f"Invalid packed QKV shape: {tuple(mixed_qkv.shape)} for state {tuple(state.shape)}")
    q_dim = qk_dim // 2
    if q_dim % k_dim != 0:
        raise ValueError(f"Q dimension {q_dim} is not divisible by K dimension {k_dim}")
    h = q_dim // k_dim
    if h <= 0 or hv % h != 0:
        raise ValueError(f"Invalid grouped-head layout: H={h}, HV={hv}")

    bk = triton.next_power_of_2(k_dim)
    if triton.cdiv(k_dim, bk) != 1:
        raise ValueError(f"Only NK=1 is supported, got K={k_dim}, BK={bk}")
    bv = min(triton.next_power_of_2(v_dim), 32)

    output = torch.empty((batch_size, hv, v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    grid = (triton.cdiv(v_dim, bv), batch_size * hv)
    _fused_linear_attn_decode_kv_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        output=output,
        state=state,
        state_indices=state_indices,
        scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_output_tok=output.stride(0),
        stride_state_token=state.stride(0),
        stride_state_head=state.stride(1),
        stride_state_k=state.stride(2),
        stride_indices_tok=state_indices.stride(0),
        H=h,
        HV=hv,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    return output


def fused_linear_attn_prefill_sglang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    *,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for the sglang linear attention backend")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            f"Expected q/k/v to be 3D, got q.shape={tuple(q.shape)}, k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}"
        )
    if gate.ndim != 2 or beta.ndim != 2:
        raise ValueError(
            f"Expected gate/beta to be 2D, got gate.shape={tuple(gate.shape)}, beta.shape={tuple(beta.shape)}"
        )
    if state.ndim != 3:
        raise ValueError(f"Expected state to be 3D, got shape={tuple(state.shape)}")

    t, hv, k_dim = q.shape
    if k.shape != (t, hv, k_dim):
        raise ValueError(f"Expected k shape {(t, hv, k_dim)}, got {tuple(k.shape)}")
    if gate.shape != (t, hv) or beta.shape != (t, hv):
        raise ValueError(
            f"Expected gate/beta shape {(t, hv)}, got {tuple(gate.shape)} and {tuple(beta.shape)}"
        )
    if state.shape[0] != hv or state.shape[1] != k_dim:
        raise ValueError(
            f"Expected state leading shape {(hv, k_dim)}, got {tuple(state.shape[:2])}"
        )
    v_dim = state.shape[2]
    if v.shape != (t, hv, v_dim):
        raise ValueError(f"Expected v shape {(t, hv, v_dim)}, got {tuple(v.shape)}")

    bk = triton.next_power_of_2(k_dim)
    if triton.cdiv(k_dim, bk) != 1:
        raise ValueError(f"Only NK=1 is supported, got K={k_dim}, BK={bk}")
    bv = min(triton.next_power_of_2(v_dim), 32)

    output = torch.empty((t, hv, v_dim), dtype=v.dtype, device=v.device)
    grid = (triton.cdiv(v_dim, bv), hv)
    _fused_linear_attn_prefill_kv_kernel[grid](
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        output=output,
        state=state,
        scale=scale,
        stride_q_tok=q.stride(0),
        stride_q_head=q.stride(1),
        stride_k_tok=k.stride(0),
        stride_k_head=k.stride(1),
        stride_v_tok=v.stride(0),
        stride_v_head=v.stride(1),
        stride_gate_tok=gate.stride(0),
        stride_beta_tok=beta.stride(0),
        stride_output_tok=output.stride(0),
        stride_output_head=output.stride(1),
        stride_state_head=state.stride(0),
        stride_state_k=state.stride(1),
        T=t,
        HV=hv,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    return output
