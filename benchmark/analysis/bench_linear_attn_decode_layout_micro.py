from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from minisgl.linear_attention import fused_linear_attn_decode_sglang


@triton.jit
def _old_decode_kernel(
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

    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
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


def old_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch_size = mixed_qkv.shape[0]
    hv, k_dim, v_dim = state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - hv * v_dim
    q_dim = qk_dim // 2
    h = q_dim // k_dim
    bk = triton.next_power_of_2(k_dim)
    bv = min(triton.next_power_of_2(v_dim), 32)

    output = torch.empty((batch_size, hv, v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    grid = (triton.cdiv(v_dim, bv), batch_size * hv)
    _old_decode_kernel[grid](
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
        num_warps=1,
        num_stages=3,
    )
    return output


def bench(fn, iters: int = 300) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def main() -> None:
    torch.manual_seed(0)
    device = "cuda"
    batch_size = 1
    hv = 128
    k_dim = 128
    v_dim = 128
    h = 4

    mixed_qkv = torch.randn(
        batch_size, h * k_dim * 2 + hv * v_dim, device=device, dtype=torch.bfloat16
    ).contiguous()
    a = torch.randn(batch_size, hv, device=device, dtype=torch.float32).contiguous()
    b = torch.randn(batch_size, hv, device=device, dtype=torch.float32).contiguous()
    A_log = torch.randn(hv, device=device, dtype=torch.float32).contiguous()
    dt_bias = torch.randn(hv, device=device, dtype=torch.float32).contiguous()
    state_old = torch.randn(1, hv, k_dim, v_dim, device=device, dtype=torch.float32).contiguous()
    state_new = state_old.transpose(-1, -2).contiguous()
    state_indices = torch.tensor([0], device=device, dtype=torch.int32)
    scale = 1.0 / (k_dim**0.5)

    old_ms = bench(
        lambda: old_decode(mixed_qkv, a, b, A_log, dt_bias, state_old, state_indices, scale)
    )
    new_ms = bench(
        lambda: fused_linear_attn_decode_sglang(
            mixed_qkv, a, b, A_log, dt_bias, state_new, state_indices, scale
        )
    )
    print({"old_ms": old_ms, "new_ms": new_ms, "speedup": old_ms / new_ms})


if __name__ == "__main__":
    main()
