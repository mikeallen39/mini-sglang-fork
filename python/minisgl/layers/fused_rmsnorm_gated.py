from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_gated_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    stride_x_row,
    stride_gate_row,
    stride_out_row,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = x_ptr + row * stride_x_row + cols
    g_ptrs = gate_ptr + row * stride_gate_row + cols
    o_ptrs = out_ptr + row * stride_out_row + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    rstd = tl.rsqrt(var + eps)
    y = x * rstd * w
    y = y * g * tl.sigmoid(g)

    tl.store(o_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask)


def fused_rmsnorm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert x.is_cuda and gate.is_cuda and weight.is_cuda
    assert x.ndim == 2 and gate.ndim == 2
    assert x.shape == gate.shape
    assert x.stride(-1) == 1 and gate.stride(-1) == 1
    assert weight.ndim == 1 and weight.shape[0] == x.shape[-1]
    M, N = x.shape
    if out is None:
        out = torch.empty_like(gate)
    else:
        assert out.shape == x.shape
        assert out.stride(-1) == 1

    block_n = min(65536 // x.element_size(), triton.next_power_of_2(N))
    if N > block_n:
        raise RuntimeError("fused_rmsnorm_gated does not support feature dim >= 64KB")
    num_warps = min(max(block_n // 256, 1), 8)

    _fused_rmsnorm_gated_kernel[(M,)](
        x,
        gate,
        weight,
        out,
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        M,
        N,
        eps,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out
