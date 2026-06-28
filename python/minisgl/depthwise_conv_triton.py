"""Fused depthwise conv for GDN decode step (single token, kernel_size=4).

Replaces the generic PyTorch F.conv1d + SiLU sequence with a single
Triton kernel, saving kernel-launch overhead and intermediate memory.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _depthwise_conv_decode_kernel(
    x_ptr,
    conv_state_ptr,
    conv_weight_ptr,
    out_ptr,
    next_state_ptr,
    stride_x: tl.constexpr,
    stride_state: tl.constexpr,
    stride_weight: tl.constexpr,
    stride_out: tl.constexpr,
    stride_next_state: tl.constexpr,
    DIM: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < DIM

    x_val = tl.load(x_ptr + offs * stride_x, mask=mask, other=0).to(tl.float32)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(KERNEL_SIZE - 1):
        s_val = tl.load(conv_state_ptr + offs * stride_state + k, mask=mask, other=0).to(tl.float32)
        w_val = tl.load(conv_weight_ptr + offs * stride_weight + k, mask=mask, other=0).to(tl.float32)
        acc += s_val * w_val
    w_last = tl.load(conv_weight_ptr + offs * stride_weight + (KERNEL_SIZE - 1), mask=mask, other=0).to(tl.float32)
    acc += x_val * w_last

    if ACTIVATION == 1:
        acc = acc * tl.sigmoid(acc)
    elif ACTIVATION == 2:
        acc = acc * tl.sigmoid(acc)

    tl.store(out_ptr + offs * stride_out, acc.to(x_ptr.dtype.element_ty), mask=mask)

    for k in tl.static_range(KERNEL_SIZE - 2):
        s_next = tl.load(conv_state_ptr + offs * stride_state + (k + 1), mask=mask, other=0)
        tl.store(next_state_ptr + offs * stride_next_state + k, s_next, mask=mask)
    tl.store(next_state_ptr + offs * stride_next_state + (KERNEL_SIZE - 2),
             x_val.to(conv_state_ptr.dtype.element_ty), mask=mask)


def depthwise_conv_decode(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 2:
        x = x.squeeze(0).contiguous()
    dim = x.shape[0]
    kernel_size = conv_weight.shape[1]

    # Ensure contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not conv_state.is_contiguous():
        conv_state = conv_state.contiguous()
    if not conv_weight.is_contiguous():
        conv_weight = conv_weight.contiguous()

    out = torch.empty(dim, dtype=x.dtype, device=x.device)
    next_state = torch.empty_like(conv_state)

    act_map = {"identity": 0, "silu": 1, "swish": 2}
    act_val = act_map.get(activation, 0)

    BLOCK = min(triton.next_power_of_2(dim), 1024)
    grid = (triton.cdiv(dim, BLOCK),)

    _depthwise_conv_decode_kernel[grid](
        x, conv_state, conv_weight, out, next_state,
        stride_x=x.stride(0),
        stride_state=conv_state.stride(0),
        stride_weight=conv_weight.stride(0),
        stride_out=out.stride(0),
        stride_next_state=next_state.stride(0),
        DIM=dim, KERNEL_SIZE=kernel_size, ACTIVATION=act_val, BLOCK=BLOCK,
        num_warps=4, num_stages=2,
    )
    return out.unsqueeze(0), next_state
