from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


@triton.jit
def _fused_sigmoid_mul_kernel(
    output_ptr,
    attn_output_ptr,
    gate_ptr,
    gate_stride_row,
    gate_stride_head,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1)

    offsets = pid_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    d = offsets - head * HEAD_DIM

    attn_off = pid_row * hidden_dim + offsets
    attn = tl.load(attn_output_ptr + attn_off, mask=mask, other=0.0).to(tl.float32)

    gate_off = pid_row * gate_stride_row + head * gate_stride_head + d
    g = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

    result = attn * tl.sigmoid(g)
    tl.store(output_ptr + attn_off, result, mask=mask)


def fused_sigmoid_mul(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for fused_sigmoid_mul")

    if gate.ndim == 3 and attn_output.ndim == 2:
        num_tokens, num_heads, head_dim = gate.shape
        hidden_dim = num_heads * head_dim
        assert attn_output.shape == (num_tokens, hidden_dim)
        gate_stride_row = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        assert attn_output.shape == gate.shape
        hidden_dim = attn_output.shape[-1]
        num_tokens = attn_output.numel() // hidden_dim
        head_dim = hidden_dim
        gate_stride_row = hidden_dim
        gate_stride_head = hidden_dim

    out = attn_output if inplace else torch.empty_like(attn_output)
    block_h = 1024 if num_tokens < 1024 else 2048
    grid = (num_tokens, triton.cdiv(hidden_dim, block_h))
    _fused_sigmoid_mul_kernel[grid](
        out,
        attn_output,
        gate,
        gate_stride_row,
        gate_stride_head,
        hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


@triton.jit
def _fused_gate_sigmoid_mul_add_kernel(
    hidden_states_ptr,
    gate_weight_ptr,
    shared_output_ptr,
    final_hidden_states_ptr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    row_offset = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    w = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(hidden_states_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(shared_output_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    f = tl.load(final_hidden_states_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))
    result = f + gate_val * s

    tl.store(final_hidden_states_ptr + row_offset + offsets, result, mask=mask)


def fused_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for fused_gate_sigmoid_mul_add")

    assert hidden_states.is_contiguous()
    assert gate_weight.is_contiguous()
    assert shared_output.is_contiguous()
    assert final_hidden_states.is_contiguous()

    num_tokens, hidden_dim = hidden_states.shape
    assert gate_weight.shape == (hidden_dim,)
    assert shared_output.shape == (num_tokens, hidden_dim)
    assert final_hidden_states.shape == (num_tokens, hidden_dim)

    block_size = triton.next_power_of_2(hidden_dim)
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), 8), 4)

    _fused_gate_sigmoid_mul_add_kernel[(num_tokens,)](
        hidden_states,
        gate_weight,
        shared_output,
        final_hidden_states,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

@triton.jit
def _fused_sigmoid_mul_flat_kernel(
    attn_output_ptr,
    gate_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_off = pid * N
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(attn_output_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    result = a * tl.sigmoid(g)
    tl.store(out_ptr + row_off + offs, result, mask=mask)


def fused_sigmoid_mul_flat(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for fused_sigmoid_mul_flat")
    assert attn_output.ndim == 2 and gate.ndim == 2
    assert attn_output.shape == gate.shape
    assert attn_output.is_contiguous()
    assert gate.is_contiguous()
    assert out.is_contiguous()
    assert out.shape == attn_output.shape
    N = attn_output.shape[1]
    block_size = min(triton.next_power_of_2(N), 4096)
    num_warps = min(max(block_size // 128, 1), 8)
    _fused_sigmoid_mul_flat_kernel[(attn_output.shape[0],)](
        attn_output,
        gate,
        out,
        N=N,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out
