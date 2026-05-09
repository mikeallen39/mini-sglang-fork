from __future__ import annotations

import torch


def per_token_quant_int8_triton(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    from .triton.activation_quant import per_token_quant_int8_kernel

    assert input.is_contiguous()
    assert output_q.is_contiguous()
    assert output_s.is_contiguous()
    assert input.ndim == 2
    assert output_q.ndim == 2
    assert output_s.ndim == 2
    assert output_q.shape == input.shape
    assert output_s.shape[0] == input.shape[0]
    assert output_s.shape[1] == 1

    M, N = input.shape
    grid = (M,)

    per_token_quant_int8_kernel[grid](
        input,
        output_q,
        output_s,
        input.stride(0),
        input.stride(1),
        output_q.stride(0),
        output_q.stride(1),
        output_s.stride(0),
        M,
        N,
        BLOCK_N=256,
        num_warps=4,
    )


def silu_and_mul_quant_int8_triton(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    from .triton.activation_quant import silu_and_mul_quant_int8_kernel

    assert input.is_contiguous()
    assert output_q.is_contiguous()
    assert output_s.is_contiguous()
    assert input.ndim == 2
    assert output_q.ndim == 2
    assert output_s.ndim == 2
    assert input.shape[1] % 2 == 0
    assert output_q.shape[0] == input.shape[0]
    assert output_q.shape[1] * 2 == input.shape[1]
    assert output_s.shape[0] == input.shape[0]
    assert output_s.shape[1] == 1

    M = input.shape[0]
    N2 = output_q.shape[1]
    BLOCK_N = 256
    grid = (M,)

    silu_and_mul_quant_int8_kernel[grid](
        input,
        output_q,
        output_s,
        input.stride(0),
        input.stride(1),
        output_q.stride(0),
        output_q.stride(1),
        output_s.stride(0),
        M,
        N2,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
