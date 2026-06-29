from __future__ import annotations

import torch


def _select_per_token_quant_config(M: int, N: int) -> tuple[int, int]:
    if M <= 8 and N >= 1024:
        return 1024, 8
    if M >= 32:
        return 256, 2
    return 256, 4


def _select_gemma_rmsnorm_quant_config(M: int, N: int) -> tuple[int, int]:
    if M <= 8 and N >= 1024:
        return 1024, 8
    if M >= 32:
        return 256, 4
    return 256, 4


def _select_silu_and_mul_quant_config(M: int, N2: int) -> tuple[int, int]:
    if M <= 8 and N2 >= 1024:
        return 1024, 2
    if M >= 32:
        return 256, 8
    return 256, 4


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
    block_n, num_warps = _select_per_token_quant_config(M, N)

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
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


def gemma_rmsnorm_quant_int8_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    from .triton.activation_quant import gemma_rmsnorm_quant_int8_kernel

    assert input.is_contiguous()
    assert weight.is_contiguous()
    assert output_q.is_contiguous()
    assert output_s.is_contiguous()
    assert input.ndim == 2
    assert weight.ndim == 1
    assert output_q.ndim == 2
    assert output_s.ndim == 2
    assert output_q.shape == input.shape
    assert output_s.shape[0] == input.shape[0]
    assert output_s.shape[1] == 1
    assert weight.shape[0] == input.shape[1]

    M, N = input.shape
    grid = (M,)
    block_n, num_warps = _select_gemma_rmsnorm_quant_config(M, N)

    gemma_rmsnorm_quant_int8_kernel[grid](
        input,
        weight,
        output_q,
        output_s,
        input.stride(0),
        input.stride(1),
        output_q.stride(0),
        output_q.stride(1),
        output_s.stride(0),
        M,
        N,
        eps,
        BLOCK_N=block_n,
        num_warps=num_warps,
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
    block_n, num_warps = _select_silu_and_mul_quant_config(M, N2)
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
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
