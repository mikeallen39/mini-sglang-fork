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


def _select_decode_quant_gemm_config(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    if M <= 2:
        if N >= 4096:
            return 256, 128, 4, 4
        return 128, 128, 4, 4
    if M <= 8:
        return 128, 64, 4, 4
    return 128, 64, 4, 4


def _select_weight_only_int8_gemm_config(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    if M <= 2:
        if N >= 4096:
            return 1, 128, 64, 4
        return 1, 64, 64, 4
    if M <= 8:
        return 8, 128, 64, 4
    return 16, 128, 64, 4


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


def decode_quant_int8_gemm_triton(
    input: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    from .triton.activation_quant import decode_quant_int8_gemm_kernel

    assert input.is_contiguous()
    assert qweight_t.is_contiguous()
    assert weight_scale.is_contiguous()
    assert output.is_contiguous()
    assert input.ndim == 2
    assert qweight_t.ndim == 2
    assert weight_scale.ndim == 2
    assert output.ndim == 2
    assert input.shape[1] == qweight_t.shape[0]
    assert qweight_t.shape[1] == output.shape[1]
    assert output.shape[0] == input.shape[0]
    assert weight_scale.shape[0] == output.shape[1]
    assert weight_scale.shape[1] == 1
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert output.dtype == torch.bfloat16
    if bias is not None:
        assert bias.ndim == 1
        assert bias.shape[0] == output.shape[1]

    M, K = input.shape
    N = output.shape[1]
    block_n, block_k, num_warps, num_stages = _select_decode_quant_gemm_config(M, N, K)
    block_m = 1
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    decode_quant_int8_gemm_kernel[grid](
        input,
        qweight_t,
        weight_scale.view(-1),
        bias,
        output,
        input.stride(0),
        input.stride(1),
        qweight_t.stride(0),
        qweight_t.stride(1),
        output.stride(0),
        output.stride(1),
        M,
        N,
        K,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def weight_only_int8_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    import triton
    from .triton.activation_quant import weight_only_int8_gemm_kernel

    assert input.is_contiguous()
    assert qweight.is_contiguous()
    assert weight_scale.is_contiguous()
    assert output.is_contiguous()
    assert input.ndim == 2
    assert qweight.ndim == 2
    assert weight_scale.ndim == 2
    assert output.ndim == 2
    assert input.shape[1] == qweight.shape[1]
    assert qweight.shape[0] == output.shape[1]
    assert output.shape[0] == input.shape[0]
    assert weight_scale.shape == (qweight.shape[0], 1)
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert qweight.dtype == torch.int8
    assert weight_scale.dtype == torch.float32
    assert output.dtype in (torch.float16, torch.bfloat16)
    if bias is not None:
        assert bias.ndim == 1
        assert bias.shape[0] == output.shape[1]

    M, K = input.shape
    N = output.shape[1]
    block_m, block_n, block_k, num_warps = _select_weight_only_int8_gemm_config(M, N, K)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    weight_only_int8_gemm_kernel[grid](
        input,
        qweight,
        weight_scale.view(-1),
        bias,
        output,
        input.stride(0),
        input.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        output.stride(0),
        output.stride(1),
        M,
        N,
        K,
        HAS_BIAS=bias is not None,
        OUT_DTYPE_BF16=output.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=4,
    )
