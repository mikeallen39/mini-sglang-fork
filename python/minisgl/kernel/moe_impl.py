from typing import Any, Dict
import torch


def fused_moe_silu_down_triton(
    intermediate1: torch.Tensor,
    w2: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    config: Dict[str, Any],
    compute_type: torch.dtype,
    filter_expert: bool = False,
) -> None:
    import triton
    from .triton.fused_moe import fused_moe_silu_down_kernel

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert intermediate1.shape[-1] % 2 == 0, "intermediate1 must be packed [gate, up]"
    N = intermediate1.shape[-1] // 2  # intermediate size
    K = w2.shape[1]  # hidden size (w2 output dim)

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(K, META["BLOCK_SIZE_N"]),
    )
    if N % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False

    fused_moe_silu_down_kernel[grid](
        intermediate1,
        w2,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N=N,
        K=K,
        EM=sorted_token_ids.shape[0],
        num_valid_tokens=topk_ids.numel(),
        stride_inter1m=intermediate1.stride(0),
        stride_inter1n=intermediate1.stride(1),
        stride_w2e=w2.stride(0),
        stride_w2k_out=w2.stride(1),  # hidden dim (output)
        stride_w2k_in=w2.stride(2),   # intermediate dim (input)
        stride_cm=C.stride(1),
        stride_cn=C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=triton.language.bfloat16 if compute_type == torch.bfloat16 else triton.language.float16,
        even_Ks=even_Ks,
        filter_expert=filter_expert,
        **config,
    )
from typing import Any, Dict

import torch


def fused_moe_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
    filter_expert: bool = False,
) -> None:
    import triton
    import triton.language as tl

    from .triton.fused_moe import fused_moe_kernel
    from minisgl.quantization import is_w8a16_int8_enabled, quantize_activation_per_token_int8

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    padded_size = 0
    use_int8_weight = B.dtype == torch.int8
    use_int8_w8a8 = use_int8_weight and not is_w8a16_int8_enabled()
    use_int8_w8a16 = use_int8_weight and is_w8a16_int8_enabled()
    per_channel_quant = use_int8_w8a8
    if use_int8_weight:
        if B_scale is None:
            raise ValueError("B_scale must be provided for int8 fused MoE")
        if use_int8_w8a8 and A_scale is None:
            A, A_scale = quantize_activation_per_token_int8(A)
        else:
            A = A.contiguous()
    A_scale_arg = A if A_scale is None else A_scale
    B_scale_arg = B if B_scale is None else B_scale
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False
    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16
    fused_moe_kernel[grid](
        A,
        B,
        C,
        A_scale_arg,
        B_scale_arg,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale_arg.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale_arg.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale_arg.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale_arg.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        use_int8_w8a8=use_int8_w8a8,  # type: ignore
        use_weight_only_int8=use_int8_w8a16,  # type: ignore
        per_channel_quant=per_channel_quant,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        filter_expert=filter_expert,  # type: ignore
        **config,
    )


def fused_moe_w2_silu_int8_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    config: Dict[str, Any],
    compute_type: torch.dtype,
    filter_expert: bool = False,
) -> None:
    import triton
    import triton.language as tl

    from .triton.fused_moe import fused_moe_w2_silu_int8_kernel

    assert A.is_contiguous()
    assert B.dtype == torch.int8
    assert B_scale is not None
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2]
    even_Ks = K % config["BLOCK_SIZE_K"] == 0
    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16
    fused_moe_w2_silu_int8_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        filter_expert=filter_expert,  # type: ignore
        **config,
    )


def moe_sum_reduce_triton(input: torch.Tensor, output: torch.Tensor) -> None:
    import triton

    from .triton.fused_moe import moe_sum_reduce_kernel

    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 8

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,  # type: ignore
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,  # type: ignore
    )
