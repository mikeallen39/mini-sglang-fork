from __future__ import annotations

import torch

from ._routing_loader import ensure_routing_ops_loaded


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
) -> None:
    ensure_routing_ops_loaded()
    torch.ops.sgl_kernel.moe_align_block_size.default(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: torch.Tensor | None = None,
) -> None:
    ensure_routing_ops_loaded()
    torch.ops.sgl_kernel.topk_softmax.default(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        moe_softcapping,
        correction_bias,
    )
