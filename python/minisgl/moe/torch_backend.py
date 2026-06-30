from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from minisgl.moe import BaseMoeBackend
from minisgl.moe.dispatch import build_local_expert_dispatch_plan
from minisgl.quantization import apply_w8a16_int8_linear, apply_w8a8_int8_linear, is_w8a16_int8_enabled

from .fused import fused_topk, grouped_topk


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    if activation == "silu":
        return F.silu(gate) * up
    if activation == "gelu":
        return F.gelu(gate) * up
    raise ValueError(f"Unsupported activation: {activation}")


class TorchMoe(BaseMoeBackend):
    """
    A conservative MoE backend implemented with plain PyTorch ops.
    This is slower than fused kernels but is used as a stable fallback.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w1_scale: torch.Tensor | None,
        w2: torch.Tensor,
        w2_scale: torch.Tensor | None,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        # Grouped TopK parameters
        use_grouped_topk: bool = False,
        num_expert_group: int = 0,
        topk_group: int = 0,
        routed_scaling_factor: float = 1.0,
        correction_bias: Optional[torch.Tensor] = None,
        num_fused_shared_experts: int = 0,
        local_expert_start: int = 0,
        num_global_experts: int | None = None,
        num_dispatch_experts: int | None = None,
    ) -> torch.Tensor:
        if use_grouped_topk:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=gating_output,
                topk=topk,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                correction_bias=correction_bias,
            )
        else:
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=gating_output,
                topk=topk,
                renormalize=renormalize,
            )

        dispatch_plan = build_local_expert_dispatch_plan(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            local_expert_start=local_expert_start,
            num_local_experts=w1.shape[0] if num_dispatch_experts is None else num_dispatch_experts,
            num_global_experts=num_global_experts,
        )
        topk_weights = dispatch_plan.topk_weights
        topk_ids = dispatch_plan.topk_ids

        num_tokens, hidden_size = hidden_states.shape
        num_experts = w1.shape[0]
        output = torch.zeros_like(hidden_states)

        # Route token slices expert-by-expert to avoid custom kernels.
        for expert_id in range(num_experts):
            token_idx, topk_pos = torch.where(topk_ids == expert_id)
            if token_idx.numel() == 0:
                continue

            routed_x = hidden_states[token_idx]
            routed_w = topk_weights[token_idx, topk_pos].to(hidden_states.dtype)

            if apply_router_weight_on_input:
                routed_x = routed_x * routed_w.unsqueeze(-1)

            if w1.dtype == torch.int8:
                assert w1_scale is not None
                if is_w8a16_int8_enabled():
                    inter = apply_w8a16_int8_linear(
                        routed_x,
                        w1[expert_id].contiguous(),
                        w1_scale[expert_id].unsqueeze(-1),
                        None,
                    )
                else:
                    inter = apply_w8a8_int8_linear(
                        routed_x,
                        w1[expert_id].transpose(0, 1).contiguous(),
                        w1_scale[expert_id].unsqueeze(-1),
                        None,
                    )
            else:
                inter = F.linear(routed_x, w1[expert_id])
            inter = _apply_activation(inter, activation)
            if w2.dtype == torch.int8:
                assert w2_scale is not None
                if is_w8a16_int8_enabled():
                    routed_out = apply_w8a16_int8_linear(
                        inter,
                        w2[expert_id].contiguous(),
                        w2_scale[expert_id].unsqueeze(-1),
                        None,
                    )
                else:
                    routed_out = apply_w8a8_int8_linear(
                        inter,
                        w2[expert_id].transpose(0, 1).contiguous(),
                        w2_scale[expert_id].unsqueeze(-1),
                        None,
                    )
            else:
                routed_out = F.linear(inter, w2[expert_id])

            if not apply_router_weight_on_input:
                routed_out = routed_out * routed_w.unsqueeze(-1)

            output.index_add_(0, token_idx, routed_out)

        assert output.shape == (num_tokens, hidden_size)
        return output
