from typing import Optional

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import (
    DistributedCommunicator,
    get_local_expert_range,
    get_moe_tp_info,
    get_tp_info,
)
from minisgl.quantization import is_w8a8_int8_enabled, quantize_weight_per_channel_int8
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        # Grouped TopK parameters
        use_grouped_topk: bool = False,
        num_expert_group: int = 0,
        topk_group: int = 0,
        routed_scaling_factor: float = 1.0,
        num_fused_shared_experts: int = 0,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.moe_tp_size = get_moe_tp_info(tp_info).size
        self.local_expert_start, local_expert_end = get_local_expert_range(num_experts)
        self.num_local_experts = local_expert_end - self.local_expert_start
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        # Grouped TopK
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts

        intermediate_size_per_partition = div_even(intermediate_size, self.moe_tp_size)
        self.gate_up_proj = torch.empty(
            self.num_local_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        )
        self.gate_up_proj_scale: torch.Tensor | None = None
        self.down_proj = torch.empty(
            self.num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
        )
        self.down_proj_scale: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ):
        ctx = get_global_ctx()
        moe_backend = ctx.moe_backend

        final_hidden_states = moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w1_scale=self.gate_up_proj_scale,
            w2=self.down_proj,
            w2_scale=self.down_proj_scale,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            # Grouped TopK parameters
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            correction_bias=correction_bias,
            num_fused_shared_experts=self.num_fused_shared_experts,
            local_expert_start=self.local_expert_start,
            num_global_experts=self.num_experts,
            num_dispatch_experts=self.num_local_experts,
        )
        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states

    def process_weights_after_loading(self) -> None:
        if not is_w8a8_int8_enabled() or self.gate_up_proj.dtype == torch.int8:
            return

        gate_up_q = []
        gate_up_scale = []
        down_q = []
        down_scale = []
        for expert_id in range(self.num_local_experts):
            q_w1, s_w1 = quantize_weight_per_channel_int8(self.gate_up_proj[expert_id])
            q_w2, s_w2 = quantize_weight_per_channel_int8(self.down_proj[expert_id])
            gate_up_q.append(q_w1)
            gate_up_scale.append(s_w1)
            down_q.append(q_w2)
            down_scale.append(s_w2)

        self.gate_up_proj = torch.stack(gate_up_q, dim=0).contiguous()
        self.gate_up_proj_scale = torch.stack(gate_up_scale, dim=0).contiguous()
        self.down_proj = torch.stack(down_q, dim=0).contiguous()
        self.down_proj_scale = torch.stack(down_scale, dim=0).contiguous()
