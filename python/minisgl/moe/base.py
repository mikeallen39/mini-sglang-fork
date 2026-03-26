from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
        local_expert_start: int = 0,
        num_global_experts: int | None = None,
        num_dispatch_experts: int | None = None,
    ) -> torch.Tensor: ...
