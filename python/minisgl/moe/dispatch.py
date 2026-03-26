from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalExpertDispatchPlan:
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    local_mask: torch.Tensor


def build_local_expert_dispatch_plan(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    num_global_experts: int | None,
) -> LocalExpertDispatchPlan:
    if num_global_experts is None or (
        local_expert_start == 0 and num_local_experts == num_global_experts
    ):
        return LocalExpertDispatchPlan(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            local_mask=torch.ones_like(topk_ids, dtype=torch.bool),
        )

    local_expert_end = local_expert_start + num_local_experts
    local_mask = (topk_ids >= local_expert_start) & (topk_ids < local_expert_end)
    remapped_ids = torch.where(
        local_mask,
        topk_ids - local_expert_start,
        torch.full_like(topk_ids, -1),
    )
    remapped_weights = torch.where(local_mask, topk_weights, torch.zeros_like(topk_weights))
    return LocalExpertDispatchPlan(
        topk_weights=remapped_weights,
        topk_ids=remapped_ids,
        local_mask=local_mask,
    )
