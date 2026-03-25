from __future__ import annotations

import torch

from minisgl.moe.dispatch import build_local_expert_dispatch_plan


def test_local_expert_dispatch_plan_is_identity_without_ep():
    topk_weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    topk_ids = torch.tensor([[1, 3]], dtype=torch.int32)

    plan = build_local_expert_dispatch_plan(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        local_expert_start=0,
        num_local_experts=4,
        num_global_experts=4,
    )

    assert torch.equal(plan.topk_weights, topk_weights)
    assert torch.equal(plan.topk_ids, topk_ids)
    assert torch.equal(plan.local_mask, torch.ones_like(topk_ids, dtype=torch.bool))


def test_local_expert_dispatch_plan_remaps_and_masks_remote_experts():
    topk_weights = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    topk_ids = torch.tensor([[1, 3, 6]], dtype=torch.int32)

    plan = build_local_expert_dispatch_plan(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        local_expert_start=2,
        num_local_experts=2,
        num_global_experts=8,
    )

    assert torch.equal(plan.local_mask, torch.tensor([[False, True, False]]))
    assert torch.equal(plan.topk_ids, torch.tensor([[2, 1, 2]], dtype=torch.int32))
    assert torch.allclose(plan.topk_weights, torch.tensor([[0.0, 0.2, 0.0]], dtype=torch.float32))
