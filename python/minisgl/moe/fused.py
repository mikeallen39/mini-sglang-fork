import functools
from typing import Dict, Optional, Tuple

import torch
from minisgl.moe import BaseMoeBackend
from minisgl.utils import div_ceil


def _remap_global_experts_to_local(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    num_global_experts: int | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_global_experts is None or (
        local_expert_start == 0 and num_local_experts == num_global_experts
    ):
        return topk_weights, topk_ids

    local_expert_end = local_expert_start + num_local_experts
    local_mask = (topk_ids >= local_expert_start) & (topk_ids < local_expert_end)
    remapped_ids = torch.where(
        local_mask,
        topk_ids - local_expert_start,
        torch.full_like(topk_ids, num_local_experts),
    )
    remapped_weights = torch.where(local_mask, topk_weights, torch.zeros_like(topk_weights))
    return remapped_weights, remapped_ids


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import topk_softmax

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights, topk_ids


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    correction_bias: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Grouped TopK routing for MoE.

    Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:467-490

    This implements the GLM4 MoE Lite routing:
    1. Apply sigmoid to get scores
    2. Add correction_bias for group selection only
    3. For each group, compute group score as topk(2).sum()
    4. Select topk_group groups
    5. Within selected groups, select topk experts
    6. Get weights from original sigmoid scores (without correction_bias)
    7. Renormalize and apply routed_scaling_factor

    Parameters:
    - hidden_states: Input tensor [num_tokens, hidden_size]
    - gating_output: Router logits [num_tokens, num_experts]
    - topk: Total number of experts to select
    - renormalize: Whether to renormalize topk weights
    - num_expert_group: Number of expert groups (n_group)
    - topk_group: Number of groups to select
    - num_fused_shared_experts: Number of fused shared experts
    - routed_scaling_factor: Scaling factor for routed experts
    - correction_bias: e_score_correction_bias tensor
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    num_tokens = gating_output.shape[0]
    num_experts = gating_output.shape[1]
    experts_per_group = num_experts // num_expert_group

    # Step 1: Apply sigmoid to get scores (transformers uses sigmoid, not softmax)
    scores = gating_output.sigmoid()

    # Step 2: Add correction_bias for group/choice selection only
    # correction_bias is used only for selecting experts, not for final weights
    scores_for_choice = scores
    if correction_bias is not None:
        scores_for_choice = scores + correction_bias

    # Step 3: Compute group scores using topk(2).sum() (not max!)
    # Reference: transformers line 470-474
    group_scores = (
        scores_for_choice.view(num_tokens, num_expert_group, experts_per_group)
        .topk(2, dim=-1)[0]  # Get top 2 scores in each group
        .sum(dim=-1)  # Sum them to get group score
    )  # [num_tokens, num_expert_group]

    # Step 4: Select topk_group groups
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]

    # Step 5: Create mask for selected groups
    group_mask = torch.zeros_like(group_scores)  # [num_tokens, num_expert_group]
    group_mask.scatter_(1, group_idx, 1)

    # Expand group mask to expert level
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_expert_group, experts_per_group)
        .reshape(num_tokens, num_experts)
    )  # [num_tokens, num_experts]

    # Step 6: Apply mask to scores_for_choice for expert selection
    masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

    # Step 7: Select topk experts from masked scores
    topk_ids = torch.topk(masked_scores, k=topk, dim=-1, sorted=False)[1]

    # Step 8: Get weights from ORIGINAL sigmoid scores (not scores_for_choice!)
    # This is critical: correction_bias only affects selection, not weights
    topk_weights = scores.gather(1, topk_ids)

    # Step 9: Renormalize weights
    if renormalize:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator

    # Step 10: Apply routed_scaling_factor
    if routed_scaling_factor is not None:
        topk_weights = topk_weights * routed_scaling_factor

    # Convert to expected dtype
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # Handle fused shared experts (if applicable)
    if num_fused_shared_experts:
        # Replace last expert ID with shared expert
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )

    # Mask padded region if needed
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1

    return topk_weights, topk_ids


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
) -> Dict[str, int]:
    E, _, N = w2_shape
    config = get_default_config(M, E, N, w1_shape[2], top_k)
    return config


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    from minisgl.kernel import fused_moe_kernel_triton, moe_sum_reduce_triton
    from minisgl.layers import gelu_and_mul, silu_and_mul

    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )
    compute_type = hidden_states.dtype

    out_hidden_states = hidden_states
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    begin_token_idx, end_token_idx = 0, num_tokens

    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    config = get_config_func(tokens_num)

    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], E
    )

    fused_moe_kernel_triton(
        curr_hidden_states,
        w1,
        intermediate_cache1,
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
    )
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    fused_moe_kernel_triton(
        intermediate_cache2,
        w2,
        (intermediate_cache3),
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
    )

    moe_sum_reduce_triton(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    return out_hidden_states


class FusedMoe(BaseMoeBackend):
    """
    Stateless MoE backend - all configuration passed through forward().
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
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

        topk_weights, topk_ids = _remap_global_experts_to_local(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            local_expert_start=local_expert_start,
            num_local_experts=w1.shape[0],
            num_global_experts=num_global_experts,
        )

        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
