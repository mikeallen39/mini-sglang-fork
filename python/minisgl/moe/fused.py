import functools
from typing import Dict, Optional, Tuple

import torch
from minisgl.env import ENV
from minisgl.moe import BaseMoeBackend
from minisgl.moe.dispatch import build_local_expert_dispatch_plan
from minisgl.utils import div_ceil
from minisgl.utils.logger import init_logger

_FUSED_MOE_WORKSPACE: dict[tuple[int, torch.dtype], dict[str, torch.Tensor]] = {}
_FUSED_MOE_PROFILE = {
    "w1_ms": 0.0,
    "stage2_ms": 0.0,
    "w2_ms": 0.0,
    "reduce_ms": 0.0,
    "count": 0,
}
_FUSED_MOE_PROFILE_INTERVAL = 100
_ENABLE_FUSED_W2_SILU_INT8 = True
logger = init_logger(__name__)


def reset_fused_moe_profile() -> None:
    _FUSED_MOE_PROFILE["w1_ms"] = 0.0
    _FUSED_MOE_PROFILE["stage2_ms"] = 0.0
    _FUSED_MOE_PROFILE["w2_ms"] = 0.0
    _FUSED_MOE_PROFILE["reduce_ms"] = 0.0
    _FUSED_MOE_PROFILE["count"] = 0


def get_fused_moe_profile() -> dict[str, float | int]:
    return dict(_FUSED_MOE_PROFILE)


def _workspace_key(device: torch.device, dtype: torch.dtype) -> tuple[int, torch.dtype]:
    assert device.type == "cuda"
    return (device.index if device.index is not None else torch.cuda.current_device(), dtype)


def _get_workspace_tensor(
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    key = _workspace_key(device, dtype)
    bucket = _FUSED_MOE_WORKSPACE.setdefault(key, {})
    needed_numel = 1
    for dim in shape:
        needed_numel *= dim

    cached = bucket.get(name)
    if cached is None or cached.numel() < needed_numel:
        cached = torch.empty(needed_numel, device=device, dtype=dtype)
        bucket[name] = cached

    return cached[:needed_numel].view(shape)


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    try:
        from sgl_kernel import topk_softmax

        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
        if renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    except Exception as exc:
        raise RuntimeError(
            "sgl_kernel.topk_softmax failed in fused_topk; refusing to fall back "
            "to torch.softmax/topk on the performance-critical fused MoE path"
        ) from exc

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
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

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


def _moe_align_block_size_torch(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_topk = topk_ids.reshape(-1).to(torch.int32)
    sentinel_token = flat_topk.numel()

    max_num_tokens_padded = flat_topk.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.full(
        (max_num_tokens_padded,),
        sentinel_token,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32, device=topk_ids.device)

    write_offset = 0
    for expert in range(-1, num_experts):
        token_positions = torch.nonzero(flat_topk == expert, as_tuple=False).flatten().to(torch.int32)
        count = token_positions.numel()
        if count == 0:
            continue

        padded_count = div_ceil(count, block_size) * block_size
        end_offset = write_offset + padded_count
        sorted_ids[write_offset : write_offset + count] = token_positions
        expert_ids[write_offset // block_size : end_offset // block_size] = expert
        write_offset = end_offset

    num_tokens_post_pad = torch.tensor([write_offset], dtype=torch.int32, device=topk_ids.device)
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
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    filter_expert: bool = False,
) -> torch.Tensor:
    from minisgl.kernel import (
        fused_moe_kernel_triton,
        fused_moe_w2_silu_int8_kernel_triton,
        moe_sum_reduce_triton,
        silu_and_mul_quant_int8_triton,
    )
    from minisgl.layers import gelu_and_mul, silu_and_mul

    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    if w1.dtype == torch.int8 or w2.dtype == torch.int8:
        assert w1.dtype == torch.int8 and w2.dtype == torch.int8
        assert w1_scale is not None and w2_scale is not None
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

    cache = _get_workspace_tensor(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
        name="cache",
        shape=(M * topk_ids.shape[1] * max(N, w2.shape[1]),),
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    use_int8_stage2_int8 = w2.dtype == torch.int8 and activation == "silu"
    if use_int8_stage2_int8:
        intermediate_cache2 = _get_workspace_tensor(
            device=hidden_states.device,
            dtype=torch.int8,
            name="stage2_q",
            shape=(M * topk_ids.shape[1], N // 2),
        )
        intermediate_cache2_scale = _get_workspace_tensor(
            device=hidden_states.device,
            dtype=torch.float32,
            name="stage2_s",
            shape=(M * topk_ids.shape[1], 1),
        )
    else:
        intermediate_cache2 = _get_workspace_tensor(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            name="stage2_fp",
            shape=(M * topk_ids.shape[1], N // 2),
        )
        intermediate_cache2_scale = None
    intermediate_cache3 = _get_workspace_tensor(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
        name="stage3_out",
        shape=(M, topk_ids.shape[1], w2.shape[1]),
    )
    compute_type = hidden_states.dtype

    out_hidden_states = torch.empty_like(hidden_states)
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
    profile_enabled = ENV.PROFILE_FUSED_MOE.value
    if profile_enabled:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)
        e4 = torch.cuda.Event(enable_timing=True)
        e0.record()

    fused_moe_kernel_triton(
        curr_hidden_states,
        w1,
        intermediate_cache1,
        None,
        w1_scale,
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
        filter_expert=filter_expert,
    )
    if profile_enabled:
        e1.record()
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
    if use_int8_stage2_int8:
        silu_and_mul_quant_int8_triton(
            intermediate_cache1.view(-1, N),
            intermediate_cache2,
            intermediate_cache2_scale,
        )
    else:
        FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    if profile_enabled:
        e2.record()
    if use_int8_stage2_int8 and _ENABLE_FUSED_W2_SILU_INT8:
        fused_moe_w2_silu_int8_kernel_triton(
            intermediate_cache1.view(-1, N),
            w2,
            intermediate_cache3,
            w2_scale,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            config,
            compute_type=compute_type,
            filter_expert=filter_expert,
        )
    else:
        fused_moe_kernel_triton(
            intermediate_cache2,
            w2,
            (intermediate_cache3),
            intermediate_cache2_scale,
            w2_scale,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            filter_expert=filter_expert,
        )
    if profile_enabled:
        e3.record()

    moe_sum_reduce_triton(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    if profile_enabled:
        e4.record()
        e4.synchronize()
        _FUSED_MOE_PROFILE["w1_ms"] += e0.elapsed_time(e1)
        _FUSED_MOE_PROFILE["stage2_ms"] += e1.elapsed_time(e2)
        _FUSED_MOE_PROFILE["w2_ms"] += e2.elapsed_time(e3)
        _FUSED_MOE_PROFILE["reduce_ms"] += e3.elapsed_time(e4)
        _FUSED_MOE_PROFILE["count"] += 1
        if _FUSED_MOE_PROFILE["count"] % _FUSED_MOE_PROFILE_INTERVAL == 0:
            count = _FUSED_MOE_PROFILE["count"]
            logger.info_rank0(
                "FusedMoE profile avg: w1=%.4f ms, stage2=%.4f ms, w2=%.4f ms, reduce=%.4f ms over %d calls",
                _FUSED_MOE_PROFILE["w1_ms"] / count,
                _FUSED_MOE_PROFILE["stage2_ms"] / count,
                _FUSED_MOE_PROFILE["w2_ms"] / count,
                _FUSED_MOE_PROFILE["reduce_ms"] / count,
                count,
            )
            _FUSED_MOE_PROFILE["w1_ms"] = 0.0
            _FUSED_MOE_PROFILE["stage2_ms"] = 0.0
            _FUSED_MOE_PROFILE["w2_ms"] = 0.0
            _FUSED_MOE_PROFILE["reduce_ms"] = 0.0
            _FUSED_MOE_PROFILE["count"] = 0
    return out_hidden_states


class FusedMoe(BaseMoeBackend):
    """
    Stateless MoE backend - all configuration passed through forward().
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
        filter_expert = (
            num_global_experts is not None
            and (w1.shape[0] if num_dispatch_experts is None else num_dispatch_experts)
            != num_global_experts
        )

        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            dispatch_plan.topk_weights,
            dispatch_plan.topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            filter_expert=filter_expert,
        )
