import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import triton
from minisgl.env import ENV
from minisgl.moe import BaseMoeBackend
from minisgl.moe.dispatch import build_local_expert_dispatch_plan
from minisgl.core import get_global_ctx
from minisgl.quantization import quantize_activation_per_token_int8
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
_FUSED_MOE_PROFILE_INTERVAL = 20
# The specialized fused w2+silu int8 kernel regressed badly on the current
# Qwen3.6 routed-expert shapes, especially for prefill-sized token counts.
# Keep it disabled by default until a shape-aware heuristic or a fixed kernel
# is available.
_ENABLE_FUSED_W2_SILU_INT8 = False
logger = init_logger(__name__)
_SGLANG_MOE_CONFIG_FALLBACKS = (
    "NVIDIA_A800-SXM4-80GB",
    "NVIDIA_A100-SXM4-80GB",
    "NVIDIA_H20",
)


def _use_moe_sum_reduce_torch_compile(num_tokens: int) -> bool:
    return num_tokens <= 32


@torch.compile
def moe_sum_reduce_torch_compile(x: torch.Tensor, out: torch.Tensor, routed_scaling_factor: float) -> None:
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


def _use_moe_topk_workspace() -> bool:
    return ENV.MOE_REUSE_WORKSPACE.value or ENV.MOE_REUSE_TOPK_WORKSPACE.value


def _use_moe_align_workspace() -> bool:
    return ENV.MOE_REUSE_WORKSPACE.value or ENV.MOE_REUSE_ALIGN_WORKSPACE.value


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


def _get_workspace_tensor_by_key(
    *,
    device: torch.device,
    dtype: torch.dtype,
    key_name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return _get_workspace_tensor(device=device, dtype=dtype, name=key_name, shape=shape)


@functools.lru_cache(maxsize=None)
def _warn_moe_config_fallback_once(triton_version: str, direct_path: str, fallback_path: str) -> None:
    logger.warning(
        "MoE config file not found for triton %s at %s; fallback to %s",
        triton_version,
        direct_path,
        fallback_path,
    )


@functools.lru_cache(maxsize=None)
def _candidate_sglang_moe_config_dirs() -> tuple[Path, ...]:
    env_dir = os.environ.get("MINISGL_SGLANG_MOE_CONFIG_DIR")
    candidate_dirs: list[Path] = []
    if env_dir:
        candidate_dirs.append(Path(env_dir))
    try:
        import sglang  # type: ignore

        candidate_dirs.append(
            Path(sglang.__file__).resolve().parent
            / "srt"
            / "layers"
            / "moe"
            / "moe_runner"
            / "triton_utils"
            / "configs"
        )
    except Exception:
        pass
    candidate_dirs.append(
        Path("/mnt/42_store/zxz/aiinfra/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs")
    )
    existing: list[Path] = []
    seen: set[Path] = set()
    for path in candidate_dirs:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        existing.append(resolved)
        seen.add(resolved)
    return tuple(existing)


@functools.lru_cache(maxsize=None)
def _get_moe_device_name() -> str:
    return torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")


def _iter_moe_device_name_candidates(device_name: str) -> tuple[str, ...]:
    candidates = [device_name]
    compact_name = device_name.replace("_PCIe", "").replace("_80GB_PCIe", "")
    if compact_name not in candidates:
        candidates.append(compact_name)
    for fallback in _SGLANG_MOE_CONFIG_FALLBACKS:
        if fallback not in candidates:
            candidates.append(fallback)
    return tuple(candidates)


@functools.lru_cache(maxsize=None)
def _load_sglang_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    triton_version: str,
    device_name: str,
) -> Optional[Dict[int, Dict[str, Any]]]:
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    for config_root in _candidate_sglang_moe_config_dirs():
        direct_path = config_root / version_dir / f"E={E},N={N},device_name={device_name}{dtype_selector}.json"
        if direct_path.exists():
            with direct_path.open() as f:
                return {int(key): val for key, val in json.load(f).items()}

        available_versions = sorted(
            (
                path.name.removeprefix("triton_").replace("_", ".")
                for path in config_root.iterdir()
                if path.is_dir() and path.name.startswith("triton_")
            ),
            key=lambda value: tuple(int(x) for x in value.split(".")),
            reverse=True,
        )
        for try_version in available_versions:
            if try_version == triton_version:
                continue
            fallback_path = (
                config_root
                / f"triton_{try_version.replace('.', '_')}"
                / f"E={E},N={N},device_name={device_name}{dtype_selector}.json"
            )
            if fallback_path.exists():
                _warn_moe_config_fallback_once(
                    triton_version, str(direct_path), str(fallback_path)
                )
                with fallback_path.open() as f:
                    return {int(key): val for key, val in json.load(f).items()}
    return None


@functools.lru_cache(maxsize=None)
def _get_sglang_style_moe_config(
    E: int,
    N: int,
    dtype: Optional[str],
) -> Optional[Dict[int, Dict[str, Any]]]:
    triton_version = triton.__version__
    for device_name in _iter_moe_device_name_candidates(_get_moe_device_name()):
        configs = _load_sglang_moe_configs(E, N, dtype, triton_version, device_name)
        if configs is not None:
            return configs
    return None


@functools.lru_cache(maxsize=None)
def _load_sglang_moe_down_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    triton_version: str,
    device_name: str,
) -> Optional[Dict[int, Dict[str, Any]]]:
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    for config_root in _candidate_sglang_moe_config_dirs():
        direct_path = (
            config_root / version_dir / f"E={E},N={N},device_name={device_name}{dtype_selector}_down.json"
        )
        if direct_path.exists():
            with direct_path.open() as f:
                return {int(key): val for key, val in json.load(f).items()}

        available_versions = sorted(
            (
                path.name.removeprefix("triton_").replace("_", ".")
                for path in config_root.iterdir()
                if path.is_dir() and path.name.startswith("triton_")
            ),
            key=lambda value: tuple(int(x) for x in value.split(".")),
            reverse=True,
        )
        for try_version in available_versions:
            if try_version == triton_version:
                continue
            fallback_path = (
                config_root
                / f"triton_{try_version.replace('.', '_')}"
                / f"E={E},N={N},device_name={device_name}{dtype_selector}_down.json"
            )
            if fallback_path.exists():
                _warn_moe_config_fallback_once(
                    triton_version, str(direct_path), str(fallback_path)
                )
                with fallback_path.open() as f:
                    return {int(key): val for key, val in json.load(f).items()}
    return None


@functools.lru_cache(maxsize=None)
def _get_sglang_style_moe_down_config(
    E: int,
    N: int,
    dtype: Optional[str],
) -> Optional[Dict[int, Dict[str, Any]]]:
    triton_version = triton.__version__
    for device_name in _iter_moe_device_name_candidates(_get_moe_device_name()):
        configs = _load_sglang_moe_down_configs(E, N, dtype, triton_version, device_name)
        if configs is not None:
            return configs
    return None


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

        if _use_moe_topk_workspace():
            topk_weights = _get_workspace_tensor_by_key(
                device=hidden_states.device,
                dtype=torch.float32,
                key_name="topk_weights",
                shape=(M, topk),
            )
            topk_ids = _get_workspace_tensor_by_key(
                device=hidden_states.device,
                dtype=torch.int32,
                key_name="topk_ids",
                shape=(M, topk),
            )
        else:
            topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
            topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        gating_input = gating_output if ENV.MOE_SKIP_TOPK_FP32_CAST.value else gating_output.float()
        topk_softmax(topk_weights, topk_ids, gating_input, renormalize)
        if renormalize and not ENV.MOE_SKIP_TOPK_POST_RENORM.value:
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
    if ENV.MOE_ALIGN_SMALL_CAP.value and topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    if _use_moe_align_workspace():
        sorted_ids = _get_workspace_tensor_by_key(
            device=topk_ids.device,
            dtype=torch.int32,
            key_name="moe_sorted_ids",
            shape=(max_num_tokens_padded,),
        )
    else:
        sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    if _use_moe_align_workspace():
        expert_ids = _get_workspace_tensor_by_key(
            device=topk_ids.device,
            dtype=torch.int32,
            key_name="moe_expert_ids",
            shape=(max_num_m_blocks,),
        )
        num_tokens_post_pad = _get_workspace_tensor_by_key(
            device=topk_ids.device,
            dtype=torch.int32,
            key_name="moe_num_tokens_post_pad",
            shape=(1,),
        )
        cumsum_buffer = _get_workspace_tensor_by_key(
            device=topk_ids.device,
            dtype=torch.int32,
            key_name="moe_cumsum_buffer",
            shape=(num_experts + 2,),
        )
    else:
        expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
        num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
        cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)

    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
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
    if ENV.MOE_SGLANG_CONFIG_LOOKUP.value:
        configs = _get_sglang_style_moe_config(E, N, None)
        if configs:
            nearest = min(configs.keys(), key=lambda x: abs(x - M))
            return dict(configs[nearest])
    config = get_default_config(M, E, N, w1_shape[2], top_k)
    return config


def try_get_optimal_moe_down_config(
    w2_shape: Tuple[int, ...],
    M: int,
    up_config: Dict[str, int],
) -> Optional[Dict[str, int]]:
    E, _, N = w2_shape
    if not ENV.MOE_SGLANG_DOWN_CONFIG.value:
        return None
    configs = _get_sglang_style_moe_down_config(E, N, None)
    if not configs:
        return None
    down_config = dict(configs[min(configs.keys(), key=lambda x: abs(x - M))])
    down_config.pop("USE_TMA", None)
    if down_config.get("BLOCK_SIZE_M") != up_config["BLOCK_SIZE_M"]:
        down_config["BLOCK_SIZE_M"] = up_config["BLOCK_SIZE_M"]
    return down_config


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
    routed_scaling_factor: float = 1.0,
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    from minisgl.kernel import (
        fused_moe_kernel_triton,
        fused_moe_silu_down_triton,
        fused_moe_w2_silu_int8_kernel_triton,
        moe_sum_reduce_triton,
    )
    from minisgl.layers import (
        fused_gelu_and_mul,
        fused_silu_and_mul,
        gelu_and_mul,
        silu_and_mul,
    )

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

    if (
        output_buffer is not None
        and output_buffer.shape == hidden_states.shape
        and output_buffer.dtype == hidden_states.dtype
        and output_buffer.device == hidden_states.device
        and output_buffer.is_contiguous()
    ):
        out_hidden_states = output_buffer
    else:
        out_hidden_states = torch.empty_like(hidden_states)
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    begin_token_idx, end_token_idx = 0, num_tokens

    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    config = get_config_func(tokens_num)
    down_config = try_get_optimal_moe_down_config(
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        tokens_num,
        config,
    )

    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], E
    )
    batch = get_global_ctx().batch
    profile_enabled = (
        ENV.PROFILE_FUSED_MOE.value
        and curr_hidden_states.is_cuda
        and not torch.cuda.is_current_stream_capturing()
        and (
            not ENV.PROFILE_MOE_DECODE_ONLY.value
            or batch.is_decode
        )
    )
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
    if use_int8_stage2_int8 and _ENABLE_FUSED_W2_SILU_INT8:
        if profile_enabled:
            e2.record()
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
            down_config or config,
            compute_type=compute_type,
            filter_expert=filter_expert,
        )
    elif ENV.MOE_SINGLE_KERNEL.value and not use_int8_stage2_int8:
        # Fused silu_and_mul + down_proj in a single kernel.
        # Skips the intermediate_cache2 buffer and the separate silu_and_mul launch.
        if profile_enabled:
            e1.record()
        fused_moe_silu_down_triton(
            intermediate_cache1.view(-1, N),
            w2,
            intermediate_cache3,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            down_config or config,
            compute_type=compute_type,
            filter_expert=filter_expert,
        )
        if profile_enabled:
            e2.record()
            e3.record()  # stage2 and w2 merged; stage2_ms will be ~0
    else:
        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        FUSED_FN_MAP = {"silu": fused_silu_and_mul, "gelu": fused_gelu_and_mul}
        if use_int8_stage2_int8:
            gate_up_fp = torch.empty(
                (intermediate_cache1.shape[0] * intermediate_cache1.shape[1], N // 2),
                device=intermediate_cache1.device,
                dtype=intermediate_cache1.dtype,
            )
            fused_silu_and_mul(intermediate_cache1.view(-1, N), gate_up_fp)
            q, s = quantize_activation_per_token_int8(gate_up_fp)
            intermediate_cache2.copy_(q)
            assert intermediate_cache2_scale is not None
            intermediate_cache2_scale.copy_(s)
        else:
            if ENV.MOE_FUSED_ACTIVATION.value:
                FUSED_FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
        if profile_enabled:
            e2.record()
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
            down_config or config,
            compute_type=compute_type,
            filter_expert=filter_expert,
        )
    if profile_enabled:
        e3.record()

    if (
        ENV.MOE_FASTPATH_TOPK2_REDUCE.value
        and curr_topk_ids.shape[1] == 2
        and routed_scaling_factor == 1.0
    ):
        torch.add(
            intermediate_cache3[:, 0],
            intermediate_cache3[:, 1],
            out=out_hidden_states[begin_token_idx:end_token_idx],
        )
    elif ENV.MOE_SGL_REDUCE.value and intermediate_cache3.is_cuda:
        try:
            import sgl_kernel

            if ENV.MOE_TORCH_COMPILE_REDUCE.value and _use_moe_sum_reduce_torch_compile(
                intermediate_cache3.shape[0]
            ):
                moe_sum_reduce_torch_compile(
                    intermediate_cache3,
                    out_hidden_states[begin_token_idx:end_token_idx],
                    routed_scaling_factor,
                )
            else:
                sgl_kernel.moe_sum_reduce(
                    intermediate_cache3,
                    out_hidden_states[begin_token_idx:end_token_idx],
                    routed_scaling_factor,
                )
        except Exception:
            moe_sum_reduce_triton(
                intermediate_cache3,
                out_hidden_states[begin_token_idx:end_token_idx],
            )
    else:
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
        output_buffer: torch.Tensor | None = None,
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

        fastpath_no_dispatch = (
            ENV.MOE_DIRECT_FASTPATH.value
            and (num_global_experts is None or (w1.shape[0] if num_dispatch_experts is None else num_dispatch_experts) == num_global_experts)
            and local_expert_start == 0
        )
        if fastpath_no_dispatch:
            dispatch_topk_weights = topk_weights
            dispatch_topk_ids = topk_ids
        else:
            dispatch_plan = build_local_expert_dispatch_plan(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                local_expert_start=local_expert_start,
                num_local_experts=w1.shape[0] if num_dispatch_experts is None else num_dispatch_experts,
                num_global_experts=num_global_experts,
            )
            dispatch_topk_weights = dispatch_plan.topk_weights
            dispatch_topk_ids = dispatch_plan.topk_ids
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
            dispatch_topk_weights,
            dispatch_topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            filter_expert=filter_expert,
            routed_scaling_factor=routed_scaling_factor,
            output_buffer=output_buffer if ENV.MOE_REUSE_OUTPUT_BUFFER.value else None,
        )
