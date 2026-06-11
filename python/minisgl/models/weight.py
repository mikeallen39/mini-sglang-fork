from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, Generator, Tuple

import safetensors
import torch
from tqdm import tqdm
from minisgl.distributed import get_ep_info, get_local_expert_range, get_moe_tp_info, get_tp_info
from minisgl.layers.base import BaseOP
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight

_STREAM_SPLIT_DIM_0 = [
    ".q_proj",
    ".k_proj",
    ".v_proj",
    ".gate_proj",
    ".up_proj",
    ".q_b_proj",
    ".kv_b_proj",
    ".conv1d.weight",
]
_STREAM_SPLIT_LINEAR_ATTN_HEAD_PARAMS = [
    ".A_log",
    ".dt_bias",
]
_STREAM_SPLIT_DIM_1 = [".o_proj", ".down_proj"]
_STREAM_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_STREAM_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_STREAM_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _is_packed_routed_expert_key(key: str) -> bool:
    return ".mlp.experts." in key and _STREAM_EXPERT_PATTERN.match(key) is None


def _stream_shard_packed_routed_expert_tensor(
    key: str,
    value: torch.Tensor,
    moe_tp_rank: int,
    moe_tp_size: int,
) -> torch.Tensor:
    ep_info = get_ep_info()

    if ep_info.size > 1:
        if value.shape[0] % ep_info.size != 0:
            raise ValueError(
                f"Packed routed expert tensor {key} has {value.shape[0]} experts, "
                f"which is not divisible by ep_size={ep_info.size}"
            )
        value = value.chunk(ep_info.size, dim=0)[ep_info.rank].contiguous()

    if moe_tp_size == 1:
        return value.clone()

    if ".gate_up_proj" in key or ".gate_proj" in key or ".up_proj" in key:
        return value.chunk(moe_tp_size, dim=1)[moe_tp_rank].contiguous()
    if ".down_proj" in key:
        return value.chunk(moe_tp_size, dim=2)[moe_tp_rank].contiguous()

    return value.clone()


def _get_safetensors_files(model_path: str) -> Tuple[str, list]:
    """Get model folder and sorted safetensors files.

    Returns:
        (model_folder, files)
    """
    model_folder = download_hf_weight(model_path)
    files = sorted(glob.glob(f"{model_folder}/*.safetensors"))
    return model_folder, files


def _get_model_info(model_path: str) -> Tuple[int, int]:
    """Get weight count and file count from model checkpoint."""
    model_folder, files = _get_safetensors_files(model_path)

    # Try index file first
    index_file = os.path.join(model_folder, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            weight_count = len(json.load(f).get("weight_map", {}))
            return weight_count, len(files)

    # Fallback: count manually
    weight_count = sum(
        len(safetensors.safe_open(file, framework="pt", device="cpu").keys())
        for file in files
    )
    return weight_count, len(files)


def _weights_iterator(
    model_path: str,
    show_progress: bool = False,
    total: int = 0,
    desc: str = "Loading weights",
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Stream weights directly from safetensors files."""
    _, files = _get_safetensors_files(model_path)

    tp_info = get_tp_info()
    disable_tqdm = not show_progress or (tp_info.size > 1 and tp_info.rank != 0)

    pbar = tqdm(total=total, desc=desc, disable=disable_tqdm, unit="tensor") if total else None

    for file in files:
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if pbar:
                    pbar.update(1)
                yield name, tensor

    if pbar:
        pbar.close()


def _normalize_checkpoint_key(name: str, num_layers: int) -> str | None:
    if name.startswith(("vision_tower.", "multi_modal_projector.")):
        return None

    if name.startswith("model.language_model."):
        name = f"model.{name.removeprefix('model.language_model.')}"
    elif name.startswith("language_model."):
        name = f"model.{name.removeprefix('language_model.')}"
    if name.startswith("model.layers."):
        parts = name.split(".")
        if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) >= num_layers:
            return None
    return name


def _stream_shard_tensor(
    key: str,
    value: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    moe_tp_rank: int,
    moe_tp_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    if _is_packed_routed_expert_key(key):
        return _stream_shard_packed_routed_expert_tensor(
            key,
            value,
            moe_tp_rank,
            moe_tp_size,
        )

    shard_rank = moe_tp_rank if ".experts." in key else tp_rank
    shard_size = moe_tp_size if ".experts." in key else tp_size
    if any(key.count(sub) for sub in _STREAM_SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads < shard_size:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = shard_rank * num_kv_heads // shard_size
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(shard_size, dim=0)[shard_rank].clone()
    if any(key.count(sub) for sub in _STREAM_SPLIT_LINEAR_ATTN_HEAD_PARAMS):
        return value.chunk(tp_size, dim=0)[tp_rank].clone()
    if any(key.count(sub) for sub in _STREAM_SPLIT_DIM_1):
        return value.chunk(shard_size, dim=1)[shard_rank].clone()
    if key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, tp_size)
        vocab_start_idx = tp_rank * num_embeddings_per_partition
        vocab_end_idx = min((tp_rank + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    return value


def _stream_get_merge_info(key: str):
    if key.count(".qkv_proj") and any(key.count(sub) for sub in (".q_proj", ".k_proj", ".v_proj")):
        return None
    for suffix, (fused_suffix, slots) in _STREAM_MERGE_GROUPS.items():
        if key.count(suffix):
            if suffix in (".gate_proj", ".up_proj") and ".shared_expert." not in key and ".experts." not in key:
                continue
            return key.replace(suffix, fused_suffix), _STREAM_SLOT_NAMES[suffix], slots
    return None


def _stream_get_expert_stack_info(key: str) -> tuple[str, int] | None:
    match = _STREAM_EXPERT_PATTERN.match(key)
    if match is None:
        return None
    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def _set_module_tensor(module_dict, checkpoint_key: str, weight: torch.Tensor) -> bool:
    if checkpoint_key not in module_dict:
        return False

    module, attr_name = module_dict[checkpoint_key]
    if hasattr(module, "weight_loader") and attr_name == "weight":
        module.weight_loader(weight)
        return True

    current = getattr(module, attr_name)
    if (
        isinstance(current, torch.Tensor)
        and not current.is_meta
        and current.shape == weight.shape
        and current.dtype == weight.dtype
        and current.device == weight.device
    ):
        current.copy_(weight)
    else:
        setattr(module, attr_name, weight)
    return True


def _stream_qwen3_5_local_qkv(
    config,
    model,
    module_dict,
    raw_name: str,
    tensor: torch.Tensor,
    device: torch.device,
) -> bool:
    if config.architectures[0] not in {"Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"}:
        return False
    if ".self_attn." not in raw_name:
        return False
    if not any(raw_name.endswith(suffix) for suffix in (".q_proj.weight", ".k_proj.weight", ".v_proj.weight")):
        return False

    base_name = raw_name
    if base_name.startswith("model.language_model."):
        base_name = f"model.{base_name.removeprefix('model.language_model.')}"
    elif base_name.startswith("language_model."):
        base_name = f"model.{base_name.removeprefix('language_model.')}"
    if base_name.startswith("model.layers."):
        parts = base_name.split(".")
        if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) >= config.num_layers:
            return False

    if base_name.endswith(".q_proj.weight"):
        slot = "q"
    elif base_name.endswith(".k_proj.weight"):
        slot = "k"
    else:
        slot = "v"
    fused_key = base_name.replace(f".{slot}_proj.weight", ".qkv_proj.weight")
    if fused_key not in module_dict:
        return False

    module, attr_name = module_dict[fused_key]
    if attr_name != "weight" or type(module).__name__ != "Qwen3_5LocalQKVProj":
        return False

    if not hasattr(module, "_stacked_params"):
        module._stacked_params = {}
    module._stacked_params[slot] = tensor
    if len(module._stacked_params) < 3:
        return True

    # load_weight_to_model() already shards Q/K/V via _stream_shard_tensor().
    # Re-sharding here would halve the local projection size again under TP>1.
    q = module._stacked_params.pop("q").contiguous()
    k = module._stacked_params.pop("k").contiguous()
    v = module._stacked_params.pop("v").contiguous()
    fused = torch.cat([q, k, v], dim=0).to(device=device)
    return _set_module_tensor(module_dict, fused_key, fused)


def _stream_qwen3_5_linear_attn_proj(
    config,
    model,
    module_dict,
    raw_name: str,
    tensor: torch.Tensor,
    device: torch.device,
) -> bool:
    if config.architectures[0] not in {"Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"}:
        return False
    if ".linear_attn." not in raw_name:
        return False

    base_name = raw_name
    if base_name.startswith("model.language_model."):
        base_name = f"model.{base_name.removeprefix('model.language_model.')}"
    elif base_name.startswith("language_model."):
        base_name = f"model.{base_name.removeprefix('language_model.')}"
    if base_name.startswith("model.layers."):
        parts = base_name.split(".")
        if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) >= config.num_layers:
            return False

    if base_name.endswith(".in_proj_qkv.weight"):
        fused_key = base_name.replace(".in_proj_qkv.weight", ".in_proj_qkvz.weight")
        slot = "qkv"
        expected_type = "LinearColParallelMerged"
    elif base_name.endswith(".in_proj_z.weight"):
        fused_key = base_name.replace(".in_proj_z.weight", ".in_proj_qkvz.weight")
        slot = "z"
        expected_type = "LinearColParallelMerged"
    elif base_name.endswith(".in_proj_b.weight"):
        fused_key = base_name.replace(".in_proj_b.weight", ".in_proj_ba.weight")
        slot = "b"
        expected_type = "LinearColParallelMerged"
    elif base_name.endswith(".in_proj_a.weight"):
        fused_key = base_name.replace(".in_proj_a.weight", ".in_proj_ba.weight")
        slot = "a"
        expected_type = "LinearColParallelMerged"
    else:
        return False

    if fused_key not in module_dict:
        return False

    module, attr_name = module_dict[fused_key]
    if attr_name != "weight" or type(module).__name__ != expected_type:
        return False

    if not hasattr(module, "_stacked_params"):
        module._stacked_params = {}
    module._stacked_params[slot] = tensor

    if slot in {"qkv", "z"}:
        if "qkv" not in module._stacked_params or "z" not in module._stacked_params:
            return True
        fused = torch.cat(
            [module._stacked_params.pop("qkv"), module._stacked_params.pop("z")],
            dim=0,
        ).to(device=device)
        return _set_module_tensor(module_dict, fused_key, fused)

    if "b" not in module._stacked_params or "a" not in module._stacked_params:
        return True
    fused = torch.cat(
        [module._stacked_params.pop("b"), module._stacked_params.pop("a")],
        dim=0,
    ).to(device=device)
    return _set_module_tensor(module_dict, fused_key, fused)


def _get_local_expert_layout(num_experts: int) -> tuple[int, int, int]:
    ep_info = get_ep_info()
    if ep_info.size == 1:
        return 0, num_experts, num_experts
    start, end = get_local_expert_range(num_experts, ep_info)
    return start, end, end - start


def load_weight(
    model_path: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Streaming GLM-compatible loader that returns a merged, TP-sharded state dict."""
    from .config import ModelConfig

    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()
    moe_tp_info = get_moe_tp_info(tp_info)
    local_expert_start, local_expert_end, num_local_experts = _get_local_expert_layout(
        config.num_experts
    )
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, tuple[torch.Tensor, set[int]]] = {}
    result: Dict[str, torch.Tensor] = {}

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _normalize_checkpoint_key(raw_name, config.num_layers)
                if name is None:
                    continue
                if config.is_moe and (expert_info := _stream_get_expert_stack_info(name)) is not None:
                    _, expert_idx = expert_info
                    if not (local_expert_start <= expert_idx < local_expert_end):
                        continue
                raw = f.get_tensor(raw_name)
                tensor = _stream_shard_tensor(
                    name,
                    raw,
                    tp_info.rank,
                    tp_info.size,
                    moe_tp_info.rank,
                    moe_tp_info.size,
                    config.num_kv_heads,
                )
                del raw

                if (info := _stream_get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _stream_get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx - local_expert_start] = out[1]
                    if len(slots) != num_local_experts:
                        continue
                    experts = [slots[idx] for idx in range(num_local_experts)]
                    del expert_buf[packed_key]
                    result[packed_key] = torch.stack(experts, dim=0)
                else:
                    result[out[0]] = out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
    return result


def _build_module_dict(model: BaseOP, prefix: str = "") -> Dict[str, Tuple[BaseOP, str]]:
    """Build a mapping from parameter names to (module, attr_name) tuples."""
    from minisgl.layers.base import OPList

    module_dict = {}

    for name, value in model.__dict__.items():
        if name.startswith("_"):
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(value, (torch.Tensor, torch.nn.Parameter)):
            module_dict[full_name] = (model, name)
        elif isinstance(value, OPList):
            for i, op in enumerate(value.op_list):
                module_dict.update(_build_module_dict(op, f"{full_name}.{i}"))
        elif isinstance(value, BaseOP):
            module_dict.update(_build_module_dict(value, full_name))

    return module_dict


def load_weight_to_model(
    model_path: str,
    model,
    device: torch.device = None,
):
    """Load model weights directly into the model on GPU via streaming shard iteration."""
    import logging
    logger = logging.getLogger(__name__)
    from .config import ModelConfig

    if device is None:
        device = torch.device("cuda:0")

    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    model_folder, files = _get_safetensors_files(model_path)
    logger.info(f"Loading model weights to {device}: {len(files)} files")

    module_dict = _build_module_dict(model)
    tp_info = get_tp_info()
    moe_tp_info = get_moe_tp_info(tp_info)
    local_expert_start, local_expert_end, num_local_experts = _get_local_expert_layout(
        config.num_experts
    )
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    unmatched_keys: list[str] = []
    disable_tqdm = tp_info.size > 1 and tp_info.rank != 0

    with tqdm(total=len(files), desc="Loading weights to GPU", disable=disable_tqdm, unit="file") as pbar:
        for file in files:
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for raw_name in f.keys():
                    name = _normalize_checkpoint_key(raw_name, config.num_layers)
                    if name is None:
                        continue
                    if config.is_moe and (expert_info := _stream_get_expert_stack_info(name)) is not None:
                        _, expert_idx = expert_info
                        if not (local_expert_start <= expert_idx < local_expert_end):
                            continue

                    tensor = _stream_shard_tensor(
                        name,
                        f.get_tensor(raw_name),
                        tp_info.rank,
                        tp_info.size,
                        moe_tp_info.rank,
                        moe_tp_info.size,
                        config.num_kv_heads,
                    )

                    if _stream_qwen3_5_local_qkv(
                        config,
                        model,
                        module_dict,
                        raw_name,
                        tensor,
                        device,
                    ):
                        continue
                    if _stream_qwen3_5_linear_attn_proj(
                        config,
                        model,
                        module_dict,
                        raw_name,
                        tensor,
                        device,
                    ):
                        continue

                    if (info := _stream_get_merge_info(name)) is None:
                        out_key, out_tensor = name, tensor
                    else:
                        merged_key, slot, all_slots = info
                        merge_buf.setdefault(merged_key, {})[slot] = tensor
                        if not all(s in merge_buf[merged_key] for s in all_slots):
                            continue
                        parts = [merge_buf[merged_key][s] for s in all_slots]
                        del merge_buf[merged_key]
                        out_key, out_tensor = merged_key, torch.cat(parts, dim=0)

                    if config.is_moe and (expert_info := _stream_get_expert_stack_info(out_key)) is not None:
                        packed_key, expert_idx = expert_info
                        if packed_key not in expert_buf:
                            packed = torch.empty(
                                (num_local_experts,) + out_tensor.shape,
                                dtype=out_tensor.dtype,
                                device=out_tensor.device,
                            )
                            expert_buf[packed_key] = (packed, set())

                        packed, seen = expert_buf[packed_key]
                        local_expert_idx = expert_idx - local_expert_start
                        packed[local_expert_idx].copy_(out_tensor)
                        seen.add(local_expert_idx)
                        if len(seen) != num_local_experts:
                            continue
                        del expert_buf[packed_key]
                        out_key, out_tensor = packed_key, packed

                    if out_tensor.device != device:
                        out_tensor = out_tensor.to(device=device)
                    if not _set_module_tensor(module_dict, out_key, out_tensor):
                        unmatched_keys.append(out_key)
                    del out_tensor
            pbar.update(1)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
    if unmatched_keys:
        logger.warning(
            "Skipped %d checkpoint tensors without matching module entries",
            len(set(unmatched_keys)),
        )

    logger.info(f"Model weights loaded successfully to {device}")


def _shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    use_mla: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Shard state dict across tensor parallel ranks.
    """
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size

    # Standard split lists
    # Note: gate_up_proj must come before gate_proj/up_proj to match correctly
    SPLIT_DIM_0_LIST = [".q_proj", ".k_proj", ".v_proj", ".gate_up_proj", ".gate_proj", ".up_proj"]
    # MLA: q_a_proj and kv_a_proj_with_mqa are replicated
    MLA_REPLICATED_LIST = [".q_a_proj", ".kv_a_proj_with_mqa", ".q_a_layernorm", ".kv_a_layernorm"]
    # MLA: q_b_proj and kv_b_proj are column parallel
    MLA_SPLIT_DIM_0_LIST = [".q_b_proj", ".kv_b_proj"]
    SPLIT_DIM_1_LIST = [".o_proj", ".down_proj"]
    LINEAR_ATTN_HEAD_PARAM_LIST = [".A_log", ".dt_bias"]

    for key, value in state_dict.items():
        # Check if MLA replicated
        if use_mla and any(sub in key for sub in MLA_REPLICATED_LIST):
            shard_state_dict[key] = value
            continue

        # Check if MLA column parallel
        if use_mla and any(sub in key for sub in MLA_SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
            continue

        # Standard sharding
        if any(sub in key for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(sub in key for sub in LINEAR_ATTN_HEAD_PARAM_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(sub in key for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif "lm_head" in key or "embed_tokens" in key:
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = div_ceil(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value

    return shard_state_dict


def _merge_state_dict(
    state_dict: Dict[str, torch.Tensor],
    use_mla: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Merge state dict by concatenating split weights.
    """
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    processed_keys = set()

    for original_key in list(state_dict.keys()):
        if original_key in processed_keys:
            continue

        key = original_key  # Keep the original key (with model. prefix)

        # GLM4 MoE gate weight (router) - keep as is (not gate_proj)
        # Example: model.layers.0.mlp.gate.weight
        if ".mlp.gate." in key and ".gate_proj" not in key and ".gate_up_proj" not in key:
            filtered_state_dict[key] = state_dict[original_key]
            processed_keys.add(original_key)
            continue

        # GLM4 shared experts - keep as is (do not merge gate_proj + up_proj)
        # Example: model.layers.0.mlp.shared_experts.gate_proj
        if ".mlp.shared_experts." in key:
            filtered_state_dict[key] = state_dict[original_key]
            processed_keys.add(original_key)
            continue

        # Skip MLA keys - they don't need merging
        if use_mla and (
            ".q_a_proj" in key
            or ".kv_a_proj_with_mqa" in key
            or ".q_a_layernorm" in key
            or ".kv_a_layernorm" in key
            or ".q_b_proj" in key
            or ".kv_b_proj" in key
        ):
            filtered_state_dict[key] = state_dict[original_key]
            processed_keys.add(original_key)
            continue

        # Standard QKV merge
        if ".q_proj" in key and ".q_a_proj" not in key:
            q_proj = state_dict[original_key]
            k_proj_key = original_key.replace(".q_proj", ".k_proj")
            v_proj_key = original_key.replace(".q_proj", ".v_proj")

            if k_proj_key in state_dict and v_proj_key in state_dict:
                k_proj = state_dict[k_proj_key]
                v_proj = state_dict[v_proj_key]
                new_key = key.replace(".q_proj", ".qkv_proj")
                filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
                processed_keys.add(original_key)
                processed_keys.add(k_proj_key)
                processed_keys.add(v_proj_key)
                continue
            else:
                filtered_state_dict[key] = state_dict[original_key]
                processed_keys.add(original_key)
                continue

        # Standard MLP: keep gate_proj and up_proj separate (no merge)
        # Skip merging for gate_proj - load directly instead
        if ".gate_proj" in key and ".mlp.experts." not in key and ".mlp.shared_experts." not in key:
            # Direct load without merging
            filtered_state_dict[key] = state_dict[original_key]
            processed_keys.add(original_key)
            continue

        # Expert weights merge: gate_proj + up_proj → gate_up_proj
        if ".mlp.experts." in key and ".gate_proj" in key:
            gate_proj = state_dict[original_key]
            up_proj_key = original_key.replace(".gate_proj", ".up_proj")
            down_proj_key = original_key.replace(".gate_proj", ".down_proj")

            if up_proj_key in state_dict and down_proj_key in state_dict:
                up_proj = state_dict[up_proj_key]
                down_proj = state_dict[down_proj_key]
                gate_up_key = key.replace(".gate_proj", ".gate_up_proj")
                down_key = key.replace(".gate_proj", ".down_proj")
                filtered_state_dict[gate_up_key] = torch.cat([gate_proj, up_proj], dim=0)
                filtered_state_dict[down_key] = down_proj
                processed_keys.add(original_key)
                processed_keys.add(up_proj_key)
                processed_keys.add(down_proj_key)
                continue
            else:
                filtered_state_dict[key] = state_dict[original_key]
                processed_keys.add(original_key)
                continue

        # Skip keys already processed
        if original_key in processed_keys:
            continue

        # Default: copy as is
        filtered_state_dict[key] = state_dict[original_key]
        processed_keys.add(original_key)

    return filtered_state_dict
