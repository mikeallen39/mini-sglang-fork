from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, Generator, Tuple

import safetensors
import torch
from tqdm import tqdm
from minisgl.distributed import get_tp_info
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

    name = name.removeprefix("language_model.")
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
    num_kv_heads: int,
) -> torch.Tensor:
    if any(key.count(sub) for sub in _STREAM_SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads < tp_size:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = tp_rank * num_kv_heads // tp_size
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(tp_size, dim=0)[tp_rank].clone()
    if any(key.count(sub) for sub in _STREAM_SPLIT_DIM_1):
        return value.chunk(tp_size, dim=1)[tp_rank].clone()
    if key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, tp_size)
        vocab_start_idx = tp_rank * num_embeddings_per_partition
        vocab_end_idx = min((tp_rank + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    return value


def _stream_get_merge_info(key: str):
    for suffix, (fused_suffix, slots) in _STREAM_MERGE_GROUPS.items():
        if key.count(suffix):
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

    setattr(module, attr_name, weight)
    return True


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
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, tuple[torch.Tensor, set[int]]] = {}
    result: Dict[str, torch.Tensor] = {}

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _normalize_checkpoint_key(raw_name, config.num_layers)
                if name is None:
                    continue
                raw = f.get_tensor(raw_name)
                tensor = _stream_shard_tensor(
                    name,
                    raw,
                    tp_info.rank,
                    tp_info.size,
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
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
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
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    unmatched_keys: list[str] = []
    disable_tqdm = tp_info.size > 1 and tp_info.rank != 0

    with tqdm(total=len(files), desc="Loading weights to GPU", disable=disable_tqdm, unit="file") as pbar:
        for file in files:
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = _normalize_checkpoint_key(raw_name, config.num_layers)
                    if name is None:
                        continue

                    tensor = _stream_shard_tensor(
                        name,
                        f.get_tensor(raw_name),
                        tp_info.rank,
                        tp_info.size,
                        config.num_kv_heads,
                    )

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
                                (config.num_experts,) + out_tensor.shape,
                                dtype=out_tensor.dtype,
                                device=out_tensor.device,
                            )
                            expert_buf[packed_key] = (packed, set())

                        packed, seen = expert_buf[packed_key]
                        packed[expert_idx].copy_(out_tensor)
                        seen.add(expert_idx)
                        if len(seen) != config.num_experts:
                            continue
                        del expert_buf[packed_key]
                        out_key, out_tensor = packed_key, packed

                    if not _set_module_tensor(module_dict, out_key, out_tensor):
                        unmatched_keys.append(out_key)
            pbar.update(1)

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
