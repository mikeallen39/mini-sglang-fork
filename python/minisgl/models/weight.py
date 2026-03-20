from __future__ import annotations

import glob
import json
import os
from typing import Dict, Generator, Tuple

import safetensors
import torch
from tqdm import tqdm
from minisgl.distributed import get_tp_info
from minisgl.layers.base import BaseOP
from minisgl.utils import div_ceil, download_hf_weight


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


def load_weight(
    model_path: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Load model weights from HuggingFace checkpoint.

    This is a legacy function that loads all weights to CPU first, then merges.
    For faster loading, use load_weight_to_model instead.
    """
    # Collect all weights
    state_dict: Dict[str, torch.Tensor] = {}
    for name, weight in _weights_iterator(model_path):
        state_dict[name] = weight

    # Detect MLA
    use_mla = False
    for key in state_dict.keys():
        if ".q_a_proj" in key or ".kv_a_proj_with_mqa" in key:
            use_mla = True
            break

    # Apply sharding
    tp_info = get_tp_info()
    if tp_info.size > 1:
        state_dict = _shard_state_dict(state_dict, use_mla=use_mla)

    # Merge weights
    merged_dict = _merge_state_dict(state_dict, use_mla=use_mla)

    # Move to target device
    if device.type == "cuda":
        merged_dict = {k: v.to(device) for k, v in merged_dict.items()}

    return merged_dict


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
    """Load model weights directly into the model on GPU (layer-by-layer, memory efficient).

    This function handles:
    1. Direct loading to GPU to avoid CPU memory overhead
    2. Layer-by-layer processing with immediate memory cleanup
    3. Stacked params (gate_up_proj): Merge gate_proj + up_proj
    4. MoE experts: Stack individual expert weights into unified tensor (per-layer)

    Args:
        model_path: Path to the model checkpoint
        model: The model to load weights into
        device: Target device (defaults to cuda:0)
    """
    import gc
    import logging
    logger = logging.getLogger(__name__)

    if device is None:
        device = torch.device("cuda:0")

    # Get model info for progress bar
    total_weights, num_files = _get_model_info(model_path)
    logger.info(f"Loading model weights to {device}: {total_weights} tensors in {num_files} files")

    # Build module dict for finding weight_loader
    module_dict = _build_module_dict(model)

    # Collect weight names and organize by layer
    model_folder, files = _get_safetensors_files(model_path)

    # Read config to get num_hidden_layers (to skip MTP layer)
    config_file = os.path.join(model_folder, "config.json")
    with open(config_file) as f:
        hf_config = json.load(f)
    if "num_hidden_layers" not in hf_config:
        raise KeyError("'num_hidden_layers' not found in model config")
    num_hidden_layers = hf_config["num_hidden_layers"]

    # Build weight index: {layer_idx: [weight_names]}
    layer_weights: Dict[int, list] = {}
    global_weights = []

    # Iterate through files and collect weight names, organized by layer index or global
    for file in files:
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                # Skip MTP layer special weights
                if ".enorm." in name or ".hnorm." in name or ".shared_head." in name:
                    continue

                if name.startswith("model.layers."):
                    layer_idx = int(name.split(".")[2])
                    if layer_idx >= num_hidden_layers:
                        continue  # Skip MTP layers
                    layer_weights.setdefault(layer_idx, []).append(name)
                else:
                    global_weights.append(name)

    num_layers = len(layer_weights)
    logger.info(f"Found {num_layers} layers, {len(global_weights)} global weights")

    # Helper: load tensor directly to GPU
    def _load_tensor(name: str, file_handles: list) -> torch.Tensor:
        for f in file_handles:
            if name in f.keys():
                return f.get_tensor(name).to(device)
        raise KeyError(f"Weight {name} not found")

    # Helper: load weight into module
    def _set_weight(checkpoint_key: str, weight: torch.Tensor):
        if checkpoint_key not in module_dict:
            return False
        module, attr_name = module_dict[checkpoint_key]
        if hasattr(module, "weight_loader") and attr_name in ("weight", "bias"):
            module.weight_loader(weight)
        else:
            current = getattr(module, attr_name)
            if isinstance(current, torch.nn.Parameter):
                current.to_(weight.device)
            elif isinstance(current, torch.Tensor):
                setattr(module, attr_name, weight)
        return True

    def _get_or_init_tensor(
        checkpoint_key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if checkpoint_key not in module_dict:
            return None
        module, attr_name = module_dict[checkpoint_key]
        current = getattr(module, attr_name)
        if (
            isinstance(current, torch.Tensor)
            and not current.is_meta
            and current.shape == shape
            and current.dtype == dtype
            and current.device == device
        ):
            return current
        materialized = torch.empty(shape, dtype=dtype, device=device)
        setattr(module, attr_name, materialized)
        return materialized

    # Process a single layer's weights
    def _process_layer(layer_idx: int, weight_names: list, file_handles: list):
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        tp_size = tp_info.size
        expert_weights: Dict[int, Dict[str, str]] = {}
        direct_weights = []

        for name in weight_names:
            # process routed expert weights specifically because they need to be stacked into a single tensor per layer
            if ".mlp.experts." in name:
                # MoE expert weights
                parts = name.split(".")
                expert_idx = int(parts[5])
                proj_type = parts[6].replace(".weight", "")
                expert_weights.setdefault(expert_idx, {})[proj_type] = name
            else:
                direct_weights.append(name)

        # Direct weights
        for name in direct_weights:
            tensor = _load_tensor(name, file_handles)
            _set_weight(name, tensor)
            del tensor

        # MoE expert weights: merge gate_proj + up_proj → gate_up_proj
        if expert_weights:
            num_experts = len(expert_weights)
            gate_up_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"
            down_key = f"model.layers.{layer_idx}.mlp.experts.down_proj"
            gate_up_tensor = None
            down_tensor = None

            for idx in range(num_experts):
                if idx not in expert_weights:
                    continue
                e = expert_weights[idx]
                if "gate_proj" in e and "up_proj" in e:
                    gate = _load_tensor(e["gate_proj"], file_handles)
                    up = _load_tensor(e["up_proj"], file_handles)
                    if tp_size > 1:
                        gate = gate.chunk(tp_size, dim=0)[tp_rank].contiguous()
                        up = up.chunk(tp_size, dim=0)[tp_rank].contiguous()
                    if gate_up_tensor is None:
                        gate_up_tensor = _get_or_init_tensor(
                            gate_up_key,
                            (num_experts, gate.shape[0] + up.shape[0], gate.shape[1]),
                            gate.dtype,
                        )
                    assert gate_up_tensor is not None
                    gate_up_tensor[idx, : gate.shape[0]].copy_(gate)
                    gate_up_tensor[idx, gate.shape[0] :].copy_(up)
                    del gate, up

                if "down_proj" in e:
                    down = _load_tensor(e["down_proj"], file_handles)
                    if tp_size > 1:
                        down = down.chunk(tp_size, dim=1)[tp_rank].contiguous()
                    if down_tensor is None:
                        down_tensor = _get_or_init_tensor(
                            down_key,
                            (num_experts, down.shape[0], down.shape[1]),
                            down.dtype,
                        )
                    assert down_tensor is not None
                    down_tensor[idx].copy_(down)
                    del down

            if gate_up_tensor is not None:
                torch.cuda.empty_cache()

            if down_tensor is not None:
                torch.cuda.empty_cache()

    # Open all files
    file_handles = [
        safetensors.safe_open(file, framework="pt", device="cpu") for file in files
    ]

    try:
        tp_info = get_tp_info()
        disable_tqdm = tp_info.size > 1 and tp_info.rank != 0

        with tqdm(total=num_layers + 1, desc="Loading layers to GPU", disable=disable_tqdm, unit="layer") as pbar:
            for layer_idx in sorted(layer_weights.keys()):
                _process_layer(layer_idx, layer_weights[layer_idx], file_handles)
                pbar.update(1)

            for name in global_weights:
                tensor = _load_tensor(name, file_handles)
                _set_weight(name, tensor)
                del tensor
            pbar.update(1)

        gc.collect()
    finally:
        del file_handles

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
