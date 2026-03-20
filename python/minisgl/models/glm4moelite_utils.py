"""
GLM4.7 Lite Weight Loading Utilities

Reference: sglang/python/sglang/srt/models/glm4_moe_lite.py:547-806

This module handles the weight mapping specific to GLM4.7 Lite models:
1. MLA weight fusion (q_a_proj + kv_a_proj_with_mqa)
2. Expert weight mapping
3. Shared expert weight fusion
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def glm4_merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Merge GLM4.7 Lite specific weights.

    This handles:
    1. Standard QKV merge: q_proj + k_proj + v_proj -> qkv_proj
    2. Standard MLP merge: gate_proj + up_proj -> gate_up_proj
    3. MLA weights:
       - q_a_proj: keep as is
       - kv_a_proj_with_mqa: keep as is (contains both k and v compressed)
       - q_b_proj: keep as is
       - kv_b_proj: keep as is
    4. Expert weights: expert_0.gate_proj + expert_0.up_proj -> expert_0.gate_up_proj

    Returns:
        Merged state dict with renamed keys
    """
    filtered_state_dict: Dict[str, torch.Tensor] = {}

    # First, handle MLA specific weights (q_a_proj, kv_a_proj_with_mqa, q_b_proj, kv_b_proj)
    # These need special handling and should not be merged with standard QKV

    # Collect MLA keys
    mla_keys = set()
    for key in list(state_dict.keys()):
        if ".q_a_proj" in key or ".kv_a_proj_with_mqa" in key:
            mla_keys.add(key.rsplit(".", 1)[0])
        elif ".q_b_proj" in key:
            mla_keys.add(key.rsplit(".", 1)[0])
        elif ".kv_b_proj" in key:
            mla_keys.add(key.rsplit(".", 1)[0])

    # Process each key
    processed_keys = set()
    for key in list(state_dict.keys()):
        if key in processed_keys:
            continue

        # Skip if this is part of MLA (handled separately)
        prefix = key.rsplit(".", 1)[0]
        if prefix in mla_keys:
            # MLA weights: copy as is
            # Transform: model.layers.X.self_attn.q_a_proj -> q_a_proj
            # The actual mapping happens when loading into model
            new_key = _transform_qlm4_key(key)
            filtered_state_dict[new_key] = state_dict[key]
            processed_keys.add(key)
            continue

        # Standard QKV merge
        if ".q_proj" in key:
            base_key = key.replace(".q_proj", "")
            q_proj = state_dict[key]
            k_proj_key = key.replace(".q_proj", ".k_proj")
            v_proj_key = key.replace(".q_proj", ".v_proj")

            if k_proj_key in state_dict and v_proj_key in state_dict:
                k_proj = state_dict[k_proj_key]
                v_proj = state_dict[v_proj_key]
                # Merge q, k, v
                new_key = key.replace(".q_proj", ".qkv_proj")
                filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
                processed_keys.add(key)
                processed_keys.add(k_proj_key)
                processed_keys.add(v_proj_key)
                continue
            else:
                # No k/v_proj, copy as is
                filtered_state_dict[key] = state_dict[key]
                processed_keys.add(key)
                continue

        # Standard MLP merge
        if ".gate_proj" in key:
            base_key = key.replace(".gate_proj", "")
            gate_proj = state_dict[key]
            up_proj_key = key.replace(".gate_proj", ".up_proj")

            if up_proj_key in state_dict:
                up_proj = state_dict[up_proj_key]
                new_key = key.replace(".gate_proj", ".gate_up_proj")
                filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
                processed_keys.add(key)
                processed_keys.add(up_proj_key)
                continue
            else:
                filtered_state_dict[key] = state_dict[key]
                processed_keys.add(key)
                continue

        # Expert weights: merge within each expert
        if ".mlp.experts." in key and ".gate_proj" in key:
            # Example: model.layers.0.mlp.experts.0.gate_proj
            base = key.rsplit(".gate_proj", 1)[0]
            gate_proj = state_dict[key]
            up_proj_key = base + ".up_proj"
            down_proj_key = base + ".down_proj"

            if up_proj_key in state_dict and down_proj_key in state_dict:
                up_proj = state_dict[up_proj_key]
                # Merge gate and up
                gate_up_key = base + ".gate_up_proj"
                filtered_state_dict[gate_up_key] = torch.cat([gate_proj, up_proj], dim=0)
                # Copy down_proj as is
                filtered_state_dict[down_proj_key] = state_dict[down_proj_key]
                processed_keys.add(key)
                processed_keys.add(up_proj_key)
                processed_keys.add(down_proj_key)
                continue

        # Skip already processed keys
        if key in processed_keys:
            continue

        # Default: copy as is
        filtered_state_dict[key] = state_dict[key]
        processed_keys.add(key)

    return filtered_state_dict


def _transform_qlm4_key(key: str) -> str:
    """
    Transform HuggingFace checkpoint key to internal model key.

    Examples:
    - model.layers.0.self_attn.q_a_proj.weight -> layers.0.self_attn.q_a_proj.weight
    - model.layers.0.self_attn.q_b_proj.weight -> layers.0.self_attn.q_b_proj.weight
    - model.layers.0.mlp.gate.up_proj.weight -> layers.0.mlp.gate_up_proj.weight
    - model.layers.0.mlp.experts.0.gate_up_proj.weight -> layers.0.mlp.experts.0.gate_up_proj.weight
    """
    # Remove "model." prefix
    if key.startswith("model."):
        key = key[6:]

    # Transform attention weights
    if ".self_attn.q_a_proj" in key:
        return key.replace(".q_a_proj", ".q_a_proj")
    if ".self_attn.kv_a_proj_with_mqa" in key:
        return key.replace(".kv_a_proj_with_mqa", ".kv_a_proj_with_mqa")
    if ".self_attn.q_b_proj" in key:
        return key.replace(".q_b_proj", ".q_b_proj")
    if ".self_attn.kv_b_proj" in key:
        return key.replace(".kv_b_proj", ".kv_b_proj")
    if ".self_attn.q_a_layernorm" in key:
        return key.replace(".q_a_layernorm", ".q_a_layernorm")
    if ".self_attn.kv_a_layernorm" in key:
        return key.replace(".kv_a_layernorm", ".kv_a_layernorm")
    if ".self_attn.o_proj" in key:
        return key.replace(".o_proj", ".o_proj")

    # Transform MoE gate (router) weights
    # GLM4 uses ".mlp.gate.weight" for router, not ".mlp.gate_proj"
    if ".mlp.gate." in key and ".gate_proj" not in key and ".gate_up_proj" not in key:
        return key

    # Transform MLP weights
    if ".mlp.gate_proj" in key:
        return key.replace(".gate_proj", ".gate_up_proj").replace(".up_proj", "")
    if ".mlp.down_proj" in key:
        return key

    # Transform expert weights
    if ".mlp.experts." in key and ".gate_proj" in key:
        return key.replace(".gate_proj", ".gate_up_proj").replace(".up_proj", "")
    if ".mlp.experts." in key and ".down_proj" in key:
        return key

    # Transform shared experts
    if ".mlp.shared_experts." in key:
        # For shared experts, we keep them separate
        return key

    # Transform layernorm
    if "input_layernorm" in key:
        return key.replace("input_layernorm", "input_layernorm")
    if "post_attention_layernorm" in key:
        return key.replace("post_attention_layernorm", "post_attention_layernorm")

    # Transform embedding and lm_head
    if "embed_tokens" in key:
        return key.replace("embed_tokens", "embed_tokens")
    if "lm_head" in key:
        return key.replace("lm_head", "lm_head")

    return key


def create_glm4_weight_mapping(config) -> Dict[str, str]:
    """
    Create weight mapping from HuggingFace checkpoint to internal model.

    Returns a dictionary mapping internal model keys to checkpoint keys.
    """
    mapping = {}

    num_layers = config.num_hidden_layers

    for layer_id in range(num_layers):
        # MLA Attention weights
        prefix = f"model.layers.{layer_id}.self_attn"

        # Q path
        mapping[f"model.layers.{layer_id}.self_attn.q_a_proj.weight"] = f"{prefix}.q_a_proj.weight"
        mapping[f"model.layers.{layer_id}.self_attn.q_a_proj.bias"] = f"{prefix}.q_a_proj.bias"

        # KV path
        mapping[f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"] = f"{prefix}.kv_a_proj_with_mqa.weight"
        mapping[f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.bias"] = f"{prefix}.kv_a_proj_with_mqa.bias"

        # Q and KV projection after layernorm
        mapping[f"model.layers.{layer_id}.self_attn.q_b_proj.weight"] = f"{prefix}.q_b_proj.weight"
        mapping[f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"] = f"{prefix}.kv_b_proj.weight"

        # Layernorms
        mapping[f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"] = f"{prefix}.q_a_layernorm.weight"
        mapping[f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"] = f"{prefix}.kv_a_layernorm.weight"

        # O projection
        mapping[f"model.layers.{layer_id}.self_attn.o_proj.weight"] = f"{prefix}.o_proj.weight"

    return mapping


def get_glm4_stacked_params_mapping() -> list:
    """
    Get the stacked params mapping for GLM4.

    This maps checkpoint weight names to internal model parameter names.

    Returns:
        List of tuples: (param_name, shard_name, shard_id)
    """
    return [
        # QKV (non-MLA)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        # MLP
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]


def get_glm4_expert_params_mapping(num_experts: int, num_fused_shared_experts: int = 0) -> list:
    """
    Get the expert params mapping for GLM4 MoE.

    This maps checkpoint expert weight names to internal model parameter names.

    Args:
        num_experts: Number of routed experts
        num_fused_shared_experts: Number of fused shared experts

    Returns:
        List of tuples: (param_name, weight_name, expert_id, shard_id)
    """
    total_experts = num_experts + num_fused_shared_experts
    mapping = []

    for expert_id in range(total_experts):
        # gate_up_proj: gate_proj -> 0, up_proj -> 1
        mapping.append((f"experts.{expert_id}.gate_up_proj", "gate_proj", expert_id, 0))
        mapping.append((f"experts.{expert_id}.gate_up_proj", "up_proj", expert_id, 1))
        # down_proj
        mapping.append((f"experts.{expert_id}.down_proj", "down_proj", expert_id, 0))

    return mapping


__all__ = [
    "glm4_merge_state_dict",
    "create_glm4_weight_mapping",
    "get_glm4_stacked_params_mapping",
    "get_glm4_expert_params_mapping",
]
