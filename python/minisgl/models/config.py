from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None
    is_neox: bool = True


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]

    # ===== GLM4.7 / DeepSeek V2 MLA related =====
    # Multi-Latent Attention (MLA) configuration
    qk_lora_rank: Optional[int] = None       # Q low-rank compression dimension
    kv_lora_rank: Optional[int] = None       # KV low-rank compression dimension
    qk_nope_head_dim: int = 0                # QK head dimension without RoPE
    qk_rope_head_dim: int = 0                # QK head dimension with RoPE
    v_head_dim: int = 0                      # V head dimension

    # ===== GLM4.7 MoE related =====
    n_routed_experts: int = 0               # Number of routed experts
    n_shared_experts: int = 0                # Number of shared experts
    num_expert_group: int = 0               # Number of expert groups (for Grouped TopK)
    topk_group: int = 0                      # Number of experts per group (for Grouped TopK)
    routed_scaling_factor: float = 1.0       # Routing scaling factor
    use_mla_backend: bool = False           # Whether to use MLA kv cache backend (else MHA)
    partial_rotary_factor: float = 1.0
    attn_output_gate: bool = False
    layer_types: list[str] | None = None
    shared_expert_intermediate_size: int = 0
    linear_conv_kernel_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @property
    def use_mla(self) -> bool:
        """Whether to use Multi-Latent Attention"""
        return self.qk_lora_rank is not None and self.kv_lora_rank is not None

    @property
    def attn_head_dim(self) -> int:
        """The Q/K head size expected by MHA-style attention backends.

        GLM-4.7 Flash uses MLA-specific config fields and does not expose a meaningful
        ``head_dim`` in its HuggingFace config. When we fall back to standard MHA
        kernels, the backend still needs the real Q/K head size.
        """
        if self.use_mla and self.qk_nope_head_dim and self.qk_rope_head_dim:
            return self.qk_nope_head_dim + self.qk_rope_head_dim
        return self.head_dim

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        parent_config = config
        if (
            getattr(config, "text_config", None) is not None
            and hasattr(config.text_config, "num_hidden_layers")
        ):
            config = config.text_config

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(
            config, "tie_word_embeddings", getattr(parent_config, "tie_word_embeddings", False)
        )
        model_type = getattr(config, "model_type", "llama")
        n_routed_experts = getattr(config, "n_routed_experts", 0)
        num_experts = getattr(
            config,
            "num_local_experts",
            getattr(config, "num_experts", n_routed_experts),
        )
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        architectures = getattr(config, "architectures", None) or getattr(
            parent_config, "architectures", ["LlamaForCausalLM"]
        )
        architectures = [
            {
                "Qwen3_5ForConditionalGeneration": "Qwen3_5ForCausalLM",
                "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeForCausalLM",
            }.get(arch, arch)
            for arch in architectures
        ]

        # ===== GLM4.7 / DeepSeek V2 MLA related =====
        qk_lora_rank = getattr(config, "q_lora_rank", None)
        kv_lora_rank = getattr(config, "kv_lora_rank", None)
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)

        # Handle MLA head dimensions
        if qk_lora_rank is not None and qk_nope_head_dim == 0:
            # If MLA is used but qk_nope_head_dim not set, derive from config
            qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128)  # default value
        if qk_lora_rank is not None and qk_rope_head_dim == 0:
            qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)  # default value
        if qk_lora_rank is not None and v_head_dim == 0:
            v_head_dim = getattr(config, "v_head_dim", 128)  # default value

        # ===== GLM4.7 MoE related =====
        n_shared_experts = getattr(config, "n_shared_experts", 0)
        num_expert_group = getattr(config, "n_group", 0)
        topk_group = getattr(config, "topk_group", 0)
        routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        attn_output_gate = getattr(config, "attn_output_gate", False)
        layer_types = getattr(config, "layer_types", None)
        shared_expert_intermediate_size = getattr(config, "shared_expert_intermediate_size", 0)
        linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 0)
        linear_num_key_heads = getattr(config, "linear_num_key_heads", 0)
        linear_num_value_heads = getattr(config, "linear_num_value_heads", 0)
        linear_key_head_dim = getattr(config, "linear_key_head_dim", 0)
        linear_value_head_dim = getattr(config, "linear_value_head_dim", 0)

        # Auto-enable MLA backend if MLA dimensions are present
        # Can be overridden by environment variable DISABLE_MLA_BACKEND=1
        import os
        disable_mla = os.environ.get("DISABLE_MLA_BACKEND", "0") == "1"
        use_mla_backend = (qk_lora_rank is not None and kv_lora_rank is not None) and not disable_mla

        # Handle rope_theta / rope_scaling / rope_interleave
        # Try to get from attributes first, then fall back to config.to_dict()
        config_dict = config.to_dict()
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = config_dict.get("rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = getattr(config, "rope_parameters", None)
        if rope_scaling is None:
            rope_scaling = config_dict.get("rope_parameters", None)
        if isinstance(rope_scaling, dict):
            rope_scaling = dict(rope_scaling)

        # Extract rope_theta from rope_scaling if present (GLM-4 style)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = config_dict.get("rope_theta", 10000.0)

        rope_interleave = getattr(config, "rope_interleave", None)
        if rope_interleave is None:
            rope_interleave = config_dict.get("rope_interleave")
        if rope_interleave is None and isinstance(rope_scaling, dict):
            rope_interleave = rope_scaling.get("mrope_interleaved")
        if rope_interleave is None and model_type == "glm4_moe_lite":
            rope_interleave = True
        is_neox = not rope_interleave
        if model_type.startswith("qwen3"):
            # Qwen3.x keeps neox-style rotary layout even when MRoPE uses
            # interleaved multimodal sections.
            is_neox = True

        # GLM-4 puts rope_theta inside rope_scaling
        if rope_scaling is not None and isinstance(rope_scaling, dict) and "rope_theta" in rope_scaling:
            rope_theta = rope_scaling["rope_theta"]

        # Check if rope_scaling is a default value (added by transformers)
        if rope_scaling is not None and isinstance(rope_scaling, dict) and "rope_type" in rope_scaling:
            # Preserve meaningful default rope_parameters such as mrope metadata.
            useful_rope_keys = set(rope_scaling) - {"rope_type", "type"}
            if rope_scaling.get("rope_type") == "default" and not useful_rope_keys:
                rope_scaling = None
        if isinstance(rope_scaling, dict):
            partial_rotary_factor = rope_scaling.get("partial_rotary_factor", partial_rotary_factor)

        norm_topk_prob = getattr(config, "norm_topk_prob", None)
        if norm_topk_prob is None:
            norm_topk_prob = bool(num_experts_per_tok > 0 and model_type.startswith("qwen3"))
        else:
            norm_topk_prob = bool(norm_topk_prob)

        # For MLA models, use qk_rope_head_dim for RoPE; otherwise use head_dim
        rope_head_dim = qk_rope_head_dim if qk_lora_rank is not None else head_dim
        rotary_dim = (
            qk_rope_head_dim
            if qk_lora_rank is not None
            else int(round(head_dim * partial_rotary_factor))
        )

        intermediate_size = getattr(
            config,
            "intermediate_size",
            shared_expert_intermediate_size or moe_intermediate_size,
        )

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=rope_head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
                is_neox=is_neox,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            # ===== GLM4.7 / MLA related =====
            qk_lora_rank=qk_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            # ===== GLM4.7 MoE related =====
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            use_mla_backend=use_mla_backend,
            partial_rotary_factor=partial_rotary_factor,
            attn_output_gate=attn_output_gate,
            layer_types=layer_types,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
        )
