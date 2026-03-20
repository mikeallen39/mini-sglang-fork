"""
GLM4.7 Lite Model Implementation

Reference:
- transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import math
import torch.nn as nn
import torch
from minisgl.core import get_global_ctx
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
    silu_and_mul,
)
from minisgl.models.base import BaseLLMModel
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

if TYPE_CHECKING:
    from minisgl.models import ModelConfig, RotaryConfig


class Glm4MoeLiteAttention(BaseOP):
    """
    Multi-Latent Attention (MLA) for GLM4.7

    Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:229-347
    """

    def __init__(
        self,
        layer_id: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        tp_info = get_tp_info()

        # MLA dimension calculation
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_lora_rank = config.qk_lora_rank

        # Total qk head dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # TP parallel
        self.num_heads = config.num_qo_heads
        self.num_qo_heads = div_even(config.num_qo_heads, tp_info.size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_info.size)

        # Q path: hidden_size -> qk_lora_rank -> q_a_layernorm -> num_qo_heads * qk_head_dim
        self.q_a_proj = LinearReplicated(
            input_size=config.hidden_size,
            output_size=config.qk_lora_rank,
            has_bias=False,
        )
        self.q_a_layernorm = RMSNorm(
            size=config.qk_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.q_b_proj = LinearColParallelMerged(
            input_size=config.qk_lora_rank,
            output_sizes=[config.num_qo_heads * self.qk_head_dim],
            has_bias=False,
        )

        # KV path: hidden_size -> kv_lora_rank + qk_rope_head_dim (latent representation)
        self.kv_a_proj_with_mqa = LinearReplicated(
            input_size=config.hidden_size,
            output_size=config.kv_lora_rank + self.qk_rope_head_dim,
            has_bias=False,
        )
        self.kv_a_layernorm = RMSNorm(
            size=config.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        # kv_b_proj: kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
        kv_output_size = config.num_kv_heads * (self.qk_nope_head_dim + self.v_head_dim)
        self.kv_b_proj = LinearColParallelMerged(
            input_size=config.kv_lora_rank,
            output_sizes=[kv_output_size],
            has_bias=False,
        )

        # RoPE
        from minisgl.layers import get_rope
        self.rotary = get_rope(
            head_dim=self.qk_rope_head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=tuple(config.rotary_config.scaling.items())
            if config.rotary_config.scaling
            else None,
            is_neox=config.rotary_config.is_neox,
        )

        # O projection
        self.o_proj = LinearOProj(
            input_size=config.num_qo_heads * self.v_head_dim,
            output_size=config.hidden_size,
            has_bias=False,
        )

        # Scaling factor for attention
        self.scaling = 1.0 / math.sqrt(self.qk_head_dim)

    def __repr__(self):
        """Print attention layer structure."""
        lines = [f"{self.__class__.__name__}(layer_id={self.layer_id})"]
        for name, child in self.named_children():
            lines.append(f"  ({name}): {child.__class__.__name__}")
        lines.append(f"  num_qo_heads={self.num_qo_heads}, num_kv_heads={self.num_kv_heads}")
        lines.append(f"  qk_head_dim={self.qk_head_dim}, kv_lora_rank={self.kv_lora_rank}")
        lines.append(")")
        return "\n".join(lines)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        ctx = get_global_ctx()
        positions = ctx.batch.positions

        # ===== Q processing =====
        q_a = self.q_a_proj.forward(hidden_states)
        self.q_a_layernorm.forward_inplace(q_a)
        q_states = self.q_b_proj.forward(q_a)

        # Reshape to [num_tokens, num_qo_heads, qk_head_dim]
        q_states = q_states.view(num_tokens, self.num_qo_heads, self.qk_head_dim)
        # HF GLM-4.7 Lite splits Q as [nope, rope].
        q_pass, q_rot = torch.split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        # ===== KV processing =====
        compressed_kv = self.kv_a_proj_with_mqa.forward(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        self.kv_a_layernorm.forward_inplace(k_pass)
        k_pass = self.kv_b_proj.forward(k_pass)

        # Reshape to [num_tokens, num_kv_heads, qk_nope_head_dim + v_head_dim]
        key_shape = (num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
        kv_proj = k_pass.view(key_shape)

        # Split into k_pass (k_nope) and value_states
        k_pass, value_states = kv_proj.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Reshape k_rot to [num_tokens, 1, qk_rope_head_dim]
        k_rot = k_rot.view(num_tokens, 1, self.qk_rope_head_dim)

        # ===== Apply RoPE to q_rot and k_rot =====
        q_rot, k_rot = self.rotary.forward(positions, q_rot, k_rot)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)


        # GLM-4 attention kernels consume [nope, rope] after rotary is applied.
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        # ===== Attention =====
        attn_output = ctx.attn_backend.forward(query_states, key_states, value_states, self.layer_id, ctx.batch)

        # Reshape and project output
        attn_output = attn_output.reshape(num_tokens, -1).contiguous()
        output = self.o_proj(attn_output)
        return output


class Glm4MoeLiteMLP(BaseOP):
    """
    Standard MLP for GLM4 (used for shared experts or Dense layer)
    Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:350-363
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # TP MLP follows col-parallel gate/up + row-parallel down projection.
        self.gate_proj = LinearColParallelMerged(
            input_size=hidden_size,
            output_sizes=[intermediate_size],
            has_bias=False,
        )
        self.up_proj = LinearColParallelMerged(
            input_size=hidden_size,
            output_sizes=[intermediate_size],
            has_bias=False,
        )
        self.down_proj = LinearRowParallel(
            input_size=intermediate_size,
            output_size=hidden_size,
            has_bias=False,
        )

    def __repr__(self):
        """Print MLP layer structure."""
        lines = [f"{self.__class__.__name__}("]
        for name, child in self.named_children():
            lines.append(f"  ({name}): {child.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MLP Forward - following transformers logic: down_proj(act_fn(gate_proj(x)) * up_proj(x))

        Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:361-363
        """
        gate = self.gate_proj.forward(hidden_states)
        up = self.up_proj.forward(hidden_states)
        # Apply activation to gate, then multiply with up, then down_proj
        down = self.down_proj.forward(torch.nn.functional.silu(gate) * up)
        return down


class Glm4MoeLiteTopkRouter(BaseOP):
    """
    TopK Router for GLM4 MoE - contains gate weight and e_score_correction_bias

    Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:366-383
    """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts

        # Gate weight for computing router logits
        self.weight = torch.empty(
            n_routed_experts,
            hidden_size,
            dtype=torch.float32,
        )

        # e_score_correction_bias for routing score correction
        # Registered as buffer (not parameter, but saved in state_dict)
        self.e_score_correction_bias = torch.empty(
            n_routed_experts,
            dtype=torch.float32,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_size={self.hidden_size}, n_routed_experts={self.n_routed_experts})"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute router logits from hidden states.

        Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:380-383
        """
        # Reshape hidden_states to [num_tokens, hidden_size]
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # Compute router logits using float32 for precision
        router_logits = torch.nn.functional.linear(
            hidden_states.to(torch.float32),
            self.weight.to(torch.float32),
        )
        return router_logits


class Glm4MoeLiteSparseMoeBlock(BaseOP):
    """
    GLM4 MoE layer (shared experts + routed experts)

    Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:447-500
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        from minisgl.layers import MoELayer

        # Number of routed experts
        num_routed_experts = config.n_routed_experts

        # Gate (router) - use custom TopkRouter class
        self.gate = Glm4MoeLiteTopkRouter(
            hidden_size=config.hidden_size,
            n_routed_experts=num_routed_experts,
        )

        # Determine if we should use grouped topk
        use_grouped_topk = config.num_expert_group > 0 and config.topk_group > 0

        # Routed Experts
        self.experts = MoELayer(
            num_experts=num_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=config.num_expert_group if use_grouped_topk else 0,
            topk_group=config.topk_group if use_grouped_topk else 0,
            routed_scaling_factor=config.routed_scaling_factor,
            num_fused_shared_experts=0,
        )

        # Shared Experts
        if config.n_shared_experts > 0:
            self.shared_experts = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            )
        else:
            self.shared_experts = None

        self.n_routed_experts = num_routed_experts
        self.n_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def __repr__(self):
        """Print MoE block structure."""
        lines = [f"{self.__class__.__name__}(layer_id={self.layer_id})"]
        for name, child in self.named_children():
            lines.append(f"  ({name}): {child.__class__.__name__}")
        if hasattr(self.config, "n_routed_experts"):
            lines.append(f"  n_routed_experts={self.config.n_routed_experts}")
        if hasattr(self.config, "n_shared_experts"):
            lines.append(f"  n_shared_experts={self.config.n_shared_experts}")
        lines.append(")")
        return "\n".join(lines)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MoE Forward - following transformers logic

        Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:492-500
        """
        # Routing - gate produces logits for expert selection
        router_logits = self.gate(hidden_states)

        # Route tokens to experts (using MoELayer which handles grouped topk)
        routed_output = self.experts(
            hidden_states,
            router_logits,
            correction_bias=self.gate.e_score_correction_bias,
        )

        # Shared experts computation - always add shared experts output
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            output = routed_output + shared_output
        else:
            output = routed_output

        return output


class Glm4MoeLiteDecoderLayer(BaseOP):
    """
    GLM4 decoder layer

    Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:503-548
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()

        self.config = config
        self.layer_id = layer_id

        # Attention
        self.self_attn = Glm4MoeLiteAttention(layer_id, config)

        # MoE / MLP
        if layer_id == 0:
            # Layer 0 is always a dense MLP
            self.mlp = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
        else:
            self.mlp = Glm4MoeLiteSparseMoeBlock(config, layer_id)

        # Layer Norms
        self.input_layernorm = RMSNorm(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    def __repr__(self):
        """Print decoder layer structure."""
        lines = [f"{self.__class__.__name__}(layer_id={self.layer_id})"]
        for name, child in self.named_children():
            lines.append(f"  ({name}): {child.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Decoder Layer Forward - following transformers logic

        Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:517-548
        """
        # Pre-attention norm
        residual = x
        x = self.input_layernorm.forward(x)

        # Self Attention
        x = self.self_attn.forward(x)
        # Residual connection after attention
        x = x + residual

        # Post-attention norm
        residual = x
        x = self.post_attention_layernorm.forward(x)

        # MLP / MoE
        x = self.mlp.forward(x)
        # Residual connection after MLP
        x = x + residual

        return x


class Glm4MoeLiteModel(BaseOP):
    """
    GLM4 complete model

    Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:584-661
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        self.layers = OPList(
            [
                Glm4MoeLiteDecoderLayer(config, layer_id)
                for layer_id in range(config.num_layers)
            ]
        )

        self.norm = RMSNorm(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def __repr__(self):
        """Print model structure."""
        lines = [f"{self.__class__.__name__}("]
        for name, child in self.named_children():
            if name == "layers":
                lines.append(f"  ({name}): OPList(len={len(child.op_list)})")
            else:
                lines.append(f"  ({name}): {child.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Model Forward - following transformers logic

        Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:604-661
        """
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class Glm4MoeLiteForCausalLM(BaseLLMModel):
    """
    GLM4 For Causal LM top-level class

    Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:665-737
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = Glm4MoeLiteModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens
            if config.tie_word_embeddings
            else None,
        )

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits

    def __repr__(self):
        """Print model structure recursively."""
        lines = [f"{self.__class__.__name__}("]

        # Helper function to print module structure recursively
        def print_structure(module, indent=0, max_depth=3):
            if indent >= max_depth:
                return "  " * indent + "..."

            space = "  " * indent
            sub_lines = []

            # Print config if available
            if hasattr(module, "config") and module.config is not None:
                config_info = str(module.config)
                if len(config_info) > 100:
                    config_info = config_info[:100] + "..."
                sub_lines.append(f"{space}  config: {config_info}")

            # Print children
            for name, child in module.named_children():
                child_info = print_structure(child, indent + 1, max_depth)
                sub_lines.append(f"{space}  ({name}): {child.__class__.__name__}")
                if child_info and not child_info.startswith("  " * (indent + 2) + "..."):
                    for line in child_info.split("\n"):
                        if line.strip():
                            sub_lines.append(f"{space}    {line}")

            return "\n".join(sub_lines)

        # Print top-level structure
        for name, child in self.named_children():
            lines.append(f"  ({name}): {child.__class__.__name__}")
            child_structure = print_structure(child, 1)
            if child_structure:
                for line in child_structure.split("\n"):
                    if line.strip():
                        lines.append(f"    {line}")

        lines.append(")")
        return "\n".join(lines)


__all__ = [
    "Glm4MoeLiteForCausalLM",
]
