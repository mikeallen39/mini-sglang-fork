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
        self.q_lora_rank = config.qk_lora_rank

        # Total qk head dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # TP parallel
        self.num_heads = config.num_qo_heads
        self.num_qo_heads = div_even(config.num_qo_heads, tp_info.size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_info.size)

        # Determine if we use MLA backend or standard attention
        # This allows switching between implementations for debugging
        self.use_mla_backend = config.use_mla_backend

        # Q path: hidden_size -> q_lora_rank -> q_a_layernorm -> num_qo_heads * qk_head_dim
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

        # MLA absorbed weights (initialized lazily on first forward)
        self.w_kc = None
        self.w_vc = None
        self._mla_weights_initialized = False

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

    def init_mla_weights(self, device: torch.device):
        """
        Initialize MLA absorbed weights from kv_b_proj.

        kv_b_proj.weight shape: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]

        kv_b_proj computes: y = x @ W^T
        where x: [batch, kv_lora_rank], W: [(qk_nope + v), kv_lora_rank]
        Result: [batch, (qk_nope + v)]

        For MLA backward projection (qk_nope -> kv_lora_rank):
        We need: y = x @ W
        where x: [batch, qk_nope], W: [qk_nope, kv_lora_rank]
        Result: [batch, kv_lora_rank]

        So w_kc should be the K part of kv_b_proj.weight (no transpose needed for bmm)
        """
        if self.kv_b_proj.weight is not None:
            weight = self.kv_b_proj.weight  # [num_heads * (qk_nope + v), kv_lora_rank]

            # Reshape to [num_heads, qk_nope + v, kv_lora_rank]
            weight = weight.view(
                self.num_kv_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank
            )

            # Split into K and V parts
            # w_k: [num_heads, qk_nope, kv_lora_rank]
            # w_v: [num_heads, v, kv_lora_rank]
            w_k, w_v = weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=1
            )

            # w_kc for backward projection: qk_nope -> kv_lora_rank
            # Shape: [num_heads, qk_nope_head_dim, kv_lora_rank]
            self.w_kc = w_k.to(device)

            # w_vc for forward projection: kv_lora_rank -> v
            # Transpose to [num_heads, kv_lora_rank, v_head_dim] for bmm
            self.w_vc = w_v.transpose(1, 2).contiguous().to(device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MLA Forward - supports both MLA backend and standard attention

        Two modes:
        1. MLA Backend (use_mla_backend=True):
           - Use w_kc/w_vc for weight absorption
           - Pass compressed latent to MLA backend
           - MLA backend handles RoPE internally

        2. Standard Attention (use_mla_backend=False):
           - Use kv_b_proj to expand latent to full K and V
           - Apply RoPE manually
           - Pass to standard attention backend
        """
        if self.use_mla_backend:
            return self._forward_mla_backend(hidden_states)
        else:
            return self._forward_standard(hidden_states)

    def _forward_mla_backend(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward with MLA backend (weight absorption)"""
        num_tokens = hidden_states.shape[0]
        ctx = get_global_ctx()
        positions = ctx.batch.positions

        # Initialize MLA weights on first forward
        if not self._mla_weights_initialized:
            self.init_mla_weights(hidden_states.device)
            self._mla_weights_initialized = True

        # ===== Q processing =====
        q_a = self.q_a_proj.forward(hidden_states)
        self.q_a_layernorm.forward_inplace(q_a)
        q = self.q_b_proj.forward(q_a)

        # Reshape to [num_tokens, num_qo_heads, qk_head_dim]
        q = q.view(num_tokens, self.num_qo_heads, self.qk_head_dim)
        # HF GLM-4.7 Lite splits Q as [nope, rope].
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        # ===== Q_nope projection to kv_lora_rank =====
        if self.w_kc is not None:
            q_nope_out = torch.bmm(
                q_nope.transpose(0, 1),
                self.w_kc,
            ).transpose(0, 1)
        else:
            q_nope_out = q_nope

        # ===== KV processing =====
        kv_a = self.kv_a_proj_with_mqa.forward(hidden_states)
        c_kv = kv_a[:, : self.kv_lora_rank]
        k_pe = kv_a[:, self.kv_lora_rank:]
        self.kv_a_layernorm.forward_inplace(c_kv)

        # ===== Prepare inputs for MLA backend =====
        q_input = torch.cat([q_nope_out, q_pe], dim=-1)
        q_flat = q_input.reshape(num_tokens, self.num_qo_heads * (self.kv_lora_rank + self.qk_rope_head_dim))
        k_flat = torch.cat([c_kv, k_pe], dim=-1)
        v_flat = None

        # ===== MLA Attention =====
        attn_output = ctx.attn_backend.forward(q_flat, k_flat, v_flat, self.layer_id, ctx.batch)

        # Reshape and project
        attn_output = attn_output.view(num_tokens, self.num_qo_heads, self.kv_lora_rank)
        if self.w_vc is not None:
            attn_output = torch.bmm(
                attn_output.transpose(0, 1),
                self.w_vc,
            ).transpose(0, 1)

        attn_output = attn_output.reshape(num_tokens, self.num_qo_heads * self.v_head_dim)
        output = self.o_proj(attn_output)
        return output

    def _forward_standard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward with standard attention (no weight absorption)"""
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

        # Gate (router)
        self.gate = LinearReplicated(
            input_size=config.hidden_size,
            output_size=num_routed_experts,
            has_bias=False,
        )

        # e_score_correction_bias for routing score correction
        self.gate.e_score_correction_bias = torch.empty(num_routed_experts, dtype=torch.float32)

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
        residuals = hidden_states
        orig_shape = hidden_states.shape

        # Routing
        router_logits = self.gate(hidden_states)

        # Route tokens to experts (using MoELayer which handles grouped topk)
        routed_output = self.experts(
            hidden_states,
            router_logits,
            correction_bias=self.gate.e_score_correction_bias,
        )

        # Shared experts computation
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
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decoder Layer Forward - following transformers logic

        Reference: transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:517-548
        """
        # Pre-attention norm
        if residual is None:
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
        residual = x  # Return as residual for interface consistency

        return x, residual


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
            x, residual = layer.forward(x, residual)
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
    "Glm4MoeLiteAttention",
    "Glm4MoeLiteMLP",
    "Glm4MoeLiteSparseMoeBlock",
    "Glm4MoeLiteDecoderLayer",
    "Glm4MoeLiteModel",
]
