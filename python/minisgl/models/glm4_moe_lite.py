from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import (
    DistributedCommunicator,
    get_local_expert_range,
    get_moe_tp_info,
    get_tp_info,
    get_world_info,
)
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
    get_rope,
    silu_and_mul,
)
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class Glm4MoeLiteAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp_info = get_tp_info()
        self.layer_id = layer_id
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.num_qo_heads = div_even(config.num_qo_heads, tp_info.size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_info.size)

        self.q_a_proj = LinearReplicated(config.hidden_size, config.qk_lora_rank, has_bias=False)
        self.q_a_layernorm = RMSNorm(config.qk_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = LinearColParallelMerged(
            config.qk_lora_rank,
            [config.num_qo_heads * self.qk_head_dim],
            has_bias=False,
        )

        self.kv_a_proj_with_mqa = LinearReplicated(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            has_bias=False,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = LinearColParallelMerged(
            config.kv_lora_rank,
            [config.num_kv_heads * (config.qk_nope_head_dim + config.v_head_dim)],
            has_bias=False,
        )

        self.rotary = get_rope(
            head_dim=config.qk_rope_head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=tuple(config.rotary_config.scaling.items())
            if config.rotary_config.scaling
            else None,
            is_neox=config.rotary_config.is_neox,
        )
        self.o_proj = LinearOProj(
            config.num_qo_heads * config.v_head_dim,
            config.hidden_size,
            has_bias=False,
        )
        self._mla_proj_k = None
        self._mla_proj_v = None

    def _init_mla_proj(self) -> None:
        if self._mla_proj_k is not None and self._mla_proj_v is not None:
            return
        weight = self.kv_b_proj.weight.view(
            self.num_kv_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_k, w_v = weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self._mla_proj_k = w_k.contiguous()
        self._mla_proj_v = w_v.transpose(1, 2).contiguous()

    @nvtx_annotate("MLA")
    def _forward_mla(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        num_tokens = hidden_states.shape[0]
        self._init_mla_proj()
        assert self._mla_proj_k is not None and self._mla_proj_v is not None
        assert self.num_qo_heads == self.num_kv_heads, "MLA path expects local Q/KV heads to match"

        q_states = self.q_a_proj.forward(hidden_states)
        self.q_a_layernorm.forward_inplace(q_states)
        q_states = self.q_b_proj.forward(q_states)
        q_states = q_states.view(num_tokens, self.num_qo_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        kv_states = self.kv_a_proj_with_mqa.forward(hidden_states)
        ckv, k_rope = torch.split(
            kv_states,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        self.kv_a_layernorm.forward_inplace(ckv)

        q_nope = torch.bmm(q_nope.transpose(0, 1), self._mla_proj_k).transpose(0, 1)

        k_rope = k_rope.view(num_tokens, 1, self.qk_rope_head_dim)
        q_rope, k_rope = self.rotary.forward(positions, q_rope, k_rope)

        q_states = torch.cat((q_nope, q_rope), dim=-1).reshape(num_tokens, -1)
        latent_k = torch.cat((ckv, k_rope.squeeze(1)), dim=-1)

        attn_output = ctx.attn_backend.forward(
            q_states,
            latent_k,
            None,
            self.layer_id,
            ctx.batch,
        )
        attn_output = attn_output.view(num_tokens, self.num_qo_heads, self.kv_lora_rank)
        attn_output = torch.bmm(attn_output.transpose(0, 1), self._mla_proj_v).transpose(0, 1)
        return self.o_proj.forward(attn_output.reshape(num_tokens, -1).contiguous())

    @nvtx_annotate("MHA")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        num_tokens = hidden_states.shape[0]
        positions = ctx.batch.positions
        if type(ctx.attn_backend).__name__ == "MLABackend":
            return self._forward_mla(hidden_states, positions)

        q_states = self.q_a_proj.forward(hidden_states)
        self.q_a_layernorm.forward_inplace(q_states)
        q_states = self.q_b_proj.forward(q_states)
        q_states = q_states.view(num_tokens, self.num_qo_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        kv_states = self.kv_a_proj_with_mqa.forward(hidden_states)
        k_latent, k_rope = torch.split(
            kv_states,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        self.kv_a_layernorm.forward_inplace(k_latent)
        kv_proj = self.kv_b_proj.forward(k_latent)
        kv_proj = kv_proj.view(num_tokens, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = kv_proj.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rope = k_rope.view(num_tokens, 1, self.qk_rope_head_dim)
        q_rope, k_rope = self.rotary.forward(positions, q_rope, k_rope)
        k_rope = k_rope.expand(*k_nope.shape[:-1], -1)

        query_states = torch.cat((q_nope, q_rope), dim=-1)
        key_states = torch.cat((k_nope, k_rope), dim=-1)
        key_states = key_states.reshape(num_tokens, -1)
        value_states = value_states.reshape(num_tokens, -1)
        attn_output = ctx.attn_backend.forward(
            query_states,
            key_states,
            value_states,
            self.layer_id,
            ctx.batch,
        )
        return self.o_proj.forward(attn_output.reshape(num_tokens, -1).contiguous())


class Glm4MoeLiteMLP(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size,
            [intermediate_size, intermediate_size],
            has_bias=False,
        )
        self.down_proj = LinearRowParallel(
            intermediate_size,
            hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Glm4MoeLiteTopkRouter(BaseOP):
    def __init__(self, hidden_size: int, n_routed_experts: int):
        self.weight = torch.empty(n_routed_experts, hidden_size)
        self.e_score_correction_bias = torch.empty(n_routed_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(torch.float32), self.weight.to(torch.float32))


class Glm4MoeLiteExperts(BaseOP):
    def __init__(self, config: ModelConfig):
        tp_info = get_tp_info()
        moe_tp_info = get_moe_tp_info(tp_info)
        intermediate_size = div_even(config.moe_intermediate_size, moe_tp_info.size)
        self.num_experts = config.n_routed_experts
        self.local_expert_start, local_expert_end = get_local_expert_range(self.num_experts)
        self.num_local_experts = local_expert_end - self.local_expert_start
        self.gate_up_proj = torch.empty(
            self.num_local_experts,
            intermediate_size * 2,
            config.hidden_size,
        )
        self.down_proj = torch.empty(
            self.num_local_experts,
            config.hidden_size,
            intermediate_size,
        )
        self._world_size = get_world_info().size
        self._comm = DistributedCommunicator(kind="world")

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        correction_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        output = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            activation="silu",
            apply_router_weight_on_input=False,
            use_grouped_topk=True,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=correction_bias,
            local_expert_start=self.local_expert_start,
            num_global_experts=self.num_experts,
            num_dispatch_experts=self.num_local_experts,
        )
        if self._world_size > 1:
            output = self._comm.all_reduce(output)
        return output


class Glm4MoeLiteSparseMoeBlock(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate = Glm4MoeLiteTopkRouter(config.hidden_size, config.n_routed_experts)
        self.experts = Glm4MoeLiteExperts(config)
        self.shared_experts = (
            Glm4MoeLiteMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
            )
            if config.n_shared_experts > 0
            else None
        )
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.num_expert_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate.forward(hidden_states)
        output = self.experts.forward(
            hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            correction_bias=self.gate.e_score_correction_bias,
        )
        if self.shared_experts is not None:
            output = output + self.shared_experts.forward(hidden_states)
        return output


class Glm4MoeLiteDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Glm4MoeLiteAttention(config, layer_id)
        self.mlp = (
            Glm4MoeLiteMLP(config.hidden_size, config.intermediate_size)
            if layer_id == 0
            else Glm4MoeLiteSparseMoeBlock(config)
        )
        self.input_layernorm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Glm4MoeLiteModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Glm4MoeLiteDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Glm4MoeLiteForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Glm4MoeLiteModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Glm4MoeLiteForCausalLM"]
