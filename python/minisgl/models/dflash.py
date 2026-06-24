from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    RMSNorm,
    get_rope,
)

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class DFlashAttention(BaseOP):
    """Minimal loadable DFlash attention skeleton.

    The draft checkpoint stores Q/K/V as separate tensors, but the runtime fast path
    in mini-sglang naturally uses a fused QKV projection. We therefore expose a
    merged `qkv_proj` so streaming load can reuse the existing Q/K/V merge path.
    """

    def __init__(self, config: ModelConfig):
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=False,
        )
        self.o_proj = LinearRowParallel(
            config.num_qo_heads * config.head_dim,
            config.hidden_size,
            has_bias=False,
        )
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.num_qo_heads = config.num_qo_heads // get_tp_info().size
        self.num_kv_heads = config.num_kv_heads // get_tp_info().size
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.rotary = get_rope(
            head_dim=config.rotary_config.head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=tuple(config.rotary_config.scaling.items())
            if config.rotary_config.scaling
            else None,
            is_neox=config.rotary_config.is_neox,
        )
        # DFlash draft runtime currently uses a small fixed block. Prefer the pure
        # Torch RoPE path to avoid FlashInfer-specific constraints during early
        # speculative integration and unit-style sanity checks.
        self.rotary.apply_rope_with_cos_sin_cache_inplace = None

    def forward(self, positions: torch.Tensor, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        q_dim = self.num_qo_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(positions, q, k)

        q = q.view(batch_size, seq_len, self.num_qo_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if self.num_qo_heads != self.num_kv_heads:
            repeat = self.num_qo_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(q.dtype)
        out = torch.einsum("bhts,bshd->bthd", attn_probs, v)
        out = out.reshape(batch_size * seq_len, q_dim)
        return self.o_proj.forward(out)


class DFlashMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj.forward(F.silu(gate) * up)


class DFlashDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig):
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DFlashAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = DFlashMLP(config)

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
            x = self.input_layernorm.forward(x)
        else:
            merged = x + residual
            x = self.input_layernorm.forward(merged)
            residual = merged
        x = self.self_attn.forward(positions, x, batch_size, seq_len)
        merged = x + residual
        x = self.post_attention_layernorm.forward(merged)
        residual = merged
        x = self.mlp.forward(x)
        return x, residual


class DFlashDraftModel(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.layers = OPList([DFlashDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # The released Qwen3.6 DFlash checkpoint uses fc.weight with shape
        # [hidden_size, K * hidden_size]. We keep the local model simple and infer
        # K from the input dimension instead of relying on raw HF config objects here.
        self.num_context_features = 8
        self.mask_token_id = None
        self.fc = LinearReplicated(
            input_size=config.hidden_size * self.num_context_features,
            output_size=config.hidden_size,
            has_bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def supports_cuda_graph(self) -> bool:
        return False

    def forward_block(
        self,
        *,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        x = input_embeds
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(positions, x, batch_size, seq_len, residual)
        if residual is None:
            return self.norm.forward(x)
        return self.norm.forward(x + residual)

    def forward(self) -> torch.Tensor:
        raise NotImplementedError(
            "DFlash runtime speculative decoding is not integrated into the scheduler yet. "
            "Use draft_block_greedy() for the current minimal runtime path."
        )

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(self.fc.full_input_size)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}], got {tuple(target_hidden.shape)}."
            )
        projected = self.fc.forward(target_hidden)
        return self.hidden_norm.forward(projected)

    def draft_block_greedy(
        self,
        *,
        verified_id: torch.Tensor,
        draft_context: torch.Tensor,
        prefix_lens: torch.Tensor,
        target_embedding,
        target_lm_head,
        block_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if get_tp_info().size != 1:
            raise NotImplementedError("Minimal DFLASH draft runtime currently supports tp_size == 1 only.")
        if self.mask_token_id is None:
            raise RuntimeError("DFLASH mask_token_id is not initialized.")
        if verified_id.ndim != 1:
            raise ValueError(f"verified_id must have shape [bs], got {tuple(verified_id.shape)}")
        if draft_context.ndim != 2:
            raise ValueError(
                f"draft_context must have shape [bs, hidden_size], got {tuple(draft_context.shape)}"
            )
        if prefix_lens.ndim != 1 or prefix_lens.shape[0] != verified_id.shape[0]:
            raise ValueError(
                f"prefix_lens must have shape [bs], got {tuple(prefix_lens.shape)} "
                f"for bs={verified_id.shape[0]}"
            )

        bs = int(verified_id.shape[0])
        block = int(block_size or self.block_size)
        device = verified_id.device
        hidden_size = int(draft_context.shape[-1])
        if hidden_size != int(self.norm.weight.shape[0]):
            raise ValueError(
                f"draft_context hidden mismatch: expected {int(self.norm.weight.shape[0])}, got {hidden_size}"
            )

        block_ids = torch.full(
            (bs, block),
            int(self.mask_token_id),
            dtype=torch.long,
            device=device,
        )
        block_ids[:, 0].copy_(verified_id.to(torch.long))
        input_embeds = target_embedding.forward(block_ids.reshape(-1)).view(bs, block, -1)
        input_embeds = input_embeds + draft_context[:, None, :]

        pos_offsets = torch.arange(block, device=device, dtype=prefix_lens.dtype)
        positions = (prefix_lens[:, None] + pos_offsets[None, :]).reshape(-1)
        hidden = self.forward_block(
            input_embeds=input_embeds.reshape(bs * block, -1),
            positions=positions,
            batch_size=bs,
            seq_len=block,
        ).view(bs, block, -1)

        lm_weight_module = target_lm_head.tied_embedding or target_lm_head
        logits = F.linear(
            hidden[:, 1:, :].reshape(-1, hidden.shape[-1]),
            lm_weight_module.weight,
            target_lm_head.bias,
        )
        next_ids = torch.argmax(logits, dim=-1).view(bs, block - 1)
        draft_tokens = block_ids.clone()
        draft_tokens[:, 1:].copy_(next_ids)
        return draft_tokens, hidden


__all__ = ["DFlashDraftModel"]
