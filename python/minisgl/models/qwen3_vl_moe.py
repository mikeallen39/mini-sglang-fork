from __future__ import annotations

import itertools
import math
from typing import Any

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import ParallelLMHead
from minisgl.models.qwen3_5_moe import Qwen3_5Model
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

from .base import BaseLLMModel
from .config import ModelConfig


class Qwen3_5VLMoeModel(Qwen3_5Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        assert config.vision_config is not None
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_config = Qwen3_5MoeVisionConfig(**config.vision_config)
        self.visual = Qwen3_5MoeVisionModel(self.vision_config)
        self._materialize_visual_runtime_buffers()
        self.rope_deltas: torch.Tensor | None = None

    def _materialize_visual_runtime_buffers(self) -> None:
        # HF vision rotary uses a non-persistent buffer, so it is not restored from checkpoint.
        # When the model is initialized on meta, we need to rebuild it explicitly.
        self._ensure_visual_runtime_buffers(torch.device("cpu"))

    def _ensure_visual_runtime_buffers(self, device: torch.device) -> None:
        rotary = self.visual.rotary_pos_emb
        inv_freq = getattr(rotary, "inv_freq", None)
        if inv_freq is not None and not getattr(inv_freq, "is_meta", False) and inv_freq.device == device:
            return
        dim = rotary.dim
        theta = rotary.theta
        rebuilt = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        rotary.register_buffer("inv_freq", rebuilt, persistent=False)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        self._ensure_visual_runtime_buffers(pixel_values.device)
        self.visual = self.visual.to(device=pixel_values.device, dtype=self.visual.dtype)
        pixel_values = pixel_values.to(dtype=self.visual.dtype, device=pixel_values.device)
        image_grid_thw = image_grid_thw.to(device=pixel_values.device)
        outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return torch.split(outputs.pooler_output, split_sizes)

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.image_token_id is not None
        special_image_mask = input_ids == self.image_token_id
        num_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match, tokens={num_image_tokens}, "
                f"features={image_features.shape[0]}"
            )
        return special_image_mask

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        *,
        spatial_merge_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        llm_grid_t = int(grid_thw[0].item())
        llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
        llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size
        pos_t = torch.arange(llm_grid_t, device=device)
        pos_w = torch.arange(llm_grid_w, device=device) + start_position
        pos_h = torch.arange(llm_grid_h, device=device) + start_position
        pos_w = pos_w.repeat(llm_grid_h * llm_grid_t)
        pos_h = pos_h.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
        pos_t = pos_t.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
        return torch.stack([pos_t, pos_h, pos_w], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required for multimodal rope positions")
        spatial_merge_size = self.vision_config.spatial_merge_size
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_iter = iter(image_grid_thw)
        deltas = []
        for batch_idx, current_input_ids in enumerate(input_ids):
            token_types = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                mask = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[mask]
                token_types = token_types[mask]
            groups = []
            for key, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
                group = list(group)
                groups.append((key, group[0][0], group[-1][0] + 1))
            current_pos = 0
            pos_parts = []
            for modality_type, start_idx, end_idx in groups:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    pos_parts.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1)
                        + current_pos
                    )
                    current_pos += text_len
                elif modality_type == 1:
                    grid = next(image_iter)
                    pos_parts.append(
                        self.get_vision_position_ids(
                            current_pos,
                            grid,
                            spatial_merge_size=spatial_merge_size,
                            device=input_ids.device,
                        )
                    )
                    current_pos += max(int(grid[1].item()), int(grid[2].item())) // spatial_merge_size
                else:
                    raise NotImplementedError("Video multimodal requests are not supported yet.")
            llm_positions = torch.cat(pos_parts, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions
            deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        rope_deltas = torch.tensor(deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, rope_deltas

    def forward_multimodal(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens.forward(input_ids)
        image_embeds = torch.cat(self.get_image_features(pixel_values, image_grid_thw), dim=0)
        image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        positions, rope_deltas = self.get_rope_index(
            input_ids.unsqueeze(0),
            mm_token_type_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )
        batch = get_global_ctx().batch
        batch.mrope_positions = positions[:, 0].contiguous()
        if batch.reqs:
            batch.reqs[0].rope_delta = rope_deltas[0].to(device="cpu")
        self.rope_deltas = rope_deltas[:, 0]
        original_positions = batch.positions
        batch.positions = batch.mrope_positions
        try:
            return super().forward(inputs_embeds=inputs_embeds)
        finally:
            batch.positions = original_positions


class Qwen3_5VLMoeForConditionalGeneration(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5VLMoeModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    @property
    def supports_prefix_cache(self) -> bool:
        return False

    @property
    def supports_cuda_graph(self) -> bool:
        return False

    @property
    def supports_prefix_cache(self) -> bool:
        return False

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.pixel_values is None:
            return self.lm_head.forward(self.model.forward(batch.input_ids))
        if batch.is_decode:
            if batch.reqs[0].rope_delta is None:
                raise RuntimeError("Missing multimodal rope delta for decode step.")
            batch.mrope_positions = batch.text_positions.unsqueeze(0).expand(3, -1)
            batch.mrope_positions = batch.mrope_positions + batch.reqs[0].rope_delta.to(
                device=batch.input_ids.device, dtype=batch.text_positions.dtype
            ).view(1, 1)
            original_positions = batch.positions
            batch.positions = batch.mrope_positions
            try:
                output = self.model.forward(batch.input_ids)
            finally:
                batch.positions = original_positions
            return self.lm_head.forward(output)
        output = self.model.forward_multimodal(
            batch.input_ids,
            batch.pixel_values,
            batch.image_grid_thw,
            batch.mm_token_type_ids,
        )
        return self.lm_head.forward(output)


__all__ = ["Qwen3_5VLMoeForConditionalGeneration"]
