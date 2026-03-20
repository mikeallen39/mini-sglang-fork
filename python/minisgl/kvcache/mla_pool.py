"""
MLA (Multi-Latent Attention) KV Cache Implementation

MLA stores compressed latent representation instead of expanded multi-head format:
- ckv_cache (compressed KV): shape [..., kv_lora_rank]
- kpe_cache (key position embedding): shape [..., qk_rope_head_dim]
- Total KV cache dimension = kv_lora_rank + qk_rope_head_dim

FlashInfer MLA API separates these two parts:
- q_nope @ ckv_cache^T for the non-RoPE part
- q_pe @ kpe_cache^T for the RoPE part

Reference:
- sglang/python/sglang/srt/mem_cache/memory_pool.py (MLATokenToKVPool)
- flashinfer.mla.BatchMLAPagedAttentionWrapper
"""

from __future__ import annotations

import torch

from .base import BaseKVCache


class MLAKVCache(BaseKVCache):
    """
    KV Cache for Multi-Latent Attention (MLA) models like GLM-4.7, DeepSeek V2/V3.

    MLA uses a compressed latent representation for KV cache:
    - kv_lora_rank: compressed KV dimension (latent representation)
    - qk_rope_head_dim: RoPE position embedding dimension

    The KV cache is stored as two parts (for FlashInfer MLA API):
    - ckv_cache: [..., kv_lora_rank] - compressed KV (also used as V)
    - kpe_cache: [..., qk_rope_head_dim] - key position embedding
    """

    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        # MLA specific parameters
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ) -> None:
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._num_layers = num_layers
        self._device = device
        self._num_pages = num_pages
        self._page_size = page_size

        # Unified buffer for both ckv and kpe
        # Shape: [num_layers, num_pages * page_size, 1, kv_lora_rank + qk_rope_head_dim]
        self._kv_buffer = torch.empty(
            (num_layers, num_pages * page_size, 1, kv_lora_rank + qk_rope_head_dim),
            device=device,
            dtype=dtype,
        )

    @property
    def kv_lora_rank(self) -> int:
        return self._kv_lora_rank

    @property
    def qk_rope_head_dim(self) -> int:
        return self._qk_rope_head_dim

    @property
    def kv_cache_dim(self) -> int:
        return self._kv_lora_rank + self._qk_rope_head_dim

    def k_cache(self, layer_id: int) -> torch.Tensor:
        """
        Get full K cache for a layer (latent representation).
        Returns: [num_pages * page_size, 1, kv_lora_rank + qk_rope_head_dim]
        """
        return self._kv_buffer[layer_id]

    def v_cache(self, layer_id: int) -> torch.Tensor:
        """
        Get V cache for a layer (ckv part only).
        Returns: [num_pages * page_size, 1, kv_lora_rank]
        """
        return self._kv_buffer[layer_id][..., : self._kv_lora_rank]

    def ckv_cache(self, layer_id: int) -> torch.Tensor:
        """
        Get compressed KV cache for FlashInfer MLA.
        Returns: [num_pages * page_size, 1, kv_lora_rank]
        """
        return self._kv_buffer[layer_id][..., : self._kv_lora_rank]

    def kpe_cache(self, layer_id: int) -> torch.Tensor:
        """
        Get key position embedding cache for FlashInfer MLA.
        Returns: [num_pages * page_size, 1, qk_rope_head_dim]
        """
        return self._kv_buffer[layer_id][..., self._kv_lora_rank :]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        """
        Store K (latent representation) into cache.

        For MLA:
        - k should be the latent representation: [num_tokens, kv_lora_rank + qk_rope_head_dim]
          - k[..., :kv_lora_rank] = ckv (compressed KV)
          - k[..., kv_lora_rank:] = kpe (key position embedding)
        - v is ignored (V = ckv, extracted from K)

        Args:
            k: Latent tensor [num_tokens, kv_lora_rank + qk_rope_head_dim]
            v: Ignored (V is extracted from K)
            out_loc: Output locations [num_tokens]
            layer_id: Layer index
        """
        # Store the latent representation directly
        # k shape: [num_tokens, kv_cache_dim]
        self._kv_buffer[layer_id][out_loc] = k.unsqueeze(1)

    def store_latent(
        self,
        ckv: torch.Tensor,
        kpe: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        """
        Store ckv and kpe separately into cache.

        Args:
            ckv: Compressed KV [num_tokens, kv_lora_rank]
            kpe: Key position embedding [num_tokens, qk_rope_head_dim]
            out_loc: Output locations [num_tokens]
            layer_id: Layer index
        """
        # Concatenate and store
        latent = torch.cat([ckv, kpe], dim=-1)  # [num_tokens, kv_cache_dim]
        self._kv_buffer[layer_id][out_loc] = latent.unsqueeze(1)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
