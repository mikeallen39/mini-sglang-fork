from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseCacheManager,
    BaseKVCache,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BaseCacheManager: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCache:
    # Check if model uses MLA (Multi-Latent Attention)
    if model_config.use_mla_backend:
        from .mla_pool import MLAKVCache

        return MLAKVCache(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=model_config.num_layers,
            kv_lora_rank=model_config.kv_lora_rank,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            device=device,
            dtype=dtype,
        )

    # Standard MHA
    from .mha_pool import MHAKVCache

    # For MLA models, use v_head_dim as the K/V dimension (head_dim is not meaningful)
    kv_dim = model_config.v_head_dim if hasattr(model_config, "v_head_dim") else model_config.head_dim
    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=kv_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache_manager(device: torch.device):
    from .naive_manager import NaiveCacheManager

    return NaiveCacheManager(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache_manager(device: torch.device):
    from .radix_manager import RadixCacheManager

    return RadixCacheManager(device=device)


def create_cache_manager(device: torch.device, type: str) -> BaseCacheManager:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache",
    "create_cache_manager",
    "BaseKVCache",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
    "SUPPORTED_CACHE_MANAGER",
]
