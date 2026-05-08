from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


def _apply_interleaved_rope(x: torch.Tensor, mrope_section: tuple[int, ...]) -> torch.Tensor:
    x_t = x[0].clone()
    if len(mrope_section) > 1:
        x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    if len(mrope_section) > 2:
        x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox: bool = True,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.is_neox = is_neox
        assert 0 < rotary_dim <= head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        self.apply_rope_with_cos_sin_cache_inplace = None
        if rotary_dim == head_size:
            assert self.head_size in [64, 128, 256, 512]
            from flashinfer import apply_rope_with_cos_sin_cache_inplace

            self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def _forward_partial(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1] // self.head_size, self.head_size)
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]

        cos_sin = self._cos_sin_cache[positions]
        half_dim = self.rotary_dim // 2
        cos = cos_sin[..., :half_dim].to(dtype=x.dtype, device=x.device)
        sin = cos_sin[..., half_dim:].to(dtype=x.dtype, device=x.device)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        if self.is_neox:
            x1 = x_rot[..., :half_dim]
            x2 = x_rot[..., half_dim:]
            x_rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        else:
            x_even = x_rot[..., ::2]
            x_odd = x_rot[..., 1::2]
            x_rot = torch.stack((x_even * cos - x_odd * sin, x_odd * cos + x_even * sin), dim=-1)
            x_rot = x_rot.flatten(start_dim=-2)

        return torch.cat((x_rot, x_pass), dim=-1).view(orig_shape)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.apply_rope_with_cos_sin_cache_inplace is not None:
            self.apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self._cos_sin_cache,
                is_neox=self.is_neox,
            )
            return query, key
        return self._forward_partial(positions, query), self._forward_partial(positions, key)


class MRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox: bool = True,
        *,
        mrope_section: tuple[int, ...] | None = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox=is_neox)
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

    def _get_cos_sin(self, positions: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self._cos_sin_cache[positions]
        half_dim = self.rotary_dim // 2
        cos = cos_sin[..., :half_dim].to(dtype=x.dtype, device=x.device)
        sin = cos_sin[..., half_dim:].to(dtype=x.dtype, device=x.device)
        if positions.ndim == 2:
            assert self.mrope_section is not None
            if self.mrope_interleaved:
                cos = _apply_interleaved_rope(cos, self.mrope_section)
                sin = _apply_interleaved_rope(sin, self.mrope_section)
            else:
                cos = torch.cat(
                    [chunk[i] for i, chunk in enumerate(cos.split(self.mrope_section, dim=-1))],
                    dim=-1,
                )
                sin = torch.cat(
                    [chunk[i] for i, chunk in enumerate(sin.split(self.mrope_section, dim=-1))],
                    dim=-1,
                )
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def _forward_partial(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1] // self.head_size, self.head_size)
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]

        cos, sin = self._get_cos_sin(positions, x)
        half_dim = self.rotary_dim // 2

        if self.is_neox:
            x1 = x_rot[..., :half_dim]
            x2 = x_rot[..., half_dim:]
            x_rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        else:
            x_even = x_rot[..., ::2]
            x_odd = x_rot[..., 1::2]
            x_rot = torch.stack((x_even * cos - x_odd * sin, x_odd * cos + x_even * sin), dim=-1)
            x_rot = x_rot.flatten(start_dim=-2)

        return torch.cat((x_rot, x_pass), dim=-1).view(orig_shape)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1 and self.apply_rope_with_cos_sin_cache_inplace is not None:
            return super().forward(positions, query, key)
        return self._forward_partial(positions, query), self._forward_partial(positions, key)


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
    is_neox: bool = True,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base, is_neox=is_neox)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            if "mrope_section" in rope_scaling:
                return MRotaryEmbedding(
                    head_dim,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox=is_neox,
                    mrope_section=tuple(rope_scaling["mrope_section"]),
                    mrope_interleaved=bool(rope_scaling.get("mrope_interleaved", False)),
                )
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, is_neox=is_neox)
        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                is_neox=is_neox,
                post_process=post_process,
            )

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
    is_neox: bool = True,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, is_neox)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, is_neox)


__all__ = ["get_rope", "RotaryEmbedding", "MRotaryEmbedding", "set_rope_device"]
