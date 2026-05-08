from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import torch as _torch


def silu_and_mul(x: _torch.Tensor, out: _torch.Tensor | None = None):
    gate, up = x.chunk(2, dim=-1)
    y = F.silu(gate) * up
    if out is not None:
        out.copy_(y)
        return out
    return y


def gelu_and_mul(x: _torch.Tensor, out: _torch.Tensor | None = None):
    gate, up = x.chunk(2, dim=-1)
    y = F.gelu(gate) * up
    if out is not None:
        out.copy_(y)
        return out
    return y


__all__ = ["silu_and_mul", "gelu_and_mul"]
