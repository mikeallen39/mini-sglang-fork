from __future__ import annotations

import os
from typing import Tuple

import torch

from .base import BaseOP


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    in_dtype = x.dtype
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_fp = x.to(compute_dtype)
    variance = x_fp.square().mean(dim=-1, keepdim=True)
    y = x_fp * torch.rsqrt(variance + eps)
    y = y * weight.to(compute_dtype)
    return y.to(in_dtype)


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self._use_flashinfer = os.environ.get("MINISGL_USE_FLASHINFER_RMSNORM", "0") == "1"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_flashinfer:
            from flashinfer import rmsnorm

            return rmsnorm(x, self.weight, self.eps)
        return _torch_rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if self._use_flashinfer:
            from flashinfer import rmsnorm

            rmsnorm(x, self.weight, self.eps, out=x)
            return
        x.copy_(_torch_rmsnorm(x, self.weight, self.eps))


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self._use_flashinfer = os.environ.get("MINISGL_USE_FLASHINFER_RMSNORM", "0") == "1"

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            if self._use_flashinfer:
                from flashinfer import rmsnorm

                return rmsnorm(x, self.weight, self.eps), x
            return _torch_rmsnorm(x, self.weight, self.eps), x

        if self._use_flashinfer:
            from flashinfer import fused_add_rmsnorm

            fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual

        merged = x + residual
        normalized = _torch_rmsnorm(merged, self.weight, self.eps)
        return normalized, merged


def _torch_gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    in_dtype = x.dtype
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_fp = x.to(compute_dtype)
    variance = x_fp.square().mean(dim=-1, keepdim=True)
    y = x_fp * torch.rsqrt(variance + eps)
    y = y * (1.0 + weight.to(compute_dtype))
    return y.to(in_dtype)


class GemmaRMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _torch_gemma_rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        x.copy_(_torch_gemma_rmsnorm(x, self.weight, self.eps))


class GemmaRMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _torch_gemma_rmsnorm(x, self.weight, self.eps), x

        merged = x + residual
        normalized = _torch_gemma_rmsnorm(merged, self.weight, self.eps)
        return normalized, merged
