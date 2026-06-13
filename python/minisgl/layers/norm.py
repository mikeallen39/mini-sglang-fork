from __future__ import annotations

import os
from typing import Tuple

import torch
from minisgl.env import ENV
from minisgl.utils.logger import init_logger

from .base import BaseOP

logger = init_logger(__name__)
_INT8_DENSE_PROFILE_INTERVAL = 100
_GEMMA_RMSNORM_PROFILE = {
    "gemma_rmsnorm_ms": 0.0,
    "count": 0,
}


def _can_profile_int8_dense(x: torch.Tensor) -> bool:
    return (
        ENV.PROFILE_INT8_DENSE.value
        and x.is_cuda
        and not torch.cuda.is_current_stream_capturing()
    )


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
        if not _can_profile_int8_dense(x):
            return _torch_gemma_rmsnorm(x, self.weight, self.eps)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        y = _torch_gemma_rmsnorm(x, self.weight, self.eps)
        e1.record()
        e1.synchronize()
        _GEMMA_RMSNORM_PROFILE["gemma_rmsnorm_ms"] += e0.elapsed_time(e1)
        _GEMMA_RMSNORM_PROFILE["count"] += 1
        if _GEMMA_RMSNORM_PROFILE["count"] % _INT8_DENSE_PROFILE_INTERVAL == 0:
            count = _GEMMA_RMSNORM_PROFILE["count"]
            logger.info_rank0(
                "Int8Dense profile avg: gemma_rmsnorm=%.4f ms over %d calls",
                _GEMMA_RMSNORM_PROFILE["gemma_rmsnorm_ms"] / count,
                count,
            )
            _GEMMA_RMSNORM_PROFILE["gemma_rmsnorm_ms"] = 0.0
            _GEMMA_RMSNORM_PROFILE["count"] = 0
        return y

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
            if not _can_profile_int8_dense(x):
                return _torch_gemma_rmsnorm(x, self.weight, self.eps), x
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            y = _torch_gemma_rmsnorm(x, self.weight, self.eps)
            e1.record()
            e1.synchronize()
            _GEMMA_RMSNORM_PROFILE["gemma_rmsnorm_ms"] += e0.elapsed_time(e1)
            _GEMMA_RMSNORM_PROFILE["count"] += 1
            if _GEMMA_RMSNORM_PROFILE["count"] % _INT8_DENSE_PROFILE_INTERVAL == 0:
                count = _GEMMA_RMSNORM_PROFILE["count"]
                logger.info_rank0(
                    "Int8Dense profile avg: gemma_rmsnorm=%.4f ms over %d calls",
                    _GEMMA_RMSNORM_PROFILE["gemma_rmsnorm_ms"] / count,
                    count,
                )
                _GEMMA_RMSNORM_PROFILE["gemma_rmsnorm_ms"] = 0.0
                _GEMMA_RMSNORM_PROFILE["count"] = 0
            return y, x

        merged = x + residual
        if not _can_profile_int8_dense(merged):
            normalized = _torch_gemma_rmsnorm(merged, self.weight, self.eps)
            return normalized, merged
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normalized = _torch_gemma_rmsnorm(merged, self.weight, self.eps)
        e1.record()
        e1.synchronize()
        _GEMMA_RMSNORM_PROFILE["gemma_rmsnorm_ms"] += e0.elapsed_time(e1)
        _GEMMA_RMSNORM_PROFILE["count"] += 1
        if _GEMMA_RMSNORM_PROFILE["count"] % _INT8_DENSE_PROFILE_INTERVAL == 0:
            count = _GEMMA_RMSNORM_PROFILE["count"]
            logger.info_rank0(
                "Int8Dense profile avg: gemma_rmsnorm=%.4f ms over %d calls",
                _GEMMA_RMSNORM_PROFILE["gemma_rmsnorm_ms"] / count,
                count,
            )
            _GEMMA_RMSNORM_PROFILE["gemma_rmsnorm_ms"] = 0.0
            _GEMMA_RMSNORM_PROFILE["count"] = 0
        return normalized, merged
