from __future__ import annotations

from typing import Iterable

import torch

SUPPORTED_QUANTIZATION = {None, "w8a8_int8"}

_CURRENT_QUANTIZATION: str | None = None


def set_quantization(quantization: str | None) -> None:
    if quantization not in SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    global _CURRENT_QUANTIZATION
    _CURRENT_QUANTIZATION = quantization


def get_quantization() -> str | None:
    return _CURRENT_QUANTIZATION


def is_w8a8_int8_enabled() -> bool:
    return _CURRENT_QUANTIZATION == "w8a8_int8"


def quantize_weight_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape={tuple(weight.shape)}")
    weight_fp32 = weight.to(torch.float32)
    scales = weight_fp32.abs().amax(dim=1, keepdim=True).clamp_min_(1e-10) / 127.0
    qweight = torch.round(weight_fp32 / scales).clamp_(-128, 127).to(torch.int8)
    return qweight.t().contiguous(), scales.contiguous()


def quantize_activation_per_token_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 2:
        raise ValueError(f"Expected an input with ndim >= 2, got ndim={x.ndim}")
    x = x.contiguous()
    x_fp32 = x.to(torch.float32)
    scales = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-10) / 127.0
    x_q = torch.round(x_fp32 / scales).clamp_(-128, 127).to(torch.int8)
    return x_q, scales.contiguous()


def apply_w8a8_int8_linear(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from sgl_kernel import int8_scaled_mm

    x_q, x_scale = quantize_activation_per_token_int8(x)
    x_q_2d = x_q.view(-1, x_q.shape[-1])
    x_scale_2d = x_scale.view(-1, x_scale.shape[-1])
    output = int8_scaled_mm(
        x_q_2d,
        qweight_t,
        x_scale_2d,
        weight_scale,
        out_dtype=x.dtype,
        bias=bias,
    )
    return output.view(*x.shape[:-1], qweight_t.shape[1])


def process_weights_after_loading(root: object) -> None:
    for module in _iter_ops(root, visited=set()):
        fn = getattr(module, "process_weights_after_loading", None)
        if callable(fn):
            fn()


def _iter_ops(root: object, visited: set[int]) -> Iterable[object]:
    root_id = id(root)
    if root_id in visited:
        return
    visited.add(root_id)

    if isinstance(root, torch.Tensor):
        return

    if _is_minisgl_object(root):
        yield root
        for name, value in root.__dict__.items():
            if name.startswith("_"):
                continue
            yield from _iter_ops(value, visited)
        return

    if isinstance(root, dict):
        for value in root.values():
            yield from _iter_ops(value, visited)
        return

    if isinstance(root, (list, tuple)):
        for value in root:
            yield from _iter_ops(value, visited)


def _is_minisgl_object(root: object) -> bool:
    module_name = getattr(root.__class__, "__module__", "")
    return hasattr(root, "__dict__") and module_name.startswith("minisgl")
