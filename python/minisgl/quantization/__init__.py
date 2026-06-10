from __future__ import annotations

from typing import Iterable

import torch

SUPPORTED_QUANTIZATION = {None, "w8a8_int8", "w8a8_int8_moe_only"}

_CURRENT_QUANTIZATION: str | None = None


def set_quantization(quantization: str | None) -> None:
    if quantization not in SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    global _CURRENT_QUANTIZATION
    _CURRENT_QUANTIZATION = quantization


def get_quantization() -> str | None:
    return _CURRENT_QUANTIZATION


def is_w8a8_int8_enabled() -> bool:
    return _CURRENT_QUANTIZATION in {"w8a8_int8", "w8a8_int8_moe_only"}


def is_w8a8_int8_full_linear_enabled() -> bool:
    return _CURRENT_QUANTIZATION == "w8a8_int8"


def is_w8a8_int8_moe_only_enabled() -> bool:
    return _CURRENT_QUANTIZATION == "w8a8_int8_moe_only"


def quantize_weight_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape={tuple(weight.shape)}")
    weight_fp32 = weight.to(torch.float32)
    scales = weight_fp32.abs().amax(dim=1, keepdim=True).clamp_min_(1e-10) / 127.0
    qweight = torch.round(weight_fp32 / scales).clamp_(-128, 127).to(torch.int8)
    return qweight.t().contiguous(), scales.contiguous()


def quantize_weight_per_channel_int8_row_major(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape={tuple(weight.shape)}")
    weight_fp32 = weight.to(torch.float32)
    scales = weight_fp32.abs().amax(dim=1, keepdim=True).clamp_min_(1e-10) / 127.0
    qweight = torch.round(weight_fp32 / scales).clamp_(-128, 127).to(torch.int8)
    return qweight.contiguous(), scales.contiguous()


def quantize_activation_per_token_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 2:
        raise ValueError(f"Expected an input with ndim >= 2, got ndim={x.ndim}")
    x = x.contiguous()
    if x.is_cuda and x.ndim == 2:
        try:
            from minisgl.kernel.activation_quant import per_token_quant_int8_triton

            x_q = torch.empty_like(x, dtype=torch.int8)
            scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
            per_token_quant_int8_triton(x, x_q, scales)
            return x_q, scales
        except Exception:
            pass
    x_fp32 = x.to(torch.float32)
    scales = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-10) / 127.0
    x_q = torch.round(x_fp32 / scales).clamp_(-128, 127).to(torch.int8)
    return x_q, scales.contiguous()


def _apply_int8_scaled_mm(
    x_q_2d: torch.Tensor,
    x_scale_2d: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    int8_scaled_mm = None
    try:
        from .int8_cuda_ext import int8_scaled_mm as _int8_scaled_mm

        int8_scaled_mm = _int8_scaled_mm
    except ImportError:
        int8_scaled_mm = None

    if int8_scaled_mm is not None:
        output = int8_scaled_mm(
            x_q_2d,
            qweight_t,
            x_scale_2d,
            weight_scale,
            out_dtype=out_dtype,
            bias=bias,
        )
        if output is not None:
            return output

    try:
        from sgl_kernel import int8_scaled_mm as _int8_scaled_mm

        int8_scaled_mm = _int8_scaled_mm
    except ImportError:
        try:
            from sgl_kernel.gemm import int8_scaled_mm as _int8_scaled_mm

            if hasattr(torch.ops.sgl_kernel, "int8_scaled_mm"):
                int8_scaled_mm = _int8_scaled_mm
            else:
                int8_scaled_mm = None
        except ImportError:
            int8_scaled_mm = None

    if int8_scaled_mm is not None:
        return int8_scaled_mm(
            x_q_2d,
            qweight_t,
            x_scale_2d,
            weight_scale,
            out_dtype=out_dtype,
            bias=bias,
        )

    # Prefer PyTorch's int8 GEMM when the dedicated sgl_kernel op is
    # unavailable. torch._int_mm requires M > 16 on CUDA, so pad decode-size
    # batches to keep the fast path available for small token counts.
    use_torch_int_mm = (
        x_q_2d.is_cuda
        and hasattr(torch, "_int_mm")
        and x_q_2d.shape[1] % 16 == 0
        and qweight_t.shape[0] % 16 == 0
        and qweight_t.shape[1] > 0
        and qweight_t.shape[1] % 8 == 0
    )
    if use_torch_int_mm:
        try:
            padded_m = max(x_q_2d.shape[0], 17)
            if padded_m != x_q_2d.shape[0]:
                padded_x_q = torch.zeros(
                    (padded_m, x_q_2d.shape[1]),
                    dtype=x_q_2d.dtype,
                    device=x_q_2d.device,
                )
                padded_x_q[: x_q_2d.shape[0]] = x_q_2d
                output = torch._int_mm(padded_x_q, qweight_t)[: x_q_2d.shape[0]]
            else:
                output = torch._int_mm(x_q_2d, qweight_t)
            output = output.to(torch.float32)
        except Exception:
            output = torch.matmul(x_q_2d.to(torch.float32), qweight_t.to(torch.float32))
    else:
        # Conservative fallback for environments without CUDA int8 GEMM.
        output = torch.matmul(x_q_2d.to(torch.float32), qweight_t.to(torch.float32))
    output = output * x_scale_2d.to(torch.float32)
    output = output * weight_scale.view(1, -1).to(torch.float32)
    if bias is not None:
        output = output + bias.to(torch.float32)
    return output.to(out_dtype)


def apply_w8a8_int8_linear_from_prequantized(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    x_q_2d = x_q.view(-1, x_q.shape[-1])
    x_scale_2d = x_scale.view(-1, x_scale.shape[-1])
    output = _apply_int8_scaled_mm(
        x_q_2d,
        x_scale_2d,
        qweight_t,
        weight_scale,
        out_dtype,
        bias,
    )
    return output.view(*x_q.shape[:-1], qweight_t.shape[1])


def apply_w8a8_int8_linear(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    x_q, x_scale = quantize_activation_per_token_int8(x)
    return apply_w8a8_int8_linear_from_prequantized(
        x_q,
        x_scale,
        qweight_t,
        weight_scale,
        out_dtype=x.dtype,
        bias=bias,
    )


def apply_w8a8_int8_linear_with_prequantized_fallback(
    x: torch.Tensor,
    x_q: torch.Tensor | None,
    x_scale: torch.Tensor | None,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if x_q is None or x_scale is None:
        return apply_w8a8_int8_linear(x, qweight_t, weight_scale, bias)
    return apply_w8a8_int8_linear_from_prequantized(
        x_q,
        x_scale,
        qweight_t,
        weight_scale,
        out_dtype=x.dtype,
        bias=bias,
    )


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
