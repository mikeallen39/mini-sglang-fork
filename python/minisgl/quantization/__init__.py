from __future__ import annotations

from typing import Iterable

import torch
from minisgl.core import get_global_ctx
from minisgl.env import ENV
from minisgl.kernel import decode_quant_int8_gemm_triton, weight_only_int8_gemm_triton
from minisgl.utils.logger import init_logger

SUPPORTED_QUANTIZATION = {None, "w8a8_int8", "w8a8_int8_moe_only", "w8a16_int8"}

_CURRENT_QUANTIZATION: str | None = None
logger = init_logger(__name__)
_INT8_DENSE_PROFILE_INTERVAL = 100
_INT8_DENSE_PROFILE = {
    "activation_quant_ms": 0.0,
    "int8_gemm_ms": 0.0,
    "count_quant": 0,
    "count_gemm": 0,
}


def _can_profile_int8_dense(x: torch.Tensor) -> bool:
    return (
        ENV.PROFILE_INT8_DENSE.value
        and x.is_cuda
        and not torch.cuda.is_current_stream_capturing()
    )


def set_quantization(quantization: str | None) -> None:
    if quantization not in SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    global _CURRENT_QUANTIZATION
    _CURRENT_QUANTIZATION = quantization


def get_quantization() -> str | None:
    return _CURRENT_QUANTIZATION


def is_w8a8_int8_enabled() -> bool:
    return _CURRENT_QUANTIZATION in {"w8a8_int8", "w8a8_int8_moe_only"}


def is_w8a16_int8_enabled() -> bool:
    return _CURRENT_QUANTIZATION == "w8a16_int8"


def is_int8_weight_quant_enabled() -> bool:
    return _CURRENT_QUANTIZATION in {"w8a8_int8", "w8a8_int8_moe_only", "w8a16_int8"}


def is_w8a8_int8_full_linear_enabled() -> bool:
    return _CURRENT_QUANTIZATION == "w8a8_int8"


def is_w8a8_int8_moe_only_enabled() -> bool:
    return _CURRENT_QUANTIZATION == "w8a8_int8_moe_only"


def is_w8a16_int8_full_linear_enabled() -> bool:
    return _CURRENT_QUANTIZATION == "w8a16_int8"


def supports_sgl_kernel_int8_linear(input_size: int, output_size: int) -> bool:
    # The environment sgl_kernel int8 GEMM requires:
    # - K % 16 == 0
    # - N % 8 == 0
    # It also expects the quantized weight to be fed as a column-major [K, N]
    # tensor. We keep the shape check here and let runtime fall back if a
    # particular backend rejects the layout for other reasons.
    return input_size % 16 == 0 and output_size % 8 == 0


def quantize_weight_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape={tuple(weight.shape)}")
    weight_fp32 = weight.to(torch.float32)
    scales = weight_fp32.abs().amax(dim=1, keepdim=True).clamp_min_(1e-10) / 127.0
    qweight = torch.round(weight_fp32 / scales).clamp_(-128, 127).to(torch.int8)
    # sgl_kernel.int8_scaled_mm expects mat_b to be a column-major [K, N]
    # tensor, i.e. stride(0) == 1 after transpose.
    return qweight.t().contiguous().t().contiguous().t(), scales.contiguous()


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
    if not x.is_cuda:
        raise RuntimeError("W8A8 activation quantization requires CUDA sgl_kernel")
    if x.ndim != 2:
        raise RuntimeError(
            f"W8A8 activation quantization only supports 2D CUDA tensors via sgl_kernel, got shape={tuple(x.shape)}"
        )
    x = x.contiguous()
    profile = _can_profile_int8_dense(x)
    e0 = e1 = None
    if profile:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
    x_q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    try:
        from sgl_kernel import sgl_per_token_group_quant_int8

        sgl_per_token_group_quant_int8(
            x,
            x_q,
            scales,
            x.shape[1],
            1e-10,
            -128,
            127,
        )
    except Exception as exc:
        raise RuntimeError(
            "sgl_kernel.sgl_per_token_group_quant_int8 failed; W8A8 activation quantization "
            "does not allow fallback paths"
        ) from exc
    if profile:
        assert e0 is not None and e1 is not None
        e1.record()
        e1.synchronize()
        _record_int8_dense_profile("activation_quant_ms", e0.elapsed_time(e1), "count_quant")
    return x_q, scales


def _apply_int8_scaled_mm(
    x_q_2d: torch.Tensor,
    x_scale_2d: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    profile = _can_profile_int8_dense(x_q_2d)
    e0 = e1 = None
    if profile:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
    try:
        from sgl_kernel import int8_scaled_mm as _int8_scaled_mm
    except ImportError:
        try:
            from sgl_kernel.gemm import int8_scaled_mm as _int8_scaled_mm
        except ImportError:
            raise RuntimeError(
                "sgl_kernel.int8_scaled_mm is unavailable; W8A8 int8 GEMM does not allow fallback paths"
            ) from None

    try:
        output = _int8_scaled_mm(
            x_q_2d,
            qweight_t,
            x_scale_2d,
            weight_scale,
            out_dtype=out_dtype,
            bias=bias,
        )
    except Exception as exc:
        raise RuntimeError(
            "sgl_kernel.int8_scaled_mm failed; W8A8 int8 GEMM does not allow fallback paths"
        ) from exc
    if profile:
        assert e0 is not None and e1 is not None
        e1.record()
        e1.synchronize()
        _record_int8_dense_profile("int8_gemm_ms", e0.elapsed_time(e1), "count_gemm")
    return output


def _can_use_decode_fused_quant_gemm(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> bool:
    if not ENV.W8A8_DECODE_FUSED_QUANT_GEMM.value:
        return False
    if out_dtype != torch.bfloat16:
        return False
    if not (x.is_cuda and x.ndim == 2 and x.is_contiguous()):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not (qweight_t.is_cuda and qweight_t.ndim == 2 and qweight_t.is_contiguous()):
        return False
    if qweight_t.dtype != torch.int8:
        return False
    if not (weight_scale.is_cuda and weight_scale.ndim == 2 and weight_scale.is_contiguous()):
        return False
    if weight_scale.dtype != torch.float32:
        return False
    if x.shape[1] != qweight_t.shape[0]:
        return False
    if weight_scale.shape != (qweight_t.shape[1], 1):
        return False
    if x.shape[0] > 8:
        return False
    try:
        batch = get_global_ctx().batch
    except Exception:
        return False
    return batch.is_decode


def _apply_decode_fused_quant_gemm(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if bias is not None:
        if not (bias.is_cuda and bias.ndim == 1 and bias.dtype == out_dtype and bias.is_contiguous()):
            raise RuntimeError("Decode fused quant+gemm requires a contiguous CUDA bias matching out_dtype")
    output = torch.empty((x.shape[0], qweight_t.shape[1]), device=x.device, dtype=out_dtype)
    decode_quant_int8_gemm_triton(
        x,
        qweight_t,
        weight_scale,
        output,
        bias=bias,
    )
    return output


def _record_int8_dense_profile(key: str, value_ms: float, count_key: str) -> None:
    _INT8_DENSE_PROFILE[key] += value_ms
    _INT8_DENSE_PROFILE[count_key] += 1
    count = _INT8_DENSE_PROFILE[count_key]
    if count % _INT8_DENSE_PROFILE_INTERVAL != 0:
        return
    quant_count = max(_INT8_DENSE_PROFILE["count_quant"], 1)
    gemm_count = max(_INT8_DENSE_PROFILE["count_gemm"], 1)
    logger.info_rank0(
        "Int8Dense profile avg: activation_quant=%.4f ms over %d calls, int8_gemm=%.4f ms over %d calls",
        _INT8_DENSE_PROFILE["activation_quant_ms"] / quant_count,
        quant_count,
        _INT8_DENSE_PROFILE["int8_gemm_ms"] / gemm_count,
        gemm_count,
    )
    _INT8_DENSE_PROFILE["activation_quant_ms"] = 0.0
    _INT8_DENSE_PROFILE["int8_gemm_ms"] = 0.0
    _INT8_DENSE_PROFILE["count_quant"] = 0
    _INT8_DENSE_PROFILE["count_gemm"] = 0


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
    if _can_use_decode_fused_quant_gemm(x, qweight_t, weight_scale, x.dtype):
        return _apply_decode_fused_quant_gemm(
            x,
            qweight_t,
            weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
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


def apply_w8a16_int8_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"Expected an input with ndim >= 2, got ndim={x.ndim}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"W8A16 only supports fp16/bf16 activations, got dtype={x.dtype}")
    if qweight.dtype != torch.int8:
        raise RuntimeError(f"W8A16 expects int8 weights, got dtype={qweight.dtype}")
    if not x.is_cuda or not qweight.is_cuda or not weight_scale.is_cuda:
        raise RuntimeError("W8A16 requires CUDA tensors")
    x_2d = x.view(-1, x.shape[-1]).contiguous()
    output = torch.empty(
        (x_2d.shape[0], qweight.shape[0]),
        device=x.device,
        dtype=x.dtype,
    )
    try:
        weight_only_int8_gemm_triton(
            x_2d,
            qweight.contiguous(),
            weight_scale.contiguous(),
            output,
            bias=bias,
        )
    except Exception as exc:
        raise RuntimeError("weight_only_int8_gemm_triton failed in W8A16 path") from exc
    return output.view(*x.shape[:-1], qweight.shape[0])


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
