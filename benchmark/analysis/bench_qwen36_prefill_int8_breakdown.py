from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from minisgl.benchmark.perf import perf_cuda
from minisgl.quantization import (
    quantize_activation_per_token_int8,
    quantize_weight_per_channel_int8,
    supports_sgl_kernel_int8_linear,
)


def _require_sgl_kernel() -> None:
    os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
    try:
        import sgl_kernel  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "sgl_kernel is required for this breakdown benchmark because it is meant to "
            "measure the real online int8 path."
        ) from exc

    if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "int8_scaled_mm"):
        raise RuntimeError("torch.ops.sgl_kernel.int8_scaled_mm is not available")


def _sgl_kernel_int8_scaled_mm(
    x_q: torch.Tensor,
    qweight_t_col_major: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.int8_scaled_mm.default(
        x_q,
        qweight_t_col_major,
        x_scale,
        weight_scale,
        out_dtype,
        None,
    )


def _torch_int_mm_path(
    x_q: torch.Tensor,
    qweight_t_row_major: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    x_q_2d = x_q.view(-1, x_q.shape[-1]).contiguous()
    original_m = x_q_2d.shape[0]
    if original_m < 17:
        padded = torch.zeros((17, x_q_2d.shape[1]), dtype=x_q_2d.dtype, device=x_q_2d.device)
        padded[:original_m] = x_q_2d
        acc = torch._int_mm(padded, qweight_t_row_major)[:original_m].contiguous()
    else:
        acc = torch._int_mm(x_q_2d, qweight_t_row_major)
    out = acc.to(torch.float32)
    out = out * x_scale.view(-1, 1).to(torch.float32)
    out = out * weight_scale.view(1, -1).to(torch.float32)
    return out.to(out_dtype)


def _epilogue_only(
    acc: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out = acc.to(torch.float32)
    out = out * x_scale.view(-1, 1).to(torch.float32)
    out = out * weight_scale.view(1, -1).to(torch.float32)
    return out.to(out_dtype)


def _bench_bucket(
    *,
    name: str,
    tokens: int,
    input_size: int,
    output_size: int,
    dtype: torch.dtype,
    repetitions: int,
) -> dict:
    x = torch.randn(tokens, input_size, device="cuda", dtype=dtype)
    w = torch.randn(output_size, input_size, device="cuda", dtype=dtype)

    quant_ms = perf_cuda(
        lambda: quantize_activation_per_token_int8(x),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )
    x_q, x_scale = quantize_activation_per_token_int8(x)

    bf16_linear_ms = perf_cuda(
        lambda: F.linear(x, w),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    supported = supports_sgl_kernel_int8_linear(input_size, output_size)
    if not supported:
        return {
            "bucket": name,
            "tokens": tokens,
            "input_size": input_size,
            "output_size": output_size,
            "supported_by_sgl_kernel_int8": False,
            "bf16_linear_ms": bf16_linear_ms,
            "quant_ms": quant_ms,
        }

    qweight_t_col_major, weight_scale = quantize_weight_per_channel_int8(w)
    qweight_t_row_major = qweight_t_col_major.contiguous()

    sgl_int8_total_ms = perf_cuda(
        lambda: _sgl_kernel_int8_scaled_mm(
            x_q,
            qweight_t_col_major,
            x_scale,
            weight_scale,
            dtype,
        ),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    torch_int8_total_ms = perf_cuda(
        lambda: _torch_int_mm_path(
            x_q,
            qweight_t_row_major,
            x_scale,
            weight_scale,
            dtype,
        ),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    x_q_2d = x_q.view(-1, x_q.shape[-1]).contiguous()
    x_scale_2d = x_scale.view(-1, x_scale.shape[-1]).contiguous()
    weight_scale_1d = weight_scale.view(-1).contiguous()
    original_m = x_q_2d.shape[0]
    if original_m < 17:
        padded = torch.zeros((17, x_q_2d.shape[1]), dtype=x_q_2d.dtype, device=x_q_2d.device)
        padded[:original_m] = x_q_2d
        acc = torch._int_mm(padded, qweight_t_row_major)[:original_m].contiguous()
    else:
        acc = torch._int_mm(x_q_2d, qweight_t_row_major)

    torch_int8_acc_ms = perf_cuda(
        lambda: (
            torch._int_mm(
                torch.cat(
                    [
                        x_q_2d,
                        torch.zeros(
                            (max(17 - original_m, 0), x_q_2d.shape[1]),
                            dtype=x_q_2d.dtype,
                            device=x_q_2d.device,
                        ),
                    ],
                    dim=0,
                )[: max(original_m, 17)],
                qweight_t_row_major,
            )[:original_m].contiguous()
            if original_m < 17
            else torch._int_mm(x_q_2d, qweight_t_row_major)
        ),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    epilogue_ms = perf_cuda(
        lambda: _epilogue_only(acc, x_scale_2d, weight_scale_1d, dtype),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    return {
        "bucket": name,
        "tokens": tokens,
        "input_size": input_size,
        "output_size": output_size,
        "supported_by_sgl_kernel_int8": True,
        "bf16_linear_ms": bf16_linear_ms,
        "quant_ms": quant_ms,
        "sgl_int8_total_ms": sgl_int8_total_ms,
        "torch_int8_total_ms": torch_int8_total_ms,
        "torch_int8_acc_ms": torch_int8_acc_ms,
        "epilogue_ms_proxy": epilogue_ms,
        "sgl_int8_speedup_vs_bf16": bf16_linear_ms / sgl_int8_total_ms if sgl_int8_total_ms > 0 else None,
        "torch_int8_speedup_vs_bf16": bf16_linear_ms / torch_int8_total_ms if torch_int8_total_ms > 0 else None,
        "quant_share_vs_sgl_total": quant_ms / sgl_int8_total_ms if sgl_int8_total_ms > 0 else None,
        "epilogue_proxy_share_vs_torch_total": epilogue_ms / torch_int8_total_ms if torch_int8_total_ms > 0 else None,
        "notes": (
            "sgl_int8_total_ms is the real online fused kernel path. "
            "epilogue_ms_proxy is measured from the torch._int_mm reference path and is only "
            "a proxy for dequant/scale/cast cost; it is not a precise internal phase split of "
            "sgl_kernel.int8_scaled_mm."
        ),
    }


def main() -> None:
    _require_sgl_kernel()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 1024],
        help="Token counts to benchmark. Use 1 for decode-like and 1024 for prefill-like.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    hidden = 2048
    q_dim = 16 * 256
    kv_dim = 2 * 256
    full_qkv_out = q_dim * 2 + kv_dim * 2
    linear_attn_qkvz_out = 16 * 128 + 16 * 128 + 32 * 128 + 32 * 128
    linear_attn_ba_out = 32 + 32
    linear_attn_out_proj_in = 32 * 128
    moe_router_out = 256
    shared_gate_out = 1
    shared_gate_up_out = 512 + 512
    shared_down_in = 512

    buckets = [
        ("full_attn_qkv_proj", hidden, full_qkv_out),
        ("full_attn_o_proj", q_dim, hidden),
        ("linear_attn_in_proj_qkvz", hidden, linear_attn_qkvz_out),
        ("linear_attn_in_proj_ba", hidden, linear_attn_ba_out),
        ("linear_attn_out_proj", linear_attn_out_proj_in, hidden),
        ("moe_router_gate", hidden, moe_router_out),
        ("shared_expert_gate", hidden, shared_gate_out),
        ("shared_expert_gate_up", hidden, shared_gate_up_out),
        ("shared_expert_down", shared_down_in, hidden),
        ("dense_gate_up_proj", hidden, 512 + 512),
        ("dense_down_proj", 512, hidden),
    ]

    print(
        json.dumps(
            {
                "device": "cuda",
                "dtype": args.dtype,
                "repetitions": args.repetitions,
                "tokens": args.tokens,
                "measurement_note": "sgl_int8_total_ms measures the real online fused int8 path. quant_ms is measured separately. epilogue_ms_proxy comes from the torch._int_mm reference path.",
            }
        )
    )
    for tokens in args.tokens:
        for name, input_size, output_size in buckets:
            print(
                json.dumps(
                    _bench_bucket(
                        name=name,
                        tokens=tokens,
                        input_size=input_size,
                        output_size=output_size,
                        dtype=dtype,
                        repetitions=args.repetitions,
                    )
                )
            )


if __name__ == "__main__":
    main()
