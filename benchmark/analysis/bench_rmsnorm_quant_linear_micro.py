from __future__ import annotations

import argparse
import json

import torch

from minisgl.benchmark.perf import perf_cuda
from minisgl.quantization import (
    _apply_int8_scaled_mm,
    quantize_activation_per_token_int8,
    quantize_weight_per_channel_int8,
)


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_fp = x.to(compute_dtype)
    variance = x_fp.square().mean(dim=-1, keepdim=True)
    y = x_fp * torch.rsqrt(variance + eps)
    y = y * (1.0 + weight.to(compute_dtype))
    return y.to(x.dtype)


def baseline_path(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y = gemma_rmsnorm(x, norm_weight, eps)
    y_q, y_scale = quantize_activation_per_token_int8(y)
    return _apply_int8_scaled_mm(
        y_q,
        y_scale,
        qweight_t,
        weight_scale,
        out_dtype=x.dtype,
        bias=None,
    )


def fused_ref_path(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_fp = x.to(compute_dtype)
    variance = x_fp.square().mean(dim=-1, keepdim=True)
    y = x_fp * torch.rsqrt(variance + eps)
    y = y * (1.0 + norm_weight.to(compute_dtype))
    y_scale = y.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-10) / 127.0
    y_q = torch.round(y / y_scale).clamp_(-128, 127).to(torch.int8)
    return _apply_int8_scaled_mm(
        y_q,
        y_scale.contiguous(),
        qweight_t,
        weight_scale,
        out_dtype=x.dtype,
        bias=None,
    )


def run_case(
    *,
    name: str,
    tokens: int,
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
    eps: float,
    repetitions: int,
) -> dict:
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=dtype)
    norm_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    w = torch.randn(output_size, hidden_size, device="cuda", dtype=dtype)
    qweight_t, weight_scale = quantize_weight_per_channel_int8(w)

    baseline_ms = perf_cuda(
        lambda: baseline_path(x, norm_weight, qweight_t, weight_scale, eps),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )
    fused_ref_ms = perf_cuda(
        lambda: fused_ref_path(x, norm_weight, qweight_t, weight_scale, eps),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )
    return {
        "case": name,
        "tokens": tokens,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "baseline_ms": baseline_ms,
        "fused_ref_ms": fused_ref_ms,
        "speedup": baseline_ms / fused_ref_ms if fused_ref_ms > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    hidden_size = 2048
    cases = [
        ("router", 256),
        ("shared_expert_gate", 1),
        ("linear_attn_in_proj_qkvz", 8192),
    ]

    print(
        json.dumps(
            {
                "device": "cuda",
                "dtype": args.dtype,
                "hidden_size": hidden_size,
                "repetitions": args.repetitions,
                "eps": args.eps,
            }
        )
    )

    for tokens in args.tokens:
        for name, output_size in cases:
            print(
                json.dumps(
                    run_case(
                        name=name,
                        tokens=tokens,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        dtype=dtype,
                        eps=args.eps,
                        repetitions=args.repetitions,
                    )
                )
            )


if __name__ == "__main__":
    main()
