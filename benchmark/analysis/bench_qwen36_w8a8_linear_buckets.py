from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from minisgl.benchmark.perf import perf_cuda
from minisgl.quantization import (
    _apply_int8_scaled_mm,
    quantize_activation_per_token_int8,
    quantize_weight_per_channel_int8,
)


def bench_bucket(
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
    qweight_t, weight_scale = quantize_weight_per_channel_int8(w)

    quant_ms = perf_cuda(
        lambda: quantize_activation_per_token_int8(x),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    x_q, x_scale = quantize_activation_per_token_int8(x)

    int8_gemm_ms = perf_cuda(
        lambda: _apply_int8_scaled_mm(
            x_q,
            x_scale,
            qweight_t,
            weight_scale,
            out_dtype=dtype,
            bias=None,
        ),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    bf16_linear_ms = perf_cuda(
        lambda: F.linear(x, w),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    int8_linear_ms = perf_cuda(
        lambda: _apply_int8_scaled_mm(
            *quantize_activation_per_token_int8(x),
            qweight_t,
            weight_scale,
            out_dtype=dtype,
            bias=None,
        ),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    return {
        "bucket": name,
        "tokens": tokens,
        "input_size": input_size,
        "output_size": output_size,
        "quant_ms": quant_ms,
        "int8_gemm_ms": int8_gemm_ms,
        "bf16_linear_ms": bf16_linear_ms,
        "int8_linear_ms": int8_linear_ms,
        "int8_gemm_speedup_vs_bf16_linear": bf16_linear_ms / int8_gemm_ms if int8_gemm_ms > 0 else None,
        "int8_linear_speedup_vs_bf16_linear": bf16_linear_ms / int8_linear_ms if int8_linear_ms > 0 else None,
        "quant_share_of_int8_linear": quant_ms / int8_linear_ms if int8_linear_ms > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--repetitions", type=int, default=100)
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
            }
        )
    )
    for tokens in args.tokens:
        for name, input_size, output_size in buckets:
            print(
                json.dumps(
                    bench_bucket(
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
