from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from minisgl.benchmark.perf import perf_cuda
from minisgl.quantization import quantize_activation_per_token_int8


def _run_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_activation_per_token_int8(x)


def _run_int8_acc_only(x_q: torch.Tensor, qweight_t: torch.Tensor) -> torch.Tensor:
    x_q_2d = x_q.view(-1, x_q.shape[-1]).contiguous()
    qweight_t = qweight_t.contiguous()
    original_m = x_q_2d.shape[0]
    if original_m < 17:
        padded = torch.zeros((17, x_q_2d.shape[1]), dtype=x_q_2d.dtype, device=x_q_2d.device)
        padded[:original_m] = x_q_2d
        acc = torch._int_mm(padded, qweight_t)[:original_m].contiguous()
    else:
        acc = torch._int_mm(x_q_2d, qweight_t)
    return acc


def _run_epilogue(
    acc: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out = acc.to(torch.float32)
    out = out * x_scale.view(-1, 1).to(torch.float32)
    out = out * weight_scale.view(1, -1).to(torch.float32)
    return out.to(out_dtype)


def bench_bucket(
    *,
    name: str,
    tokens: int,
    input_size: int,
    output_size: int,
    dtype: torch.dtype,
    repetitions: int,
) -> dict:
    if input_size % 16 != 0 or output_size % 8 != 0:
        return {
            "bucket": name,
            "tokens": tokens,
            "input_size": input_size,
            "output_size": output_size,
            "supported_by_sgl_kernel_int8": False,
        }

    x = torch.randn(tokens, input_size, device="cuda", dtype=dtype)
    w = torch.randn(output_size, input_size, device="cuda", dtype=dtype)

    w_fp32 = w.to(torch.float32)
    weight_scale = w_fp32.abs().amax(dim=1, keepdim=True).clamp_min_(1e-10) / 127.0
    qweight_t = torch.round(w_fp32 / weight_scale).clamp_(-128, 127).to(torch.int8).t().contiguous()

    quant_ms = perf_cuda(
        lambda: _run_quant(x),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    x_q, x_scale = _run_quant(x)
    x_scale_2d = x_scale.view(-1, 1).contiguous()
    weight_scale_1d = weight_scale.view(-1).contiguous()

    int8_acc_ms = perf_cuda(
        lambda: _run_int8_acc_only(x_q, qweight_t),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    acc = _run_int8_acc_only(x_q, qweight_t)

    epilogue_ms = perf_cuda(
        lambda: _run_epilogue(acc, x_scale_2d, weight_scale_1d, dtype),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    bf16_linear_ms = perf_cuda(
        lambda: F.linear(x, w),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    )

    int8_total_ms = perf_cuda(
        lambda: _run_epilogue(_run_int8_acc_only(x_q, qweight_t), x_scale_2d, weight_scale_1d, dtype),
        repetitions=repetitions,
        cuda_graph_repetitions=None,
    ) + quant_ms

    return {
        "bucket": name,
        "tokens": tokens,
        "input_size": input_size,
        "output_size": output_size,
        "supported_by_sgl_kernel_int8": True,
        "quant_ms": quant_ms,
        "int8_acc_ms": int8_acc_ms,
        "epilogue_ms": epilogue_ms,
        "bf16_linear_ms": bf16_linear_ms,
        "int8_total_ms": int8_total_ms,
        "int8_acc_speedup_vs_bf16": bf16_linear_ms / int8_acc_ms if int8_acc_ms > 0 else None,
        "int8_total_speedup_vs_bf16": bf16_linear_ms / int8_total_ms if int8_total_ms > 0 else None,
        "quant_share_of_total": quant_ms / int8_total_ms if int8_total_ms > 0 else None,
        "epilogue_share_of_total": epilogue_ms / int8_total_ms if int8_total_ms > 0 else None,
        "gemm_share_of_total": int8_acc_ms / int8_total_ms if int8_total_ms > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--repetitions", type=int, default=30)
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

    print(json.dumps({"device": "cuda", "dtype": args.dtype, "repetitions": args.repetitions}))
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
