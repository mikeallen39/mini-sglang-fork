from __future__ import annotations

import argparse
import json

import torch

from minisgl.benchmark.perf import perf_cuda
from minisgl.kernel import gemma_rmsnorm_quant_int8_triton
from minisgl.quantization import quantize_activation_per_token_int8


def gemma_rmsnorm_then_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_fp = x.to(compute_dtype)
    variance = x_fp.square().mean(dim=-1, keepdim=True)
    y = x_fp * torch.rsqrt(variance + eps)
    y = y * (1.0 + weight.to(compute_dtype))
    return quantize_activation_per_token_int8(y.to(x.dtype))


def gemma_rmsnorm_quant_fused_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_fp = x.to(compute_dtype)
    variance = x_fp.square().mean(dim=-1, keepdim=True)
    y = x_fp * torch.rsqrt(variance + eps)
    y = y * (1.0 + weight.to(compute_dtype))
    scales = y.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-10) / 127.0
    q = torch.round(y / scales).clamp_(-128, 127).to(torch.int8)
    return q, scales.contiguous()


def gemma_rmsnorm_quant_fused_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    gemma_rmsnorm_quant_int8_triton(x, weight, eps, q, scales)
    return q, scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda"
    weight = torch.randn(args.hidden_size, device=device, dtype=dtype)

    print(
        json.dumps(
            {
                "device": device,
                "dtype": args.dtype,
                "hidden_size": args.hidden_size,
                "repetitions": args.repetitions,
                "eps": args.eps,
            }
        )
    )

    for tokens in args.tokens:
        x = torch.randn(tokens, args.hidden_size, device=device, dtype=dtype)

        baseline_ms = perf_cuda(
            lambda: gemma_rmsnorm_then_quant(x, weight, args.eps),
            repetitions=args.repetitions,
            cuda_graph_repetitions=None,
        )
        fused_ref_ms = perf_cuda(
            lambda: gemma_rmsnorm_quant_fused_ref(x, weight, args.eps),
            repetitions=args.repetitions,
            cuda_graph_repetitions=None,
        )
        fused_triton_ms = perf_cuda(
            lambda: gemma_rmsnorm_quant_fused_triton(x, weight, args.eps),
            repetitions=args.repetitions,
            cuda_graph_repetitions=None,
        )

        print(
            json.dumps(
                {
                    "tokens": tokens,
                    "baseline_ms": baseline_ms,
                    "fused_ref_ms": fused_ref_ms,
                    "fused_triton_ms": fused_triton_ms,
                    "speedup_ref": baseline_ms / fused_ref_ms if fused_ref_ms > 0 else None,
                    "speedup_triton": baseline_ms / fused_triton_ms if fused_triton_ms > 0 else None,
                }
            )
        )


if __name__ == "__main__":
    main()
